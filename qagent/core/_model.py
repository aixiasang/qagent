import os
import uuid
import json
import hashlib
import asyncio
import logging
import numpy as np
from time import time
from datetime import datetime
from openai import AsyncOpenAI, AsyncStream
from itertools import accumulate
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncGenerator, Callable, Literal, Optional, Union


StreamCallback = Callable[["ChatResponse"], None]
CompleteCallback = Callable[["ChatResponse"], None]
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from openai import (
    APIError,
    OpenAIError,
    ConflictError,
    NotFoundError,
    APIStatusError,
    RateLimitError,
    APITimeoutError,
    BadRequestError,
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    UnprocessableEntityError,
    APIResponseValidationError,
    LengthFinishReasonError,
    InvalidWebhookSignatureError,
    ContentFilterFinishReasonError,
)
from ._exceptions import ModelError
from ._trace import generation_span, embedder_span, get_current_trace, SpanError

RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    APIError,
    APIResponseValidationError,
)

NON_RETRYABLE_EXCEPTIONS = [
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
    ConflictError,
    LengthFinishReasonError,
    InvalidWebhookSignatureError,
    ContentFilterFinishReasonError,
]

NON_RETRYABLE_EXCEPTIONS = tuple(NON_RETRYABLE_EXCEPTIONS)


@dataclass
class TextBlock:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageBlock:
    type: Literal["image"] = "image"
    url: Optional[str] = None
    base64: Optional[str] = None
    detail: Literal["auto", "low", "high"] = "auto"


@dataclass
class AudioBlock:
    type: Literal["audio"] = "audio"
    url: Optional[str] = None
    base64: Optional[str] = None
    format: str = "mp3"


@dataclass
class VideoBlock:
    type: Literal["video"] = "video"
    url: Optional[str] = None
    base64: Optional[str] = None


@dataclass
class FileBlock:
    type: Literal["file"] = "file"
    url: Optional[str] = None
    filename: Optional[str] = None


ContentBlock = Union[TextBlock, ImageBlock, AudioBlock, VideoBlock, FileBlock]


# todo: current not support file block and just test with ali multimodal model
def block_to_openai(block: ContentBlock) -> dict:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    elif isinstance(block, ImageBlock):
        if block.url:
            return {
                "type": "image_url",
                "image_url": {"url": block.url, "detail": block.detail},
            }
        elif block.base64:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{block.base64}",
                    "detail": block.detail,
                },
            }
    elif isinstance(block, AudioBlock):
        if block.url:
            return {
                "type": "input_audio",
                "input_audio": {"data": block.url, "format": block.format},
            }
        elif block.base64:
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": f"data:audio/{block.format};base64,{block.base64}",
                    "format": block.format,
                },
            }
    elif isinstance(block, VideoBlock):
        if block.url:
            return {"type": "video_url", "video_url": {"url": block.url}}
        elif block.base64:
            return {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{block.base64}"},
            }
    elif isinstance(block, FileBlock):
        filename = block.filename or "file"
        if block.url:
            return {"type": "text", "text": f"[File: {filename} - {block.url}]"}
        return {"type": "text", "text": f"[File: {filename}]"}

    return {"type": "text", "text": str(block)}


@dataclass
class MultimodalContent:
    blocks: list[ContentBlock] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "MultimodalContent":
        return cls(blocks=[TextBlock(text=text)])

    @classmethod
    def from_blocks(cls, blocks: list[ContentBlock]) -> "MultimodalContent":
        return cls(blocks=blocks)

    def add_text(self, text: str) -> None:
        self.blocks.append(TextBlock(text=text))

    def add_image(
        self,
        url: Optional[str] = None,
        base64: Optional[str] = None,
        detail: str = "auto",
    ) -> None:
        self.blocks.append(ImageBlock(url=url, base64=base64, detail=detail))

    def add_video(self, url: Optional[str] = None, base64: Optional[str] = None) -> None:
        self.blocks.append(VideoBlock(url=url, base64=base64))

    def add_audio(
        self,
        url: Optional[str] = None,
        base64: Optional[str] = None,
        format: str = "mp3",
    ) -> None:
        self.blocks.append(AudioBlock(url=url, base64=base64, format=format))

    def get_text_content(self) -> str:
        texts = []
        for block in self.blocks:
            if isinstance(block, TextBlock):
                texts.append(block.text)
        return "".join(texts)

    def get_blocks_by_type(self, block_type: str) -> list[ContentBlock]:
        return [b for b in self.blocks if b.type == block_type]

    def to_openai(self) -> Union[str, list[dict]]:
        if len(self.blocks) == 0:
            return ""

        if len(self.blocks) == 1 and isinstance(self.blocks[0], TextBlock):
            return self.blocks[0].text

        return [block_to_openai(block) for block in self.blocks]

    def __str__(self) -> str:
        return self.get_text_content()

    def __bool__(self) -> bool:
        return len(self.blocks) > 0


def normalize_content(
    content: Union[str, list[ContentBlock], MultimodalContent, None],
) -> MultimodalContent:
    if content is None or content == "":
        return MultimodalContent()

    if isinstance(content, str):
        return MultimodalContent.from_text(content)

    if isinstance(content, MultimodalContent):
        return content

    if isinstance(content, list):
        return MultimodalContent.from_blocks(content)

    return MultimodalContent.from_text(str(content))


@dataclass
class EmbedUsage:
    time: float
    token: int


@dataclass
class EmbedResponse:
    source: Literal["api", "cache"]
    embedding: list[float]
    usage: Optional[EmbedUsage] = field(default_factory=lambda: EmbedUsage(time=0, token=0))

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}

    @classmethod
    def from_dict(cls, data: dict) -> "EmbedResponse":
        return cls(**data)


@dataclass
class ToolCall:
    fn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    fn_name: str = field(default_factory=lambda: "")
    fn_args: Union[dict, str] = field(default_factory=lambda: {})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        return cls(**data)

    def to_openai(self) -> dict:
        args = self.fn_args if isinstance(self.fn_args, str) else json.dumps(self.fn_args)
        return {
            "id": self.fn_id,
            "type": "function",
            "function": {"name": self.fn_name, "arguments": args},
        }

    def get_args_dict(self) -> dict:
        if isinstance(self.fn_args, dict):
            return self.fn_args
        try:
            return json.loads(self.fn_args) if self.fn_args else {}
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass
class ToolResult:
    fn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    fn_name: str = field(default_factory=lambda: "")
    fn_args: dict = field(default_factory=lambda: {})
    fn_output: str = field(default_factory=lambda: "")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ToolResult":
        return cls(**data)

    def to_openai(self) -> dict:
        return {"role": "tool", "tool_call_id": self.fn_id, "content": self.fn_output}


@dataclass
class ChatUsage:
    input_tokens: int = field(default=0)
    output_tokens: int = field(default=0)
    time: float = field(default=0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatUsage":
        return cls(**data)


@dataclass
class ClientCfg:
    api_key: str
    base_url: Optional[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChatCfg:
    model: str
    tools: Optional[list[dict]] | None = None
    tool_choice: Optional[str] | None = None
    temperature: Optional[float] | None = None
    max_tokens: Optional[int] | None = None
    top_p: Optional[float] | None = None
    frequency_penalty: Optional[float] | None = None
    presence_penalty: Optional[float] | None = None
    stop: Optional[list[str]] | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class EmbedCfg:
    model: str
    encoding_format: Optional[str] | None = None
    normalize: Optional[bool] | None = None
    dimensions: Optional[int] | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class ChaterCfg:
    client_cfg: ClientCfg
    chat_cfg: ChatCfg


@dataclass
class EmbedderCfg:
    client_cfg: ClientCfg
    embed_cfg: EmbedCfg


@dataclass
class ChatResponse:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created: str = field(default_factory=lambda: str(int(time())))
    role: Literal["system", "user", "assistant", "tool"] = field(
        default_factory=lambda: "assistant"
    )
    content: Union[str, MultimodalContent] = field(default_factory=lambda: "")
    reasoning_content: str = field(default_factory=lambda: "")
    tool_call: ToolCall = field(default_factory=lambda: None)
    tool_calls: list[ToolCall] = field(default_factory=lambda: [])
    tool_result: ToolResult = field(default_factory=lambda: None)
    tool_results: list[ToolResult] = field(default_factory=lambda: [])
    usage: Optional[ChatUsage] = field(default_factory=lambda: None)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}

    @classmethod
    def from_dict(cls, data: dict) -> "ChatResponse":
        return cls(
            id=data.get("id", uuid.uuid4().hex),
            created=data.get("created", str(int(time()))),
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            reasoning_content=data.get("reasoning_content", ""),
            tool_call=(ToolCall.from_dict(data["tool_call"]) if data.get("tool_call") else None),
            tool_calls=[ToolCall.from_dict(tool_call) for tool_call in data.get("tool_calls", [])],
            tool_results=[
                ToolResult.from_dict(tool_result) for tool_result in data.get("tool_results", [])
            ],
            usage=ChatUsage.from_dict(data["usage"]) if data.get("usage") else None,
        )

    def _normalize_content_for_openai(self) -> Union[str, list[dict]]:
        if isinstance(self.content, MultimodalContent):
            return self.content.to_openai()
        return self.content

    def to_openai(self, include_reasoning: bool = False) -> Union[dict, list[dict]]:
        if self.role == "system":
            return {"role": "system", "content": self._normalize_content_for_openai()}
        elif self.role == "user":
            return {"role": "user", "content": self._normalize_content_for_openai()}
        elif self.role == "assistant":
            base_msg = {}
            content_openai = self._normalize_content_for_openai()

            if self.tool_call:
                base_msg = {
                    "role": "assistant",
                    "content": content_openai or None,
                    "tool_calls": [self.tool_call.to_openai()],
                }
            elif self.tool_calls:
                base_msg = {
                    "role": "assistant",
                    "content": content_openai or None,
                    "tool_calls": [tool_call.to_openai() for tool_call in self.tool_calls],
                }
            else:
                base_msg = {"role": "assistant", "content": content_openai or None}

            if include_reasoning and self.reasoning_content:
                base_msg["reasoning_content"] = self.reasoning_content

            return base_msg
        elif self.role == "tool":
            if self.tool_result:
                return self.tool_result.to_openai()
            if self.tool_results:
                return [tool_result.to_openai() for tool_result in self.tool_results]
            raise ValueError(f"Invalid tool_result: {self.tool_result}")
        else:
            raise ValueError(f"Invalid role: {self.role}")


class FileCache:
    def __init__(
        self,
        cache_dir: str = "./cache",
        max_size: int | None = None,
        max_num: int | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.max_num = max_num
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_file_name(self, identifier: dict | object) -> str:
        json_str = json.dumps(identifier, ensure_ascii=False, indent=2)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest() + ".npy"

    def _get_file_path(self, file_name: str) -> str:
        return os.path.join(self.cache_dir, file_name)

    async def store(
        self,
        embeddings: list[list[float]],
        identifier: dict | object,
        overwrite: bool = False,
    ) -> str:
        file_path = self._get_file_path(self._get_file_name(identifier))
        if os.path.exists(file_path):
            if not os.path.isfile(file_path):
                raise ValueError(f"Cache path {file_path} exists but is not a file")
            if overwrite:
                np.save(file_path, embeddings)
                await self._rebuild()
        else:
            np.save(file_path, embeddings)
            await self._rebuild()

    async def retrieve(
        self,
        identifier: dict | object,
    ) -> np.ndarray | None:
        file_path = self._get_file_path(self._get_file_name(identifier))
        if os.path.exists(file_path):
            try:
                data = np.load(file_path, allow_pickle=True)
                return data
            except Exception as e:
                print(f"Warning: Could not load cache file {file_path}: {e}")
                return None
        return None

    async def delete(
        self,
        identifier: dict | object,
    ) -> None:
        file_path = self._get_file_path(self._get_file_name(identifier))
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            raise ValueError(f"Cache file {file_path} does not exist")

    async def clear(self) -> bool:
        for _, _, files in os.walk(self.cache_dir):
            for file in files:
                file_path = os.path.join(self.cache_dir, file)
                if file_path.endswith(".npy") and os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except (PermissionError, OSError) as e:
                        print(f"Warning: Could not delete cache file {file_path}: {e}")
                        continue

    async def _rebuild(self):
        try:
            files = [
                (_.name, _.stat().st_size, _.stat().st_mtime)
                for _ in os.scandir(self.cache_dir)
                if _.is_file() and _.name.endswith(".npy")
            ]
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not scan cache directory {self.cache_dir}: {e}")
            return False

        files.sort(key=lambda x: x[2])
        if self.max_num and len(files) > self.max_num:
            for file in files[: -self.max_num]:
                try:
                    os.remove(os.path.join(self.cache_dir, file[0]))
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not delete cache file {file[0]}: {e}")
                    continue
            files = files[-self.max_num :]
        file_size_sum = accumulate([_[1] for _ in files], initial=0)
        if self.max_size and file_size_sum[-1] > self.max_size:
            for file in files:
                if file_size_sum[-1] - file[1] < self.max_size:
                    break
                try:
                    os.remove(os.path.join(self.cache_dir, file[0]))
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not delete cache file {file[0]}: {e}")
                    continue
            files = files[files.index(file) :]
        return True


class Memory:
    def __init__(self, max_messages: Optional[int] = None, filepath: Optional[str] = None):
        self.messages: list[ChatResponse] = []
        self.max_messages = max_messages
        self.filepath = filepath

    def add(self, message: ChatResponse | list[ChatResponse]) -> None:
        if isinstance(message, list):
            self.messages.extend(message)
        else:
            self.messages.append(message)

        if self.max_messages is not None and len(self.messages) > self.max_messages:
            overflow = len(self.messages) - self.max_messages
            self.messages = self.messages[overflow:]

    def add_user(self, content: str) -> None:
        self.add(ChatResponse(role="user", content=content))

    def get(
        self,
        n: Optional[int] = None,
        filter_func: Optional[Callable[[ChatResponse], bool]] = None,
    ) -> list[ChatResponse]:
        messages = self.messages.copy() if n is None else self.messages[-n:] if n > 0 else []
        if filter_func:
            messages = [m for m in messages if filter_func(m)]
        return messages

    def get_by_role(self, role: str) -> list[ChatResponse]:
        return [m for m in self.messages if m.role == role]

    def get_with_reasoning(self) -> list[ChatResponse]:
        return [m for m in self.messages if m.reasoning_content]

    def get_with_tools(self) -> list[ChatResponse]:
        return [m for m in self.messages if m.tool_call or m.tool_calls]

    def filter(self, predicate: Callable[[ChatResponse], bool]) -> list[ChatResponse]:
        return [m for m in self.messages if predicate(m)]

    def export_json(self, filepath: str) -> None:
        import json

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in self.messages], f, ensure_ascii=False, indent=2)

    def load_json(self, filepath: str, overwrite: bool = False) -> None:
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if overwrite:
            self.messages.clear()

        for item in data:
            self.messages.append(ChatResponse.from_dict(item))

    def to_openai(self, include_reasoning: bool = False, n: Optional[int] = None) -> list[dict]:
        messages = self.get(n)
        result = []

        for msg in messages:
            openai_msg = msg.to_openai(include_reasoning=include_reasoning)

            if isinstance(openai_msg, list):
                result.extend(openai_msg)
            else:
                result.append(openai_msg)

        return result

    def clear(self) -> None:
        self.messages.clear()

    def remove_at(self, index: int) -> None:
        if 0 <= index < len(self.messages):
            self.messages.pop(index)

    def remove_range(self, start: int, end: int) -> None:
        if 0 <= start < len(self.messages) and start < end:
            del self.messages[start:end]

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, index: int) -> ChatResponse:
        return self.messages[index]

    def __iter__(self):
        return iter(self.messages)

    async def save(self, filepath: Optional[str] = None) -> None:
        save_path = filepath or self.filepath
        if not save_path:
            raise ValueError("No filepath specified for save operation")

        def _sync_save():
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([m.to_dict() for m in self.messages], f, ensure_ascii=False, indent=2)

        await asyncio.to_thread(_sync_save)

    async def load(self, filepath: Optional[str] = None, overwrite: bool = True) -> None:
        load_path = filepath or self.filepath
        if not load_path:
            raise ValueError("No filepath specified for load operation")

        if not os.path.exists(load_path):
            return

        def _sync_load():
            with open(load_path, "r", encoding="utf-8") as f:
                return json.load(f)

        data = await asyncio.to_thread(_sync_load)

        if overwrite:
            self.messages.clear()

        for item in data:
            self.messages.append(ChatResponse.from_dict(item))


class Chater:
    def __init__(
        self,
        chater_cfg: ChaterCfg,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
        on_stream: Optional[StreamCallback] = None,
        on_complete: Optional[CompleteCallback] = None,
    ):
        self.client = AsyncOpenAI(**chater_cfg.client_cfg.to_dict())
        self.chat_cfg = chater_cfg.chat_cfg
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.on_stream = on_stream
        self.on_complete = on_complete

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        stream: bool = False,
        on_stream: Optional[StreamCallback] = None,
        on_complete: Optional[CompleteCallback] = None,
        **kwargs,
    ) -> ChatResponse:
        current_trace = get_current_trace()
        disabled = current_trace is None

        stream_cb = on_stream or self.on_stream
        complete_cb = on_complete or self.on_complete

        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=self.retry_min_wait, max=self.retry_max_wait),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        )

        kwargs_inner = {
            "messages": messages,
            "stream": stream,
            **self.chat_cfg.to_dict(),
            **kwargs,
        }
        if (tools and not tool_choice) or (not tools and tool_choice):
            raise ValueError("tools and tool_choice must be provided together")
        if tools:
            kwargs_inner["tools"] = tools
        if tool_choice:
            kwargs_inner["tool_choice"] = tool_choice

        params = (
            {k: v for k, v in kwargs_inner.items() if k not in ["messages", "stream", "model"]}
            if not disabled
            else None
        )

        with generation_span(
            model=kwargs_inner.get("model"),
            input_msgs=messages if not disabled else None,
            params=params,
            disabled=disabled,
        ) as span:
            try:
                @retry_decorator
                async def _do_chat():
                    start_time = datetime.now()
                    response = await self.client.chat.completions.create(**kwargs_inner)
                    if stream:
                        return await self._stream(response, start_time, stream_cb)
                    result = self._no_stream(response, start_time)
                    if complete_cb:
                        complete_cb(result)
                    return result

                result = await _do_chat()

                if not disabled:
                    span.span_data.output_msg = {
                        "role": result.role,
                        "content": result.content,
                        "reasoning_content": result.reasoning_content,
                        "tool_calls": (
                            [
                                {
                                    "fn_id": tc.fn_id,
                                    "fn_name": tc.fn_name,
                                    "fn_args": tc.fn_args,
                                }
                                for tc in result.tool_calls
                            ]
                            if result.tool_calls
                            else None
                        ),
                    }
                    if result.usage:
                        span.span_data.usage = {
                            "input_tokens": result.usage.input_tokens,
                            "output_tokens": result.usage.output_tokens,
                            "time": result.usage.time,
                        }

                return result
            except Exception as e:
                if not disabled:
                    span.set_error(
                        SpanError(
                            message=f"Model generation failed: {str(e)}",
                            data={
                                "model": kwargs_inner.get("model"),
                                "error_type": type(e).__name__,
                            },
                        )
                    )
                raise

    def _no_stream(
        self,
        response: ChatCompletion,
        start_time: datetime,
    ):
        msg = response.choices[0].message
        id, created, role, tools = response.id, response.created, msg.role, []
        reasoning = (
            msg.reasoning_content
            if hasattr(msg, "reasoning_content") and msg.reasoning_content
            else ""
        )
        content = msg.content if hasattr(msg, "content") and msg.content else ""
        tool_calls = msg.tool_calls if hasattr(msg, "tool_calls") and msg.tool_calls else []
        if tool_calls:
            tools = [
                ToolCall(
                    fn_id=tool_call.id,
                    fn_name=tool_call.function.name,
                    fn_args=tool_call.function.arguments,
                )
                for tool_call in tool_calls
            ]
        if response.usage:
            usage = ChatUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                time=(datetime.now() - start_time).total_seconds(),
            )
        else:
            usage = ChatUsage()
        return ChatResponse(
            id=id,
            created=created,
            role=role,
            reasoning_content=reasoning,
            content=content,
            tool_calls=tools,
            usage=usage,
        )

    async def _stream(
        self,
        response: AsyncStream[ChatCompletionChunk],
        start_time: datetime,
        callback: Optional[StreamCallback] = None,
    ) -> ChatResponse:
        tool_calls_map = OrderedDict()
        full_content = ""
        full_reasoning = ""
        resp_id = uuid.uuid4().hex
        created = str(int(time()))
        role = "assistant"
        usage: Optional[ChatUsage] = None

        async for chunk in response:
            delta = chunk.choices[0].delta

            if chunk.id:
                resp_id = chunk.id
            if chunk.created:
                created = str(chunk.created)
            if hasattr(delta, "role") and delta.role:
                role = delta.role
            if chunk.usage:
                usage = ChatUsage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    time=(datetime.now() - start_time).total_seconds(),
                )

            chunk_response = None

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                full_reasoning += delta.reasoning_content
                chunk_response = ChatResponse(
                    id=resp_id,
                    created=created,
                    role=role,
                    reasoning_content=delta.reasoning_content,
                    usage=usage,
                )

            if hasattr(delta, "content") and delta.content:
                full_content += delta.content
                chunk_response = ChatResponse(
                    id=resp_id,
                    created=created,
                    role=role,
                    content=delta.content,
                    usage=usage,
                )

            for tc in delta.tool_calls or []:
                if tc.index in tool_calls_map:
                    if tc.function and tc.function.arguments:
                        tool_calls_map[tc.index].fn_args += tc.function.arguments
                else:
                    tool_calls_map[tc.index] = ToolCall(
                        fn_id=tc.id or "",
                        fn_name=tc.function.name if tc.function else "",
                        fn_args=tc.function.arguments if tc.function else "",
                    )
                chunk_response = ChatResponse(
                    id=resp_id,
                    created=created,
                    role=role,
                    tool_call=tool_calls_map[tc.index],
                    usage=usage,
                )

            if callback and chunk_response:
                callback(chunk_response)

        return ChatResponse(
            id=resp_id,
            created=created,
            role=role,
            content=full_content,
            reasoning_content=full_reasoning,
            tool_calls=list(tool_calls_map.values()) if tool_calls_map else [],
            usage=usage,
        )


class Embedder:
    def __init__(
        self,
        embedder_cfg: EmbedderCfg,
        embed_cache: Optional[FileCache] = None,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
    ):
        self.client = AsyncOpenAI(**embedder_cfg.client_cfg.to_dict())
        self.embed_cfg = embedder_cfg.embed_cfg
        self.embed_cache = embed_cache or FileCache()
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait

    async def embed(self, text: list[str], **kwargs) -> EmbedResponse:
        current_trace = get_current_trace()
        disabled = current_trace is None

        kwargs_inner = {"input": text, **self.embed_cfg.to_dict(), **kwargs}

        with embedder_span(
            model=self.embed_cfg.model,
            input_texts=text if not disabled else None,
            disabled=disabled,
        ) as span:
            if self.embed_cache:
                response = await self.embed_cache.retrieve(kwargs_inner)
                if response:
                    result = EmbedResponse(
                        source="cache", embedding=[np.array(_.embedding) for _ in response]
                    )
                    if not disabled:
                        span.span_data.source = "cache"
                        if result.embedding:
                            span.span_data.output_dimensions = len(result.embedding[0])
                    return result

            retry_decorator = retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=self.retry_min_wait, max=self.retry_max_wait),
                retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
                reraise=True,
            )

            @retry_decorator
            async def _do_embed():
                start_time = datetime.now()
                response = await self.client.embeddings.create(**kwargs_inner)
                if self.embed_cache:
                    await self.embed_cache.store(response.data, kwargs_inner)
                return EmbedResponse(
                    source="api",
                    embedding=[np.array(_.embedding) for _ in response.data],
                    usage=EmbedUsage(
                        time=(datetime.now() - start_time).total_seconds(),
                        token=response.usage.total_tokens,
                    ),
                )

            try:
                result = await _do_embed()
                if not disabled:
                    span.span_data.source = "api"
                    if result.embedding:
                        span.span_data.output_dimensions = len(result.embedding[0])
                    if result.usage:
                        span.span_data.usage = {
                            "time": result.usage.time,
                            "token": result.usage.token,
                        }
                return result
            except Exception as e:
                if not disabled:
                    span.span_data.error = str(e)
                raise


class ChaterPool:
    def __init__(
        self,
        chater_cfgs: list[ChaterCfg],
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
        circuit_breaker_threshold: int = 5,
        on_stream: Optional[StreamCallback] = None,
        on_complete: Optional[CompleteCallback] = None,
    ):
        self.chaters = [
            Chater(cfg, max_retries, retry_min_wait, retry_max_wait) for cfg in chater_cfgs
        ]
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.failure_counts = [0] * len(self.chaters)
        self.success_counts = [0] * len(self.chaters)
        self.on_stream = on_stream
        self.on_complete = on_complete
        self.logger = logging.getLogger(__name__)

    def _is_circuit_open(self, idx: int) -> bool:
        return self.failure_counts[idx] >= self.circuit_breaker_threshold

    def _record_success(self, idx: int):
        self.success_counts[idx] += 1
        self.failure_counts[idx] = max(0, self.failure_counts[idx] - 1)

    def _record_failure(self, idx: int):
        self.failure_counts[idx] += 1

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        stream: bool = False,
        on_stream: Optional[StreamCallback] = None,
        on_complete: Optional[CompleteCallback] = None,
        **kwargs,
    ) -> ChatResponse:
        stream_cb = on_stream or self.on_stream
        complete_cb = on_complete or self.on_complete

        last_error = None
        for idx, chater in enumerate(self.chaters):
            if self._is_circuit_open(idx):
                self.logger.warning(f"Circuit breaker open for chater {idx}, skipping")
                continue

            try:
                response = await chater.chat(
                    messages, tools, tool_choice, stream, stream_cb, complete_cb, **kwargs
                )
                self._record_success(idx)
                return response
            except NON_RETRYABLE_EXCEPTIONS as e:
                last_error = e
                self._record_failure(idx)
                self.logger.warning(
                    f"Chater {idx} non-retryable error: {type(e).__name__}, switching"
                )
                continue
            except RetryError as e:
                last_error = e.last_attempt.exception()
                self._record_failure(idx)
                self.logger.warning(
                    f"Chater {idx} exhausted retries: {type(last_error).__name__}, switching"
                )
                continue
            except (OpenAIError, APIStatusError) as e:
                last_error = e
                self._record_failure(idx)
                self.logger.warning(f"Chater {idx} OpenAI error: {type(e).__name__}, switching")
                continue
            except Exception as e:
                last_error = e
                self._record_failure(idx)
                self.logger.warning(f"Chater {idx} unexpected error: {type(e).__name__}, switching")
                continue

        raise ModelError(f"All chaters failed. Last error: {last_error}")

    async def health_check(self) -> dict:
        health_status = {}
        for idx, chater in enumerate(self.chaters):
            health_status[f"chater_{idx}"] = {
                "success_count": self.success_counts[idx],
                "failure_count": self.failure_counts[idx],
                "circuit_open": self._is_circuit_open(idx),
            }
        return health_status


class EmbedderPool:
    def __init__(
        self,
        embedder_cfgs: list[EmbedderCfg],
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
        circuit_breaker_threshold: int = 5,
        embed_cache: Optional[FileCache] = None,
    ):
        self.embedders = [
            Embedder(cfg, embed_cache, max_retries, retry_min_wait, retry_max_wait)
            for cfg in embedder_cfgs
        ]
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.failure_counts = [0] * len(self.embedders)
        self.success_counts = [0] * len(self.embedders)
        self.logger = logging.getLogger(__name__)

    def _is_circuit_open(self, idx: int) -> bool:
        return self.failure_counts[idx] >= self.circuit_breaker_threshold

    def _record_success(self, idx: int):
        self.success_counts[idx] += 1
        self.failure_counts[idx] = max(0, self.failure_counts[idx] - 1)

    def _record_failure(self, idx: int):
        self.failure_counts[idx] += 1

    async def embed(self, text: list[str], **kwargs) -> EmbedResponse:
        last_error = None
        for idx, embedder in enumerate(self.embedders):
            if self._is_circuit_open(idx):
                self.logger.warning(f"Circuit breaker open for embedder {idx}, skipping")
                continue

            try:
                response = await embedder.embed(text, **kwargs)
                self._record_success(idx)
                return response
            except NON_RETRYABLE_EXCEPTIONS as e:
                last_error = e
                self._record_failure(idx)
                self.logger.warning(
                    f"Embedder {idx} non-retryable error: {type(e).__name__}, switching"
                )
                continue
            except RetryError as e:
                last_error = e.last_attempt.exception()
                self._record_failure(idx)
                self.logger.warning(
                    f"Embedder {idx} exhausted retries: {type(last_error).__name__}, switching"
                )
                continue
            except (OpenAIError, APIStatusError) as e:
                last_error = e
                self._record_failure(idx)
                self.logger.warning(f"Embedder {idx} OpenAI error: {type(e).__name__}, switching")
                continue
            except Exception as e:
                last_error = e
                self._record_failure(idx)
                self.logger.warning(
                    f"Embedder {idx} unexpected error: {type(e).__name__}, switching"
                )
                continue

        raise ModelError(f"All embedders failed. Last error: {last_error}")

    async def health_check(self) -> dict:
        health_status = {}
        for idx, embedder in enumerate(self.embedders):
            health_status[f"embedder_{idx}"] = {
                "success_count": self.success_counts[idx],
                "failure_count": self.failure_counts[idx],
                "circuit_open": self._is_circuit_open(idx),
            }
        return health_status


DEFAULT_CONFIGS = {
    "ali": {
        "client": ClientCfg(
            api_key=os.getenv("ali_api_key"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "chat": ChatCfg(model="qwen3-max-preview"),
        "embed": EmbedCfg(model="text-embedding-v4"),
    },
    "zhipuai": {
        "client": ClientCfg(
            api_key=os.getenv("zhipuai_api_key"),
            base_url="https://open.bigmodel.cn/api/paas/v4",
        ),
        "chat": ChatCfg(model="glm-4.5-air"),
        "embed": EmbedCfg(model="embedding-3"),
    },
    "siliconflow": {
        "client": ClientCfg(
            api_key=os.getenv("siliconflow_api_key"),
            base_url="https://api.siliconflow.cn/v1",
        ),
        "chat": ChatCfg(model="zai-org/GLM-4.5"),
        "embed": EmbedCfg(model="Qwen/Qwen3-Embedding-8B"),
    },
    "ark": {
        "client": ClientCfg(
            api_key=os.getenv("ark_api_key"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        ),
        "chat": ChatCfg(model="doubao-1-5-thinking-pro-250415"),
        "embed": EmbedCfg(model="doubao-embedding-text-240715"),
    },
}


def get_chater_cfg(provider: str) -> ChaterCfg:
    cfg = DEFAULT_CONFIGS[provider]
    return ChaterCfg(client_cfg=cfg["client"], chat_cfg=cfg["chat"])


def get_embedder_cfg(provider: str) -> EmbedderCfg:
    cfg = DEFAULT_CONFIGS[provider]
    return EmbedderCfg(client_cfg=cfg["client"], embed_cfg=cfg["embed"])


if __name__ == "__main__":

    def stream_handler(chunk: ChatResponse):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        if chunk.reasoning_content:
            print(chunk.reasoning_content, end="", flush=True)

    def complete_handler(response: ChatResponse):
        print(f"\n[Complete] content={response.content[:50]}...")

    async def main():
        chater = Chater(get_chater_cfg("ali"))

        print("=" * 50)
        print("Test 1: Non-stream with on_complete callback")
        print("=" * 50)
        response = await chater.chat(
            [{"role": "user", "content": "Hello, say hi in 10 words"}],
            stream=False,
            on_complete=complete_handler,
        )
        print(f"Return: {response.content}")

        print("\n" + "=" * 50)
        print("Test 2: Stream with on_stream callback")
        print("=" * 50)
        response = await chater.chat(
            [{"role": "user", "content": "Count from 1 to 5"}],
            stream=True,
            on_stream=stream_handler,
        )
        print(f"\nReturn: {response}")

        print("\n" + "=" * 50)
        print("Test 3: ChaterPool stream")
        print("=" * 50)
        pool = ChaterPool([get_chater_cfg("ali")])
        response = await pool.chat(
            [{"role": "user", "content": "Say hello"}],
            stream=True,
            on_stream=stream_handler,
        )
        print(f"\nReturn: {response}")

    asyncio.run(main())
