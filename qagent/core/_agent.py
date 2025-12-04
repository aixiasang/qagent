import asyncio
import uuid
import json
import logging
from abc import ABC, abstractmethod
from functools import wraps
from dataclasses import dataclass, field
from typing import Optional, Union, AsyncGenerator, Callable, Any, Dict, List

from ._model import (
    Memory,
    ChatResponse,
    Chater,
    ChaterPool,
    ToolCall,
    ToolResult,
    get_chater_cfg,
)
from ._tools import ToolKit
from ._speaker import Speaker, ConsoleSpeaker
from ._trace import (
    agent_span,
    tool_span,
    get_current_trace,
)


StreamHandler = Callable[[ChatResponse], None]


@dataclass
class LogConfig:
    enabled: bool = False
    level: str = "INFO"
    file: Optional[str] = None

    @classmethod
    def disabled(cls) -> "LogConfig":
        return cls(enabled=False)

    @classmethod
    def console(cls, level: str = "INFO") -> "LogConfig":
        return cls(enabled=True, level=level)

    @classmethod
    def to_file(cls, file: str, level: str = "INFO") -> "LogConfig":
        return cls(enabled=True, level=level, file=file)


def _run_hooks(hooks: Dict[str, Callable], value: Any, agent: Any = None) -> Any:
    current = value
    for hook in hooks.values():
        try:
            if agent is not None:
                result = hook(agent, current)
            else:
                result = hook(current)
            if result is not None:
                current = result
        except Exception as e:
            logging.warning(f"Hook {hook.__name__} failed: {e}")
    return current


def with_reply_hooks(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        current_args = args
        if args:
            msg = _run_hooks(self._pre_reply_hooks, args[0])
            msg = _run_hooks(self.__class__._class_pre_reply_hooks, msg, self)
            current_args = (msg,) + args[1:]

        return await func(self, *current_args, **kwargs)

    return wrapper


def with_observe_hooks(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, msg):
        current = _run_hooks(self._pre_observe_hooks, msg)
        current = _run_hooks(self.__class__._class_pre_observe_hooks, current, self)
        func(self, current)

    return wrapper


class _HookDecorator:
    def __init__(self, hooks: Dict[str, Callable], name: Optional[str] = None):
        self._hooks = hooks
        self._name = name

    def __call__(self, func_or_name: Union[str, Callable]) -> Union["_HookDecorator", Callable]:
        if isinstance(func_or_name, str):
            return _HookDecorator(self._hooks, func_or_name)
        func = func_or_name
        name = self._name or func.__name__
        self._hooks[name] = func
        return func


class BaseAgent(ABC):
    _class_pre_reply_hooks: Dict[str, Callable] = {}
    _class_post_reply_hooks: Dict[str, Callable] = {}
    _class_pre_observe_hooks: Dict[str, Callable] = {}

    def __init__(self, name: str):
        self.name = name
        self.agent_id = uuid.uuid4().hex
        self._pre_reply_hooks: Dict[str, Callable] = {}
        self._post_reply_hooks: Dict[str, Callable] = {}
        self._pre_observe_hooks: Dict[str, Callable] = {}

    @property
    def pre_reply(self) -> _HookDecorator:
        return _HookDecorator(self._pre_reply_hooks)

    @property
    def post_reply(self) -> _HookDecorator:
        return _HookDecorator(self._post_reply_hooks)

    @property
    def pre_observe(self) -> _HookDecorator:
        return _HookDecorator(self._pre_observe_hooks)

    def add_hook(self, phase: str, name: str, hook: Callable) -> "BaseAgent":
        hooks = getattr(self, f"_{phase}_hooks", None)
        if hooks is not None:
            hooks[name] = hook
        return self

    def remove_hook(self, name: str) -> "BaseAgent":
        self._pre_reply_hooks.pop(name, None)
        self._post_reply_hooks.pop(name, None)
        self._pre_observe_hooks.pop(name, None)
        return self

    @classmethod
    def class_pre_reply(cls, func: Callable) -> Callable:
        cls._class_pre_reply_hooks[func.__name__] = func
        return func

    @classmethod
    def class_post_reply(cls, func: Callable) -> Callable:
        cls._class_post_reply_hooks[func.__name__] = func
        return func

    def _run_post_reply_hooks(self, response: ChatResponse, user_message: str) -> ChatResponse:
        current = _run_hooks(self._post_reply_hooks, response)
        current = _run_hooks(self.__class__._class_post_reply_hooks, current, self)
        return current

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.__class__.__name__,
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.agent_id[:8]}, name={self.name})>"

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def reply(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def observe(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError



def make_stream_callback(speaker: Speaker) -> StreamHandler:
    def callback(chunk: ChatResponse) -> None:
        speaker.speak_chunk(chunk)
    return callback


def make_complete_callback(speaker: Speaker, name: str = "Agent") -> StreamHandler:
    def callback(response: ChatResponse) -> None:
        speaker.speak_complete(response, name)
    return callback


class Agent(BaseAgent):
    def __init__(
        self,
        chater: Union[Chater, ChaterPool],
        name: Optional[str] = None,
        memory: Optional[Memory] = None,
        tools: Optional[ToolKit] = None,
        tool_choice : Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_iterations: int = 5,
        tool_timeout: Optional[int] = None,
        log_config: Optional[LogConfig] = None,
        agent_type: Optional[str] = None,
        on_stream: Optional[StreamHandler] = None,
        on_complete: Optional[StreamHandler] = None,
    ):
        agent_name = name or f"Agent_{uuid.uuid4().hex[:8]}"
        super().__init__(agent_name)
        self.chater = chater
        self.memory = memory or Memory()
        self.tools = tools
        self.tool_choice = tool_choice
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.tool_timeout = tool_timeout
        self.agent_type = agent_type or self.__class__.__name__
        self.on_stream = on_stream
        self.on_complete = on_complete
        self._audience: Optional[list["Agent"]] = None

        from ._utils import AgentLogger

        log_cfg = log_config or LogConfig.disabled()
        self.logger = AgentLogger(
            name=f"Agent.{agent_name}",
            level=log_cfg.level,
            enabled=log_cfg.enabled,
            log_file=log_cfg.file,
        )

    def _build_history(self) -> list[dict]:
        system_msg = {"role": "system", "content": self.system_prompt}
        memory_msgs = self.memory.to_openai()
        return [system_msg] + memory_msgs

    @with_observe_hooks
    def observe(self, msg: Union[ChatResponse, list[ChatResponse]]) -> None:
        if isinstance(msg, list):
            for m in msg:
                self.memory.add(m)
        else:
            self.memory.add(msg)

    def _broadcast_to_audience(self, msg: ChatResponse) -> None:
        if self._audience:
            for agent in self._audience:
                agent.observe(msg)

    @with_reply_hooks
    async def __call__(
        self,
        user_message: str,
        stream: bool = False,
        on_stream: Optional[StreamHandler] = None,
        on_complete: Optional[StreamHandler] = None,
    ) -> ChatResponse:
        current_trace = get_current_trace()

        if current_trace:
            with agent_span(
                agent_name=self.name,
                agent_type=self.agent_type,
                tools=list(self.tools._tools.keys()) if self.tools else None,
                user_input=user_message,
                agent_id=self.agent_id,
            ) as span:
                response = await self.reply(user_message, stream, on_stream, on_complete)
                span.span_data.agent_output = response.content
                return response
        else:
            return await self.reply(user_message, stream, on_stream, on_complete)

    async def reply(
        self,
        user_message: str,
        stream: bool = False,
        on_stream: Optional[StreamHandler] = None,
        on_complete: Optional[StreamHandler] = None,
    ) -> ChatResponse:
        self.logger.info(f"Received message: {user_message[:50]}...")

        self.memory.add_user(user_message)
        tools = self.tools.to_openai_tools() if self.tools else None
        tool_choice = self.tool_choice or ("auto" if self.tools else None)
        stream_cb = on_stream or self.on_stream
        complete_cb = on_complete or self.on_complete

        for iteration in range(self.max_iterations):
            self.logger.debug(f"Iteration {iteration + 1}/{self.max_iterations}")
            history = self._build_history()

            response = await self.chater.chat(
                messages=history,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                on_stream=stream_cb,
                on_complete=complete_cb,
            )

            self.memory.add(response)

            tool_calls_list = (
                response.tool_calls
                if response.tool_calls
                else ([response.tool_call] if response.tool_call else [])
            )

            if tool_calls_list:
                self.logger.info(f"Executing {len(tool_calls_list)} tools")
                tool_results = await self._execute_tools_concurrent(tool_calls_list)
                self.memory.add(tool_results)
            else:
                self.logger.info("Reply completed")
                response = self._run_post_reply_hooks(response, user_message)
                self._broadcast_to_audience(response)
                return response

        return response

    async def _execute_single_tool(self, tc: ToolCall, use_trace: bool = True) -> ChatResponse:
        from ._exceptions import ToolError

        args_dict = tc.get_args_dict()
        current_trace = get_current_trace() if use_trace else None

        async def _do_execute():
            try:
                if self.tool_timeout:
                    return await asyncio.wait_for(
                        self.tools.execute(name=tc.fn_name, **args_dict), timeout=self.tool_timeout
                    )
                return await self.tools.execute(name=tc.fn_name, **args_dict)
            except asyncio.TimeoutError:
                return f"Tool timeout after {self.tool_timeout}s"
            except ToolError as e:
                return f"Error: {e.message}"
            except Exception as e:
                return f"Error: {str(e)}"

        if current_trace:
            with tool_span(tool_name=tc.fn_name, input_args=args_dict) as span:
                output = await _do_execute()
                span.span_data.output = str(output)
        else:
            output = await _do_execute()

        return ChatResponse(
            role="tool",
            tool_result=ToolResult(
                fn_id=tc.fn_id,
                fn_name=tc.fn_name,
                fn_args=args_dict,
                fn_output=str(output),
            ),
        )

    async def _execute_tools_concurrent(self, tool_calls: list[ToolCall]) -> list[ChatResponse]:
        return await asyncio.gather(*[self._execute_single_tool(tc) for tc in tool_calls])

    async def _execute_tools_sequential(self, tool_calls: list[ToolCall]) -> list[ChatResponse]:
        results = []
        for tc in tool_calls:
            result = await self._execute_single_tool(tc)
            results.append(result)
        return results

    def clear_memory(self):
        self.memory.clear()


if __name__ == "__main__":
    from ._builtin_tools import DirectoryOperations
    from datetime import datetime

    async def get_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def main():
        tools = ToolKit()
        tools.register(DirectoryOperations.list_directory, "list_directory")
        tools.register(get_time, "get_time")

        speaker = ConsoleSpeaker()

        agent = Agent(
            chater=ChaterPool([get_chater_cfg("ali")]),
            tools=tools,
        )

        print("=" * 50)
        print("Test 1: Stream with make_stream_callback")
        print("=" * 50)
        response = await agent("What time is it?", stream=True, on_stream=make_stream_callback(speaker))
        print(f"\nReturn: {response.content}")

        print("\n" + "=" * 50)
        print("Test 2: Non-stream with make_complete_callback")
        print("=" * 50)
        response = await agent("Say hello", stream=False, on_complete=make_complete_callback(speaker, "MyAgent"))
        print(f"Return: {response.content}")

    asyncio.run(main())
