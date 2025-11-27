import asyncio
import uuid
import json
from abc import ABC, abstractmethod
from functools import wraps
from collections import OrderedDict
from typing import Optional, Union, AsyncGenerator, Callable, Any, Dict

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
    generation_span,
    tool_span,
    get_current_trace,
    agent_step_span,
)


class _HookDecorator:
    def __init__(self, agent, phase: str, is_class_level=False):
        self.agent = agent
        self.phase = phase
        self.is_class_level = is_class_level

    def __call__(self, func: Callable) -> Callable:
        if self.is_class_level:
            hook_dict = getattr(self.agent, f"_class_hooks_{self.phase}")
            hook_dict[func.__name__] = func
        else:
            hook_dict = getattr(self.agent, f"_{self.phase}_hooks")
            hook_dict[func.__name__] = func
        return func


def with_reply_hooks(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        current_args, current_kwargs = args, kwargs

        for hook in self._pre_reply_hooks.values():
            try:
                if args:
                    result = hook(current_args[0])
                    if result is not None:
                        current_args = (result,) + current_args[1:]
            except Exception as e:
                import logging

                logging.warning(f"Pre-reply hook failed: {e}")

        for hook in self.__class__._class_hooks_pre_reply.values():
            try:
                if current_args:
                    result = hook(self, current_args[0])
                    if result is not None:
                        current_args = (result,) + current_args[1:]
            except Exception as e:
                import logging

                logging.warning(f"Class pre-reply hook failed: {e}")

        async for response in func(self, *current_args, **current_kwargs):
            yield response

    return wrapper


def with_observe_hooks(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, msg):
        current_msg = msg

        for hook in self._pre_observe_hooks.values():
            try:
                from copy import deepcopy

                result = hook(deepcopy(current_msg))
                if result is not None:
                    current_msg = result
            except Exception as e:
                import logging

                logging.warning(f"Pre-observe hook failed: {e}")

        for hook in self.__class__._class_hooks_pre_observe.values():
            try:
                from copy import deepcopy

                result = hook(self, deepcopy(current_msg))
                if result is not None:
                    current_msg = result
            except Exception as e:
                import logging

                logging.warning(f"Class pre-observe hook failed: {e}")

        func(self, current_msg)

        for hook in self._post_observe_hooks.values():
            try:
                hook(current_msg)
            except Exception as e:
                import logging

                logging.warning(f"Post-observe hook failed: {e}")

        for hook in self.__class__._class_hooks_post_observe.values():
            try:
                hook(self, current_msg)
            except Exception as e:
                import logging

                logging.warning(f"Class post-observe hook failed: {e}")

    return wrapper


class BaseAgent(ABC):
    _class_hooks_pre_reply: OrderedDict = OrderedDict()
    _class_hooks_post_reply: OrderedDict = OrderedDict()
    _class_hooks_pre_observe: OrderedDict = OrderedDict()
    _class_hooks_post_observe: OrderedDict = OrderedDict()
    _class_hooks_pre_speak: OrderedDict = OrderedDict()
    _class_hooks_post_speak: OrderedDict = OrderedDict()

    def __init__(self, name: str, speaker: Optional[Speaker] = None):
        self.name = name
        self.agent_id = uuid.uuid4().hex
        self.speaker = speaker or ConsoleSpeaker()
        self._pre_observe_hooks: OrderedDict = OrderedDict()
        self._post_observe_hooks: OrderedDict = OrderedDict()
        self._pre_reply_hooks: OrderedDict = OrderedDict()
        self._post_reply_hooks: OrderedDict = OrderedDict()
        self._pre_speak_hooks: OrderedDict = OrderedDict()
        self._post_speak_hooks: OrderedDict = OrderedDict()

        self.pre_reply = _HookDecorator(self, "pre_reply", is_class_level=False)
        self.post_reply = _HookDecorator(self, "post_reply", is_class_level=False)
        self.pre_observe = _HookDecorator(self, "pre_observe", is_class_level=False)
        self.post_observe = _HookDecorator(self, "post_observe", is_class_level=False)
        self.pre_speak = _HookDecorator(self, "pre_speak", is_class_level=False)
        self.post_speak = _HookDecorator(self, "post_speak", is_class_level=False)

    def _run_post_reply_hooks(self, response: ChatResponse, user_message: str) -> ChatResponse:
        from copy import deepcopy
        import logging

        current_response = response

        for hook in self._post_reply_hooks.values():
            try:
                result = hook(deepcopy(current_response))
                if result is not None:
                    current_response = result
            except Exception as e:
                logging.warning(f"Post-reply hook failed: {e}")

        for hook in self.__class__._class_hooks_post_reply.values():
            try:
                result = hook(self, deepcopy(current_response))
                if result is not None:
                    current_response = result
            except Exception as e:
                logging.warning(f"Class post-reply hook failed: {e}")

        return current_response

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
    async def reply(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def observe(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def speak(
        self,
        content: Union[str, ChatResponse, AsyncGenerator[ChatResponse, None]],
        stream: bool = False,
    ) -> None:
        current_content = content

        for hook in self._pre_speak_hooks.values():
            try:
                result = hook(current_content)
                if result is not None:
                    current_content = result
            except Exception as e:
                import logging

                logging.warning(f"Pre-speak hook failed: {e}")

        for hook in self.__class__._class_hooks_pre_speak.values():
            try:
                result = hook(self, current_content)
                if result is not None:
                    current_content = result
            except Exception as e:
                import logging

                logging.warning(f"Class pre-speak hook failed: {e}")

        if hasattr(current_content, "__aiter__"):
            import asyncio

            asyncio.create_task(self._speak_stream(current_content))
        elif stream and isinstance(current_content, ChatResponse):
            self.speaker.speak_chunk(current_content)
        else:
            self._speak_content(current_content)

        for hook in self._post_speak_hooks.values():
            try:
                hook(current_content)
            except Exception as e:
                import logging

                logging.warning(f"Post-speak hook failed: {e}")

        for hook in self.__class__._class_hooks_post_speak.values():
            try:
                hook(self, current_content)
            except Exception as e:
                import logging

                logging.warning(f"Class post-speak hook failed: {e}")

    async def _speak_stream(self, content: AsyncGenerator[ChatResponse, None]) -> None:
        self.speaker.speak_stream_start(self.name)
        async for chunk in content:
            self.speaker.speak_chunk(chunk)
        self.speaker.speak_stream_end()

    def _speak_content(self, content: Union[str, ChatResponse]) -> None:
        if isinstance(content, str):
            response = ChatResponse(role="assistant", content=content)
            self.speaker.speak_complete(response, self.name)
        elif isinstance(content, ChatResponse):
            self.speaker.speak_complete(content, self.name)


BaseAgent.pre_reply = _HookDecorator(BaseAgent, "pre_reply", is_class_level=True)
BaseAgent.post_reply = _HookDecorator(BaseAgent, "post_reply", is_class_level=True)
BaseAgent.pre_observe = _HookDecorator(BaseAgent, "pre_observe", is_class_level=True)
BaseAgent.post_observe = _HookDecorator(BaseAgent, "post_observe", is_class_level=True)
BaseAgent.pre_speak = _HookDecorator(BaseAgent, "pre_speak", is_class_level=True)
BaseAgent.post_speak = _HookDecorator(BaseAgent, "post_speak", is_class_level=True)


class Agent(BaseAgent):
    def __init__(
        self,
        name: str,
        chater: Union[Chater, ChaterPool],
        memory: Memory,
        tools: Optional[ToolKit] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_iterations: int = 5,
        tool_timeout: Optional[int] = None,
        enable_logging: bool = False,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        speaker: Optional[Speaker] = None,
    ):
        super().__init__(name, speaker=speaker)
        self.chater = chater
        self.memory = memory
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.tool_timeout = tool_timeout
        self._audience: Optional[list["Agent"]] = None

        from ._utils import AgentLogger

        self.logger = AgentLogger(
            name=f"Agent.{name}", level=log_level, enabled=enable_logging, log_file=log_file
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
    async def reply(
        self, user_message: str, stream: bool = False, auto_speak: bool = False
    ) -> Union[ChatResponse, AsyncGenerator[ChatResponse, None]]:
        current_trace = get_current_trace()

        if current_trace is not None:
            async for response in self._reply_with_trace(user_message, stream, auto_speak):
                yield response
        else:
            async for response in self._reply_impl(user_message, stream, auto_speak):
                yield response

    async def _reply_impl(
        self, user_message: str, stream: bool = False, auto_speak: bool = False
    ) -> AsyncGenerator[ChatResponse, None]:
        self.logger.info(f"Received message: {user_message[:50]}...")

        user_msg = ChatResponse(role="user", content=user_message)
        self.memory.add(user_msg)

        tools = self.tools.to_openai_tools() if self.tools else None
        tool_choice = "auto" if self.tools else None

        for iteration in range(self.max_iterations):
            self.logger.debug(f"Iteration {iteration + 1}/{self.max_iterations}")
            history = self._build_history()

            if not stream:
                response = await self.chater.chat(
                    messages=history, tools=tools, tool_choice=tool_choice, stream=False
                )

                self.memory.add(response)

                tool_calls_list = (
                    response.tool_calls
                    if response.tool_calls
                    else ([response.tool_call] if response.tool_call else [])
                )

                if tool_calls_list:
                    yield response
                    self.logger.info(f"Executing {len(tool_calls_list)} tools")
                    tool_results = await self._handle_tool_calls_concurrent(tool_calls_list)
                    self.memory.add(tool_results)
                else:
                    self.logger.info("Reply completed")
                    response = self._run_post_reply_hooks(response, user_message)

                    yield response

                    if auto_speak:
                        self.speak(response, stream=False)

                    self._broadcast_to_audience(response)
                    break

            else:
                response = ChatResponse()
                last_msg = None
                first_chunk = True

                async for msg in await self.chater.chat(
                    messages=history, stream=True, tools=tools, tool_choice=tool_choice
                ):
                    if msg.reasoning_content:
                        response.reasoning_content += msg.reasoning_content

                    if msg.content:
                        response.content += msg.content

                    if msg.tool_call:
                        response.tool_call = msg.tool_call

                    if msg.tool_calls:
                        response.tool_calls = msg.tool_calls

                    last_msg = msg
                    if auto_speak:
                        if first_chunk and (msg.content or msg.reasoning_content):
                            first_chunk = False
                        self.speaker.speak_chunk(msg)

                    yield msg

                if last_msg:
                    response.usage = last_msg.usage
                    response.id = last_msg.id
                    response.created = last_msg.created
                    response.role = last_msg.role

                self.memory.add(response)

                tool_calls_list = (
                    response.tool_calls
                    if response.tool_calls
                    else ([response.tool_call] if response.tool_call else [])
                )

                if tool_calls_list:
                    self.logger.info(f"Executing {len(tool_calls_list)} tools")
                    tool_results = await self._handle_tool_calls_concurrent(tool_calls_list)
                    self.memory.add(tool_results)
                else:
                    self.logger.info("Reply completed")
                    final_response = self._run_post_reply_hooks(response, user_message)
                    if final_response != response and hasattr(self.memory, "messages"):
                        if len(self.memory.messages) > 0:
                            self.memory.messages[-1] = final_response

                    if auto_speak:
                        print()

                    self._broadcast_to_audience(final_response)
                    break

    async def _reply_with_trace(
        self, user_message: str, stream: bool = False, auto_speak: bool = False
    ) -> AsyncGenerator[ChatResponse, None]:
        with agent_span(
            agent_name=self.name,
            agent_type=self.__class__.__name__,
            tools=list(self.tools._tools.keys()) if self.tools else None,
            user_input=user_message,
            agent_id=self.agent_id,
        ) as span:
            final_content = None

            async for response in self._reply_impl(user_message, stream, auto_speak):
                if hasattr(response, "content") and response.content:
                    final_content = response.content
                yield response

            if final_content:
                span.span_data.agent_output = final_content

    async def _execute_single_tool(self, tc: ToolCall) -> ChatResponse:
        from ._exceptions import ToolError

        args_dict = tc.get_args_dict()

        try:
            if self.tool_timeout:
                output = await asyncio.wait_for(
                    self.tools.execute(name=tc.fn_name, **args_dict), timeout=self.tool_timeout
                )
            else:
                output = await self.tools.execute(name=tc.fn_name, **args_dict)

        except asyncio.TimeoutError:
            output = f"Tool execution timeout after {self.tool_timeout}s"
        except ToolError as e:
            output = f"Error: {e.message}"
        except Exception as e:
            output = f"Unexpected error: {str(e)}"

        return ChatResponse(
            role="tool",
            tool_result=ToolResult(
                fn_id=tc.fn_id,
                fn_name=tc.fn_name,
                fn_args=args_dict,
                fn_output=str(output),
            ),
        )

    async def _handle_tool_calls_concurrent(self, tool_calls: list[ToolCall]) -> list[ChatResponse]:
        return await asyncio.gather(*[self._execute_single_tool(tc) for tc in tool_calls])

    def clear_memory(self):
        self.memory.clear()


if __name__ == "__main__":
    from ._utils import FileOperations, DirectoryOperations, SearchOperations
    from datetime import datetime

    async def get_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def ainput(prompt: str = "") -> str:
        return await asyncio.to_thread(input, prompt)

    async def main():
        tools = ToolKit()
        tools.register(FileOperations.read_file, "read_file")
        tools.register(FileOperations.write_file, "write_file")
        tools.register(DirectoryOperations.list_directory, "list_directory")
        tools.register(SearchOperations.grep_in_file, "grep_in_file")
        tools.register(get_time, "get_time")

        agent = Agent(
            name="Assistant",
            chater=ChaterPool([get_chater_cfg("siliconflow"), get_chater_cfg("zhipuai")]),
            memory=Memory(max_messages=None),
            tools=tools,
            system_prompt="You are a helpful AI assistant.",
            enable_logging=False,
        )

        @agent.post_reply
        def format_hook(response):
            if response.content and not response.tool_call and not response.tool_calls:
                response.content = f"✨ {response.content}"
            return response

        print(f"Agent: {repr(agent)}")
        print(f"Tools: {list(tools._tools.keys())}")
        print("Commands: 'quit', 'clear', 'memory', 'info'\n")

        while True:
            try:
                user_input = await ainput("You: ")

                if not user_input.strip():
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break

                if user_input.lower() == "clear":
                    agent.clear_memory()
                    print("Memory cleared.\n")
                    continue

                if user_input.lower() == "memory":
                    print(f"Memory: {len(agent.memory)} messages\n")
                    continue

                if user_input.lower() == "info":
                    print(f"{agent.to_dict()}\n")
                    continue

                print("Agent: ", end="", flush=True)
                async for msg in agent.reply(user_input, stream=True):
                    if msg.content:
                        print(msg.content, end="", flush=True)
                    if msg.reasoning_content:
                        print(msg.reasoning_content, end="", flush=True)
                    if msg.tool_call:
                        print(msg.tool_call, end="", flush=True)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    asyncio.run(main())
