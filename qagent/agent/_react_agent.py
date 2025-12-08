import asyncio
import re
import json
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Dict, List

from typing import TYPE_CHECKING

from ..core import (
    Agent,
    Memory,
    ChatResponse,
    Chater,
    ChaterPool,
    ToolKit,
)

if TYPE_CHECKING:
    from ..core._agent import LogConfig

REACT_SYSTEM_PROMPT = """You are an AI assistant that uses the ReAct (Reasoning and Acting) framework to solve problems.

For each step:
1. Think about what you need to do
2. Use available tools when necessary
3. Observe the results
4. Continue until you have the final answer

Important:
- Respond in the SAME language as the user's input
- Use tools to gather information when needed
- Explain your reasoning process clearly
- Provide accurate and helpful answers"""

CLASSIC_REACT_SYSTEM = """You are an AI assistant using the ReAct framework. You MUST follow this exact format:

<Thought>
Your reasoning about what to do next
</Thought>

<Action>
tool_name
</Action>

<ActionInput>
{{"param1": "value1", "param2": "value2"}}
</ActionInput>

After receiving observation, continue with more Thought/Action/ActionInput cycles.

When you have gathered all necessary information through tools:
<Thought>
I now have enough information to answer
</Thought>

<FinalAnswer>
Your complete answer here
</FinalAnswer>

CRITICAL Rules:
- Respond in the SAME language as the user's question
- Use EXACT format with <tags>
- ActionInput must be valid JSON
- You MUST use tools to gather real-time information
- NEVER make up information
- Only provide <FinalAnswer> after using appropriate tools

Available Tools:
{tools}"""


@dataclass
class ReActStep:
    iteration: int
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    is_final: bool = False


@dataclass
class ReActTrace:
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_iterations: int = 0

    def add_step(self, step: ReActStep):
        self.steps.append(step)
        self.total_iterations = max(self.total_iterations, step.iteration + 1)

    def to_dict(self) -> Dict:
        return {
            "total_iterations": self.total_iterations,
            "steps": [
                {
                    "iteration": s.iteration,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "is_final": s.is_final,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
        }

    def clear(self):
        self.steps.clear()
        self.final_answer = None
        self.total_iterations = 0


class ReActAgent(Agent):

    def __init__(
        self,
        chater: Union[Chater, ChaterPool],
        name: Optional[str] = None,
        memory: Optional[Memory] = None,
        tools: Optional[ToolKit] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        tool_timeout: Optional[int] = None,
        log_config: Optional["LogConfig"] = None,
        on_stream: Optional[Callable[[ChatResponse], None]] = None,
        on_complete: Optional[Callable[[ChatResponse], None]] = None,
    ):

        super().__init__(
            chater=chater,
            name=name,
            memory=memory,
            tools=tools,
            tool_choice=tool_choice,
            system_prompt=system_prompt or REACT_SYSTEM_PROMPT,
            max_iterations=max_iterations,
            tool_timeout=tool_timeout,
            log_config=log_config,
            on_stream=on_stream,
            on_complete=on_complete,
        )
        self._react_trace = ReActTrace()

    async def reply(
        self,
        user_message: str,
        stream: bool = False,
        on_stream: Optional[Callable[[ChatResponse], None]] = None,
        on_complete: Optional[Callable[[ChatResponse], None]] = None,
    ) -> ChatResponse:
        self._react_trace.clear()
        self.logger.info(f"ReAct received: {user_message[:50]}...")

        self.memory.add_user(user_message)
        tools = self.tools.to_openai_tools() if self.tools else None
        tool_choice = self.tool_choice or ("auto" if self.tools else None)
        stream_cb = on_stream or self.on_stream
        complete_cb = on_complete or self.on_complete

        for iteration in range(self.max_iterations):
            step = ReActStep(iteration=iteration)
            history = self._build_history()

            response = await self.chater.chat(
                messages=history,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                on_stream=stream_cb,
                on_complete=complete_cb,
            )

            if response.reasoning_content:
                step.thought = response.reasoning_content

            self.memory.add(response)
            tool_calls = response.tool_calls or ([response.tool_call] if response.tool_call else [])

            if tool_calls:
                step.action = ", ".join(tc.fn_name for tc in tool_calls)
                step.action_input = {tc.fn_name: tc.get_args_dict() for tc in tool_calls}

                tool_results = await self._execute_tools_concurrent(tool_calls)
                observations = [r.tool_result.fn_output for r in tool_results if r.tool_result]
                step.observation = "\n".join(observations)

                self.memory.add(tool_results)
                self._react_trace.add_step(step)
            else:
                step.is_final = True
                self._react_trace.final_answer = response.content
                self._react_trace.add_step(step)

                response = self._run_post_reply_hooks(response, user_message)
                self._broadcast_to_audience(response)
                return response

        return response

    def get_trace(self) -> Dict:
        return self._react_trace.to_dict()

    def clear_memory(self):
        super().clear_memory()
        self._react_trace.clear()


class ClassicReActAgent(Agent):

    def __init__(
        self,
        chater: Union[Chater, ChaterPool],
        name: Optional[str] = None,
        memory: Optional[Memory] = None,
        tools: Optional[ToolKit] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        tool_timeout: Optional[int] = None,
        log_config: Optional["LogConfig"] = None,
        on_stream: Optional[Callable[[ChatResponse], None]] = None,
        on_complete: Optional[Callable[[ChatResponse], None]] = None,
    ):
        from ..core import LogConfig

        tools_desc = self._build_tools_description(tools)
        final_prompt = system_prompt or CLASSIC_REACT_SYSTEM.format(tools=tools_desc)

        super().__init__(
            chater=chater,
            name=name,
            memory=memory,
            tools=tools,
            system_prompt=final_prompt,
            max_iterations=max_iterations,
            tool_timeout=tool_timeout,
            log_config=log_config,
            on_stream=on_stream,
            on_complete=on_complete,
        )
        self._react_trace = ReActTrace()

    @staticmethod
    def _build_tools_description(tools: Optional[ToolKit]) -> str:
        if not tools:
            return "No tools available"
        desc = []
        for name in tools.list_tools():
            schema = tools.get_tool_schema(name)
            params = schema.get("parameters", {}).get("properties", {})
            params_str = json.dumps(params, ensure_ascii=False, indent=2)
            desc.append(f"- {name}: {schema.get('description', 'No description')}\n  Parameters: {params_str}")
        return "\n".join(desc)

    async def reply(
        self,
        user_message: str,
        stream: bool = False,
        on_stream: Optional[Callable[[ChatResponse], None]] = None,
        on_complete: Optional[Callable[[ChatResponse], None]] = None,
    ) -> ChatResponse:
        self._react_trace.clear()
        self.logger.info(f"Classic ReAct received: {user_message[:50]}...")

        self.memory.add_user(user_message)
        stream_cb = on_stream or self.on_stream
        complete_cb = on_complete or self.on_complete

        for iteration in range(self.max_iterations):
            step = ReActStep(iteration=iteration)
            history = self._build_history()

            response = await self.chater.chat(
                messages=history,
                stream=stream,
                on_stream=stream_cb,
                on_complete=complete_cb,
            )
            self.memory.add(response)
            content = response.content or ""

            thought = self._extract_tag(content, "Thought")
            action = self._extract_tag(content, "Action")
            action_input = self._extract_tag(content, "ActionInput")
            step.thought = thought

            if "<FinalAnswer>" in content:
                if iteration == 0 and self.tools and self.tools.list_tools():
                    self.logger.warning("Model tried to answer without tools")
                    self.memory.add_user("<Observation>Error: You must use tools first.</Observation>")
                    continue

                final_answer = self._extract_tag(content, "FinalAnswer") or content
                step.is_final = True
                self._react_trace.final_answer = final_answer
                self._react_trace.add_step(step)

                final_response = ChatResponse(
                    role="assistant",
                    content=final_answer,
                    id=response.id,
                    created=response.created,
                    usage=response.usage,
                )
                final_response = self._run_post_reply_hooks(final_response, user_message)
                self._broadcast_to_audience(final_response)
                return final_response

            if action and action_input:
                step.action = action.strip()
                try:
                    step.action_input = json.loads(action_input)
                except json.JSONDecodeError:
                    step.action_input = {"raw": action_input}

                output = await self._execute_tool(action.strip(), step.action_input or {})
                step.observation = str(output)
                self._react_trace.add_step(step)
                self.memory.add_user(f"<Observation>{output}</Observation>")
            else:
                self.logger.warning("No valid action found")
                if iteration == 0 and self.tools and self.tools.list_tools():
                    self.memory.add_user("<Observation>Error: Invalid format. Use <Action> and <ActionInput>.</Observation>")
                    continue

                step.is_final = True
                self._react_trace.final_answer = content
                self._react_trace.add_step(step)
                return self._run_post_reply_hooks(
                    ChatResponse(role="assistant", content=content),
                    user_message
                )

        return ChatResponse(role="assistant", content="Max iterations reached")

    async def _execute_tool(self, tool_name: str, args: Dict):
        try:
            if self.tool_timeout:
                return await asyncio.wait_for(
                    self.tools.execute(name=tool_name, **args),
                    timeout=self.tool_timeout,
                )
            return await self.tools.execute(name=tool_name, **args)
        except asyncio.TimeoutError:
            return f"Tool timeout after {self.tool_timeout}s"
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        pattern_open = f"<{tag}>(.*?)(?=<|$)"
        match_open = re.search(pattern_open, text, re.DOTALL | re.IGNORECASE)
        if match_open:
            return match_open.group(1).strip()
        return ""

    def get_trace(self) -> Dict:
        return self._react_trace.to_dict()

    def clear_memory(self):
        super().clear_memory()
        self._react_trace.clear()


if __name__ == "__main__":
    from ..core import get_chater_cfg, ChaterPool, ConsoleSpeaker, make_stream_callback
    from datetime import datetime

    async def get_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def calculate(expression: str) -> float:
        try:
            return float(eval(expression))
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")

    async def main():
        tools = ToolKit()
        tools.register(get_time, "get_time")
        tools.register(calculate, "calculate")

        speaker = ConsoleSpeaker()

        print("=" * 60)
        print("Test: ReActAgent")
        print("=" * 60)

        agent = ReActAgent(
            chater=ChaterPool([get_chater_cfg("ali")]),
            tools=tools,
        )

        response = await agent("What is 123 * 456?", stream=True, on_stream=make_stream_callback(speaker))
        print(f"\nReturn: {response.content}")
        print(f"Trace: {agent.get_trace()['total_iterations']} iterations")

    asyncio.run(main())
