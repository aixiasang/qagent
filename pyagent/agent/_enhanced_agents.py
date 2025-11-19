import re
from typing import Optional, Union, AsyncGenerator, Dict, List

from ..core import Agent, Memory, ChatResponse, Chater, ChaterPool, ToolKit, Speaker

from ..prompt import (
    build_plan_react_prompt,
    build_reflection_prompt,
    build_self_consistency_prompt,
)


class PlanReActAgent(Agent):
    def __init__(
        self,
        name: str,
        chater: Union[Chater, ChaterPool],
        memory: Memory,
        tools: Optional[ToolKit] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        tool_timeout: Optional[int] = None,
        enable_logging: bool = False,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        speaker: Optional[Speaker] = None,
    ):
        tools_desc = self._build_tools_desc(tools)
        final_prompt = system_prompt or build_plan_react_prompt(tools_desc)

        super().__init__(
            name=name,
            chater=chater,
            memory=memory,
            tools=tools,
            system_prompt=final_prompt,
            max_iterations=max_iterations,
            tool_timeout=tool_timeout,
            enable_logging=enable_logging,
            log_file=log_file,
            log_level=log_level,
            speaker=speaker,
        )

        self.plan_history = []

    @staticmethod
    def _build_tools_desc(tools: Optional[ToolKit]) -> str:
        if not tools:
            return "No tools available"

        tools_desc = []
        for tool_name in tools.list_tools():
            schema = tools.get_tool_schema(tool_name)
            desc = f"- {tool_name}: {schema.get('description', 'No description')}"
            tools_desc.append(desc)

        return "\n".join(tools_desc)

    def _extract_plan(self, text: str) -> Optional[str]:
        plan_match = re.search(r"<Plan>(.*?)</Plan>", text, re.DOTALL | re.IGNORECASE)
        if plan_match:
            return plan_match.group(1).strip()
        return None

    def _extract_final_answer(self, text: str) -> Optional[str]:
        answer_match = re.search(
            r"<FinalAnswer>(.*?)</FinalAnswer>", text, re.DOTALL | re.IGNORECASE
        )
        if answer_match:
            return answer_match.group(1).strip()

        if "<FinalAnswer>" in text:
            idx = text.find("<FinalAnswer>")
            remaining = text[idx + len("<FinalAnswer>") :]
            return remaining.strip()

        return None

    async def reply(
        self,
        user_message: str,
        stream: bool = False,
        auto_speak: bool = True,
    ) -> AsyncGenerator[ChatResponse, None]:

        self.logger.info(f"Plan-ReAct started: {user_message}")

        full_response = ""
        final_response_obj = None

        async for response in super().reply(user_message, stream=False, auto_speak=False):
            full_response += response.content if response.content else ""
            final_response_obj = response

        plan = self._extract_plan(full_response)
        if plan:
            self.plan_history.append(plan)
            self.logger.info(f"Plan created with {len(plan.split('Step'))-1} steps")

        final_answer = self._extract_final_answer(full_response)

        if final_answer:
            result = ChatResponse(
                role="assistant",
                content=final_answer,
                id=final_response_obj.id if final_response_obj else None,
                created=final_response_obj.created if final_response_obj else None,
                usage=final_response_obj.usage if final_response_obj else None,
            )
        else:
            result = ChatResponse(
                role="assistant",
                content=full_response,
                id=final_response_obj.id if final_response_obj else None,
                created=final_response_obj.created if final_response_obj else None,
                usage=final_response_obj.usage if final_response_obj else None,
            )

        if auto_speak:
            self.speak(result, stream=False)

        yield result

    def get_last_plan(self) -> Optional[str]:
        return self.plan_history[-1] if self.plan_history else None

    def clear_plan_history(self):
        self.plan_history.clear()


class ReflectionAgent(Agent):
    def __init__(
        self,
        name: str,
        chater: Union[Chater, ChaterPool],
        memory: Memory,
        tools: Optional[ToolKit] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        tool_timeout: Optional[int] = None,
        enable_logging: bool = False,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        speaker: Optional[Speaker] = None,
    ):
        tools_desc = self._build_tools_desc(tools)
        final_prompt = system_prompt or build_reflection_prompt(tools_desc)

        super().__init__(
            name=name,
            chater=chater,
            memory=memory,
            tools=tools,
            system_prompt=final_prompt,
            max_iterations=max_iterations,
            tool_timeout=tool_timeout,
            enable_logging=enable_logging,
            log_file=log_file,
            log_level=log_level,
            speaker=speaker,
        )

        self.reflection_records = []

    @staticmethod
    def _build_tools_desc(tools: Optional[ToolKit]) -> str:
        if not tools:
            return "No tools available"

        tools_desc = []
        for tool_name in tools.list_tools():
            schema = tools.get_tool_schema(tool_name)
            desc = f"- {tool_name}: {schema.get('description', 'No description')}"
            tools_desc.append(desc)

        return "\n".join(tools_desc)

    def _extract_initial_answer(self, text: str) -> Optional[str]:
        match = re.search(r"<InitialAnswer>(.*?)</InitialAnswer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_reflection(self, text: str) -> Optional[str]:
        match = re.search(
            r"<SelfReflection>(.*?)</SelfReflection>", text, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return None

    def _extract_improved_answer(self, text: str) -> Optional[str]:
        match = re.search(
            r"<ImprovedAnswer>(.*?)</ImprovedAnswer>", text, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return None

    def _extract_final_answer(self, text: str) -> Optional[str]:
        match = re.search(r"<FinalAnswer>(.*?)</FinalAnswer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        if "<FinalAnswer>" in text:
            idx = text.find("<FinalAnswer>")
            remaining = text[idx + len("<FinalAnswer>") :]
            end_idx = remaining.find("</FinalAnswer>")
            if end_idx != -1:
                return remaining[:end_idx].strip()
            return remaining.strip()

        return None

    async def reply(
        self,
        user_message: str,
        stream: bool = False,
        auto_speak: bool = True,
    ) -> AsyncGenerator[ChatResponse, None]:

        self.logger.info(f"Reflection started: {user_message}")

        full_response = ""
        final_response_obj = None

        async for response in super().reply(user_message, stream=False, auto_speak=False):
            full_response += response.content if response.content else ""
            final_response_obj = response

        initial = self._extract_initial_answer(full_response)
        reflection = self._extract_reflection(full_response)
        improved = self._extract_improved_answer(full_response)
        final = self._extract_final_answer(full_response)

        if reflection:
            self.reflection_records.append(
                {"initial": initial, "reflection": reflection, "improved": improved, "final": final}
            )
            self.logger.info("Reflection completed")

        answer_text = final or improved or initial or full_response

        result = ChatResponse(
            role="assistant",
            content=answer_text,
            id=final_response_obj.id if final_response_obj else None,
            created=final_response_obj.created if final_response_obj else None,
            usage=final_response_obj.usage if final_response_obj else None,
        )

        if auto_speak:
            self.speak(result, stream=False)

        yield result

    def get_reflection_records(self) -> List[Dict]:
        return self.reflection_records

    def clear_reflection_records(self):
        self.reflection_records.clear()


class SelfConsistencyAgent(Agent):
    def __init__(
        self,
        name: str,
        chater: Union[Chater, ChaterPool],
        memory: Memory,
        tools: Optional[ToolKit] = None,
        system_prompt: Optional[str] = None,
        num_paths: int = 3,
        max_iterations: int = 5,
        tool_timeout: Optional[int] = None,
        enable_logging: bool = False,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        speaker: Optional[Speaker] = None,
    ):
        tools_desc = self._build_tools_desc(tools)
        final_prompt = system_prompt or build_self_consistency_prompt(tools_desc, num_paths)

        super().__init__(
            name=name,
            chater=chater,
            memory=memory,
            tools=tools,
            system_prompt=final_prompt,
            max_iterations=max_iterations,
            tool_timeout=tool_timeout,
            enable_logging=enable_logging,
            log_file=log_file,
            log_level=log_level,
            speaker=speaker,
        )

        self.num_paths = num_paths
        self.reasoning_paths = []

    @staticmethod
    def _build_tools_desc(tools: Optional[ToolKit]) -> str:
        if not tools:
            return "No tools available"

        tools_desc = []
        for tool_name in tools.list_tools():
            schema = tools.get_tool_schema(tool_name)
            desc = f"- {tool_name}: {schema.get('description', 'No description')}"
            tools_desc.append(desc)

        return "\n".join(tools_desc)

    def _extract_reasoning_paths(self, text: str) -> List[str]:
        paths = []
        for i in range(1, self.num_paths + 1):
            pattern = f"<ReasoningPath{i}>(.*?)</ReasoningPath{i}>"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                paths.append(match.group(1).strip())
        return paths

    def _extract_comparison(self, text: str) -> Optional[str]:
        match = re.search(r"<Comparison>(.*?)</Comparison>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_final_answer(self, text: str) -> Optional[str]:
        match = re.search(r"<FinalAnswer>(.*?)</FinalAnswer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        if "<FinalAnswer>" in text:
            idx = text.find("<FinalAnswer>")
            remaining = text[idx + len("<FinalAnswer>") :]
            end_idx = remaining.find("</FinalAnswer>")
            if end_idx != -1:
                return remaining[:end_idx].strip()
            return remaining.strip()

        return None

    async def reply(
        self,
        user_message: str,
        stream: bool = False,
        auto_speak: bool = True,
    ) -> AsyncGenerator[ChatResponse, None]:

        self.logger.info(f"Self-consistency started with {self.num_paths} paths: {user_message}")

        full_response = ""
        final_response_obj = None

        async for response in super().reply(user_message, stream=False, auto_speak=False):
            full_response += response.content if response.content else ""
            final_response_obj = response

        paths = self._extract_reasoning_paths(full_response)
        comparison = self._extract_comparison(full_response)
        final = self._extract_final_answer(full_response)

        if paths:
            self.reasoning_paths = paths
            self.logger.info(f"Generated {len(paths)} reasoning paths")

        answer_text = final or full_response

        result = ChatResponse(
            role="assistant",
            content=answer_text,
            id=final_response_obj.id if final_response_obj else None,
            created=final_response_obj.created if final_response_obj else None,
            usage=final_response_obj.usage if final_response_obj else None,
        )

        if auto_speak:
            self.speak(result, stream=False)

        yield result

    def get_reasoning_paths(self) -> List[str]:
        return self.reasoning_paths

    def clear_reasoning_paths(self):
        self.reasoning_paths.clear()
