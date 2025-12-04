import asyncio
from typing import Optional, Callable
from ._agent import Agent, StreamHandler
from ._model import ChatResponse
from ._trace import custom_span, get_current_trace
from ._msghub import msghub


class Runner:
    @staticmethod
    async def run(
        agent: Agent,
        user_message: str,
        stream: bool = False,
        on_stream: Optional[StreamHandler] = None,
        on_complete: Optional[StreamHandler] = None,
    ) -> ChatResponse:
        return await agent(user_message, stream=stream, on_stream=on_stream, on_complete=on_complete)

    @staticmethod
    async def run_with_reply(
        agent: Agent,
        user_message: str,
        stream: bool = False,
        on_stream: Optional[StreamHandler] = None,
        on_complete: Optional[StreamHandler] = None,
    ) -> ChatResponse:
        return await agent.reply(user_message, stream=stream, on_stream=on_stream, on_complete=on_complete)

    @staticmethod
    async def run_sequential(
        agents: list[Agent], initial_message: str, trace_name: str = "sequential_pipeline"
    ) -> ChatResponse:
        current_trace = get_current_trace()

        if current_trace:
            with custom_span(
                name=trace_name,
                data={
                    "agent_count": len(agents),
                    "agent_names": [a.name for a in agents],
                    "pattern": "sequential",
                },
            ):
                return await Runner._execute_sequential(agents, initial_message)
        return await Runner._execute_sequential(agents, initial_message)

    @staticmethod
    async def run_parallel(
        agents: list[Agent], message: str, trace_name: str = "parallel_execution"
    ) -> list[ChatResponse]:
        current_trace = get_current_trace()

        if current_trace:
            with custom_span(
                name=trace_name,
                data={
                    "agent_count": len(agents),
                    "agent_names": [a.name for a in agents],
                    "pattern": "parallel",
                },
            ) as parallel_span:
                tasks = [agent(message) for agent in agents]
                results = await asyncio.gather(*tasks)
                parallel_span.span_data.data["completed_count"] = len(results)
                return results
        tasks = [agent(message) for agent in agents]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def run_msghub(
        agents: list[Agent], announcement: ChatResponse, rounds: int = 1, trace_name: str = "msghub"
    ) -> list[ChatResponse]:
        current_trace = get_current_trace()

        async def _execute_msghub():
            results = []
            with msghub(agents, announcement=announcement):
                for round_num in range(rounds):
                    for agent in agents:
                        response = await agent.reply("")
                        if response:
                            results.append(response)
            return results

        if current_trace:
            with custom_span(
                name=trace_name,
                data={
                    "agent_count": len(agents),
                    "agent_names": [a.name for a in agents],
                    "pattern": "msghub",
                    "rounds": rounds,
                },
            ) as hub_span:
                results = await _execute_msghub()
                hub_span.span_data.data["total_responses"] = len(results)
                return results
        return await _execute_msghub()

    @staticmethod
    async def _execute_sequential(agents: list[Agent], initial_message: str) -> ChatResponse:
        current_msg = initial_message
        response = None

        for agent in agents:
            response = await agent(current_msg)
            if response and hasattr(response, "content"):
                current_msg = response.content

        return response
