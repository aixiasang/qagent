import asyncio
from typing import AsyncGenerator
from ._agent import Agent
from ._model import ChatResponse
from ._trace import agent_span, custom_span, get_current_trace, get_current_span
from ._msghub import msghub


class Runner:
    @staticmethod
    async def run(
        agent: Agent,
        user_message: str,
        auto_speak: bool = False,
    ) -> ChatResponse:
        return await Runner._run_single(agent, user_message, auto_speak)

    @staticmethod
    async def run_streamed(
        agent: Agent,
        user_message: str,
        auto_speak: bool = False,
    ):
        async for response in Runner._run_stream(agent, user_message, auto_speak):
            yield response

    @staticmethod
    async def _run_single(
        agent: Agent,
        user_message: str,
        auto_speak: bool,
    ) -> ChatResponse:
        final_response = None
        async for response in agent.reply(user_message, stream=False, auto_speak=auto_speak):
            final_response = response
        return final_response

    @staticmethod
    async def _run_stream(
        agent: Agent,
        user_message: str,
        auto_speak: bool,
    ):
        async for response in agent.reply(user_message, stream=True, auto_speak=auto_speak):
            yield response

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
        else:
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
                tasks = [Runner._consume_agent_reply(agent, message) for agent in agents]
                results = await asyncio.gather(*tasks)
                parallel_span.span_data.data["completed_count"] = len(results)
                return results
        else:
            tasks = [Runner._consume_agent_reply(agent, message) for agent in agents]
            return await asyncio.gather(*tasks)

    @staticmethod
    async def run_msghub(
        agents: list[Agent], announcement: ChatResponse, rounds: int = 1, trace_name: str = "msghub"
    ) -> list[ChatResponse]:
        current_trace = get_current_trace()

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
                results = []
                with msghub(agents, announcement=announcement):
                    for round_num in range(rounds):
                        for agent in agents:
                            final_response = None
                            async for response in agent.reply("", auto_speak=True):
                                final_response = response
                            if final_response:
                                results.append(final_response)

                hub_span.span_data.data["total_responses"] = len(results)
                return results
        else:
            results = []
            with msghub(agents, announcement=announcement):
                for round_num in range(rounds):
                    for agent in agents:
                        final_response = None
                        async for response in agent.reply("", auto_speak=True):
                            final_response = response
                        if final_response:
                            results.append(final_response)
            return results

    @staticmethod
    async def _consume_agent_reply(agent: Agent, message: str) -> ChatResponse:
        final_response = None
        async for response in agent.reply(message):
            final_response = response
        return final_response

    @staticmethod
    async def _execute_sequential(agents: list[Agent], initial_message: str) -> ChatResponse:
        current_msg = initial_message
        final_response = None

        for agent in agents:
            async for response in agent.reply(current_msg):
                final_response = response

            if final_response and hasattr(final_response, "content"):
                current_msg = final_response.content

        return final_response
