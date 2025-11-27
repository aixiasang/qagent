from typing import Union, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ._agent import Agent
    from ._model import ChatResponse

from ._trace import custom_span, get_current_trace, get_current_span


async def sequential_pipeline(
    agents: list[Union["Agent", Callable]],
    initial_message: Optional["ChatResponse"] = None,
) -> "ChatResponse":
    if len(agents) == 0:
        raise ValueError("No agents provided in pipeline")

    current_trace = get_current_trace()

    if current_trace:
        current_span = get_current_span()
        if not (
            current_span
            and hasattr(current_span.span_data, "data")
            and current_span.span_data.data.get("pattern") == "sequential"
        ):
            with custom_span(
                name="sequential_pipeline",
                data={
                    "agent_count": len(agents),
                    "agent_names": [a.name if hasattr(a, "name") else "callable" for a in agents],
                    "pattern": "sequential",
                },
            ):
                return await _execute_pipeline(agents, initial_message)
        else:
            return await _execute_pipeline(agents, initial_message)
    else:
        return await _execute_pipeline(agents, initial_message)


async def _execute_pipeline(
    agents: list[Union["Agent", Callable]],
    initial_message: Optional["ChatResponse"] = None,
) -> "ChatResponse":
    current_msg = initial_message

    for agent in agents:
        if callable(agent):
            result = agent(current_msg)
            if hasattr(result, "__await__"):
                current_msg = await result
            else:
                current_msg = result
        else:
            last_response = None
            async for response in agent.reply(current_msg.content if current_msg else ""):
                last_response = response
            current_msg = last_response

    return current_msg


async def parallel_pipeline(
    agents: list[Union["Agent", Callable]], message: "ChatResponse"
) -> list["ChatResponse"]:
    import asyncio

    async def consume_agent_reply(agent, msg_content):
        last_response = None
        async for response in agent.reply(msg_content):
            last_response = response
        return last_response

    tasks = []
    for agent in agents:
        if callable(agent):
            result = agent(message)
            if hasattr(result, "__await__"):
                tasks.append(result)
            else:

                async def wrap_sync():
                    return result

                tasks.append(wrap_sync())
        else:
            tasks.append(consume_agent_reply(agent, message.content))

    return await asyncio.gather(*tasks)


async def conditional_pipeline(
    condition: Callable[["ChatResponse"], bool],
    true_agent: Union["Agent", Callable],
    false_agent: Union["Agent", Callable],
    message: "ChatResponse",
) -> "ChatResponse":
    agent = true_agent if condition(message) else false_agent

    if callable(agent):
        result = agent(message)
        if hasattr(result, "__await__"):
            return await result
        return result

    last_response = None
    async for response in agent.reply(message.content):
        last_response = response
    return last_response


async def loop_pipeline(
    agents: list[Union["Agent", Callable]],
    initial_message: "ChatResponse",
    max_iterations: int = 5,
    stop_condition: Optional[Callable[["ChatResponse"], bool]] = None,
) -> "ChatResponse":
    current_msg = initial_message

    for i in range(max_iterations):
        for agent in agents:
            if callable(agent):
                result = agent(current_msg)
                if hasattr(result, "__await__"):
                    current_msg = await result
                else:
                    current_msg = result
            else:
                last_response = None
                async for response in agent.reply(current_msg.content):
                    last_response = response
                current_msg = last_response

        if stop_condition and stop_condition(current_msg):
            break

    return current_msg
