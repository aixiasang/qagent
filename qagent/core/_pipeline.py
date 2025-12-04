from typing import Union, Callable, Optional
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
        if callable(agent) and not hasattr(agent, "reply"):
            result = agent(current_msg)
            if hasattr(result, "__await__"):
                current_msg = await result
            else:
                current_msg = result
        else:
            msg_content = current_msg.content if current_msg else ""
            current_msg = await agent(msg_content)

    return current_msg


async def parallel_pipeline(
    agents: list[Union["Agent", Callable]], message: "ChatResponse"
) -> list["ChatResponse"]:
    import asyncio

    tasks = []
    for agent in agents:
        if callable(agent) and not hasattr(agent, "reply"):
            result = agent(message)
            if hasattr(result, "__await__"):
                tasks.append(result)
            else:
                async def wrap_sync(r=result):
                    return r
                tasks.append(wrap_sync())
        else:
            tasks.append(agent(message.content))

    return await asyncio.gather(*tasks)


async def conditional_pipeline(
    condition: Callable[["ChatResponse"], bool],
    true_agent: Union["Agent", Callable],
    false_agent: Union["Agent", Callable],
    message: "ChatResponse",
) -> "ChatResponse":
    agent = true_agent if condition(message) else false_agent

    if callable(agent) and not hasattr(agent, "reply"):
        result = agent(message)
        if hasattr(result, "__await__"):
            return await result
        return result

    return await agent(message.content)


async def loop_pipeline(
    agents: list[Union["Agent", Callable]],
    initial_message: "ChatResponse",
    max_iterations: int = 5,
    stop_condition: Optional[Callable[["ChatResponse"], bool]] = None,
) -> "ChatResponse":
    current_msg = initial_message

    for i in range(max_iterations):
        for agent in agents:
            if callable(agent) and not hasattr(agent, "reply"):
                result = agent(current_msg)
                if hasattr(result, "__await__"):
                    current_msg = await result
                else:
                    current_msg = result
            else:
                current_msg = await agent(current_msg.content)

        if stop_condition and stop_condition(current_msg):
            break

    return current_msg
