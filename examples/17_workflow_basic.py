import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyagent import StateGraph, START, END, Channel, AppendReducer, MergeReducer


async def basic_sequential():
    def step1(state):
        return {"value": state.get("input", 0) + 10}

    def step2(state):
        return {"value": state["value"] * 2}

    def step3(state):
        return {"output": f"Final: {state['value']}"}

    graph = StateGraph()
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)

    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    compiled = graph.compile()
    result = await compiled.ainvoke({"input": 5})

    print("Sequential result:", result)


async def conditional_routing():
    def analyze(state):
        score = state.get("score", 0)
        return {"category": "high" if score >= 80 else "medium" if score >= 60 else "low"}

    def high_handler(state):
        return {"message": "Excellent performance!"}

    def medium_handler(state):
        return {"message": "Good job, keep improving!"}

    def low_handler(state):
        return {"message": "Need more practice."}

    def route(state):
        return state.get("category", "low")

    graph = StateGraph()
    graph.add_node("analyze", analyze)
    graph.add_node("high", high_handler)
    graph.add_node("medium", medium_handler)
    graph.add_node("low", low_handler)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        route,
        {"high": "high", "medium": "medium", "low": "low"},
    )
    graph.add_edge("high", END)
    graph.add_edge("medium", END)
    graph.add_edge("low", END)

    compiled = graph.compile()

    for score in [95, 75, 45]:
        result = await compiled.ainvoke({"score": score})
        print(f"Score {score}: {result['message']}")


async def loop_with_condition():
    def increment(state):
        current = state.get("counter", 0)
        return {"counter": current + 1, "history": [f"Step {current + 1}"]}

    def should_continue(state):
        return "continue" if state.get("counter", 0) < 5 else END

    graph = StateGraph()
    graph._channels = {
        "counter": Channel(default=0),
        "history": Channel(default=[], reducer=AppendReducer()),
    }

    graph.add_node("increment", increment)

    graph.add_edge(START, "increment")
    graph.add_conditional_edges("increment", should_continue, {"continue": "increment"})

    compiled = graph.compile().with_config(max_iterations=20)
    result = await compiled.ainvoke({})

    print("Loop result:", result)


async def streaming_execution():
    def step1(state):
        return {"progress": 25, "status": "Started"}

    def step2(state):
        return {"progress": 50, "status": "Processing"}

    def step3(state):
        return {"progress": 75, "status": "Almost done"}

    def step4(state):
        return {"progress": 100, "status": "Complete"}

    graph = StateGraph()
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)
    graph.add_node("step4", step4)

    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", "step4")
    graph.add_edge("step4", END)

    compiled = graph.compile()

    print("Streaming execution:")
    async for node_id, state in compiled.astream({}):
        print(f"  [{node_id}] Progress: {state['progress']}% - {state['status']}")


async def reducer_example():
    graph = StateGraph()
    graph._channels = {
        "messages": Channel(default=[], reducer=AppendReducer()),
        "metadata": Channel(default={}, reducer=MergeReducer()),
        "total": Channel(default=0),
    }

    def add_user_msg(state):
        return {
            "messages": [{"role": "user", "content": state.get("input", "")}],
            "metadata": {"user_added": True},
        }

    def add_assistant_msg(state):
        return {
            "messages": [{"role": "assistant", "content": "I understand."}],
            "metadata": {"assistant_added": True},
            "total": len(state.get("messages", [])) + 1,
        }

    graph.add_node("user", add_user_msg)
    graph.add_node("assistant", add_assistant_msg)

    graph.add_edge(START, "user")
    graph.add_edge("user", "assistant")
    graph.add_edge("assistant", END)

    compiled = graph.compile()
    result = await compiled.ainvoke({"input": "Hello!"})

    print("Reducer result:")
    print(f"  Messages: {result['messages']}")
    print(f"  Metadata: {result['metadata']}")
    print(f"  Total: {result['total']}")


async def main():
    print("=" * 60)
    print("1. Basic Sequential Workflow")
    print("=" * 60)
    await basic_sequential()

    print("\n" + "=" * 60)
    print("2. Conditional Routing")
    print("=" * 60)
    await conditional_routing()

    print("\n" + "=" * 60)
    print("3. Loop with Condition")
    print("=" * 60)
    await loop_with_condition()

    print("\n" + "=" * 60)
    print("4. Streaming Execution")
    print("=" * 60)
    await streaming_execution()

    print("\n" + "=" * 60)
    print("5. Reducer Example")
    print("=" * 60)
    await reducer_example()


if __name__ == "__main__":
    asyncio.run(main())
