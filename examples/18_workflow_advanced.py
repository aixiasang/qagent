import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyagent import (
    StateGraph,
    WorkflowBuilder,
    START,
    END,
    Channel,
    AppendReducer,
    Agent,
    Memory,
    ToolKit,
    Chater,
    get_chater_cfg,
    trace,
)


async def plan_execute_review_workflow():
    planner = Agent(
        name="Planner",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a planning expert. Create a step-by-step plan for the given task. Be concise.",
    )

    executor = Agent(
        name="Executor",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are an execution expert. Execute the plan step by step. Be concise.",
    )

    reviewer = Agent(
        name="Reviewer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a review expert. Review the execution result. Reply with APPROVED if good, or REJECTED with feedback.",
    )

    graph = StateGraph()
    graph._channels = {
        "messages": Channel(default=[], reducer=AppendReducer()),
        "plan": Channel(default=""),
        "execution": Channel(default=""),
        "review": Channel(default=""),
        "status": Channel(default="pending"),
        "iteration": Channel(default=0),
    }

    async def plan_node(state):
        task = state.get("task", "")
        last_response = None
        async for response in planner.reply(f"Create a plan for: {task}"):
            last_response = response
        return {"plan": last_response.content if last_response else ""}

    async def execute_node(state):
        plan = state.get("plan", "")
        last_response = None
        async for response in executor.reply(f"Execute this plan:\n{plan}"):
            last_response = response
        return {"execution": last_response.content if last_response else ""}

    async def review_node(state):
        execution = state.get("execution", "")
        iteration = state.get("iteration", 0)
        last_response = None
        async for response in reviewer.reply(f"Review this execution (iteration {iteration + 1}):\n{execution}"):
            last_response = response

        review_content = last_response.content if last_response else ""
        status = "approved" if "APPROVED" in review_content.upper() else "rejected"

        return {
            "review": review_content,
            "status": status,
            "iteration": iteration + 1,
        }

    def route_review(state):
        status = state.get("status", "")
        iteration = state.get("iteration", 0)
        if status == "approved" or iteration >= 3:
            return END
        return "plan"

    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("review", review_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "review")
    graph.add_conditional_edges("review", route_review, {"plan": "plan"})

    compiled = graph.compile().with_config(max_iterations=15)

    with trace("plan_execute_review"):
        result = await compiled.ainvoke({"task": "Write a haiku about programming"})

    print("Final Status:", result["status"])
    print("Iterations:", result["iteration"])
    print("\nPlan:", result["plan"][:200] if result["plan"] else "N/A")
    print("\nExecution:", result["execution"][:200] if result["execution"] else "N/A")
    print("\nReview:", result["review"][:200] if result["review"] else "N/A")


async def multi_agent_debate():
    optimist = Agent(
        name="Optimist",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are an optimist. Always find positive aspects. Be concise (2-3 sentences).",
    )

    pessimist = Agent(
        name="Pessimist",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a pessimist. Always find potential problems. Be concise (2-3 sentences).",
    )

    moderator = Agent(
        name="Moderator",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a neutral moderator. Summarize both viewpoints fairly. Be concise.",
    )

    graph = StateGraph()
    graph._channels = {
        "topic": Channel(default=""),
        "optimist_view": Channel(default=""),
        "pessimist_view": Channel(default=""),
        "summary": Channel(default=""),
        "round": Channel(default=0),
        "debate_history": Channel(default=[], reducer=AppendReducer()),
    }

    async def optimist_node(state):
        topic = state.get("topic", "")
        pessimist_view = state.get("pessimist_view", "")
        prompt = f"Topic: {topic}"
        if pessimist_view:
            prompt += f"\nPessimist said: {pessimist_view}\nRespond to this."

        last_response = None
        async for response in optimist.reply(prompt):
            last_response = response

        return {
            "optimist_view": last_response.content if last_response else "",
            "debate_history": [{"speaker": "Optimist", "content": last_response.content if last_response else ""}],
        }

    async def pessimist_node(state):
        topic = state.get("topic", "")
        optimist_view = state.get("optimist_view", "")
        prompt = f"Topic: {topic}\nOptimist said: {optimist_view}\nRespond to this."

        last_response = None
        async for response in pessimist.reply(prompt):
            last_response = response

        return {
            "pessimist_view": last_response.content if last_response else "",
            "round": state.get("round", 0) + 1,
            "debate_history": [{"speaker": "Pessimist", "content": last_response.content if last_response else ""}],
        }

    async def moderator_node(state):
        optimist_view = state.get("optimist_view", "")
        pessimist_view = state.get("pessimist_view", "")
        prompt = f"Optimist view: {optimist_view}\nPessimist view: {pessimist_view}\nProvide a balanced summary."

        last_response = None
        async for response in moderator.reply(prompt):
            last_response = response

        return {"summary": last_response.content if last_response else ""}

    def should_continue(state):
        if state.get("round", 0) >= 2:
            return "moderate"
        return "optimist"

    graph.add_node("optimist", optimist_node)
    graph.add_node("pessimist", pessimist_node)
    graph.add_node("moderate", moderator_node)

    graph.add_edge(START, "optimist")
    graph.add_edge("optimist", "pessimist")
    graph.add_conditional_edges("pessimist", should_continue, {"optimist": "optimist", "moderate": "moderate"})
    graph.add_edge("moderate", END)

    compiled = graph.compile().with_config(max_iterations=20)

    result = await compiled.ainvoke({"topic": "Artificial Intelligence in education"})

    print("Debate Topic:", result["topic"])
    print("Rounds:", result["round"])
    print("\nDebate History:")
    for entry in result["debate_history"]:
        print(f"  [{entry['speaker']}]: {entry['content'][:100]}...")
    print("\nModerator Summary:", result["summary"])


async def workflow_builder_example():
    agent = Agent(
        name="Assistant",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a helpful assistant. Be concise.",
    )

    def preprocess(state):
        input_text = state.get("input", "")
        return {"processed_input": input_text.strip().lower()}

    def postprocess(state):
        output = state.get("agent_output", "")
        return {"final_output": f"[Response] {output}"}

    wf = (
        WorkflowBuilder("assistant_workflow")
        .add_channel("processed_input", default="")
        .add_channel("agent_output", default="")
        .add_channel("final_output", default="")
        .add_function("preprocess", preprocess)
        .add_agent("assistant", agent, input_key="processed_input", output_key="agent_output")
        .add_function("postprocess", postprocess)
        .chain("preprocess", "assistant", "postprocess")
        .set_entry("preprocess")
        .set_exit("postprocess")
        .build()
    )

    result = await wf.ainvoke({"input": "   What is Python?   "})

    print("Input:", result.get("input"))
    print("Processed:", result.get("processed_input"))
    print("Agent Output:", result.get("agent_output", "")[:100])
    print("Final Output:", result.get("final_output", "")[:100])


async def main():
    print("=" * 60)
    print("1. Plan-Execute-Review Workflow")
    print("=" * 60)
    await plan_execute_review_workflow()

    print("\n" + "=" * 60)
    print("2. Multi-Agent Debate")
    print("=" * 60)
    await multi_agent_debate()

    print("\n" + "=" * 60)
    print("3. WorkflowBuilder Example")
    print("=" * 60)
    await workflow_builder_example()


if __name__ == "__main__":
    asyncio.run(main())
