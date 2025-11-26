import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyagent import Flow, END, chain, Agent, Chater, Memory, get_chater_cfg, trace


async def example_plan_execute_review():
    planner = Agent(
        name="Planner",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a planning expert. Create a brief step-by-step plan for the task. Be concise.",
    )
    executor = Agent(
        name="Executor",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Execute the plan step by step. Show results briefly.",
    )
    reviewer = Agent(
        name="Reviewer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Review the execution. Say APPROVED if good, or REJECTED with feedback.",
    )

    flow = Flow("plan_execute_review")
    flow.add("plan", planner)
    flow.add("execute", executor)
    flow.add("review", reviewer)
    flow.max_loops(5)

    flow.route("plan").to("execute")
    flow.route("execute").to("review")
    flow.route("review").when(lambda r: "APPROVED" in r.content.upper()).to(END).default().to("plan")

    with trace("plan_execute_review"):
        result = await flow.reply("Write a function to calculate fibonacci numbers")

    print("Result:", result.content[:300] if result.content else "N/A")


async def example_multi_expert_debate():
    optimist = Agent(
        name="Optimist",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You see positive aspects. Give 2-3 sentences of optimistic view.",
    )
    pessimist = Agent(
        name="Pessimist",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You see risks and problems. Give 2-3 sentences of cautious view.",
    )
    moderator = Agent(
        name="Moderator",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Summarize both viewpoints fairly in 2-3 sentences. End with CONCLUSION.",
    )

    class DebateState:
        def __init__(self):
            self.round = 0

    state = DebateState()

    def track_round(response):
        state.round += 1
        return state.round >= 2

    flow = Flow("debate")
    flow.add("optimist", optimist)
    flow.add("pessimist", pessimist)
    flow.add("moderator", moderator)
    flow.max_loops(10)

    flow.route("optimist").to("pessimist")
    flow.route("pessimist").when(track_round).to("moderator").default().to("optimist")
    flow.route("moderator").to(END)

    result = await flow.reply("Should companies invest heavily in AI right now?")
    print("Debate result:", result.content)


async def example_parallel_analysis():
    tech_analyst = Agent(
        name="TechAnalyst",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Analyze technical feasibility. Give brief assessment.",
    )
    biz_analyst = Agent(
        name="BusinessAnalyst",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Analyze business value and ROI. Give brief assessment.",
    )
    risk_analyst = Agent(
        name="RiskAnalyst",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Analyze potential risks. Give brief assessment.",
    )
    synthesizer = Agent(
        name="Synthesizer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Combine all analyses into a final recommendation. Be concise.",
    )

    flow = Flow("parallel_analysis")
    flow.parallel("analysts", [tech_analyst, biz_analyst, risk_analyst])
    flow.add("synthesize", synthesizer)

    flow.route("analysts").to("synthesize")
    flow.route("synthesize").to(END)

    result = await flow.reply("Should we migrate our monolith to microservices?")
    print("Parallel analysis result:", result.content)


async def example_nested_flow():
    draft_writer = Agent(
        name="DraftWriter",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Write or improve a draft based on feedback. Keep it brief.",
    )
    draft_reviewer = Agent(
        name="DraftReviewer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Review the draft. Say GOOD if acceptable, or give specific feedback.",
    )

    inner_flow = Flow("draft_loop").add("write", draft_writer).add("review", draft_reviewer).max_loops(3)
    inner_flow.route("write").to("review")
    inner_flow.route("review").when(lambda r: "GOOD" in r.content.upper()).to(END).default().to("write")

    publisher = Agent(
        name="Publisher",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Format and finalize the content for publication.",
    )

    outer_flow = Flow("publish_flow")
    outer_flow.add("drafting", inner_flow)
    outer_flow.add("publish", publisher)

    outer_flow.route("drafting").to("publish")
    outer_flow.route("publish").to(END)

    result = await flow.reply("Write a product announcement for our new AI feature")
    print("Nested flow result:", result.content)


async def example_dynamic_routing():
    classifier = Agent(
        name="Classifier",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Classify the query as: TECHNICAL, CREATIVE, or GENERAL. Reply with just the category.",
    )
    tech_expert = Agent(
        name="TechExpert",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a technical expert. Answer technical questions concisely.",
    )
    creative_expert = Agent(
        name="CreativeExpert",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a creative writer. Respond creatively and imaginatively.",
    )
    general_expert = Agent(
        name="GeneralExpert",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a helpful assistant. Answer questions helpfully.",
    )

    flow = Flow("dynamic_routing")
    flow.add("classify", classifier)
    flow.add("tech", tech_expert)
    flow.add("creative", creative_expert)
    flow.add("general", general_expert)

    flow.route("classify").when(lambda r: "TECHNICAL" in r.content.upper()).to("tech").when(lambda r: "CREATIVE" in r.content.upper()).to("creative").default().to("general")
    flow.route("tech").to(END)
    flow.route("creative").to(END)
    flow.route("general").to(END)

    queries = [
        "How does a hash table work?",
        "Write a poem about autumn",
        "What's the weather like today?",
    ]

    for query in queries:
        result = await flow.reply(query)
        print(f"Query: {query[:30]}...")
        print(f"Response: {result.content[:100]}...")
        print()


async def example_iterative_refinement():
    generator = Agent(
        name="Generator",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Generate or improve content based on the request. Be creative.",
    )
    critic = Agent(
        name="Critic",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Critique the content. Score 1-10. If score >= 8, say EXCELLENT. Otherwise give improvement suggestions.",
    )

    flow = Flow("iterative_refinement").add("generate", generator).add("critique", critic).max_loops(4)
    flow.route("generate").to("critique")
    flow.route("critique").when(lambda r: "EXCELLENT" in r.content.upper() or "10" in r.content or "9" in r.content).to(END).default().to("generate")

    result = await flow.reply("Create a memorable tagline for a coffee shop")
    print("Iterative refinement result:", result.content)


async def example_chain_helper():
    summarizer = Agent(
        name="Summarizer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Summarize the input in one paragraph.",
    )
    translator = Agent(
        name="Translator",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Translate the summary to Chinese.",
    )
    formatter = Agent(
        name="Formatter",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Format the content with a title and bullet points.",
    )

    flow = chain(summarizer, translator, formatter)

    result = await flow.reply(
        "Artificial intelligence is transforming industries worldwide. "
        "From healthcare diagnostics to autonomous vehicles, AI applications "
        "are becoming increasingly sophisticated and widespread."
    )
    print("Chain result:", result.content)


async def main():
    print("=" * 60)
    print("1. Plan-Execute-Review Pattern")
    print("=" * 60)
    await example_plan_execute_review()

    print("\n" + "=" * 60)
    print("2. Multi-Expert Debate")
    print("=" * 60)
    await example_multi_expert_debate()

    print("\n" + "=" * 60)
    print("3. Parallel Analysis")
    print("=" * 60)
    await example_parallel_analysis()

    print("\n" + "=" * 60)
    print("4. Dynamic Routing")
    print("=" * 60)
    await example_dynamic_routing()

    print("\n" + "=" * 60)
    print("5. Iterative Refinement")
    print("=" * 60)
    await example_iterative_refinement()

    print("\n" + "=" * 60)
    print("6. Chain Helper")
    print("=" * 60)
    await example_chain_helper()


if __name__ == "__main__":
    asyncio.run(main())
