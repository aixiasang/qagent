import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyagent import Flow, END, chain, Agent, Chater, Memory, get_chater_cfg


async def test_simple_chain():
    agent_a = Agent(
        name="StepA",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You receive input and add 'Step A complete.' to it. Be very brief.",
    )
    agent_b = Agent(
        name="StepB",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You receive input and add 'Step B complete.' to it. Be very brief.",
    )

    flow = Flow("simple_chain").add("a", agent_a).add("b", agent_b)

    result = await flow.reply("Start the process")
    print("Simple chain result:")
    print(result.content)


async def test_conditional_routing():
    classifier = Agent(
        name="Classifier",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Classify the input as TECHNICAL or CREATIVE. Reply with just the category word.",
    )
    tech_agent = Agent(
        name="TechExpert",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a technical expert. Give a brief technical response.",
    )
    creative_agent = Agent(
        name="CreativeExpert",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a creative writer. Give a brief creative response.",
    )

    flow = Flow("conditional")
    flow.add("classify", classifier)
    flow.add("tech", tech_agent)
    flow.add("creative", creative_agent)

    flow.route("classify").when(lambda r: "TECHNICAL" in r.content.upper()).to("tech").when(lambda r: "CREATIVE" in r.content.upper()).to("creative").default().to("tech")
    flow.route("tech").to(END)
    flow.route("creative").to(END)
    flow.entry("classify")

    result = await flow.reply("How does a computer CPU work?")
    print("Conditional routing result:")
    print(result.content)


async def test_review_loop():
    writer = Agent(
        name="Writer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Write or improve a short haiku about the given topic. Be creative.",
    )
    reviewer = Agent(
        name="Reviewer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Review the haiku. If it's good (proper 5-7-5 syllables and meaningful), say 'APPROVED'. Otherwise say 'REJECTED' with brief feedback.",
    )

    flow = Flow("review_loop").add("write", writer).add("review", reviewer).max_loops(5)
    flow.route("write").to("review")
    flow.route("review").when(lambda r: "APPROVED" in r.content.upper()).to(END).default().to("write")

    result = await flow.reply("programming")
    print("Review loop result:")
    print(result.content)


async def test_chain_helper():
    agent1 = Agent(
        name="Analyzer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Analyze the input briefly.",
    )
    agent2 = Agent(
        name="Summarizer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Summarize the analysis in one sentence.",
    )

    flow = chain(agent1, agent2)
    result = await flow.reply("Artificial intelligence is transforming healthcare")
    print("Chain helper result:")
    print(result.content)


async def main():
    print("=" * 60)
    print("1. Simple Chain Flow")
    print("=" * 60)
    await test_simple_chain()

    print("\n" + "=" * 60)
    print("2. Conditional Routing")
    print("=" * 60)
    await test_conditional_routing()

    print("\n" + "=" * 60)
    print("3. Review Loop")
    print("=" * 60)
    await test_review_loop()

    print("\n" + "=" * 60)
    print("4. Chain Helper")
    print("=" * 60)
    await test_chain_helper()


if __name__ == "__main__":
    asyncio.run(main())
