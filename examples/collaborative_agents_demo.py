import asyncio
from qagent import (
    Memory,
    Agent,
    Chater,
    get_chater_cfg,
    Runner,
    msghub,
    ChatResponse,
)


async def demo_collaborative_agents():
    print("=" * 60)
    print("Demo: Collaborative Agents with MsgHub")
    print("=" * 60)

    chater = Chater(get_chater_cfg("ali"))

    expert1 = Agent(
        name="TechExpert",
        chater=chater,
        memory=Memory(),
        system_prompt="You are a technology expert. Give brief insights.",
    )

    expert2 = Agent(
        name="EthicsExpert",
        chater=chater,
        memory=Memory(),
        system_prompt="You are an ethics expert. Give brief insights.",
    )

    topic = ChatResponse(role="user", content="Discuss the impact of AI on society.")

    with msghub([expert1, expert2], announcement=topic):
        r1 = await expert1.reply("Share your perspective.")
        print(f"[{expert1.name}]: {r1.content}\n")

        r2 = await expert2.reply("Share your perspective.")
        print(f"[{expert2.name}]: {r2.content}\n")


async def demo_hooks():
    print("\n" + "=" * 60)
    print("Demo: Hook System")
    print("=" * 60)

    chater = Chater(get_chater_cfg("ali"))

    agent = Agent(
        name="HookedAgent",
        chater=chater,
        memory=Memory(),
        system_prompt="You are a helpful assistant.",
    )

    history = []

    @agent.pre_reply
    def log_input(msg):
        print(f"[PRE] Input: {msg[:50]}...")
        return msg

    @agent.post_reply
    def record_history(response):
        history.append(response.content)
        print(f"[POST] Recorded response")
        return response

    response = await agent("What is Python?")
    print(f"Response: {response.content[:100]}...")
    print(f"History entries: {len(history)}")


async def demo_sequential_agents():
    print("\n" + "=" * 60)
    print("Demo: Sequential Agent Pipeline")
    print("=" * 60)

    chater = Chater(get_chater_cfg("ali"))

    writer = Agent(
        name="Writer",
        chater=chater,
        memory=Memory(),
        system_prompt="You write short content.",
    )

    reviewer = Agent(
        name="Reviewer",
        chater=chater,
        memory=Memory(),
        system_prompt="You review and improve content briefly.",
    )

    draft = await writer("Write a one-sentence intro about AI.")
    print(f"[Writer]: {draft.content}\n")

    final = await reviewer(f"Review and improve: {draft.content}")
    print(f"[Reviewer]: {final.content}")


async def main():
    print("QAgent Collaborative Agents Demo")
    print("=" * 60)
    print()

    await demo_hooks()
    await demo_sequential_agents()
    await demo_collaborative_agents()


if __name__ == "__main__":
    asyncio.run(main())
