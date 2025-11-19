import asyncio
from core import Agent, Memory, get_chater_cfg, ChaterPool, ChatResponse, msghub


async def main():
    pro = Agent(
        name="Pro",
        chater=ChaterPool([
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        system_prompt="You argue FOR AI regulation. Be persuasive and brief."
    )

    con = Agent(
        name="Con",
        chater=ChaterPool([
            get_chater_cfg("zhipuai"),
        ]),
        memory=Memory(),
        system_prompt="You argue AGAINST AI regulation. Be persuasive and brief."
    )

    judge = Agent(
        name="Judge",
        chater=ChaterPool([
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        system_prompt="You evaluate arguments objectively. Summarize both sides."
    )

    print("Debate: Should AI be strictly regulated?\n")
    print("=" * 60)

    topic = ChatResponse(
        role="user",
        content="Debate topic: Should AI be strictly regulated?"
    )

    with msghub([pro, con], announcement=topic):
        for round_num in range(2):
            print(f"\n--- Round {round_num + 1} ---\n")
            
            for agent in [pro, con]:
                print(f"{agent.name}:")
                async for response in agent.reply("State your argument", stream=False):
                    agent.speak(response)
                print()

    judge.observe([
        ChatResponse(role="assistant", content=msg.content)
        for msg in pro.memory.messages + con.memory.messages
        if msg.role == "assistant"
    ])

    print("--- Judge's Decision ---\n")
    async for response in judge.reply("Summarize both arguments", stream=False):
        judge.speak(response)
    print()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
