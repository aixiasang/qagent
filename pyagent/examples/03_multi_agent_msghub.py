import asyncio
from core import Agent, Memory, get_chater_cfg, ChaterPool, ChatResponse, msghub


async def main():
    alice = Agent(
        name="Alice",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        system_prompt="You are Alice, a creative writer. Be concise and poetic."
    )

    bob = Agent(
        name="Bob",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        system_prompt="You are Bob, a logical analyst. Be precise and analytical."
    )

    charlie = Agent(
        name="Charlie",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        system_prompt="You are Charlie, a friendly mediator. Be warm and balanced."
    )

    agents = [alice, bob, charlie]

    print("Multi-Agent Discussion: 'What is the meaning of life?'\n")
    print("=" * 60)

    initial_msg = ChatResponse(
        role="user",
        content="What is the meaning of life? Share your perspective briefly. Speak in chinese."
    )

    with msghub(agents, announcement=initial_msg):
        for agent in agents:
            print(f"\n{agent.name}:")
            async for _ in agent.reply("", stream=True, auto_speak=True):
                pass
            print()

    print("\n" + "=" * 60)
    print("\nMemory check:")
    for agent in agents:
        print(f"{agent.name} memory size: {len(agent.memory)}")


if __name__ == "__main__":
    asyncio.run(main())
