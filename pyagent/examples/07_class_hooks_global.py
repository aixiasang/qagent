import asyncio
from core import Agent, BaseAgent, Memory, get_chater_cfg, ChaterPool


async def main():
    @BaseAgent.pre_reply
    def global_logger(agent, message):
        print(f"[GLOBAL] {agent.name} received: {message[:40]}...")
        return message

    @BaseAgent.post_reply
    def global_formatter(agent, response):
        if response.content:
            response.content = f"🤖 {response.content}"
        return response

    print("Class-level hooks affect ALL agents\n")
    print("=" * 60)

    agent1 = Agent(
        name="Agent1",
        chater=ChaterPool([
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        system_prompt="You are helpful."
    )

    agent2 = Agent(
        name="Agent2",
        chater=ChaterPool([
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        system_prompt="You are friendly."
    )

    for agent in [agent1, agent2]:
        print(f"\n{agent.name}:")
        async for response in agent.reply("Say hello", stream=False):
            agent.speak(response)
        print()

    BaseAgent._class_hooks_pre_reply.clear()
    BaseAgent._class_hooks_post_reply.clear()
    print("\n" + "=" * 60)
    print("Class hooks cleared")


if __name__ == "__main__":
    asyncio.run(main())
