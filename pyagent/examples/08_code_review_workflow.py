import asyncio
from core import Agent, Memory, get_chater_cfg, ChaterPool, ToolKit, FileOperations


async def main():
    tools = ToolKit()
    tools.register(FileOperations.read_file, "read_file")

    coder = Agent(
        name="Coder",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        tools=tools,
        system_prompt="You review code and suggest improvements. Be specific."
    )

    print("Code Review Workflow\n")
    print("=" * 60)

    task = "Read file '../core/_agent.py' and review the Agent class design"
    print(f"Task: {task}\n")

    print("Review:")
    async for response in coder.reply(task, stream=False):
        coder.speak(response)
    print()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
