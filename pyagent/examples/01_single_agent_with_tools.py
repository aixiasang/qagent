import asyncio
from core import (
    Agent,
    Memory,
    get_chater_cfg,
    ChaterPool,
    ToolKit,
    FileOperations,
    DirectoryOperations,
)
from datetime import datetime


async def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def calculate(expression: str) -> float:
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


async def main():
    tools = ToolKit()
    tools.register(get_current_time, "get_current_time")
    tools.register(calculate, "calculate")
    tools.register(FileOperations.read_file, "read_file")
    tools.register(DirectoryOperations.list_directory, "list_directory")

    agent = Agent(
        name="ToolAgent",
        chater=ChaterPool([get_chater_cfg("siliconflow"), get_chater_cfg("zhipuai")]),
        memory=Memory(max_messages=20),
        tools=tools,
        system_prompt="You are a helpful assistant with access to various tools.",
        max_iterations=3,
    )

    print(f"Agent: {repr(agent)}")
    print(f"Available tools: {list(tools._tools.keys())}\n")

    tasks = ["What time is it?", "Calculate 123 * 456", "List files in current directory"]

    for task in tasks:
        print(f"User: {task}")

        async for response in agent.reply(
            task,
            stream=True,
        ):
            agent.speak(response, stream=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
