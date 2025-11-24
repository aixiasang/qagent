import asyncio
from datetime import datetime
from pyagent import Agent, Memory, Chater, get_chater_cfg, ToolKit, Runner

async def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def calculate(operation: str, a: float, b: float) -> float:
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    return operations.get(operation, lambda x, y: 0)(a, b)

async def main():
    toolkit = ToolKit()
    toolkit.register(get_current_time)
    toolkit.register(calculate)
    
    agent = Agent(
        name="MathAssistant",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        tools=toolkit,
        system_prompt="You are a math assistant with access to calculation and time tools."
    )
    
    result = await Runner.run(agent, "What time is it? Also calculate 15 * 23")
    print(f"Result: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
