import asyncio
from qagent import Agent, Memory, Chater, get_chater_cfg, Runner

async def main():
    agent = Agent(
        name="Assistant",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a helpful assistant."
    )
    
    result = await Runner.run(agent, "Hello! What is 2+2?")
    print(f"Agent response: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
