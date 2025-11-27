import asyncio
from qagent import Agent, Memory, ChaterPool, get_chater_cfg, Runner

async def main():
    chater_pool = ChaterPool([
        get_chater_cfg("ali"),
    ])
    
    agent = Agent(
        name="ResilientAgent",
        chater=chater_pool,
        memory=Memory(),
        system_prompt="You are a reliable assistant with automatic failover."
    )
    
    result = await Runner.run(agent, "Explain quantum computing in simple terms")
    print(f"Response: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
