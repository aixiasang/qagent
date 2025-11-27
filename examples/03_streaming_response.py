import asyncio
from qagent import Agent, Memory, Chater, get_chater_cfg, Runner

async def main():
    agent = Agent(
        name="Storyteller",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a creative storyteller."
    )
    
    print("Streaming response:")
    async for chunk in Runner.run_streamed(agent, "Tell me a short story about a robot"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
