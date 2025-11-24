import asyncio
from pyagent import Agent, Memory, Chater, get_chater_cfg, Runner

async def main():
    agents = [
        Agent(
            name="Analyst1",
            chater=Chater(get_chater_cfg("ali")),
            memory=Memory(),
            system_prompt="You are a technical analyst. Provide technical perspective."
        ),
        Agent(
            name="Analyst2",
            chater=Chater(get_chater_cfg("ali")),
            memory=Memory(),
            system_prompt="You are a business analyst. Provide business perspective."
        ),
        Agent(
            name="Analyst3",
            chater=Chater(get_chater_cfg("ali")),
            memory=Memory(),
            system_prompt="You are a risk analyst. Provide risk perspective."
        ),
    ]
    
    results = await Runner.run_parallel(
        agents,
        "Analyze the potential of AI in healthcare"
    )
    
    for i, result in enumerate(results):
        print(f"\nAgent {i+1} ({agents[i].name}):")
        print(result.content)

if __name__ == "__main__":
    asyncio.run(main())
