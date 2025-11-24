import asyncio
from pyagent import Agent, Memory, Chater, get_chater_cfg, Runner

async def main():
    planner = Agent(
        name="Planner",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a planning agent. Create a brief plan for the task."
    )
    
    executor = Agent(
        name="Executor",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are an execution agent. Execute the plan step by step."
    )
    
    reviewer = Agent(
        name="Reviewer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a review agent. Review the execution and provide feedback."
    )
    
    result = await Runner.run_sequential(
        [planner, executor, reviewer],
        "Create a simple Python script to calculate fibonacci numbers"
    )
    
    print(f"Final output: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
