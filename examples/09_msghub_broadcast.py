import asyncio
from pyagent import Agent, Memory, Chater, get_chater_cfg, Runner, ChatResponse, msghub

async def main():
    agents = [
        Agent(
            name="Alice",
            chater=Chater(get_chater_cfg("ali")),
            memory=Memory(),
            system_prompt="You are Alice. Respond to messages in the group chat."
        ),
        Agent(
            name="Bob",
            chater=Chater(get_chater_cfg("ali")),
            memory=Memory(),
            system_prompt="You are Bob. Respond to messages in the group chat."
        ),
        Agent(
            name="Charlie",
            chater=Chater(get_chater_cfg("ali")),
            memory=Memory(),
            system_prompt="You are Charlie. Respond to messages in the group chat."
        ),
    ]
    
    announcement = ChatResponse(
        role="system",
        content="Topic: Discuss the benefits of async programming in Python"
    )
    
    results = await Runner.run_msghub(agents, announcement, rounds=1)
    
    for i, result in enumerate(results):
        print(f"\n{agents[i].name}: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
