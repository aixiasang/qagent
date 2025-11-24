import asyncio
from pyagent import Agent, Memory, Chater, get_chater_cfg, Runner, BaseAgent

@BaseAgent.pre_reply
def global_pre_reply(agent, msg):
    print(f"[GLOBAL PRE-REPLY] Agent {agent.name} received: {msg[:30]}")
    return msg

@BaseAgent.post_reply
def global_post_reply(agent, response):
    print(f"[GLOBAL POST-REPLY] Agent {agent.name} replied")
    return response

async def main():
    agent1 = Agent(
        name="Agent1",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are agent 1"
    )
    
    agent2 = Agent(
        name="Agent2",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are agent 2"
    )
    
    @agent1.post_reply
    def instance_hook(response):
        if response.content:
            response.content = f"[Agent1 Enhanced] {response.content}"
        return response
    
    result1 = await Runner.run(agent1, "Hello")
    print(f"Agent1: {result1.content}\n")
    
    result2 = await Runner.run(agent2, "Hello")
    print(f"Agent2: {result2.content}")

if __name__ == "__main__":
    asyncio.run(main())
