import asyncio
from qagent import Agent, Memory, Chater, get_chater_cfg, Runner

async def main():
    agent = Agent(
        name="HookedAgent",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a helpful assistant."
    )
    
    @agent.pre_reply
    def log_input(msg):
        print(f"[PRE-REPLY] User said: {msg}")
        return msg
    
    @agent.post_reply
    def format_output(response):
        if response.content and not response.tool_call:
            response.content = f"âœ?{response.content} âœ?
        return response
    
    @agent.pre_observe
    def log_observe(msg):
        print(f"[PRE-OBSERVE] Observing: {msg.content[:50]}")
        return msg
    
    result = await Runner.run(agent, "Hello, how are you?")
    print(f"\nFinal: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
