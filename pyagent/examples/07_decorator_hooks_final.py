import asyncio
from core import Agent, BaseAgent, ChaterPool, Memory, get_chater_cfg


async def main():
    print("=== Decorator-Based Hook System ===\n")
    
    agent = Agent(
        name="Assistant",
        chater=ChaterPool([get_chater_cfg("zhipuai")]),
        memory=Memory(),
        system_prompt="You are a helpful assistant. Keep responses very brief."
    )
    
    print("1️⃣ Object-level hooks (instance-specific)\n")
    
    @agent.pre_reply
    def log_input(message):
        print(f"  📥 Processing: {message[:40]}...")
        return message
    
    @agent.post_reply
    def add_sparkle(response):
        if response.content and not response.tool_call:
            response.content = f"✨ {response.content}"
        return response
    
    @agent.post_observe
    def track_memory(msg):
        print(f"  💾 Memory: {len(agent.memory)} messages")
    
    print(f"  Registered: {list(agent._post_reply_hooks.keys())}\n")
    
    print("2️⃣ Test object-level hooks\n")
    print("User: Hello!\n")
    
    async for response in agent.reply("Hello!", stream=False):
        if response.content:
            print(f"Agent: {response.content}\n")
    
    print("\n3️⃣ Class-level hooks (global)\n")
    
    @BaseAgent.pre_reply
    def global_logger(agent, message):
        print(f"  🌐 [{agent.name}] Incoming request")
        return message
    
    agent2 = Agent(
        name="Agent2",
        chater=ChaterPool([get_chater_cfg("zhipuai")]),
        memory=Memory(),
        system_prompt="Be helpful and brief."
    )
    
    print("User: Hi there!\n")
    
    async for response in agent2.reply("Hi there!", stream=False):
        if response.content:
            print(f"Agent2: {response.content}\n")
    
    print("=== Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
