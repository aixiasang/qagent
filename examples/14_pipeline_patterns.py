import asyncio
from qagent import (
    Agent, Memory, Chater, get_chater_cfg, ChatResponse,
    sequential_pipeline, parallel_pipeline, 
    conditional_pipeline, loop_pipeline
)

async def main():
    agent1 = Agent(
        name="Analyzer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Analyze the input briefly."
    )
    
    agent2 = Agent(
        name="Summarizer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Summarize the analysis."
    )
    
    initial_msg = ChatResponse(role="user", content="Explain machine learning")
    
    print("=== Sequential Pipeline ===")
    result = await sequential_pipeline([agent1, agent2], initial_msg)
    print(f"Result: {result.content}\n")
    
    print("=== Conditional Pipeline ===")
    def is_technical(msg):
        return "technical" in msg.content.lower()
    
    tech_agent = Agent(
        name="TechExpert",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Provide technical explanation."
    )
    
    simple_agent = Agent(
        name="SimpleExplainer",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="Provide simple explanation."
    )
    
    test_msg = ChatResponse(role="user", content="Explain neural networks")
    result = await conditional_pipeline(is_technical, tech_agent, simple_agent, test_msg)
    print(f"Result: {result.content}\n")

if __name__ == "__main__":
    asyncio.run(main())
