import asyncio
from datetime import datetime
from agent import EnhancedAgent
from core import Memory, ChaterPool, get_chater_cfg, ToolKit


async def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def calculate(expression: str) -> float:
    try:
        return float(eval(expression))
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


async def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 25°C"


async def demo_enhanced_agent():
    print("="*70)
    print("Enhanced Agent Demo - Quality Loop with Reflection")
    print("="*70)
    
    tools = ToolKit()
    tools.register(get_current_time, "get_current_time")
    tools.register(calculate, "calculate")
    tools.register(get_weather, "get_weather")
    
    agent = EnhancedAgent(
        name="EnhancedAssistant",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        tools=tools,
        max_iterations=3,
        quality_threshold=7.5,
        enable_logging=True,
        enable_reflection=True,
    )
    
    questions = [
        "现在几点？请给出完整的日期和时间。",
        "Calculate 15 * 23 and explain what this number represents",
        "告诉我北京的天气，如果温度上升5度会是多少？",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}: {question}")
        print(f"{'='*70}\n")
        
        async for response in agent.reply_enhanced(question):
            print(f"Final Answer:")
            print(response.content)
            print()
        
        summary = agent.get_execution_summary()
        
        print(f"Execution Summary:")
        print(f"  Iterations: {summary['total_iterations']}")
        print(f"  Reflections: {summary['reflections']}")
        print(f"  Final State: {summary['final_state']}")
        
        if summary['reflections'] > 0:
            print(f"\n  Quality Improvement Process:")
            for j, ref in enumerate(summary['reflection_summary'], 1):
                print(f"    Round {j}:")
                if ref['problems']:
                    print(f"      Found issues: {len(ref['problems'])}")
                    for prob in ref['problems'][:2]:
                        print(f"        - {prob}")
                if ref['improvements']:
                    print(f"      Applied improvements: {len(ref['improvements'])}")
        
        agent.clear_history()
        print()


async def demo_comparison():
    print("\n" + "="*70)
    print("Comparison: Standard Agent vs Enhanced Agent")
    print("="*70)
    
    from agent import ReActAgent
    
    tools = ToolKit()
    tools.register(get_current_time, "get_current_time")
    tools.register(calculate, "calculate")
    
    standard_agent = ReActAgent(
        name="StandardAgent",
        chater=ChaterPool([get_chater_cfg("siliconflow")]),
        memory=Memory(),
        tools=tools,
        max_iterations=3,
    )
    
    enhanced_agent = EnhancedAgent(
        name="EnhancedAgent",
        chater=ChaterPool([get_chater_cfg("siliconflow")]),
        memory=Memory(),
        tools=tools,
        max_iterations=3,
        quality_threshold=7.5,
        enable_reflection=True,
    )
    
    question = "What is 12 * 34 + 56 * 78?"
    
    print(f"\nQuestion: {question}\n")
    
    print("Standard Agent:")
    print("-" * 70)
    async for response in standard_agent.reply(question, stream=False):
        print(f"Answer: {response.content[:150]}...")
    
    print("\n\nEnhanced Agent (with quality loop):")
    print("-" * 70)
    async for response in enhanced_agent.reply_enhanced(question):
        print(f"Answer: {response.content[:150]}...")
    
    summary = enhanced_agent.get_execution_summary()
    print(f"\nEnhanced Agent went through {summary['total_iterations']} iteration(s)")
    print(f"with {summary['reflections']} reflection(s) for quality assurance")


async def main():
    await demo_enhanced_agent()
    await demo_comparison()
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
