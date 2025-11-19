import asyncio
from datetime import datetime
from agent import PlanReActAgent, ReflectionAgent, SelfConsistencyAgent
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


async def demo_plan_react():
    print("="*70)
    print("Demo 1: Plan-ReAct Agent - Planning Before Execution")
    print("="*70)
    
    tools = ToolKit()
    tools.register(get_current_time, "get_current_time")
    tools.register(calculate, "calculate")
    tools.register(get_weather, "get_weather")
    
    agent = PlanReActAgent(
        name="Planner",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        tools=tools,
        enable_logging=True,
    )
    
    question = "现在北京几点？天气如何？温度如果上升5度会是多少？"
    
    print(f"\nQuestion: {question}\n")
    
    async for response in agent.reply(question):
        print(f"Answer:\n{response.content}\n")
    
    plan = agent.get_last_plan()
    if plan:
        print(f"Generated Plan:\n{plan}\n")


async def demo_reflection():
    print("\n" + "="*70)
    print("Demo 2: Reflection Agent - Self-Assessment & Improvement")
    print("="*70)
    
    tools = ToolKit()
    tools.register(calculate, "calculate")
    
    agent = ReflectionAgent(
        name="Reflector",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        tools=tools,
        enable_logging=True,
    )
    
    question = "Calculate 15 * 23 and explain what this number could represent"
    
    print(f"\nQuestion: {question}\n")
    
    async for response in agent.reply(question):
        print(f"Answer:\n{response.content}\n")
    
    records = agent.get_reflection_records()
    if records:
        print("Reflection Process:")
        for i, record in enumerate(records, 1):
            if record.get('reflection'):
                print(f"  Self-assessment: {record['reflection'][:100]}...")


async def demo_self_consistency():
    print("\n" + "="*70)
    print("Demo 3: Self-Consistency Agent - Multiple Reasoning Paths")
    print("="*70)
    
    tools = ToolKit()
    tools.register(calculate, "calculate")
    
    agent = SelfConsistencyAgent(
        name="Consistent",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        tools=tools,
        num_paths=3,
        enable_logging=True,
    )
    
    question = "Calculate 50 + 30 * 2 following mathematical order of operations"
    
    print(f"\nQuestion: {question}\n")
    
    async for response in agent.reply(question):
        print(f"Answer:\n{response.content}\n")
    
    paths = agent.get_reasoning_paths()
    if paths:
        print(f"Generated {len(paths)} different reasoning paths for consistency check")


async def main():
    print("\nEnhanced Agents Demo\n")
    
    await demo_plan_react()
    await demo_reflection()
    await demo_self_consistency()
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
