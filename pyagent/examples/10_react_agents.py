import asyncio
from datetime import datetime
from agent import ReActAgent, ClassicReActAgent
from core import Memory, ChaterPool, get_chater_cfg, ToolKit


async def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def calculate(expression: str) -> float:
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


async def get_weather(city: str) -> str:
    weather_data = {
        "beijing": "Sunny, 22°C",
        "shanghai": "Cloudy, 18°C",
        "guangzhou": "Rainy, 25°C",
        "shenzhen": "Sunny, 26°C",
        "北京": "晴天, 22°C",
        "上海": "多云, 18°C",
        "广州": "雨天, 25°C",
        "深圳": "晴天, 26°C",
    }
    return weather_data.get(city.lower(), f"Weather data for {city} not available")


async def search_wikipedia(query: str) -> str:
    simulated_results = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "react": "ReAct (Reasoning and Acting) is a framework that combines reasoning traces and task-specific actions.",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
        "人工智能": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
        "机器学习": "机器学习是人工智能的一个子领域，通过数据和经验自动改进性能。",
    }
    
    for key, value in simulated_results.items():
        if key in query.lower():
            return value
    
    return f"Simulated Wikipedia search result for: {query}"


async def ainput(prompt: str = "") -> str:
    return await asyncio.to_thread(input, prompt)


async def demo_react_agent():
    print("="*70)
    print("ReActAgent Demo (OpenAI Tool Calling Based)")
    print("="*70)
    
    tools = ToolKit()
    tools.register(get_current_time, "get_current_time")
    tools.register(calculate, "calculate")
    tools.register(get_weather, "get_weather")
    tools.register(search_wikipedia, "search_wikipedia")
    
    agent = ReActAgent(
        name="ReActAssistant",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(max_messages=50),
        tools=tools,
        max_iterations=5,
        enable_logging=True,
        track_reasoning=True
    )
    
    print(f"\nAgent: {repr(agent)}")
    print(f"Tools: {tools.list_tools()}\n")
    
    test_tasks = [
        "现在几点了？",
        "What is 456 * 789?",
        "北京今天天气怎么样？",
        "Search for information about ReAct framework",
        "先查询当前时间，然后计算 100 + 200"
    ]
    
    for task in test_tasks:
        print(f"\n{'='*70}")
        print(f"User: {task}")
        print(f"{'='*70}")
        print("Agent: ", end="", flush=True)
        
        async for response in agent.reply(task, stream=True):
            if response.content:
                print(response.content, end="", flush=True)
            if response.reasoning_content:
                print(f"\n[Reasoning: {response.reasoning_content}]", end="", flush=True)
        
        print("\n")
        
        trace = agent.get_reasoning_trace()
        print(f"Trace Summary:")
        print(f"  - Total iterations: {trace['total_iterations']}")
        print(f"  - Thoughts recorded: {len(trace['thoughts'])}")
        print(f"  - Actions taken: {len(trace['actions'])}")
        print(f"  - Observations: {len(trace['observations'])}")
    
    print("\n" + "="*70)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*70)
    
    while True:
        try:
            user_input = await ainput("\nYou: ")
            
            if not user_input.strip():
                continue
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "clear":
                agent.clear_memory()
                print("Memory cleared.")
                continue
            
            if user_input.lower() == "trace":
                trace = agent.get_reasoning_trace()
                print(f"\nReasoning Trace:")
                import json
                print(json.dumps(trace, indent=2, ensure_ascii=False))
                continue
            
            print("Agent: ", end="", flush=True)
            async for response in agent.reply(user_input, stream=True):
                if response.content:
                    print(response.content, end="", flush=True)
                if response.reasoning_content:
                    print(f"\n[Reasoning: {response.reasoning_content}]", end="", flush=True)
            print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


async def demo_classic_react_agent():
    print("\n\n" + "="*70)
    print("ClassicReActAgent Demo (Classic ReAct Format)")
    print("="*70)
    
    tools = ToolKit()
    tools.register(get_current_time, "get_current_time")
    tools.register(calculate, "calculate")
    tools.register(get_weather, "get_weather")
    tools.register(search_wikipedia, "search_wikipedia")
    
    agent = ClassicReActAgent(
        name="ClassicReActAssistant",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(max_messages=50),
        tools=tools,
        max_iterations=5,
        enable_logging=True
    )
    
    print(f"\nAgent: {repr(agent)}")
    print(f"Tools: {tools.list_tools()}\n")
    
    test_tasks = [
        "现在是什么时间？",
        "Calculate 888 + 999",
        "上海的天气如何？"
    ]
    
    for task in test_tasks:
        print(f"\n{'='*70}")
        print(f"User: {task}")
        print(f"{'='*70}")
        
        step_count = 0
        async for response in agent.reply(task):
            step_count += 1
            print(f"\n[Step {step_count}]")
            print(response.content)
        
        history = agent.get_react_history()
        print(f"\nTotal ReAct steps: {len(history)}")
    
    print("\n" + "="*70)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*70)
    
    while True:
        try:
            user_input = await ainput("\nYou: ")
            
            if not user_input.strip():
                continue
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "clear":
                agent.clear_memory()
                print("Memory cleared.")
                continue
            
            if user_input.lower() == "history":
                history = agent.get_react_history()
                print(f"\nReAct History ({len(history)} steps):")
                import json
                print(json.dumps(history, indent=2, ensure_ascii=False))
                continue
            
            step_count = 0
            async for response in agent.reply(user_input):
                step_count += 1
                print(f"\n[Step {step_count}]")
                print(response.content)
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


async def main():
    print("\n" + "="*70)
    print("ReAct Agents Demonstration")
    print("="*70)
    print("\nChoose demo mode:")
    print("1. ReActAgent (OpenAI Tool Calling)")
    print("2. ClassicReActAgent (Classic ReAct Format)")
    print("3. Both (Sequential)")
    
    choice = await ainput("\nEnter choice (1/2/3): ")
    
    if choice == "1":
        await demo_react_agent()
    elif choice == "2":
        await demo_classic_react_agent()
    elif choice == "3":
        await demo_react_agent()
        await demo_classic_react_agent()
    else:
        print("Invalid choice. Running both demos...")
        await demo_react_agent()
        await demo_classic_react_agent()


if __name__ == "__main__":
    asyncio.run(main())
