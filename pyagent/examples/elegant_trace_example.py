"""
Elegant Trace Usage Example - Similar to openai-agents style

This shows the recommended way to use tracing:
1. Agent stays simple and clean
2. Model and Tool automatically traced
3. User only wraps with trace() for agent-level tracking
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    Agent,
    Runner,
    ToolKit,
    trace,
    enable_console_output,
    export_traces,
    get_chater_cfg,
)


def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


async def example_1_simple_usage():
    """Example 1: Simple Agent - No Trace"""
    print("\n" + "="*70)
    print("Example 1: Simple Usage - No Trace")
    print("="*70)
    
    from core import Chater, Memory
    
    try:
        chater = Chater(get_chater_cfg("ali"))
        memory = Memory()
        
        agent = Agent(
            name="Assistant",
            chater=chater,
            memory=memory,
            system_prompt="You are a helpful assistant.",
        )
        
        async for response in agent.reply("Hello!"):
            print(response.content if response.content else "")
    except Exception as e:
        print(f"⚠ Skipped (no API key): {str(e)[:80]}")
    
    print("\n✅ No trace overhead when not needed!")


async def example_2_with_runner_trace():
    """Example 2: Elegant Trace with Runner - Just like openai-agents!"""
    print("\n" + "="*70)
    print("Example 2: Elegant Trace with Runner")
    print("="*70)
    
    enable_console_output(verbose=True)
    
    toolkit = ToolKit()
    toolkit.register(add)
    toolkit.register(multiply)
    
    agent = Agent(
        name="Calculator",
        instructions="You are a helpful calculator assistant. Use tools when needed.",
        tools=toolkit,
    )
    
    with trace("calculator_workflow", group_id="session_123"):
        response = await Runner.run(agent, "Calculate 5 + 3, then multiply result by 2")
        print(f"\n📝 Final: {response.content if response else 'No response'}")
    
    export_traces("elegant_trace_output.json")
    print("\n✅ Trace automatically captured at all levels!")
    print("   - Agent span")
    print("   - Generation spans (model)")  
    print("   - Tool spans (add, multiply)")


async def example_3_streaming_with_trace():
    """Example 3: Streaming + Trace"""
    print("\n" + "="*70)
    print("Example 3: Streaming with Trace")
    print("="*70)
    
    enable_console_output(verbose=False)
    
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
    )
    
    with trace("streaming_workflow"):
        print("\nStreaming response: ", end="", flush=True)
        async for chunk in Runner.run_streamed(agent, "Tell me a very short joke"):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()
    
    print("\n✅ Streaming works perfectly with trace!")


async def example_4_multi_agent():
    """Example 4: Multi-Agent Workflow"""
    print("\n" + "="*70)
    print("Example 4: Multi-Agent Scenario")
    print("="*70)
    
    enable_console_output(verbose=True)
    
    triage_agent = Agent(
        name="TriageAgent",
        instructions="You classify and route requests.",
        agent_id="agent_triage",
    )
    
    specialist_agent = Agent(
        name="SpecialistAgent",
        instructions="You handle specialized requests.",
        agent_id="agent_specialist",
        parent_agent_id="agent_triage",
    )
    
    with trace("multi_agent_workflow", group_id="multi_session"):
        triage_response = await Runner.run(
            triage_agent, 
            "I need help with billing"
        )
        print(f"\nTriage: {triage_response.content[:50]}...")
        
        specialist_response = await Runner.run(
            specialist_agent,
            "Handle the billing issue"
        )
        print(f"\nSpecialist: {specialist_response.content[:50]}...")
    
    print("\n✅ Multi-agent hierarchy automatically tracked!")


async def main():
    print("\n" + "="*70)
    print(" ELEGANT TRACE EXAMPLES - openai-agents style! ")
    print("="*70)
    
    await example_1_simple_usage()
    await example_2_with_runner_trace()
    await example_3_streaming_with_trace()
    await example_4_multi_agent()
    
    print("\n" + "="*70)
    print(" KEY TAKEAWAYS ")
    print("="*70)
    print("""
✨ Simple & Elegant:
   - Agent code stays clean (no trace logic)
   - Just wrap with trace() when needed
   - Model & Tool auto-traced

📦 Three Levels of Control:
   1. No trace: agent.reply() directly
   2. Auto trace: Runner.run() inside trace()
   3. Custom: Manual agent_span() if needed

🎯 Just like openai-agents!
   with trace("my_workflow"):
       result = await Runner.run(agent, "question")
""")


if __name__ == "__main__":
    asyncio.run(main())

