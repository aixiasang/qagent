"""
Simple & Elegant Trace Usage Example

Shows how tracing is now completely opt-in and elegant:
1. Agent code stays clean (no trace logic inside Agent class)
2. Model & Tool automatically traced at lower levels  
3. User wraps with trace() for agent-level tracking
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    Agent,
    Runner,
    ToolKit,
    Chater,
    Memory,
    trace,
    enable_console_output,
    export_traces,
    get_chater_cfg,
    clear_traces,
)


def calculate(expression: str) -> str:
    """Calculate a math expression"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


async def demo_without_trace():
    """Demo 1: Simple usage without trace"""
    print("\n" + "="*70)
    print(" Demo 1: Without Trace - Just normal Agent usage ")
    print("="*70)
    
    try:
        toolkit = ToolKit()
        toolkit.register(calculate)
        
        chater = Chater(get_chater_cfg("ali"))
        memory = Memory()
        
        agent = Agent(
            name="Calculator",
            chater=chater,
            memory=memory,
            tools=toolkit,
            system_prompt="You are a calculator. Use the calculate tool when needed.",
        )
        
        print("\nQuestion: What is 15 * 23?")
        result = await Runner.run(agent, "What is 15 * 23?")
        print(f"Answer: {result.content if result.content else 'No response'}")
        
        print("\n✅ Works perfectly, no trace overhead!")
    except Exception as e:
        print(f"⚠ Skipped: {str(e)[:80]}")


async def demo_with_trace():
    """Demo 2: Elegant trace with Runner"""
    print("\n" + "="*70)
    print(" Demo 2: With Trace - Just wrap with trace()! ")
    print("="*70)
    
    clear_traces()
    enable_console_output(verbose=True)
    
    try:
        toolkit = ToolKit()
        toolkit.register(calculate)
        
        chater = Chater(get_chater_cfg("ali"))
        memory = Memory()
        
        agent = Agent(
            name="Calculator",
            chater=chater,
            memory=memory,
            tools=toolkit,
            system_prompt="You are a calculator. Use the calculate tool when needed.",
        )
        
        print("\nQuestion: What is (5 + 3) * 2?")
        
        with trace("calculator_workflow", group_id="demo_session"):
            result = await Runner.run(agent, "What is (5 + 3) * 2?")
            print(f"\nAnswer: {result.content if result.content else 'No response'}")
        
        export_traces("trace_output_demo.json")
        
        print("\n✅ Trace automatically captured!")
        print("   📊 Agent span - user input/output")
        print("   🤖 Generation span - model call (auto from _model.py)")
        print("   🔧 Tool span - calculate function (auto from _tools.py)")
        print("   📁 Exported to: trace_output_demo.json")
    except Exception as e:
        print(f"⚠ Skipped: {str(e)[:80]}")


async def demo_multi_agent():
    """Demo 3: Multi-agent with automatic hierarchy"""
    print("\n" + "="*70)
    print(" Demo 3: Multi-Agent - Hierarchy auto-tracked ")
    print("="*70)
    
    clear_traces()
    enable_console_output(verbose=True)
    
    try:
        chater = Chater(get_chater_cfg("ali"))
        
        triage_agent = Agent(
            name="TriageAgent",
            chater=chater,
            memory=Memory(),
            system_prompt="You route requests to specialists.",
        )
        
        specialist_agent = Agent(
            name="SpecialistAgent",
            chater=chater,
            memory=Memory(),
            system_prompt="You handle specialized requests.",
        )
        
        with trace("multi_agent_demo"):
            print("\n[Triage Agent]")
            triage_result = await Runner.run(triage_agent, "I need help")
            print(f"Triage: {triage_result.content[:50] if triage_result.content else 'None'}...")
            
            print("\n[Specialist Agent]")
            specialist_result = await Runner.run(specialist_agent, "Handle the request")
            print(f"Specialist: {specialist_result.content[:50] if specialist_result.content else 'None'}...")
        
        print("\n✅ Multi-agent hierarchy automatically tracked!")
        print("   🔗 parent_agent_id automatically linked")
    except Exception as e:
        print(f"⚠ Skipped: {str(e)[:80]}")


async def main():
    print("\n" + "="*70)
    print("   ELEGANT TRACE SYSTEM - Keep Agent Simple!   ")
    print("="*70)
    
    await demo_without_trace()
    await demo_with_trace()
    await demo_multi_agent()
    
    print("\n" + "="*70)
    print(" 🎯 KEY BENEFITS ")
    print("="*70)
    print("""
1️⃣  Agent stays clean - no trace code inside Agent.reply()
2️⃣  Model & Tool auto-traced - happens at lower levels
3️⃣  User controls - just wrap with trace() when needed
4️⃣  Zero overhead - when no trace(), completely disabled

Usage Pattern:
    # Without trace - normal usage
    result = await Runner.run(agent, "question")
    
    # With trace - just add wrapper
    with trace("my_workflow"):
        result = await Runner.run(agent, "question")

That's it! Simple & elegant! 🎉
""")


if __name__ == "__main__":
    asyncio.run(main())

