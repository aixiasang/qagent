import asyncio
from pyagent import Agent, Memory, Chater, get_chater_cfg, ToolKit, Runner, trace, export_traces

async def multiply(a: float, b: float) -> float:
    return a * b

async def add(a: float, b: float) -> float:
    return a + b

async def main():
    toolkit = ToolKit()
    toolkit.register(multiply)
    toolkit.register(add)
    
    agent = Agent(
        name="MathAgent",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        tools=toolkit,
        system_prompt="You are a math agent."
    )
    
    with trace("math_workflow") as t:
        result = await Runner.run(agent, "Calculate (5 * 3) + (10 * 2)")
        print(f"Result: {result.content}")
    
    traces = export_traces()
    print(f"\nTrace captured: {len(traces)} trace(s)")
    for trace_data in traces:
        print(f"Trace ID: {trace_data['trace_id']}")
        print(f"Total spans: {len(trace_data['spans'])}")

if __name__ == "__main__":
    asyncio.run(main())
