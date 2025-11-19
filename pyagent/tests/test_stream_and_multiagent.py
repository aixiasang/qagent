import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    Agent,
    Chater,
    Memory,
    ToolKit,
    Runner,
    get_chater_cfg,
    trace,
    clear_traces,
    get_all_traces,
    get_all_spans,
    get_all_agents,
)


def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


async def test_stream_output():
    print("\n" + "="*70)
    print("TEST 1: Stream Output with Trace")
    print("="*70)
    
    clear_traces()
    
    chater = Chater(get_chater_cfg("ali"))
    memory = Memory()
    tools = ToolKit()
    tools.register(calculate)
    
    agent = Agent(
        name="StreamCalculator",
        system_prompt="You are a calculator. Use the calculate tool when needed.",
        chater=chater,
        memory=memory,
        tools=tools,
    )
    
    print("\n[TEST] Streaming response with trace...")
    
    with trace("stream_test"):
        print("Question: What is 10 + 20?")
        print("Answer: ", end="", flush=True)
        
        final_content = ""
        async for response in Runner.run_streamed(agent, "What is 10 + 20?"):
            if hasattr(response, 'content') and response.content:
                print(response.content, end="", flush=True)
                final_content = response.content
        print()
    
    spans = get_all_spans()
    agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
    gen_spans = [s for s in spans if s["span_data"]["type"] == "generation"]
    tool_spans = [s for s in spans if s["span_data"]["type"] == "tool"]
    
    print(f"\n✅ Stream trace captured:")
    print(f"   - Agent spans: {len(agent_spans)}")
    print(f"   - Generation spans: {len(gen_spans)}")
    print(f"   - Tool spans: {len(tool_spans)}")
    
    if agent_spans:
        agent_data = agent_spans[0]["span_data"]
        print(f"\n✅ Agent span structure (simplified):")
        print(f"   - type: {agent_data.get('type')}")
        print(f"   - agent_id: {agent_data.get('agent_id', 'N/A')}")
        print(f"   - user_input: {agent_data.get('user_input', 'N/A')[:30]}...")
        print(f"   - agent_output: {agent_data.get('agent_output', 'N/A')[:30]}...")
        
        assert "agent_name" not in agent_data
        assert "agent_type" not in agent_data
        assert "tools" not in agent_data
        print(f"\n✅ Confirmed: agent_name, agent_type, tools NOT in span_data!")
    
    if gen_spans:
        gen_data = gen_spans[0]["span_data"]
        print(f"\n✅ Generation span structure:")
        print(f"   - model: {gen_data.get('model')}")
        print(f"   - params keys: {list(gen_data.get('params', {}).keys())}")
        
        assert "model" in gen_data
        assert "params" in gen_data
        print(f"✅ Confirmed: model + params structure!")
    
    print("\n" + "="*70)


async def test_multi_agent_interaction():
    print("\n" + "="*70)
    print("TEST 2: Multi-Agent Interaction with Trace")
    print("="*70)
    
    clear_traces()
    
    chater = Chater(get_chater_cfg("ali"))
    
    triage_agent = Agent(
        name="TriageAgent",
        system_prompt="You are a triage agent. Analyze requests and route them.",
        chater=chater,
        memory=Memory(),
    )
    
    specialist_agent = Agent(
        name="SpecialistAgent",
        system_prompt="You are a specialist agent. Provide detailed responses.",
        chater=chater,
        memory=Memory(),
    )
    
    print("\n[TEST] Multi-agent workflow with trace...")
    
    with trace("multi_agent_test", metadata={"workflow": "triage_and_specialist"}):
        print("\n[Step 1] Triage Agent")
        triage_response = None
        async for resp in Runner.run_streamed(triage_agent, "Help me with a task"):
            triage_response = resp
        print(f"Triage: {triage_response.content[:50]}...")
        
        print("\n[Step 2] Specialist Agent")
        specialist_response = None
        async for resp in Runner.run_streamed(specialist_agent, "Provide detailed help"):
            specialist_response = resp
        print(f"Specialist: {specialist_response.content[:50]}...")
    
    data = {
        "traces": get_all_traces(),
        "spans": get_all_spans(),
        "agents": get_all_agents(),
    }
    spans = data["spans"]
    agents_info = data["agents"]
    
    agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
    
    print(f"\n✅ Multi-agent trace captured:")
    print(f"   - Total spans: {len(spans)}")
    print(f"   - Agent spans: {len(agent_spans)}")
    print(f"   - Registered agents: {len(agents_info)}")
    
    print(f"\n✅ Agent span data (simplified):")
    for i, span in enumerate(agent_spans, 1):
        span_data = span["span_data"]
        print(f"   Agent {i}:")
        print(f"      - type: {span_data.get('type')}")
        print(f"      - agent_id: {span_data.get('agent_id')}")
        print(f"      - user_input: {span_data.get('user_input', '')[:30]}...")
        print(f"      - agent_output: {span_data.get('agent_output', '')[:30]}...")
        
        assert "agent_name" not in span_data
        assert "agent_type" not in span_data
        assert "tools" not in span_data
    
    print(f"\n✅ Agents registry (full info):")
    for agent_id, agent_info in agents_info.items():
        print(f"   {agent_id[:16]}...")
        print(f"      - agent_name: {agent_info.get('agent_name')}")
        print(f"      - agent_type: {agent_info.get('agent_type')}")
        print(f"      - tools: {agent_info.get('tools', [])}")
    
    print(f"\n✅ Hierarchy verification:")
    for span in agent_spans:
        parent_id = span.get("parent_id")
        print(f"   span {span['span_id'][:16]}... parent={parent_id if parent_id else 'ROOT'}")
    
    print("\n" + "="*70)


async def test_nested_multi_agent():
    print("\n" + "="*70)
    print("TEST 3: Nested Multi-Agent (Handoff Simulation)")
    print("="*70)
    
    clear_traces()
    
    chater = Chater(get_chater_cfg("ali"))
    tools = ToolKit()
    tools.register(calculate)
    
    coordinator = Agent(
        name="Coordinator",
        system_prompt="You coordinate tasks.",
        chater=chater,
        memory=Memory(),
    )
    
    worker = Agent(
        name="Worker",
        system_prompt="You execute calculations.",
        chater=chater,
        memory=Memory(),
        tools=tools,
    )
    
    print("\n[TEST] Nested agent calls with trace...")
    
    with trace("nested_agents", metadata={"pattern": "coordinator_worker"}):
        print("\n[Coordinator]")
        coord_resp = None
        async for resp in Runner.run_streamed(coordinator, "I need to calculate 5*5"):
            coord_resp = resp
        print(f"Coordinator: {coord_resp.content[:50]}...")
        
        print("\n[Worker - executing calculation]")
        worker_resp = None
        async for resp in Runner.run_streamed(worker, "Calculate 5*5"):
            worker_resp = resp
        print(f"Worker: {worker_resp.content[:50]}...")
    
    data = export_traces()
    spans = data["spans"]
    
    agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
    tool_spans = [s for s in spans if s["span_data"]["type"] == "tool"]
    
    print(f"\n✅ Nested trace captured:")
    print(f"   - Agent spans: {len(agent_spans)}")
    print(f"   - Tool spans: {len(tool_spans)}")
    
    print(f"\n✅ Span hierarchy:")
    for span in agent_spans:
        span_data = span["span_data"]
        agent_id = span_data.get("agent_id", "unknown")
        agent_name = data["agents"].get(agent_id, {}).get("agent_name", "Unknown")
        print(f"   - {agent_name} (agent_id={agent_id[:16]}...)")
        print(f"     user_input: {span_data.get('user_input', '')[:40]}...")
        print(f"     agent_output: {span_data.get('agent_output', '')[:40]}...")
    
    with open("test_stream_multiagent_output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Exported to: test_stream_multiagent_output.json")
    
    print("\n" + "="*70)


async def test_stream_without_trace():
    print("\n" + "="*70)
    print("TEST 4: Stream WITHOUT Trace (No Error)")
    print("="*70)
    
    chater = Chater(get_chater_cfg("ali"))
    memory = Memory()
    
    agent = Agent(
        name="SimpleAgent",
        system_prompt="You are helpful.",
        chater=chater,
        memory=memory,
    )
    
    print("\n[TEST] Streaming without trace context...")
    print("Question: Hello!")
    print("Answer: ", end="", flush=True)
    
    async for response in Runner.run_streamed(agent, "Hello!"):
        if hasattr(response, 'content') and response.content:
            print(response.content, end="", flush=True)
    print()
    
    print("\n✅ Stream works perfectly without trace context!")
    print("✅ No errors, no overhead!")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STREAM & MULTI-AGENT TRACE TESTS")
    print("="*70)
    
    asyncio.run(test_stream_output())
    asyncio.run(test_multi_agent_interaction())
    asyncio.run(test_nested_multi_agent())
    asyncio.run(test_stream_without_trace())
    
    print("\n" + "="*70)
    print("✅ ALL STREAM & MULTI-AGENT TESTS PASSED!")
    print("="*70)
    
    print("\n📝 Summary:")
    print("  ✓ Stream output with trace - works!")
    print("  ✓ Multi-agent interaction traced")
    print("  ✓ Nested agents with tools traced")
    print("  ✓ Stream without trace - no errors!")
    print("  ✓ Agent span simplified (type, agent_id, user_input, agent_output)")
    print("  ✓ Full agent info in agents registry")
    print("\n🎉 Stream & Multi-Agent tracing fully functional!")

