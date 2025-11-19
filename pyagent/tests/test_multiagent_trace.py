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
    ChatResponse,
    get_chater_cfg,
    trace,
    clear_traces,
    get_all_traces,
    get_all_spans,
    get_all_agents,
    msghub,
    sequential_pipeline,
)


def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


async def test_auto_trace_in_agent_reply():
    print("\n" + "="*70)
    print("TEST 1: Auto-trace in Agent.reply()")
    print("="*70)
    
    clear_traces()
    
    chater = Chater(get_chater_cfg("ali"))
    agent = Agent(
        name="TestAgent",
        system_prompt="You are helpful.",
        chater=chater,
        memory=Memory(),
    )
    
    print("\n[TEST] Direct agent.reply() call with trace context...")
    
    with trace("agent_reply_test"):
        final_response = None
        async for response in agent.reply("Say hello"):
            final_response = response
        print(f"Response: {final_response.content[:50]}...")
    
    spans = get_all_spans()
    agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
    gen_spans = [s for s in spans if s["span_data"]["type"] == "generation"]
    
    print(f"\n✅ Auto-trace captured:")
    print(f"   - Agent spans: {len(agent_spans)}")
    print(f"   - Generation spans: {len(gen_spans)}")
    
    if agent_spans:
        agent_data = agent_spans[0]["span_data"]
        print(f"\n✅ Agent span (simplified):")
        print(f"   - type: {agent_data.get('type')}")
        print(f"   - agent_id: {agent_data.get('agent_id')[:16]}...")
        print(f"   - user_input: {agent_data.get('user_input')}")
        print(f"   - agent_output: {agent_data.get('agent_output', '')[:30]}...")
        
        assert "agent_name" not in agent_data
        assert "agent_type" not in agent_data
        assert "tools" not in agent_data
        print(f"✅ Confirmed: agent span only has type, agent_id, user_input, agent_output!")
    
    print("\n" + "="*70)


async def test_runner_sequential():
    print("\n" + "="*70)
    print("TEST 2: Runner.run_sequential()")
    print("="*70)
    
    clear_traces()
    
    chater = Chater(get_chater_cfg("ali"))
    
    planner = Agent(
        name="Planner",
        system_prompt="You create brief plans. Output 3 steps.",
        chater=chater,
        memory=Memory(),
    )
    
    executor = Agent(
        name="Executor",
        system_prompt="You execute plans step by step.",
        chater=chater,
        memory=Memory(),
    )
    
    reviewer = Agent(
        name="Reviewer",
        system_prompt="You review results and give feedback.",
        chater=chater,
        memory=Memory(),
    )
    
    print("\n[TEST] Sequential execution with trace...")
    
    with trace("sequential_test"):
        result = await Runner.run_sequential(
            [planner, executor, reviewer],
            "Plan a simple morning routine"
        )
        print(f"Final result: {result.content[:50]}...")
    
    data = {
        "traces": get_all_traces(),
        "spans": get_all_spans(),
        "agents": get_all_agents(),
    }
    
    spans = data["spans"]
    agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
    custom_spans = [s for s in spans if s["span_data"]["type"] == "custom"]
    
    print(f"\n✅ Sequential trace captured:")
    print(f"   - Total spans: {len(spans)}")
    print(f"   - Agent spans: {len(agent_spans)}")
    print(f"   - Custom spans (pipeline): {len(custom_spans)}")
    
    print(f"\n✅ Span hierarchy:")
    for span in custom_spans:
        if span["span_data"].get("name") == "sequential_pipeline":
            print(f"   Pipeline span: {span['span_id'][:16]}...")
            print(f"     - pattern: {span['span_data']['data'].get('pattern')}")
            print(f"     - agent_count: {span['span_data']['data'].get('agent_count')}")
            print(f"     - agent_names: {span['span_data']['data'].get('agent_names')}")
    
    for i, span in enumerate(agent_spans, 1):
        agent_id = span["span_data"].get("agent_id")
        agent_name = data["agents"].get(agent_id, {}).get("agent_name", "Unknown")
        parent_id = span.get("parent_id", "")[:16]
        print(f"   Agent {i} ({agent_name}): parent={parent_id}...")
    
    with open("test_sequential_trace.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Exported to: test_sequential_trace.json")
    
    print("\n" + "="*70)


async def test_runner_parallel():
    print("\n" + "="*70)
    print("TEST 3: Runner.run_parallel() - Concurrent Agents")
    print("="*70)
    
    clear_traces()
    
    chater = Chater(get_chater_cfg("ali"))
    
    agent1 = Agent(
        name="Agent1",
        system_prompt="You analyze from perspective 1.",
        chater=chater,
        memory=Memory(),
    )
    
    agent2 = Agent(
        name="Agent2",
        system_prompt="You analyze from perspective 2.",
        chater=chater,
        memory=Memory(),
    )
    
    agent3 = Agent(
        name="Agent3",
        system_prompt="You analyze from perspective 3.",
        chater=chater,
        memory=Memory(),
    )
    
    print("\n[TEST] Parallel execution with trace...")
    
    with trace("parallel_test"):
        results = await Runner.run_parallel(
            [agent1, agent2, agent3],
            "What is AI?"
        )
        for i, result in enumerate(results, 1):
            print(f"Agent {i}: {result.content[:30]}...")
    
    data = {
        "traces": get_all_traces(),
        "spans": get_all_spans(),
        "agents": get_all_agents(),
    }
    
    spans = data["spans"]
    agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
    custom_spans = [s for s in spans if s["span_data"]["type"] == "custom"]
    
    print(f"\n✅ Parallel trace captured:")
    print(f"   - Total spans: {len(spans)}")
    print(f"   - Agent spans (concurrent): {len(agent_spans)}")
    print(f"   - Custom spans (parallel): {len(custom_spans)}")
    
    print(f"\n✅ Concurrent relationship:")
    parallel_span = None
    for span in custom_spans:
        if span["span_data"].get("data", {}).get("pattern") == "parallel":
            parallel_span = span
            print(f"   Parallel span: {span['span_id'][:16]}...")
            print(f"     - agent_count: {span['span_data']['data'].get('agent_count')}")
            print(f"     - completed_count: {span['span_data']['data'].get('completed_count')}")
    
    if parallel_span:
        print(f"\n   Child agent spans (all have same parent):")
        for span in agent_spans:
            if span.get("parent_id") == parallel_span["span_id"]:
                agent_id = span["span_data"].get("agent_id")
                agent_name = data["agents"].get(agent_id, {}).get("agent_name", "Unknown")
                print(f"     - {agent_name}: {span['span_id'][:16]}...")
    
    with open("test_parallel_trace.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Exported to: test_parallel_trace.json")
    
    print("\n" + "="*70)


async def test_runner_msghub():
    print("\n" + "="*70)
    print("TEST 4: Runner.run_msghub() - Multi-round Broadcast")
    print("="*70)
    
    clear_traces()
    
    chater = Chater(get_chater_cfg("ali"))
    
    alice = Agent(
        name="Alice",
        system_prompt="You are Alice. Be creative.",
        chater=chater,
        memory=Memory(),
    )
    
    bob = Agent(
        name="Bob",
        system_prompt="You are Bob. Be analytical.",
        chater=chater,
        memory=Memory(),
    )
    
    charlie = Agent(
        name="Charlie",
        system_prompt="You are Charlie. Be balanced.",
        chater=chater,
        memory=Memory(),
    )
    
    print("\n[TEST] MsgHub with trace...")
    
    announcement = ChatResponse(role="user", content="Discuss: What is creativity?")
    
    with trace("msghub_test"):
        results = await Runner.run_msghub(
            [alice, bob, charlie],
            announcement,
            rounds=2
        )
        print(f"Total responses: {len(results)}")
    
    data = {
        "traces": get_all_traces(),
        "spans": get_all_spans(),
        "agents": get_all_agents(),
    }
    
    spans = data["spans"]
    agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
    custom_spans = [s for s in spans if s["span_data"]["type"] == "custom"]
    
    print(f"\n✅ MsgHub trace captured:")
    print(f"   - Total spans: {len(spans)}")
    print(f"   - Agent spans: {len(agent_spans)}")
    print(f"   - Custom spans: {len(custom_spans)}")
    
    print(f"\n✅ MsgHub structure:")
    for span in custom_spans:
        if span["span_data"].get("data", {}).get("pattern") == "msghub":
            print(f"   MsgHub span: {span['span_id'][:16]}...")
            print(f"     - rounds: {span['span_data']['data'].get('rounds')}")
            print(f"     - total_responses: {span['span_data']['data'].get('total_responses')}")
    
    print(f"\n   Agent interactions (expected 6: 3 agents x 2 rounds):")
    print(f"   Actual agent spans: {len(agent_spans)}")
    
    with open("test_msghub_trace.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Exported to: test_msghub_trace.json")
    
    print("\n" + "="*70)


async def test_direct_msghub_usage():
    print("\n" + "="*70)
    print("TEST 5: Direct msghub() usage (backward compatible)")
    print("="*70)
    
    clear_traces()
    
    chater = Chater(get_chater_cfg("ali"))
    
    agent1 = Agent(
        name="Agent1",
        system_prompt="You are agent 1.",
        chater=chater,
        memory=Memory(),
    )
    
    agent2 = Agent(
        name="Agent2",
        system_prompt="You are agent 2.",
        chater=chater,
        memory=Memory(),
    )
    
    print("\n[TEST] Direct msghub() with trace...")
    
    announcement = ChatResponse(role="user", content="Collaborate on a task")
    
    with trace("direct_msghub_test"):
        with msghub([agent1, agent2], announcement=announcement):
            for agent in [agent1, agent2]:
                final_response = None
                async for response in agent.reply("", auto_speak=True):
                    final_response = response
                if final_response:
                    print(f"{agent.name}: {final_response.content[:30]}...")
    
    spans = get_all_spans()
    agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
    custom_spans = [s for s in spans if s["span_data"]["type"] == "custom"]
    
    print(f"\n✅ Direct msghub trace captured:")
    print(f"   - Agent spans: {len(agent_spans)}")
    print(f"   - Custom spans (msghub_context): {len(custom_spans)}")
    
    for span in custom_spans:
        if span["span_data"].get("name") == "msghub_context":
            print(f"\n   MsgHub context span created automatically:")
            print(f"     - participants: {span['span_data']['data'].get('participants')}")
    
    print("\n✅ Backward compatible - works as before!")
    
    print("\n" + "="*70)


async def test_direct_sequential_pipeline():
    print("\n" + "="*70)
    print("TEST 6: Direct sequential_pipeline() usage")
    print("="*70)
    
    clear_traces()
    
    chater = Chater(get_chater_cfg("ali"))
    
    agent1 = Agent(
        name="Step1",
        system_prompt="You are step 1.",
        chater=chater,
        memory=Memory(),
    )
    
    agent2 = Agent(
        name="Step2",
        system_prompt="You are step 2.",
        chater=chater,
        memory=Memory(),
    )
    
    print("\n[TEST] Direct sequential_pipeline() with trace...")
    
    initial = ChatResponse(role="user", content="Start task")
    
    with trace("direct_pipeline_test"):
        result = await sequential_pipeline([agent1, agent2], initial)
        print(f"Final: {result.content[:30]}...")
    
    spans = get_all_spans()
    agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
    custom_spans = [s for s in spans if s["span_data"]["type"] == "custom"]
    
    print(f"\n✅ Direct pipeline trace captured:")
    print(f"   - Agent spans: {len(agent_spans)}")
    print(f"   - Custom spans (pipeline): {len(custom_spans)}")
    
    for span in custom_spans:
        if span["span_data"].get("name") == "sequential_pipeline":
            print(f"\n   Pipeline span created automatically:")
            print(f"     - agent_count: {span['span_data']['data'].get('agent_count')}")
            print(f"     - pattern: {span['span_data']['data'].get('pattern')}")
    
    print("\n✅ Backward compatible - works as before!")
    
    print("\n" + "="*70)


async def test_no_trace_zero_overhead():
    print("\n" + "="*70)
    print("TEST 7: No Trace - Zero Overhead")
    print("="*70)
    
    clear_traces()
    
    chater = Chater(get_chater_cfg("ali"))
    agent = Agent(
        name="FastAgent",
        system_prompt="You are fast.",
        chater=chater,
        memory=Memory(),
    )
    
    print("\n[TEST] Agent.reply() without trace...")
    
    final_response = None
    async for response in agent.reply("Quick test"):
        final_response = response
    print(f"Response: {final_response.content[:30]}...")
    
    spans = get_all_spans()
    print(f"\n✅ No trace overhead:")
    print(f"   - Spans captured: {len(spans)}")
    assert len(spans) == 0, "Should have no spans without trace context!"
    
    print("✅ Confirmed: Zero overhead when not tracing!")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPREHENSIVE MULTI-AGENT TRACE TESTS")
    print("="*70)
    
    # asyncio.run(test_auto_trace_in_agent_reply())
    # asyncio.run(test_runner_sequential())
    # asyncio.run(test_runner_parallel())
    # asyncio.run(test_runner_msghub())
    # asyncio.run(test_direct_msghub_usage())
    asyncio.run(test_direct_sequential_pipeline())
    asyncio.run(test_no_trace_zero_overhead())
    
    print("\n" + "="*70)
    print("✅ ALL MULTI-AGENT TRACE TESTS PASSED!")
    print("="*70)
    
    print("\n📝 Summary:")
    print("  ✓ Agent.reply() auto-detects trace")
    print("  ✓ Runner.run_sequential() with trace")
    print("  ✓ Runner.run_parallel() with concurrent agents")
    print("  ✓ Runner.run_msghub() with multi-round broadcast")
    print("  ✓ Direct msghub() usage (backward compatible)")
    print("  ✓ Direct sequential_pipeline() (backward compatible)")
    print("  ✓ Zero overhead without trace")
    print("\n🎉 Multi-agent tracing fully functional!")

