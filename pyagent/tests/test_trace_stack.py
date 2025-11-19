import asyncio
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core._trace import (
    trace,
    agent_span,
    agent_step_span,
    generation_span,
    tool_span,
    custom_span,
    enable_console_output,
    export_traces,
    get_all_traces,
    get_all_spans,
    get_all_agents,
    clear_traces,
    get_current_trace,
    get_current_span,
)


def test_basic_stack_structure():
    print("\n=== Test 1: Basic Stack Structure ===")
    clear_traces()
    enable_console_output(verbose=True)
    
    with trace("test_workflow", metadata={"test": "basic"}):
        assert get_current_trace() is not None
        print(f"Current trace: {get_current_trace().trace_id}")
        
        with agent_span("TestAgent", "BaseAgent"):
            assert get_current_span() is not None
            print(f"Current span: {get_current_span().span_id}")
            
            with generation_span(model="gpt-4"):
                pass
            
            with tool_span("search_tool", input_args={"query": "test"}):
                pass
    
    traces = get_all_traces()
    spans = get_all_spans()
    
    print(f"\nCollected {len(traces)} traces, {len(spans)} spans")
    print(f"Span types: {[s['span_data']['type'] for s in spans]}")
    assert len(traces) == 1
    assert len(spans) == 3
    print("✓ Basic stack structure works!")


def test_multi_agent_scenario():
    print("\n=== Test 2: Multi-Agent Scenario ===")
    clear_traces()
    enable_console_output(verbose=True)
    
    with trace("multi_agent_workflow", group_id="session_123"):
        with agent_span(
            "TriageAgent", 
            "RouterAgent",
            agent_id="agent_a",
            tools=["handoff"]
        ) as agent_a:
            agent_a.span_data.user_input = "I need help with billing"
            
            with agent_step_span(step_index=0, has_tool_calls=True, tool_count=1):
                with generation_span(model="gpt-4"):
                    time.sleep(0.01)
                
                with tool_span("handoff_to_billing") as tool:
                    with agent_span(
                        "BillingAgent",
                        "SpecialistAgent", 
                        agent_id="agent_b",
                        parent_agent_id="agent_a",
                        handoff_from="TriageAgent"
                    ) as agent_b:
                        agent_b.span_data.user_input = "Billing issue forwarded"
                        
                        with agent_step_span(step_index=0):
                            with generation_span(model="gpt-4"):
                                time.sleep(0.01)
                            
                            with tool_span("check_billing_database"):
                                time.sleep(0.01)
                        
                        agent_b.span_data.agent_output = "Billing resolved"
                    
                    tool.span_data.output = "Agent B completed"
            
            agent_a.span_data.agent_output = "Task delegated successfully"
    
    traces = get_all_traces()
    spans = get_all_spans()
    agents = get_all_agents()
    
    print(f"\nCollected:")
    print(f"  - {len(traces)} traces")
    print(f"  - {len(spans)} spans")
    print(f"  - {len(agents)} agents")
    
    assert len(agents) == 2
    assert "agent_a" in agents
    assert "agent_b" in agents
    assert agents["agent_b"]["parent_agent_id"] == "agent_a"
    
    parent_child_found = False
    for span in spans:
        if span["span_data"].get("parent_agent_id") == "agent_a":
            parent_child_found = True
            break
    
    assert parent_child_found
    print("✓ Multi-agent scenario works with parent-child tracking!")


def test_span_hierarchy():
    print("\n=== Test 3: Span Hierarchy ===")
    clear_traces()
    
    with trace("hierarchy_test"):
        trace_id = get_current_trace().trace_id
        
        with agent_span("Agent1", "BaseAgent") as s1:
            span1_id = s1.span_id
            
            with agent_step_span(step_index=0) as s2:
                span2_id = s2.span_id
                
                with generation_span(model="gpt-4") as s3:
                    span3_id = s3.span_id
    
    spans = get_all_spans()
    
    span_map = {s["span_id"]: s for s in spans}
    
    assert span_map[span1_id]["parent_id"] is None
    assert span_map[span2_id]["parent_id"] == span1_id
    assert span_map[span3_id]["parent_id"] == span2_id
    
    print("Hierarchy verified:")
    print(f"  Trace -> Agent (parent=None)")
    print(f"  Agent -> Step (parent={span1_id[:12]}...)")
    print(f"  Step -> Generation (parent={span2_id[:12]}...)")
    print("✓ Span hierarchy correctly maintained!")


def test_error_handling():
    print("\n=== Test 4: Error Handling ===")
    clear_traces()
    
    try:
        with trace("error_test"):
            with agent_span("Agent", "BaseAgent"):
                with tool_span("failing_tool") as tool:
                    raise ValueError("Simulated error")
    except ValueError:
        pass
    
    spans = get_all_spans()
    tool_spans = [s for s in spans if s["span_data"]["type"] == "tool"]
    
    assert len(tool_spans) == 1
    assert tool_spans[0]["error"] is not None
    assert "ValueError" in tool_spans[0]["error"]["data"]["exc_type"]
    
    print("✓ Error captured in span!")


async def simulate_model_call():
    with generation_span(
        model="gpt-4",
        model_config={"temperature": 0.7}
    ) as span:
        await asyncio.sleep(0.01)
        span.span_data.input_msgs = [{"role": "user", "content": "test"}]
        span.span_data.output_msg = {"role": "assistant", "content": "response"}
        span.span_data.usage = {"input_tokens": 10, "output_tokens": 20}


async def simulate_tool_call():
    with tool_span(
        "search_api",
        input_args={"query": "test query"}
    ) as span:
        await asyncio.sleep(0.01)
        span.span_data.output = {"results": ["item1", "item2"]}


async def test_async_integration():
    print("\n=== Test 5: Async Integration (Simulating Model & Tool) ===")
    clear_traces()
    enable_console_output(verbose=True)
    
    with trace("async_workflow"):
        with agent_span("AsyncAgent", "BaseAgent"):
            with agent_step_span(step_index=0):
                await simulate_model_call()
                await simulate_tool_call()
    
    spans = get_all_spans()
    gen_spans = [s for s in spans if s["span_data"]["type"] == "generation"]
    tool_spans = [s for s in spans if s["span_data"]["type"] == "tool"]
    
    assert len(gen_spans) == 1
    assert gen_spans[0]["span_data"]["usage"] == {"input_tokens": 10, "output_tokens": 20}
    
    assert len(tool_spans) == 1
    assert tool_spans[0]["span_data"]["output"] == {"results": ["item1", "item2"]}
    
    print("✓ Async integration works!")


def test_export_to_file():
    print("\n=== Test 6: Export to File ===")
    clear_traces()
    
    with trace("export_test", group_id="group_456"):
        with agent_span("Agent", "BaseAgent", agent_id="agent_001"):
            with generation_span(model="gpt-4"):
                pass
    
    import tempfile
    import json
    from pathlib import Path
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        export_traces(temp_path)
        
        data = json.loads(Path(temp_path).read_text(encoding='utf-8'))
        
        assert "traces" in data
        assert "spans" in data
        assert "agents" in data
        assert len(data["traces"]) == 1
        assert data["traces"][0]["group_id"] == "group_456"
        
        print(f"✓ Exported to {temp_path}")
        print(f"  Traces: {len(data['traces'])}")
        print(f"  Spans: {len(data['spans'])}")
        print(f"  Agents: {len(data['agents'])}")
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_nested_agents_complex():
    print("\n=== Test 7: Complex Nested Agents (3 levels) ===")
    clear_traces()
    enable_console_output(verbose=True)
    
    with trace("complex_multi_agent"):
        with agent_span("CoordinatorAgent", "CoordinatorAgent", agent_id="coord") as coord:
            coord.span_data.user_input = "Complex task"
            
            with tool_span("delegate_to_worker_a"):
                with agent_span(
                    "WorkerA", "WorkerAgent",
                    agent_id="worker_a",
                    parent_agent_id="coord"
                ):
                    with generation_span(model="gpt-4"):
                        pass
                    
                    with tool_span("delegate_to_specialist"):
                        with agent_span(
                            "SpecialistAgent", "SpecialistAgent",
                            agent_id="specialist",
                            parent_agent_id="worker_a"
                        ):
                            with generation_span(model="gpt-4"):
                                pass
    
    agents = get_all_agents()
    spans = get_all_spans()
    
    print(f"\nAgent hierarchy:")
    print(f"  Coordinator (coord)")
    print(f"    └─> WorkerA (worker_a)")
    print(f"         └─> Specialist (specialist)")
    
    assert len(agents) == 3
    assert agents["worker_a"]["parent_agent_id"] == "coord"
    assert agents["specialist"]["parent_agent_id"] == "worker_a"
    
    print("✓ 3-level agent hierarchy works!")


def run_all_tests():
    print("\n" + "="*60)
    print("TRACE SYSTEM TESTS")
    print("="*60)
    
    test_basic_stack_structure()
    test_multi_agent_scenario()
    test_span_hierarchy()
    test_error_handling()
    asyncio.run(test_async_integration())
    test_export_to_file()
    test_nested_agents_complex()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()

