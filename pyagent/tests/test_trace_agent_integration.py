import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    trace,
    agent_span,
    agent_step_span,
    generation_span,
    tool_span,
    enable_console_output,
    export_traces,
    get_all_traces,
    get_all_spans,
    get_all_agents,
    clear_traces,
)


def search_tool(query: str) -> str:
    return f"Search results for: {query}"


def calculator_tool(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


async def test_agent_with_trace():
    print("\n" + "="*70)
    print("TEST: Simulated Agent with Trace Integration")
    print("="*70)
    
    clear_traces()
    enable_console_output(verbose=True)
    
    with trace("agent_workflow", group_id="test_session_001"):
        with agent_span(
            agent_name="TestAgent",
            agent_type="Agent",
            agent_id="agent_001",
            tools=["search", "calculator"],
        ) as span_agent:
            span_agent.span_data.user_input = "Test user input"
            
            with agent_step_span(step_index=0, has_tool_calls=True, tool_count=2):
                with generation_span(
                    model="simulated-gpt-4",
                    model_config={"temperature": 0.7}
                ) as span_gen:
                    await asyncio.sleep(0.01)
                    span_gen.span_data.input_msgs = [
                        {"role": "user", "content": "Search for Python and calculate 2+2"}
                    ]
                    span_gen.span_data.output_msg = {
                        "role": "assistant",
                        "content": "Let me help you with that",
                        "tool_calls": [
                            {"name": "search", "arguments": {"query": "Python"}},
                            {"name": "calculator", "arguments": {"expression": "2+2"}},
                        ]
                    }
                    span_gen.span_data.usage = {
                        "input_tokens": 50,
                        "output_tokens": 30,
                    }
                
                with tool_span("search", input_args={"query": "Python"}) as span_tool1:
                    result1 = search_tool("Python")
                    span_tool1.span_data.output = result1
                
                with tool_span("calculator", input_args={"expression": "2+2"}) as span_tool2:
                    result2 = calculator_tool("2+2")
                    span_tool2.span_data.output = result2
            
            span_agent.span_data.agent_output = "Here are your results..."
    
    traces = get_all_traces()
    spans = get_all_spans()
    agents = get_all_agents()
    
    print("\n" + "-"*70)
    print("RESULTS:")
    print(f"  Traces: {len(traces)}")
    print(f"  Spans: {len(spans)}")
    print(f"  Agents: {len(agents)}")
    
    assert len(traces) == 1
    assert len(spans) == 5
    assert len(agents) == 1
    
    gen_spans = [s for s in spans if s["span_data"]["type"] == "generation"]
    tool_spans = [s for s in spans if s["span_data"]["type"] == "tool"]
    
    assert len(gen_spans) == 1
    assert gen_spans[0]["span_data"]["usage"]["input_tokens"] == 50
    assert len(tool_spans) == 2
    
    print("\n✓ Agent with Trace Integration works!")
    print("="*70)
    
    export_path = "test_agent_trace.json"
    export_traces(export_path)
    print(f"\n✓ Trace exported to: {export_path}")
    
    import json
    from pathlib import Path
    data = json.loads(Path(export_path).read_text(encoding='utf-8'))
    print(f"\nExported data structure:")
    print(f"  - Traces: {len(data['traces'])}")
    print(f"  - Spans: {len(data['spans'])}")
    print(f"  - Agents: {len(data['agents'])}")
    
    Path(export_path).unlink(missing_ok=True)


async def test_multi_agent_collaboration():
    print("\n" + "="*70)
    print("TEST: Multi-Agent Collaboration with Trace")
    print("="*70)
    
    clear_traces()
    enable_console_output(verbose=True)
    
    with trace("multi_agent_collaboration", group_id="collab_session_001"):
        with agent_span(
            "CoordinatorAgent",
            "CoordinatorAgent",
            agent_id="coord_001",
        ) as coord:
            coord.span_data.user_input = "Complex multi-step task"
            
            with agent_step_span(step_index=0):
                with generation_span(model="gpt-4") as gen:
                    await asyncio.sleep(0.01)
                    gen.span_data.output_msg = {"content": "Delegating to worker"}
                
                with tool_span("delegate_to_worker") as tool:
                    with agent_span(
                        "WorkerAgent",
                        "WorkerAgent",
                        agent_id="worker_001",
                        parent_agent_id="coord_001",
                        handoff_from="CoordinatorAgent",
                    ) as worker:
                        worker.span_data.user_input = "Subtask from coordinator"
                        
                        with agent_step_span(step_index=0):
                            with generation_span(model="gpt-4") as gen2:
                                await asyncio.sleep(0.01)
                                gen2.span_data.output_msg = {"content": "Processing subtask"}
                            
                            with tool_span("process_data") as tool2:
                                await asyncio.sleep(0.01)
                                tool2.span_data.output = "Processed data"
                        
                        worker.span_data.agent_output = "Subtask completed"
                    
                    tool.span_data.output = "Worker completed successfully"
            
            coord.span_data.agent_output = "All tasks completed"
    
    traces = get_all_traces()
    spans = get_all_spans()
    agents = get_all_agents()
    
    print("\n" + "-"*70)
    print("RESULTS:")
    print(f"  Traces: {len(traces)}")
    print(f"  Spans: {len(spans)}")
    print(f"  Agents: {len(agents)}")
    
    assert len(agents) == 2
    assert agents["worker_001"]["parent_agent_id"] == "coord_001"
    
    agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
    assert len(agent_spans) == 2
    
    worker_span = [s for s in agent_spans if s["span_data"].get("agent_id") == "worker_001"][0]
    assert worker_span["span_data"]["handoff_from"] == "CoordinatorAgent"
    
    print("\n✓ Multi-Agent Collaboration with Trace works!")
    print("="*70)


async def test_error_propagation():
    print("\n" + "="*70)
    print("TEST: Error Propagation in Trace")
    print("="*70)
    
    clear_traces()
    
    try:
        with trace("error_workflow"):
            with agent_span("ErrorAgent", "Agent", agent_id="error_001"):
                with agent_step_span(step_index=0):
                    with generation_span(model="gpt-4"):
                        pass
                    
                    with tool_span("failing_tool") as tool_span_obj:
                        raise RuntimeError("Simulated tool failure")
    except RuntimeError:
        pass
    
    spans = get_all_spans()
    tool_spans = [s for s in spans if s["span_data"]["type"] == "tool"]
    
    assert len(tool_spans) == 1
    assert tool_spans[0]["error"] is not None
    assert "RuntimeError" in tool_spans[0]["error"]["data"]["exc_type"]
    assert "Simulated tool failure" in tool_spans[0]["error"]["message"]
    
    print("\n✓ Error Propagation in Trace works!")
    print("="*70)


async def test_trace_performance():
    print("\n" + "="*70)
    print("TEST: Trace Performance (1000 spans)")
    print("="*70)
    
    clear_traces()
    
    import time as time_module
    start = time_module.time()
    
    with trace("performance_test"):
        for i in range(100):
            with agent_span(f"Agent_{i}", "Agent"):
                for j in range(10):
                    with tool_span(f"tool_{j}"):
                        pass
    
    end = time_module.time()
    duration = (end - start) * 1000
    
    spans = get_all_spans()
    
    print(f"\n  Created {len(spans)} spans in {duration:.2f}ms")
    print(f"  Average time per span: {duration/len(spans):.4f}ms")
    
    assert len(spans) == 1100
    
    print("\n✓ Trace Performance test passed!")
    print("="*70)


async def run_all_integration_tests():
    print("\n" + "="*70)
    print("TRACE-AGENT INTEGRATION TESTS")
    print("="*70)
    
    await test_agent_with_trace()
    await test_multi_agent_collaboration()
    await test_error_propagation()
    await test_trace_performance()
    
    print("\n" + "="*70)
    print("ALL INTEGRATION TESTS PASSED! ✓")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(run_all_integration_tests())

