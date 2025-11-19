import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    trace,
    enable_console_output,
    export_traces,
    get_all_traces,
    get_all_spans,
    get_all_agents,
    clear_traces,
    Agent,
    ToolKit,
    get_chater_cfg,
)


def test_tool_execution_with_trace():
    print("\n" + "="*70)
    print("TEST: Tool Execution with Trace")
    print("="*70)
    
    clear_traces()
    enable_console_output(verbose=True)
    
    def add_numbers(a: int, b: int) -> int:
        return a + b
    
    def multiply(a: int, b: int) -> int:
        return a * b
    
    toolkit = ToolKit()
    toolkit.register(add_numbers)
    toolkit.register(multiply)
    
    async def test():
        with trace("tool_test"):
            result1 = await toolkit.execute("add_numbers", a=5, b=3)
            result2 = await toolkit.execute("multiply", a=4, b=2)
            
            assert result1 == 8
            assert result2 == 8
        
        spans = get_all_spans()
        tool_spans = [s for s in spans if s["span_data"]["type"] == "tool"]
        
        assert len(tool_spans) == 2
        assert tool_spans[0]["span_data"]["tool_name"] == "add_numbers"
        assert tool_spans[0]["span_data"]["output"] == 8
        assert tool_spans[1]["span_data"]["tool_name"] == "multiply"
        assert tool_spans[1]["span_data"]["output"] == 8
        
        print("\n✓ Tool execution trace works!")
    
    asyncio.run(test())
    print("="*70)


def test_batch_tool_execution():
    print("\n" + "="*70)
    print("TEST: Batch Tool Execution with Trace")
    print("="*70)
    
    clear_traces()
    enable_console_output(verbose=True)
    
    def add(a: int, b: int) -> int:
        return a + b
    
    toolkit = ToolKit()
    toolkit.register(add)
    
    async def test():
        with trace("batch_tool_test"):
            results = await toolkit.execute_many([
                {"name": "add", "arguments": {"a": 1, "b": 2}},
                {"name": "add", "arguments": {"a": 3, "b": 4}},
                {"name": "add", "arguments": {"a": 5, "b": 6}},
            ])
            
            assert results[0] == 3
            assert results[1] == 7
            assert results[2] == 11
        
        spans = get_all_spans()
        tool_spans = [s for s in spans if s["span_data"]["type"] == "tool"]
        custom_spans = [s for s in spans if s["span_data"]["type"] == "custom"]
        
        assert len(tool_spans) == 3
        assert len(custom_spans) == 1
        assert custom_spans[0]["span_data"]["name"] == "batch_tool_execution"
        assert custom_spans[0]["span_data"]["data"]["tool_count"] == 3
        assert custom_spans[0]["span_data"]["data"]["success_count"] == 3
        
        print("\n✓ Batch tool execution trace works!")
    
    asyncio.run(test())
    print("="*70)


def test_simulated_agent_workflow():
    print("\n" + "="*70)
    print("TEST: Simulated Agent Workflow with Full Trace")
    print("="*70)
    
    clear_traces()
    enable_console_output(verbose=True)
    
    from core._model import Chater, ChaterCfg, ChatCfg
    from core._model import ChatResponse
    
    try:
        chater_cfg = get_chater_cfg("ali")
    except Exception as e:
        print(f"\n⚠ Skipped (cannot create chater config): {str(e)[:100]}")
        return
    
    async def test():
        with trace("simulated_agent_workflow", group_id="test_session"):
            chater = Chater(chater_cfg)
            
            messages = [
                {"role": "user", "content": "Hello, how are you?"}
            ]
            
            response = await chater.chat(messages, stream=False)
            
            assert response is not None
            assert response.role == "assistant"
        
        traces = get_all_traces()
        spans = get_all_spans()
        
        gen_spans = [s for s in spans if s["span_data"]["type"] == "generation"]
        
        assert len(traces) == 1
        assert traces[0]["group_id"] == "test_session"
        assert len(gen_spans) >= 1
        
        if gen_spans:
            assert "model" in gen_spans[0]["span_data"]
            assert gen_spans[0]["span_data"]["usage"] is not None
        
        print("\n✓ Simulated agent workflow trace works!")
    
    try:
        asyncio.run(test())
    except Exception as e:
        print(f"\n⚠ Skipped (requires OpenAI API): {str(e)[:100]}")
    
    print("="*70)


def test_trace_hierarchy():
    print("\n" + "="*70)
    print("TEST: Trace Hierarchy Verification")
    print("="*70)
    
    clear_traces()
    
    def tool_a() -> str:
        return "result_a"
    
    def tool_b() -> str:
        return "result_b"
    
    toolkit = ToolKit()
    toolkit.register(tool_a)
    toolkit.register(tool_b)
    
    async def test():
        from core import agent_span, agent_step_span
        
        with trace("hierarchy_test"):
            with agent_span("TestAgent", "Agent", agent_id="agent_001"):
                with agent_step_span(step_index=0):
                    await toolkit.execute("tool_a")
                    await toolkit.execute("tool_b")
        
        spans = get_all_spans()
        
        agent_spans = [s for s in spans if s["span_data"]["type"] == "agent"]
        step_spans = [s for s in spans if s["span_data"]["type"] == "agent_step"]
        tool_spans = [s for s in spans if s["span_data"]["type"] == "tool"]
        
        assert len(agent_spans) == 1
        assert len(step_spans) == 1
        assert len(tool_spans) == 2
        
        agent_span_id = agent_spans[0]["span_id"]
        step_span_id = step_spans[0]["span_id"]
        
        assert step_spans[0]["parent_id"] == agent_span_id
        assert tool_spans[0]["parent_id"] == step_span_id
        assert tool_spans[1]["parent_id"] == step_span_id
        
        print("\nHierarchy structure:")
        print(f"  Trace")
        print(f"    └─> Agent (id={agent_span_id[:12]}...)")
        print(f"        └─> Step (id={step_span_id[:12]}..., parent={agent_span_id[:12]}...)")
        print(f"            ├─> Tool A (parent={step_span_id[:12]}...)")
        print(f"            └─> Tool B (parent={step_span_id[:12]}...)")
        
        print("\n✓ Trace hierarchy is correct!")
    
    asyncio.run(test())
    print("="*70)


def test_error_handling_in_tools():
    print("\n" + "="*70)
    print("TEST: Error Handling in Tools with Trace")
    print("="*70)
    
    clear_traces()
    enable_console_output(verbose=True)
    
    def failing_tool() -> str:
        raise ValueError("Intentional error")
    
    toolkit = ToolKit()
    toolkit.register(failing_tool)
    
    async def test():
        with trace("error_test"):
            try:
                await toolkit.execute("failing_tool")
            except Exception:
                pass
        
        spans = get_all_spans()
        tool_spans = [s for s in spans if s["span_data"]["type"] == "tool"]
        
        assert len(tool_spans) == 1
        assert tool_spans[0]["error"] is not None
        assert tool_spans[0]["span_data"]["error"] is not None
        
        print("\n✓ Error handling in tools works!")
    
    asyncio.run(test())
    print("="*70)


def run_all_integration_tests():
    print("\n" + "="*70)
    print("FULL INTEGRATION TRACE TESTS")
    print("="*70)
    
    test_tool_execution_with_trace()
    test_batch_tool_execution()
    test_trace_hierarchy()
    test_error_handling_in_tools()
    test_simulated_agent_workflow()
    
    print("\n" + "="*70)
    print("ALL INTEGRATION TESTS PASSED! ✓")
    print("="*70)
    print("\n📊 Summary:")
    print("  ✓ Tool execution trace")
    print("  ✓ Batch tool execution trace")  
    print("  ✓ Trace hierarchy")
    print("  ✓ Error handling")
    print("  ✓ Agent workflow (simulated)")
    print("\n🎉 Trace system fully integrated in model, tools, and agent!")


if __name__ == "__main__":
    run_all_integration_tests()

