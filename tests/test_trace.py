import pytest
from qagent import trace, export_traces, get_current_trace, custom_span, agent_span

def test_trace_creation():
    with trace("test_trace") as t:
        assert t is not None
        current = get_current_trace()
        assert current is not None
        assert current.trace_id == t.trace_id

def test_trace_disabled():
    from qagent import disable_tracing, enable_tracing
    
    disable_tracing()
    with trace("disabled") as t:
        pass
    
    enable_tracing()

def test_trace_export():
    import tempfile
    import os
    
    with trace("export_test"):
        pass
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filepath = f.name
    
    try:
        export_traces(filepath)
        assert os.path.exists(filepath)
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)

def test_custom_span():
    with trace("custom_test"):
        with custom_span("step1", data={"value": 123}):
            pass

def test_agent_span():
    with trace("agent_test"):
        with agent_span(
            agent_name="TestAgent",
            agent_type="Agent",
            tools=["tool1", "tool2"],
            user_input="test input",
            agent_id="test_id"
        ):
            pass

def test_nested_spans():
    with trace("nested_test"):
        with custom_span("outer"):
            with custom_span("inner"):
                pass

def test_trace_isolation():
    with trace("trace1") as t1:
        trace1_id = t1.trace_id
    
    with trace("trace2") as t2:
        trace2_id = t2.trace_id
    
    assert trace1_id != trace2_id

def test_no_trace_context():
    current = get_current_trace()
    assert current is None

def test_export_multiple_traces():
    import tempfile
    import os
    
    with trace("trace1"):
        pass
    
    with trace("trace2"):
        pass
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filepath = f.name
    
    try:
        export_traces(filepath)
        assert os.path.exists(filepath)
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)
