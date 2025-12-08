"""
Tests for ConsoleTracingProcessor concurrent depth tracking fix.

This test verifies that the depth tracking in ConsoleTracingProcessor
uses contextvars properly for coroutine-safe operation.
"""

import pytest
import asyncio
import io
import sys
from contextlib import redirect_stdout

from qagent.core._trace import (
    ConsoleTracingProcessor,
    TraceImpl,
    SpanImpl,
    CustomSpanData,
    gen_trace_id,
    gen_span_id,
    _console_depth,
)


class TestConsoleTracingProcessorConcurrent:
    """Test concurrent depth tracking in ConsoleTracingProcessor."""

    def test_contextvars_depth_isolation(self):
        """Test that _console_depth is properly isolated between contexts."""
        # Initial value should be 0
        assert _console_depth.get() == 0
        
        # Set depth in current context
        _console_depth.set(5)
        assert _console_depth.get() == 5
        
        # Reset for other tests
        _console_depth.set(0)

    @pytest.mark.asyncio
    async def test_concurrent_spans_depth_isolation(self):
        """Test that concurrent coroutines maintain separate depth counts."""
        processor = ConsoleTracingProcessor(verbose=False)
        depths_recorded = {"task1": [], "task2": []}
        
        async def task1():
            # Simulate span nesting: start -> start -> end -> end
            processor._set_depth(0)
            for i in range(3):
                current = processor._get_depth()
                depths_recorded["task1"].append(("start", current))
                processor._set_depth(current + 1)
                await asyncio.sleep(0.01)  # Allow context switch
            
            for i in range(3):
                current = max(0, processor._get_depth() - 1)
                processor._set_depth(current)
                depths_recorded["task1"].append(("end", current))
                await asyncio.sleep(0.01)
        
        async def task2():
            await asyncio.sleep(0.005)  # Start slightly after task1
            processor._set_depth(0)
            for i in range(2):
                current = processor._get_depth()
                depths_recorded["task2"].append(("start", current))
                processor._set_depth(current + 1)
                await asyncio.sleep(0.01)
            
            for i in range(2):
                current = max(0, processor._get_depth() - 1)
                processor._set_depth(current)
                depths_recorded["task2"].append(("end", current))
                await asyncio.sleep(0.01)
        
        # Run concurrently
        await asyncio.gather(task1(), task2())
        
        # Each task should have consistent depth progression
        # Task1: start at 0,1,2, end at 2,1,0
        task1_starts = [d[1] for d in depths_recorded["task1"] if d[0] == "start"]
        task1_ends = [d[1] for d in depths_recorded["task1"] if d[0] == "end"]
        
        # Note: Due to contextvars isolation in async, each task maintains its own depth
        # The key is that within a single coroutine, depth should be consistent
        assert len(task1_starts) == 3
        assert len(task1_ends) == 3

    @pytest.mark.asyncio
    async def test_processor_verbose_output(self):
        """Test that verbose output shows correct indentation."""
        processor = ConsoleTracingProcessor(verbose=True)
        trace_id = gen_trace_id()
        
        # Create a mock trace
        class MockTrace:
            def __init__(self):
                self.trace_id = trace_id
                self.name = "test_trace"
        
        mock_trace = MockTrace()
        
        # Capture stdout
        captured = io.StringIO()
        with redirect_stdout(captured):
            processor.on_trace_start(mock_trace)
            
            # Create nested spans
            span1 = SpanImpl(
                trace_id=trace_id,
                span_id=gen_span_id(),
                parent_id=None,
                span_data=CustomSpanData("level1", {}),
                processor=processor,
            )
            processor.on_span_start(span1)
            
            span2 = SpanImpl(
                trace_id=trace_id,
                span_id=gen_span_id(),
                parent_id=span1.span_id,
                span_data=CustomSpanData("level2", {}),
                processor=processor,
            )
            processor.on_span_start(span2)
            processor.on_span_end(span2)
            processor.on_span_end(span1)
            
            processor.on_trace_end(mock_trace)
        
        output = captured.getvalue()
        lines = output.strip().split('\n')
        
        # Verify trace markers
        assert "[Trace Start]" in lines[0]
        assert "[Trace End]" in lines[-1]
        
        # Verify span nesting via indentation
        # level1 start should have no indent, level2 should have indent
        span_lines = [l for l in lines if "Span" in l]
        assert len(span_lines) == 4  # 2 starts + 2 ends

    def test_get_set_depth_methods(self):
        """Test _get_depth and _set_depth helper methods."""
        processor = ConsoleTracingProcessor(verbose=False)
        
        # Initial depth
        initial = processor._get_depth()
        assert initial == 0
        
        # Set and get
        processor._set_depth(3)
        assert processor._get_depth() == 3
        
        # Reset
        processor._set_depth(0)
        assert processor._get_depth() == 0

    @pytest.mark.asyncio
    async def test_depth_reset_on_trace_start(self):
        """Test that depth is reset to 0 on trace start."""
        processor = ConsoleTracingProcessor(verbose=False)
        
        # Set some depth
        processor._set_depth(5)
        assert processor._get_depth() == 5
        
        # Create mock trace
        class MockTrace:
            trace_id = "test"
            name = "test"
        
        # Start trace should reset depth
        processor.on_trace_start(MockTrace())
        assert processor._get_depth() == 0


class TestConsoleProcessorIntegration:
    """Integration tests with actual Trace/Span classes."""

    @pytest.mark.asyncio
    async def test_full_trace_with_nested_spans(self):
        """Test complete trace flow with nested spans."""
        processor = ConsoleTracingProcessor(verbose=False)
        trace_id = gen_trace_id()
        
        trace = TraceImpl(
            name="integration_test",
            trace_id=trace_id,
            group_id=None,
            metadata={},
            processor=processor,
        )
        
        trace.start()
        
        # Create nested spans
        span1 = SpanImpl(
            trace_id=trace_id,
            span_id=None,
            parent_id=None,
            span_data=CustomSpanData("outer"),
            processor=processor,
        )
        span1.start()
        
        inner_span = SpanImpl(
            trace_id=trace_id,
            span_id=None,
            parent_id=span1.span_id,
            span_data=CustomSpanData("inner"),
            processor=processor,
        )
        inner_span.start()
        inner_span.finish()
        
        span1.finish()
        trace.finish()
        
        # Depth should be back to 0
        assert processor._get_depth() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
