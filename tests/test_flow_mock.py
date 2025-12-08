"""
Tests for Flow module with corrected MockAgent implementation.

This test verifies that the Flow system works correctly with agents
that return ChatResponse directly (not generators).
"""

import pytest
import asyncio

from qagent.core._flow import Flow, END, chain, FlowContext
from qagent.core._model import ChatResponse


class MockAgent:
    """Mock agent that returns ChatResponse directly (correct API)."""
    
    def __init__(self, name: str, response: str = ""):
        self.name = name
        self._response = response
        self.call_count = 0
        self.received_messages = []
    
    async def __call__(self, message: str, stream: bool = False) -> ChatResponse:
        """Make agent callable as Flow expects."""
        return await self.reply(message, stream)
    
    async def reply(self, message: str, stream: bool = False) -> ChatResponse:
        """Return ChatResponse directly, not a generator."""
        self.call_count += 1
        self.received_messages.append(message)
        content = self._response or f"{self.name} processed: {message[:20]}"
        return ChatResponse(role="assistant", content=content)


class CounterAgent:
    """Agent that counts calls and returns status."""
    
    def __init__(self, stop_at: int = 3):
        self.name = "Counter"
        self.count = 0
        self.stop_at = stop_at
    
    async def __call__(self, message: str, stream: bool = False) -> ChatResponse:
        """Make agent callable as Flow expects."""
        return await self.reply(message, stream)
    
    async def reply(self, message: str, stream: bool = False) -> ChatResponse:
        self.count += 1
        status = "DONE" if self.count >= self.stop_at else "CONTINUE"
        return ChatResponse(
            role="assistant", 
            content=f"Count: {self.count}, Status: {status}"
        )


class TestFlowBasic:
    """Basic flow tests."""

    @pytest.mark.asyncio
    async def test_single_agent_flow(self):
        """Test flow with single agent."""
        agent = MockAgent("SingleAgent", "Hello from agent")
        flow = Flow("single").add("agent", agent)
        
        result = await flow.reply("Test input")
        
        assert result is not None
        assert "Hello from agent" in result.content
        assert agent.call_count == 1

    @pytest.mark.asyncio
    async def test_sequential_flow(self):
        """Test sequential flow execution."""
        agent_a = MockAgent("AgentA", "Step 1 done")
        agent_b = MockAgent("AgentB", "Step 2 done")
        agent_c = MockAgent("AgentC", "Step 3 done")
        
        flow = (
            Flow("sequential")
            .add("a", agent_a)
            .add("b", agent_b)
            .add("c", agent_c)
        )
        
        result = await flow.reply("Start")
        
        assert "AgentC" in result.content or "Step 3" in result.content
        assert agent_a.call_count == 1
        assert agent_b.call_count == 1
        assert agent_c.call_count == 1

    @pytest.mark.asyncio
    async def test_flow_entry_point(self):
        """Test setting custom entry point."""
        agent_a = MockAgent("AgentA", "From A")
        agent_b = MockAgent("AgentB", "From B")
        
        flow = (
            Flow("custom_entry")
            .add("a", agent_a)
            .add("b", agent_b)
            .entry("b")  # Start from B
        )
        
        result = await flow.reply("Test")
        
        # Should only execute B since we start there and no routing back to A
        assert agent_b.call_count >= 1


class TestFlowConditional:
    """Conditional routing tests."""

    @pytest.mark.asyncio
    async def test_conditional_routing(self):
        """Test conditional routing based on response content."""
        planner = MockAgent("Planner", "Plan created")
        executor = MockAgent("Executor", "Executed")
        reviewer = MockAgent("Reviewer", "APPROVED - looks good")
        
        flow = (
            Flow("conditional")
            .add("plan", planner)
            .add("execute", executor)
            .add("review", reviewer)
        )
        flow.route("plan").to("execute")
        flow.route("execute").to("review")
        flow.route("review").when(
            lambda r: "REJECTED" in r.content
        ).to("plan").when(
            lambda r: "APPROVED" in r.content
        ).to(END).default().to("plan")
        
        result = await flow.reply("Build app")
        
        assert "APPROVED" in result.content

    @pytest.mark.asyncio
    async def test_default_routing(self):
        """Test default routing when no condition matches."""
        agent_a = MockAgent("A", "No match here")
        agent_b = MockAgent("B", "Default route")
        
        flow = Flow("default_route").add("a", agent_a).add("b", agent_b)
        flow.route("a").when(
            lambda r: "NEVER" in r.content
        ).to(END).default().to("b")
        flow.route("b").to(END)
        
        result = await flow.reply("Test")
        
        assert agent_b.call_count == 1


class TestFlowLoop:
    """Loop flow tests."""

    @pytest.mark.asyncio
    async def test_loop_with_condition(self):
        """Test loop that exits on condition."""
        counter = CounterAgent(stop_at=3)
        
        flow = Flow("loop").add("counter", counter).max_loops(10)
        flow.route("counter").when(
            lambda r: "DONE" in r.content
        ).to(END).default().to("counter")
        
        result = await flow.reply("Start counting")
        
        assert "Count: 3" in result.content
        assert "DONE" in result.content
        assert counter.count == 3

    @pytest.mark.asyncio
    async def test_max_loops_limit(self):
        """Test that max_loops prevents infinite loops."""
        # Agent that never says DONE
        infinite_agent = MockAgent("Infinite", "CONTINUE forever")
        
        flow = Flow("max_loop").add("agent", infinite_agent).max_loops(5)
        flow.route("agent").when(
            lambda r: "DONE" in r.content
        ).to(END).default().to("agent")
        
        result = await flow.reply("Test")
        
        # Should stop after max_loops iterations
        assert infinite_agent.call_count <= 5


class TestFlowParallel:
    """Parallel execution tests."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel agent execution."""
        tech = MockAgent("Tech", "Technical analysis")
        biz = MockAgent("Business", "Business analysis")
        legal = MockAgent("Legal", "Legal analysis")
        
        flow = Flow("parallel").parallel("experts", [tech, biz, legal])
        
        result = await flow.reply("Analyze this")
        
        # All agents should be called
        assert tech.call_count == 1
        assert biz.call_count == 1
        assert legal.call_count == 1
        
        # Result should combine all outputs
        assert "Technical" in result.content
        assert "Business" in result.content
        assert "Legal" in result.content


class TestFlowSubflow:
    """Subflow (nested flow) tests."""

    @pytest.mark.asyncio
    async def test_nested_subflow(self):
        """Test flow containing another flow."""
        inner_a = MockAgent("InnerA", "Inner step 1")
        inner_b = MockAgent("InnerB", "Inner step 2")
        outer_start = MockAgent("OuterStart", "Starting")
        outer_end = MockAgent("OuterEnd", "Finishing")
        
        inner_flow = Flow("inner").add("a", inner_a).add("b", inner_b)
        
        outer_flow = (
            Flow("outer")
            .add("start", outer_start)
            .add("inner", inner_flow)
            .add("end", outer_end)
        )
        
        result = await outer_flow.reply("Begin")
        
        assert outer_start.call_count == 1
        assert outer_end.call_count == 1
        assert "OuterEnd" in result.content or "Finishing" in result.content


class TestChainHelper:
    """Tests for chain() helper function."""

    @pytest.mark.asyncio
    async def test_chain_creates_sequential_flow(self):
        """Test chain helper creates sequential flow."""
        a = MockAgent("A", "A done")
        b = MockAgent("B", "B done")
        c = MockAgent("C", "C done")
        
        flow = chain(a, b, c)
        result = await flow.reply("Go")
        
        assert a.call_count == 1
        assert b.call_count == 1
        assert c.call_count == 1
        assert "C" in result.content

    @pytest.mark.asyncio
    async def test_chain_with_single_agent(self):
        """Test chain with single agent."""
        agent = MockAgent("Solo", "Solo response")
        
        flow = chain(agent)
        result = await flow.reply("Test")
        
        assert agent.call_count == 1
        assert "Solo" in result.content


class TestFlowContext:
    """FlowContext tests."""

    def test_flow_context_initialization(self):
        """Test FlowContext initialization."""
        ctx = FlowContext()
        
        assert ctx.current_node == ""
        assert ctx.messages == {}
        assert ctx.history == []
        assert ctx.loop_count == 0
        assert ctx.input_message == ""

    def test_flow_context_with_input(self):
        """Test FlowContext with input message."""
        ctx = FlowContext(input_message="test input")
        
        assert ctx.input_message == "test input"
        assert ctx.current_node == ""
        assert ctx.loop_count == 0

    def test_flow_context_last_response(self):
        """Test last_response method."""
        from qagent.core._model import ChatResponse
        
        ctx = FlowContext()
        
        # No history, should return None
        assert ctx.last_response() is None
        
        # Add history
        response = ChatResponse(role="assistant", content="test")
        ctx.history.append(("node1", response))
        
        assert ctx.last_response() is not None
        assert ctx.last_response().content == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
