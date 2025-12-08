"""
Tests for ReActAgent module.

Tests the ReActAgent creation, reply, and ReActTrace functionality.
"""

import pytest
from qagent import ReActAgent, Memory, ToolKit, ChatResponse
from qagent.agent._react_agent import ReActTrace, ReActStep


async def sample_tool(x: int) -> int:
    """Double a number.
    
    Args:
        x: Number to double
    
    Returns:
        The doubled value
    """
    return x * 2


@pytest.fixture
def react_agent(mock_chater):
    """Create a ReActAgent with a sample tool."""
    toolkit = ToolKit()
    toolkit.register(sample_tool)
    
    return ReActAgent(
        name="ReActBot",
        chater=mock_chater,
        memory=Memory(),
        tools=toolkit,
        max_iterations=5
    )


@pytest.mark.asyncio
async def test_react_agent_creation(react_agent):
    """Test ReActAgent is created correctly."""
    assert react_agent.name == "ReActBot"
    assert react_agent.max_iterations == 5
    # ReActAgent should have internal trace
    assert hasattr(react_agent, '_react_trace')


@pytest.mark.asyncio
async def test_react_agent_reply(react_agent):
    """Test ReActAgent.reply returns ChatResponse."""
    # ReActAgent.reply() returns ChatResponse directly
    response = await react_agent.reply("Test question")
    
    assert response is not None
    assert isinstance(response, ChatResponse)


@pytest.mark.asyncio
async def test_react_agent_trace(react_agent):
    """Test ReActAgent internal trace tracking."""
    await react_agent.reply("Test")
    
    # Check internal trace
    trace = react_agent._react_trace
    assert isinstance(trace, ReActTrace)


class TestReActTrace:
    """Tests for ReActTrace dataclass."""

    def test_trace_creation(self):
        """Test ReActTrace creation."""
        trace = ReActTrace()
        assert trace.steps == []
        assert trace.final_answer is None
        assert trace.total_iterations == 0

    def test_trace_add_step(self):
        """Test adding steps to trace."""
        trace = ReActTrace()
        
        step = ReActStep(
            iteration=0,
            thought="I need to search",
            action="search",
            action_input={"query": "test"},
            observation="Results found"
        )
        trace.add_step(step)
        
        assert len(trace.steps) == 1
        assert trace.total_iterations == 1

    def test_trace_to_dict(self):
        """Test trace serialization."""
        trace = ReActTrace()
        step = ReActStep(iteration=0, thought="Thinking")
        trace.add_step(step)
        trace.final_answer = "Answer"
        
        result = trace.to_dict()
        
        assert result["total_iterations"] == 1
        assert len(result["steps"]) == 1
        assert result["final_answer"] == "Answer"

    def test_trace_clear(self):
        """Test clearing trace."""
        trace = ReActTrace()
        trace.add_step(ReActStep(iteration=0))
        trace.final_answer = "Some answer"
        
        trace.clear()
        
        assert trace.steps == []
        assert trace.final_answer is None
        assert trace.total_iterations == 0


class TestReActStep:
    """Tests for ReActStep dataclass."""

    def test_step_creation(self):
        """Test ReActStep creation with all fields."""
        step = ReActStep(
            iteration=1,
            thought="Need to calculate",
            action="calculate",
            action_input={"expr": "2+2"},
            observation="4",
            is_final=True
        )
        
        assert step.iteration == 1
        assert step.thought == "Need to calculate"
        assert step.action == "calculate"
        assert step.action_input == {"expr": "2+2"}
        assert step.observation == "4"
        assert step.is_final is True

    def test_step_defaults(self):
        """Test ReActStep default values."""
        step = ReActStep(iteration=0)
        
        assert step.thought is None
        assert step.action is None
        assert step.action_input is None
        assert step.observation is None
        assert step.is_final is False
