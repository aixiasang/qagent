import pytest
from pyagent import ReActAgent, Memory, ToolKit, ChatResponse


async def sample_tool(x: int) -> int:
    return x * 2


@pytest.fixture
def react_agent(mock_chater):
    toolkit = ToolKit()
    toolkit.register(sample_tool)
    
    return ReActAgent(
        name="ReActBot",
        chater=mock_chater,
        memory=Memory(),
        tools=toolkit,
        track_reasoning=True
    )

@pytest.mark.asyncio
async def test_react_agent_creation(react_agent):
    assert react_agent.name == "ReActBot"
    assert react_agent.track_reasoning is True
    assert react_agent.tracker is not None

@pytest.mark.asyncio
async def test_react_agent_reply(react_agent):
    response = None
    async for resp in react_agent.reply("Test question"):
        response = resp
    
    assert response is not None

@pytest.mark.asyncio
async def test_react_tracker(react_agent):
    async for _ in react_agent.reply("Test"):
        pass
    
    trace = react_agent.tracker.get_full_trace()
    assert "total_iterations" in trace
    assert "thoughts" in trace
    assert "actions" in trace
    assert "observations" in trace

def test_react_tracker_clear(react_agent):
    react_agent.tracker.record_thought("Test thought")
    react_agent.tracker.clear()
    
    trace = react_agent.tracker.get_full_trace()
    assert len(trace["thoughts"]) == 0
    assert trace["total_iterations"] == 0

def test_react_tracker_record_thought(react_agent):
    react_agent.tracker.record_thought("Thinking...")
    
    assert len(react_agent.tracker.thoughts) == 1
    assert react_agent.tracker.thoughts[0]["content"] == "Thinking..."

def test_react_tracker_next_iteration(react_agent):
    initial_iter = react_agent.tracker.iteration
    react_agent.tracker.next_iteration()
    
    assert react_agent.tracker.iteration == initial_iter + 1
