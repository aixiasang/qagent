"""
Tests for Runner module.

Tests the various static methods for running agents.
"""

import pytest
from qagent import Agent, Memory, Runner, ChatResponse


@pytest.fixture
def mock_agent(mock_chater):
    return Agent(
        name="TestAgent",
        chater=mock_chater,
        memory=Memory(),
        system_prompt="Test"
    )


@pytest.mark.asyncio
async def test_runner_single_agent(mock_agent):
    """Test Runner.run with a single agent."""
    result = await Runner.run(mock_agent, "Test message")
    
    assert result is not None
    assert result.content == "Mock response"


@pytest.mark.asyncio
async def test_runner_with_reply(mock_agent):
    """Test Runner.run_with_reply method."""
    result = await Runner.run_with_reply(mock_agent, "Test message")
    
    assert result is not None
    assert result.content == "Mock response"


@pytest.mark.asyncio
async def test_runner_sequential(mock_chater):
    """Test Runner.run_sequential with multiple agents."""
    agent1 = Agent(
        name="Agent1",
        chater=mock_chater,
        memory=Memory(),
        system_prompt="First"
    )
    agent2 = Agent(
        name="Agent2",
        chater=mock_chater,
        memory=Memory(),
        system_prompt="Second"
    )
    
    result = await Runner.run_sequential([agent1, agent2], "Test")
    
    assert result is not None
    assert result.content == "Mock response"


@pytest.mark.asyncio
async def test_runner_parallel(mock_chater):
    """Test Runner.run_parallel with multiple agents."""
    agents = [
        Agent(
            name=f"Agent{i}",
            chater=mock_chater,
            memory=Memory(),
            system_prompt=f"Agent {i}"
        )
        for i in range(3)
    ]
    
    results = await Runner.run_parallel(agents, "Test")
    
    assert len(results) == 3
    for result in results:
        assert result.content == "Mock response"


@pytest.mark.asyncio
async def test_runner_with_stream_callback(mock_agent):
    """Test Runner.run with stream callback."""
    completed = []
    
    def on_complete(response):
        completed.append(response)
    
    result = await Runner.run(
        mock_agent, 
        "Test", 
        stream=False, 
        on_complete=on_complete
    )
    
    assert result is not None
    assert len(completed) == 1
