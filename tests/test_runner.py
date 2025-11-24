import pytest
from pyagent import Agent, Memory, Runner, ChatResponse


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
    result = await Runner.run(mock_agent, "Test message")
    
    assert result is not None
    assert result.content == "Mock response"

@pytest.mark.asyncio
async def test_runner_streamed(mock_agent):
    chunks = []
    async for chunk in Runner.run_streamed(mock_agent, "Test"):
        chunks.append(chunk)
    
    assert len(chunks) > 0

@pytest.mark.asyncio
async def test_runner_sequential(mock_chater):
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

@pytest.mark.asyncio
async def test_runner_parallel(mock_chater):
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

@pytest.mark.asyncio
async def test_runner_auto_speak(mock_agent):
    result = await Runner.run(mock_agent, "Test", auto_speak=True)
    assert result is not None
