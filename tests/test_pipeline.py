import pytest
from qagent import (
    Agent, Memory, ChatResponse,
    sequential_pipeline, parallel_pipeline,
    conditional_pipeline, loop_pipeline
)


@pytest.fixture
def agents(mock_chater):
    return [
        Agent(
            name=f"Agent{i}",
            chater=mock_chater,
            memory=Memory(),
            system_prompt=f"Agent {i}"
        )
        for i in range(3)
    ]

@pytest.mark.asyncio
async def test_sequential_pipeline(agents):
    msg = ChatResponse(role="user", content="Test")
    result = await sequential_pipeline(agents, msg)
    
    assert result is not None

@pytest.mark.asyncio
async def test_parallel_pipeline(agents):
    msg = ChatResponse(role="user", content="Test")
    results = await parallel_pipeline(agents, msg)
    
    assert len(results) == 3

@pytest.mark.asyncio
async def test_conditional_pipeline(agents):
    def condition(msg):
        return True
    
    msg = ChatResponse(role="user", content="Test")
    result = await conditional_pipeline(condition, agents[0], agents[1], msg)
    
    assert result is not None

@pytest.mark.asyncio
async def test_loop_pipeline(agents):
    msg = ChatResponse(role="user", content="Test")
    result = await loop_pipeline(agents[:2], msg, max_iterations=2)
    
    assert result is not None

@pytest.mark.asyncio
async def test_loop_with_stop_condition(agents):
    iteration_count = []
    
    def stop_condition(msg):
        iteration_count.append(1)
        return len(iteration_count) >= 2
    
    msg = ChatResponse(role="user", content="Test")
    result = await loop_pipeline(
        agents[:1],
        msg,
        max_iterations=10,
        stop_condition=stop_condition
    )
    
    assert result is not None
    assert len(iteration_count) == 2

@pytest.mark.asyncio
async def test_sequential_with_callable():
    async def processor(msg):
        return ChatResponse(role="assistant", content="Processed")
    
    msg = ChatResponse(role="user", content="Test")
    result = await sequential_pipeline([processor], msg)
    
    assert result.content == "Processed"

@pytest.mark.asyncio
async def test_empty_pipeline():
    msg = ChatResponse(role="user", content="Test")
    
    with pytest.raises(ValueError):
        await sequential_pipeline([], msg)
