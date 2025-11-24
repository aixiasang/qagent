import pytest
from pyagent import Agent, Memory, ChatResponse


@pytest.fixture
def basic_agent(mock_chater):
    return Agent(
        name="TestAgent",
        chater=mock_chater,
        memory=Memory(),
        system_prompt="You are a test agent"
    )

@pytest.mark.asyncio
async def test_agent_creation(basic_agent):
    assert basic_agent.name == "TestAgent"
    assert basic_agent.system_prompt == "You are a test agent"
    assert basic_agent.agent_id is not None

@pytest.mark.asyncio
async def test_agent_reply(basic_agent):
    response = None
    async for resp in basic_agent.reply("Hello"):
        response = resp
    
    assert response is not None
    assert response.content == "Mock response"

@pytest.mark.asyncio
async def test_agent_memory(basic_agent):
    async for _ in basic_agent.reply("Test message"):
        pass
    
    assert len(basic_agent.memory) >= 2

@pytest.mark.asyncio
async def test_agent_observe(basic_agent):
    msg = ChatResponse(role="user", content="Observation test")
    basic_agent.observe(msg)
    
    assert len(basic_agent.memory) == 1

@pytest.mark.asyncio
async def test_agent_clear_memory(basic_agent):
    async for _ in basic_agent.reply("Test"):
        pass
    
    basic_agent.clear_memory()
    assert len(basic_agent.memory) == 0

@pytest.mark.asyncio
async def test_agent_hooks(basic_agent):
    hook_called = []
    
    @basic_agent.pre_reply
    def pre_hook(msg):
        hook_called.append("pre")
        return msg
    
    @basic_agent.post_reply
    def post_hook(response):
        hook_called.append("post")
        return response
    
    async for _ in basic_agent.reply("Test"):
        pass
    
    assert "pre" in hook_called
    assert "post" in hook_called

@pytest.mark.asyncio
async def test_agent_to_dict(basic_agent):
    agent_dict = basic_agent.to_dict()
    
    assert "agent_id" in agent_dict
    assert "name" in agent_dict
    assert agent_dict["name"] == "TestAgent"
    assert agent_dict["type"] == "Agent"
