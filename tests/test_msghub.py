import pytest
from qagent import Agent, Memory, MsgHub, ChatResponse, msghub


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

def test_msghub_creation(agents):
    hub = MsgHub(agents)
    
    assert len(hub.participants) == 3

def test_msghub_context_manager(agents):
    with msghub(agents):
        for agent in agents:
            assert agent._audience is not None
    
    for agent in agents:
        assert agent._audience is None

def test_msghub_broadcast(agents):
    hub = MsgHub(agents)
    
    with hub:
        msg = ChatResponse(role="user", content="Broadcast test")
        hub.broadcast(msg)
        
        for agent in agents:
            assert len(agent.memory) == 1

def test_msghub_add_agent(agents):
    hub = MsgHub(agents[:2])
    
    assert len(hub.participants) == 2
    
    hub.add(agents[2])
    assert len(hub.participants) == 3

def test_msghub_remove_agent(agents):
    hub = MsgHub(agents)
    
    hub.remove(agents[0])
    assert len(hub.participants) == 2

def test_msghub_with_announcement(agents):
    announcement = ChatResponse(role="system", content="Welcome")
    
    with msghub(agents, announcement=announcement):
        for agent in agents:
            assert len(agent.memory) == 1

def test_msghub_audience_exclusion(agents):
    with msghub(agents):
        for i, agent in enumerate(agents):
            assert len(agent._audience) == len(agents) - 1
            assert agent not in agent._audience

def test_msghub_reset_audience(agents, mock_chater):
    hub = MsgHub(agents)
    
    with hub:
        initial_count = len(agents[0]._audience)
        hub.add(Agent(
            name="NewAgent",
            chater=mock_chater,
            memory=Memory(),
            system_prompt="New"
        ))
        
        assert len(agents[0]._audience) > initial_count
