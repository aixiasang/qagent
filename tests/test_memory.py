import pytest
from qagent import Memory, ChatResponse

@pytest.fixture
def memory():
    return Memory(max_messages=10)

def test_memory_creation(memory):
    assert len(memory) == 0
    assert memory.max_messages == 10

def test_memory_add_message(memory):
    msg = ChatResponse(role="user", content="Hello")
    memory.add(msg)
    
    assert len(memory) == 1

def test_memory_add_multiple_messages(memory):
    messages = [
        ChatResponse(role="user", content="Message 1"),
        ChatResponse(role="assistant", content="Response 1"),
        ChatResponse(role="user", content="Message 2"),
    ]
    
    for msg in messages:
        memory.add(msg)
    
    assert len(memory) == 3

def test_memory_max_messages():
    memory = Memory(max_messages=3)
    
    for i in range(5):
        memory.add(ChatResponse(role="user", content=f"Message {i}"))
    
    assert len(memory) == 3

def test_memory_to_openai(memory):
    memory.add(ChatResponse(role="user", content="Hello"))
    memory.add(ChatResponse(role="assistant", content="Hi"))
    
    openai_msgs = memory.to_openai()
    
    assert len(openai_msgs) == 2
    assert openai_msgs[0]["role"] == "user"
    assert openai_msgs[0]["content"] == "Hello"

def test_memory_clear(memory):
    memory.add(ChatResponse(role="user", content="Test"))
    memory.clear()
    
    assert len(memory) == 0

def test_memory_get_messages(memory):
    msg1 = ChatResponse(role="user", content="First")
    msg2 = ChatResponse(role="assistant", content="Second")
    
    memory.add(msg1)
    memory.add(msg2)
    
    messages = memory.messages
    assert len(messages) == 2
    assert messages[0].content == "First"

def test_memory_unlimited():
    memory = Memory(max_messages=None)
    
    for i in range(100):
        memory.add(ChatResponse(role="user", content=f"Msg {i}"))
    
    assert len(memory) == 100
