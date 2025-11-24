import pytest
import asyncio
from pyagent import ChatResponse, ToolCall


class MockChater:
    """Mock Chater for testing without real API calls"""
    
    def __init__(self, response_content="Mock response", tool_calls=None):
        self.response_content = response_content
        self.tool_calls = tool_calls or []
        self.call_count = 0
    
    async def chat(self, messages, stream=False, tools=None, tool_choice=None):
        self.call_count += 1
        
        if stream:
            async def generate():
                yield ChatResponse(
                    role="assistant",
                    content=self.response_content,
                    tool_calls=self.tool_calls if self.tool_calls else None
                )
            return generate()
        
        return ChatResponse(
            role="assistant",
            content=self.response_content,
            tool_calls=self.tool_calls if self.tool_calls else None
        )


@pytest.fixture
def mock_chater():
    """Fixture providing a mock chater"""
    return MockChater()


@pytest.fixture
def mock_chater_with_tools():
    """Fixture providing a mock chater that returns tool calls"""
    tool_call = ToolCall(
        fn_id="call_123",
        fn_name="test_tool",
        fn_args='{"x": 5, "y": 3}'
    )
    return MockChater(response_content="Using tools", tool_calls=[tool_call])


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_api_key(monkeypatch):
    """Mock API keys to avoid real API calls"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test_key")
    monkeypatch.setenv("ZHIPUAI_API_KEY", "test_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
