import pytest
from qagent import ToolKit

async def sample_tool(x: int, y: int) -> int:
    return x + y

async def another_tool(text: str) -> str:
    return text.upper()

@pytest.fixture
def toolkit():
    return ToolKit()

def test_toolkit_creation(toolkit):
    assert len(toolkit._tools) == 0
    assert len(toolkit._schemas) == 0

def test_toolkit_register(toolkit):
    toolkit.register(sample_tool)
    
    assert "sample_tool" in toolkit._tools
    assert "sample_tool" in toolkit._schemas

def test_toolkit_register_with_name(toolkit):
    toolkit.register(sample_tool, name="add_numbers")
    
    assert "add_numbers" in toolkit._tools

def test_toolkit_register_all(toolkit):
    toolkit.register_all(sample_tool, another_tool)
    
    assert len(toolkit._tools) == 2
    assert "sample_tool" in toolkit._tools
    assert "another_tool" in toolkit._tools

def test_toolkit_decorator(toolkit):
    @toolkit.tool
    async def decorated_tool(value: float) -> float:
        return value * 2
    
    assert "decorated_tool" in toolkit._tools

@pytest.mark.asyncio
async def test_toolkit_execute(toolkit):
    toolkit.register(sample_tool)
    
    result = await toolkit.execute("sample_tool", x=5, y=3)
    assert result == 8

@pytest.mark.asyncio
async def test_toolkit_execute_not_found(toolkit):
    with pytest.raises(Exception):
        await toolkit.execute("nonexistent_tool")

def test_toolkit_to_openai_tools(toolkit):
    toolkit.register(sample_tool)
    
    openai_tools = toolkit.to_openai_tools()
    
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert "function" in openai_tools[0]

def test_toolkit_list_tools(toolkit):
    toolkit.register(sample_tool)
    toolkit.register(another_tool)
    
    tools = list(toolkit._tools.keys())
    assert len(tools) == 2
    assert "sample_tool" in tools
    assert "another_tool" in tools

@pytest.mark.asyncio
async def test_toolkit_with_string_return(toolkit):
    async def string_tool(greeting: str) -> str:
        return f"Hello, {greeting}"
    
    toolkit.register(string_tool)
    result = await toolkit.execute("string_tool", greeting="World")
    
    assert result == "Hello, World"
