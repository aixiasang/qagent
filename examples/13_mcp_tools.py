import asyncio
from pyagent import Agent, Memory, Chater, get_chater_cfg, ToolKit, Runner, MCPServerConfig

async def main():
    toolkit = ToolKit()
    
    mcp_config = MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        transport="stdio"
    )
    
    await toolkit.add_mcp_server(mcp_config)
    
    agent = Agent(
        name="FileAgent",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        tools=toolkit,
        system_prompt="You are a file system assistant with MCP tools."
    )
    
    result = await Runner.run(agent, "List files in the directory")
    print(f"Result: {result.content}")
    
    await toolkit.disconnect_all_mcp()

if __name__ == "__main__":
    asyncio.run(main())
