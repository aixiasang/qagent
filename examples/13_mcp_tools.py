import asyncio
from pyagent import Agent, Memory, Chater, get_chater_cfg, ToolKit, Runner


async def main():
    toolkit = ToolKit()

    await toolkit.add_mcp_stdio_server(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        prefix="fs",
    )

    print(f"Registered tools: {toolkit.list_tools()}")

    agent = Agent(
        name="FileAgent",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        tools=toolkit,
        system_prompt="You are a file system assistant with MCP tools.",
    )

    result = await Runner.run(agent, "List files in the directory")
    print(f"Result: {result.content}")

    await toolkit.disconnect_all_mcp_servers()


if __name__ == "__main__":
    asyncio.run(main())
