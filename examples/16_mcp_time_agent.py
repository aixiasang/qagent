import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyagent import Agent, Memory, Chater, get_chater_cfg, ToolKit, Runner


async def main():
    toolkit = ToolKit()

    print("Connecting to MCP Time server via stdio...")
    print("-" * 50)

    await toolkit.add_mcp_stdio_server(
        name="time",
        command="py",
        args=["-m", "mcp_server_time"],
        prefix="time",
    )

    tools = toolkit.list_tools()
    print(f"Registered MCP tools: {tools}")

    for tool_name in tools:
        schema = toolkit.get_tool_schema(tool_name)
        desc = schema.get("description", "N/A")
        print(f"  - {tool_name}: {desc[:60]}...")

    print("-" * 50)

    agent = Agent(
        name="TimeAgent",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        tools=toolkit,
        system_prompt="You are a helpful assistant. Use MCP time tools to answer questions.",
    )

    print("Query: What is the current time in Asia/Shanghai?")
    response = await Runner.run(agent, "What is the current time in Asia/Shanghai?")
    print(f"Agent: {response.content}")

    await toolkit.disconnect_all_mcp_servers()
    print("\nMCP servers disconnected.")


if __name__ == "__main__":
    asyncio.run(main())
