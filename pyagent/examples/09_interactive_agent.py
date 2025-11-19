import asyncio
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (
    Agent, Memory, get_chater_cfg, ChaterPool, ToolKit,
    FileOperations, DirectoryOperations, SearchOperations,
    trace, export_traces, get_all_spans, clear_traces
)


async def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def calculate(expression: str) -> float:
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


async def ainput(prompt: str = "") -> str:
    return await asyncio.to_thread(input, prompt)


async def main():
    tools = ToolKit()
    tools.register(get_current_time, "get_current_time")
    tools.register(calculate, "calculate")
    tools.register(FileOperations.read_file, "read_file")
    tools.register(FileOperations.write_file, "write_file")
    tools.register(DirectoryOperations.list_directory, "list_directory")
    tools.register(SearchOperations.grep_in_file, "grep_in_file")

    agent = Agent(
        name="Assistant",
        chater=ChaterPool([
            get_chater_cfg("ali"),
            get_chater_cfg("siliconflow"),
        ]),
        memory=Memory(max_messages=50),
        tools=tools,
        system_prompt="You are a helpful AI assistant with access to various tools.",
        max_iterations=5,
        tool_timeout=30,
        enable_logging=False
    )

    @agent.post_reply
    def response_formatter(response):
        if response.content and not response.tool_call and not response.tool_calls:
            response.content = f"💬 {response.content}"
        return response

    print("=" * 60)
    print("Interactive Agent Demo (with Tracing)")
    print("=" * 60)
    print(f"\nAgent: {repr(agent)}")
    print(f"Tools: {', '.join(tools._tools.keys())}")
    print(f"\nCommands:")
    print("  - 'quit' / 'exit'  : Exit and export trace")
    print("\n" + "=" * 60 + "\n")

    conversation_count = 0
    session_start = datetime.now()
    
    with trace("Interactive Session", metadata={
        "agent_name": agent.name,
        "start_time": session_start.isoformat(),
        "tools": list(tools._tools.keys())
    }):
        while True:
            try:
                user_input = await ainput("You: ")

                if not user_input.strip():
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    break

                conversation_count += 1
                print("Agent: ", end="", flush=True)
                async for _ in agent.reply(user_input, stream=True, auto_speak=True):
                    pass
                print("\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted.")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.json"
    export_traces(filename)
    
    spans = get_all_spans()
    agent_spans = [s for s in spans if s['span_data']['type'] == 'agent']
    gen_spans = [s for s in spans if s['span_data']['type'] == 'generation']
    tool_spans = [s for s in spans if s['span_data']['type'] == 'tool']
    
    total_tokens = sum(
        s['span_data'].get('usage', {}).get('input_tokens', 0) + 
        s['span_data'].get('usage', {}).get('output_tokens', 0)
        for s in gen_spans
    )
    
    session_duration = (datetime.now() - session_start).total_seconds()
    
    print(f"\n{'='*60}")
    print(f"Session Summary")
    print(f"{'='*60}")
    print(f"Duration: {session_duration:.1f}s")
    print(f"Conversations: {conversation_count}")
    print(f"Total spans: {len(spans)}")
    print(f"  - Agent spans: {len(agent_spans)}")
    print(f"  - Generations: {len(gen_spans)}")
    print(f"  - Tool calls: {len(tool_spans)}")
    print(f"Total tokens: {total_tokens}")
    print(f"\nTrace exported to: {filename}")
    print(f"{'='*60}")
    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
