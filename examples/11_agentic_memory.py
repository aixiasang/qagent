import asyncio
from qagent import AgenticMemoryAgent, Memory, Chater, Embedder, get_chater_cfg, get_embedder_cfg, Runner


async def main():
    agent = AgenticMemoryAgent(
        name="MemoryBot",
        chater=Chater(get_chater_cfg("ali")),
        embedder=Embedder(get_embedder_cfg("ali")),
        memory=Memory(),
        system_prompt="You are an assistant with self-evolving memory.",
        evo_threshold=5
    )

    await agent.add_memory_note("The user prefers Python over Java")
    await agent.add_memory_note("The user is working on a machine learning project")
    await agent.add_memory_note("The user likes to use async/await patterns")

    result = await agent.reply_with_memory("What do you know about my preferences?")
    print(f"Response: {result.content}")

    print(f"\nTotal memories stored: {len(agent.agentic_memories)}")
    print(f"Memory stats: {agent.get_memory_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
