import asyncio

from agent import AgenticMemoryAgent
from core import Memory, ChaterPool, EmbedderPool, get_chater_cfg, get_embedder_cfg


async def demo_basic_usage():
    print("="*70)
    print("Demo: Agentic Memory System - Basic Usage")
    print("="*70)
    
    agent = AgenticMemoryAgent(
        name="MemoryAssistant",
        chater=ChaterPool([
            get_chater_cfg("ali"),
            get_chater_cfg("zhipuai")
        ]),
        embedder=EmbedderPool([get_embedder_cfg("ali")]),
        memory=Memory(),
        enable_logging=True,
    )
    
    print("\n1. Adding memories with automatic analysis...")
    
    memory1 = await agent.add_memory_note(
        "Machine learning algorithms use neural networks to process complex datasets."
    )
    
    mem = agent.read_memory(memory1)
    print(f"\nMemory added:")
    print(f"  Content: {mem.content}")
    print(f"  Auto-generated keywords: {mem.keywords}")
    print(f"  Auto-generated context: {mem.context}")
    print(f"  Auto-generated tags: {mem.tags}")
    
    print("\n2. Adding more related memories...")
    
    await agent.add_memory_note(
        "Python is the most popular language for machine learning and data science."
    )
    
    await agent.add_memory_note(
        "Deep learning models require GPU acceleration for efficient training."
    )
    
    print("\n3. Searching for related memories...")
    
    results = await agent.search_memory("machine learning with Python", k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  {result['content']}")
        print(f"  Relevance: {result['score']:.4f}")
    
    print("\n4. Using memories in conversation...")
    
    question = "What do I know about machine learning?"
    print(f"\nQuestion: {question}\n")
    
    async for response in agent.reply_with_memory(question, k=3):
        print(f"Answer: {response.content}")
    
    stats = agent.get_memory_stats()
    print(f"\n5. Memory Statistics:")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Memory connections: {stats['total_links']}")
    print(f"  Unique tags: {stats['unique_tags']}")


async def demo_memory_evolution():
    print("\n\n" + "="*70)
    print("Demo: Memory Evolution and Automatic Linking")
    print("="*70)
    
    agent = AgenticMemoryAgent(
        name="EvoAgent",
        chater=ChaterPool([
            get_chater_cfg("ali"),
            get_chater_cfg("zhipuai")
        ]),
        embedder=EmbedderPool([get_embedder_cfg("ali")]),
        memory=Memory(),
        enable_logging=True,
        evo_threshold=2,
    )
    
    print("\nAdding memories that will trigger evolution...")
    
    contents = [
        "Python is a versatile programming language used in many domains.",
        "Web development in Python uses frameworks like Django and Flask.",
        "Data science and machine learning are major Python applications.",
        "Python's syntax is clean and easy to learn for beginners.",
    ]
    
    ids = []
    for content in contents:
        mid = await agent.add_memory_note(content)
        ids.append(mid)
        print(f"  Added: {content[:60]}...")
    
    print("\nMemories after evolution:")
    
    for mid in ids:
        mem = agent.read_memory(mid)
        if mem.links:
            print(f"\n  {mem.content[:50]}...")
            print(f"    Connected to {len(mem.links)} other memories")
            print(f"    Tags: {mem.tags[:3]}")
    
    stats = agent.get_memory_stats()
    print(f"\nEvolution Statistics:")
    print(f"  Evolution events: {stats['evolution_count']}")
    print(f"  Average connections: {stats['avg_links_per_memory']:.2f}")


async def main():
    await demo_basic_usage()
    await demo_memory_evolution()
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
