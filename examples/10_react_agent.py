import asyncio
from qagent import ReActAgent, Memory, Chater, get_chater_cfg, ToolKit, Runner

async def search_database(query: str) -> str:
    database = {
        "python": "Python is a high-level programming language known for simplicity.",
        "java": "Java is a class-based, object-oriented programming language.",
        "javascript": "JavaScript is a scripting language for web pages."
    }
    return database.get(query.lower(), "No information found")

async def calculate_stats(numbers: list) -> dict:
    if not numbers:
        return {"error": "Empty list"}
    return {
        "mean": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers),
        "count": len(numbers)
    }

async def main():
    toolkit = ToolKit()
    toolkit.register(search_database)
    toolkit.register(calculate_stats)
    
    agent = ReActAgent(
        name="ReActBot",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        tools=toolkit,
        track_reasoning=True
    )
    
    result = await Runner.run(agent, "Search for information about Python and calculate stats for [10, 20, 30, 40, 50]")
    print(f"Result: {result.content}")
    
    if agent.tracker:
        trace = agent.tracker.get_full_trace()
        print(f"\nReAct Trace:")
        print(f"Iterations: {trace['total_iterations']}")
        print(f"Thoughts: {len(trace['thoughts'])}")
        print(f"Actions: {len(trace['actions'])}")

if __name__ == "__main__":
    asyncio.run(main())
