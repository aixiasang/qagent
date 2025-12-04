import asyncio
from qagent import Agent, Memory, Chater, get_chater_cfg, Runner, ConsoleSpeaker, make_stream_callback


async def main():
    agent = Agent(
        name="Storyteller",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a creative storyteller."
    )

    print("Streaming response:")
    speaker = ConsoleSpeaker()
    response = await Runner.run(
        agent,
        "Tell me a short story about a robot",
        stream=True,
        on_stream=make_stream_callback(speaker)
    )
    print(f"\nFinal: {response.content[:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
