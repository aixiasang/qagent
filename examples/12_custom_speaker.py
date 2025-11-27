import asyncio
from qagent import Agent, Memory, Chater, get_chater_cfg, Runner, Speaker, ChatResponse

class CustomSpeaker(Speaker):
    def speak_complete(self, response: ChatResponse, agent_name: str):
        print(f"\n{'='*50}")
        print(f"[{agent_name}] says:")
        print(f"{response.content}")
        print(f"{'='*50}\n")
    
    def speak_chunk(self, chunk: ChatResponse):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    
    def speak_stream_start(self, agent_name: str):
        print(f"\n{'='*50}")
        print(f"[{agent_name}] is speaking...")
        print(f"{'='*50}")
    
    def speak_stream_end(self):
        print(f"\n{'='*50}\n")

async def main():
    agent = Agent(
        name="VerboseAgent",
        chater=Chater(get_chater_cfg("ali")),
        memory=Memory(),
        system_prompt="You are a helpful assistant.",
        speaker=CustomSpeaker()
    )
    
    await Runner.run(agent, "Tell me a fun fact", auto_speak=True)

if __name__ == "__main__":
    asyncio.run(main())
