import asyncio

from core import Agent, ChaterPool, Memory, get_chater_cfg


async def main():
    agent = Agent(
        name="HookDemo",
        chater=ChaterPool([
            get_chater_cfg("siliconflow"),
            get_chater_cfg("zhipuai")
        ]),
        memory=Memory(),
        system_prompt="You are a helpful assistant."
    )

    call_log = []

    @agent.pre_reply
    def pre_reply_hook(message):
        call_log.append(f"[Pre-Reply] Received: {message[:30]}...")
        return message

    @agent.post_reply
    def post_reply_hook(response):
        call_log.append(f"[Post-Reply] Generated: {response.content[:30]}...")
        if response.content:
            response.content = f"✨ {response.content}"
        return response

    @agent.pre_observe
    def pre_observe_hook(msg):
        call_log.append(f"[Pre-Observe] Observing message")
        return msg

    @agent.post_observe
    def post_observe_hook(msg):
        call_log.append(f"[Post-Observe] Memory size: {len(agent.memory)}")

    print("Hooks registered:")
    print(f"- Pre-reply: {list(agent._pre_reply_hooks.keys())}")
    print(f"- Post-reply: {list(agent._post_reply_hooks.keys())}")
    print(f"- Pre-observe: {list(agent._pre_observe_hooks.keys())}")
    print(f"- Post-observe: {list(agent._post_observe_hooks.keys())}\n")

    message = "Hello! How are you?"
    print(f"User: {message}\n")

    async for response in agent.reply(message, stream=True):
        agent.speak(response, stream=True)
    print()

    print("Hook execution log:")
    for log in call_log:
        print(f"  {log}")


if __name__ == "__main__":
    asyncio.run(main())
