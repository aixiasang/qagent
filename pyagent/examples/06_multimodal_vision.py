import asyncio
import os
from core import Agent, Memory, ChatResponse, MultimodalContent, Chater, ChaterCfg, ClientCfg, ChatCfg, image_to_base64


async def main():
    vision_cfg = ChaterCfg(
        client_cfg=ClientCfg(
            api_key=os.getenv("ali_api_key"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        chat_cfg=ChatCfg(model="qwen3-vl-plus")
    )
    
    agent = Agent(
        name="VisionAgent",
        chater=Chater(vision_cfg),
        memory=Memory(),
        system_prompt="You are a vision AI that describes images in detail."
    )

    image_path = os.path.join(os.path.dirname(__file__), "../data/ikun1.png")
    image_base64 = image_to_base64(image_path)
    
    content = MultimodalContent()
    content.add_text("Please describe the image.")
    content.add_image(base64=image_base64)

    user_msg = ChatResponse(
        role="user",
        content=content
    )

    agent.observe(user_msg)

    print("Vision Analysis Demo\n")
    print("=" * 60)
    print(f"Image: {image_path}\n")

    async for response in agent.reply("", stream=False):
        agent.speak(response)
    print()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
