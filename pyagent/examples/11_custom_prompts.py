import asyncio
from datetime import datetime
from agent import ReActAgent, ClassicReActAgent
from core import Memory, ChaterPool, get_chater_cfg, ToolKit, PromptTemplate
from prompt import build_classic_react_system_prompt


async def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def calculate(expression: str) -> float:
    try:
        return float(eval(expression))
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


async def demo_custom_react_prompt():
    print("="*70)
    print("Demo: Custom ReAct Prompt using PromptTemplate")
    print("="*70)
    
    custom_prompt_template = PromptTemplate("""You are a {role} that helps users with {task}.

You should:
1. {instruction_1}
2. {instruction_2}
3. {instruction_3}

Language Rule:
- Respond in {language} language

Personality:
- Be {personality}
- Use {tone} tone
""")
    
    custom_prompt = custom_prompt_template.format(
        role="helpful AI assistant",
        task="time and calculation queries",
        instruction_1="Think carefully about the user's request",
        instruction_2="Use available tools when needed",
        instruction_3="Provide clear and accurate answers",
        language="the SAME as user's input",
        personality="friendly and professional",
        tone="casual yet informative"
    )
    
    print("\nCustom Prompt:")
    print("-" * 70)
    print(custom_prompt.totext())
    print("-" * 70)
    
    tools = ToolKit()
    tools.register(get_current_time, "get_current_time")
    tools.register(calculate, "calculate")
    
    agent = ReActAgent(
        name="CustomAgent",
        chater=ChaterPool([get_chater_cfg("siliconflow")]),
        memory=Memory(),
        tools=tools,
        system_prompt=custom_prompt.totext(),
        max_iterations=3
    )
    
    print("\nTest Query: 现在几点了？")
    print("-" * 70)
    
    async for response in agent.reply("现在几点了？", stream=False):
        print(f"Response: {response.content}")
    
    print("\n" + "="*70)


async def demo_composable_prompts():
    print("\n" + "="*70)
    print("Demo: Composable Prompts with PromptTemplate")
    print("="*70)
    
    base_instruction = PromptTemplate("""You are a specialized AI assistant.""")
    
    react_instruction = PromptTemplate("""
Use ReAct framework:
- Think before acting
- Use tools when needed
- Observe results""")
    
    language_rule = PromptTemplate("""
Language: Always respond in {language}""")
    
    combined = base_instruction + react_instruction + language_rule
    
    final_prompt = combined.format(language="user's input language")
    
    print("\nComposed Prompt:")
    print("-" * 70)
    print(final_prompt.totext())
    print("-" * 70)
    
    print("\n" + "="*70)


async def demo_classic_prompt_customization():
    print("\n" + "="*70)
    print("Demo: Classic ReAct with Custom Tool Description")
    print("="*70)
    
    tools = ToolKit()
    tools.register(get_current_time, "get_current_time")
    tools.register(calculate, "calculate")
    
    custom_tools_desc = """
- get_current_time: Returns current date and time in format YYYY-MM-DD HH:MM:SS
  Use this when user asks about time, date, or current moment
  Parameters: None required
  
- calculate: Evaluates mathematical expressions
  Use this for any calculation requests
  Parameters: {{"expression": "math expression as string"}}
  Example: {{"expression": "123 + 456"}}
"""
    
    custom_prompt = build_classic_react_system_prompt(custom_tools_desc)
    
    print("\nCustom Classic ReAct Prompt (excerpt):")
    print("-" * 70)
    print(custom_prompt[:500] + "...")
    print("-" * 70)
    
    agent = ClassicReActAgent(
        name="CustomClassicAgent",
        chater=ChaterPool([get_chater_cfg("siliconflow")]),
        memory=Memory(),
        tools=tools,
        system_prompt=custom_prompt,
        max_iterations=3
    )
    
    print("\nTest Query: Calculate 100 * 50")
    print("-" * 70)
    
    async for response in agent.reply("Calculate 100 * 50"):
        if len(response.content) > 200:
            print(f"Response: {response.content[:200]}...")
        else:
            print(f"Response: {response.content}")
    
    print("\n" + "="*70)


async def demo_dynamic_prompt_building():
    print("\n" + "="*70)
    print("Demo: Dynamic Prompt Building")
    print("="*70)
    
    prompt_parts = []
    
    prompt_parts.append(PromptTemplate("You are {role}."))
    
    prompt_parts.append(PromptTemplate("""
Skills:
{skills}"""))
    
    prompt_parts.append(PromptTemplate("""
Rules:
{rules}"""))
    
    full_prompt = prompt_parts[0]
    for part in prompt_parts[1:]:
        full_prompt = full_prompt + part
    
    final = full_prompt.format(
        role="a problem-solving assistant",
        skills="- Time checking\n- Math calculations\n- Logical reasoning",
        rules="- Be accurate\n- Use tools\n- Respond in user's language"
    )
    
    print("\nDynamically Built Prompt:")
    print("-" * 70)
    print(final.totext())
    print("-" * 70)
    
    print("\n" + "="*70)


async def demo_prompt_template_operators():
    print("\n" + "="*70)
    print("Demo: PromptTemplate Operators")
    print("="*70)
    
    part1 = PromptTemplate("Hello {name}")
    part2 = PromptTemplate("You are {age} years old")
    
    print("\n1. Addition (+):")
    combined = part1 + part2
    result = combined.format(name="Alice", age=25)
    print(f"   Result: {result.totext()}")
    
    print("\n2. Pipe (|):")
    piped = part1 | part2
    result_piped = piped.format(name="Bob", age=30)
    print(f"   Result: {result_piped.totext()}")
    
    print("\n3. Left Shift (<<) for positional args:")
    template = PromptTemplate("User {0} is {1} years old from {2}")
    filled = template << ["Charlie", 35, "Beijing"]
    print(f"   Result: {filled.totext()}")
    
    print("\n4. Format method:")
    template2 = PromptTemplate("City: {city}, Country: {country}")
    formatted = template2.format(city="Shanghai", country="China")
    print(f"   Result: {formatted.totext()}")
    
    print("\n" + "="*70)


async def main():
    print("\n" + "="*70)
    print("Custom Prompts with PromptTemplate System")
    print("="*70)
    
    await demo_custom_react_prompt()
    await demo_composable_prompts()
    await demo_classic_prompt_customization()
    await demo_dynamic_prompt_building()
    await demo_prompt_template_operators()
    
    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
