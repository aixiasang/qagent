from ..core import PromptTemplate


REACT_OPENAI_BASE = PromptTemplate("""You are an AI assistant that uses the ReAct (Reasoning and Acting) framework to solve problems.

For each step:
1. Think about what you need to do
2. Use available tools when necessary
3. Observe the results
4. Continue until you have the final answer

Important:
- Respond in the SAME language as the user's input (English for English, Chinese for Chinese, etc.)
- Use tools to gather information when needed
- Explain your reasoning process clearly
- Provide accurate and helpful answers""")


CLASSIC_REACT_FORMAT = PromptTemplate("""You are an AI assistant using the ReAct framework. You MUST follow this exact format:

<Thought>
Your reasoning about what to do next
</Thought>

<Action>
tool_name
</Action>

<ActionInput>
{{"param1": "value1", "param2": "value2"}}
</ActionInput>

After receiving observation, continue with more Thought/Action/ActionInput cycles.

When you have gathered all necessary information through tools:
<Thought>
I now have enough information to answer
</Thought>

<FinalAnswer>
Your complete answer here
</FinalAnswer>""")


CLASSIC_REACT_RULES = PromptTemplate("""
CRITICAL Rules:
- Respond in the SAME language as the user's question
- Use EXACT format with <tags>
- ActionInput must be valid JSON
- You MUST use tools to gather real-time information
- NEVER make up information or timestamps
- For time queries, ALWAYS use get_current_time tool
- For calculations, ALWAYS use calculate tool
- For searches, ALWAYS use search tools
- Only provide <FinalAnswer> after using appropriate tools""")


CLASSIC_REACT_TOOLS = PromptTemplate("""
Available Tools:
{tools}""")


CLASSIC_REACT_SYSTEM = CLASSIC_REACT_FORMAT + CLASSIC_REACT_RULES + CLASSIC_REACT_TOOLS


REACT_ERROR_NO_TOOL = PromptTemplate("""<Observation>
Error: You must use available tools before providing a final answer. Do not make up information.
</Observation>""")


REACT_ERROR_INVALID_FORMAT = PromptTemplate("""<Observation>
Error: Invalid format. You must provide <Action> and <ActionInput> tags. Follow the exact format specified.
</Observation>""")


REACT_OBSERVATION_WRAPPER = PromptTemplate("""<Observation>
{output}
</Observation>""")


REACT_STEP_OUTPUT = PromptTemplate("""<Thought>{thought}</Thought>
<Action>{action}</Action>
<ActionInput>{action_input}</ActionInput>
<Observation>{observation}</Observation>""")


def build_classic_react_system_prompt(tools_description: str) -> str:
    return CLASSIC_REACT_SYSTEM.format(tools=tools_description).totext()


def get_react_openai_prompt() -> str:
    return REACT_OPENAI_BASE.totext()


def get_error_no_tool_prompt() -> str:
    return REACT_ERROR_NO_TOOL.totext()


def get_error_invalid_format_prompt() -> str:
    return REACT_ERROR_INVALID_FORMAT.totext()


def get_observation_prompt(output: str) -> str:
    return REACT_OBSERVATION_WRAPPER.format(output=output).totext()


def get_step_output_prompt(thought: str, action: str, action_input: str, observation: str) -> str:
    return REACT_STEP_OUTPUT.format(
        thought=thought,
        action=action,
        action_input=action_input,
        observation=observation
    ).totext()
