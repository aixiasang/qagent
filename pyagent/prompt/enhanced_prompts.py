from ..core import PromptTemplate


ENHANCED_QUALITY_EVALUATOR = PromptTemplate(
    """You are a quality evaluator. Evaluate the following answer strictly.

Question: {question}
Answer: {answer}

Evaluate on these dimensions (score 0-10):
1. Correctness: Are facts accurate? Is logic sound?
2. Completeness: Are all aspects addressed? Any missing info?
3. Clarity: Is it clear and well-structured?

Output ONLY valid JSON:
{{
    "overall_score": <float 0-10>,
    "correctness": {{"score": <int 0-10>, "issue": "<description or null>"}},
    "completeness": {{"score": <int 0-10>, "issue": "<description or null>"}},
    "clarity": {{"score": <int 0-10>, "issue": "<description or null>"}},
    "needs_improvement": <boolean>,
    "main_issues": ["<issue1>", "<issue2>"]
}}

Be strict. If score < 7, set needs_improvement to true."""
)


ENHANCED_REFLECTION = PromptTemplate(
    """You are in reflection mode. Analyze your previous answer deeply.

Question: {question}
Previous Answer: {previous_answer}
Identified Issues: {issues}

Reflect in three layers:

1. WHAT went wrong?
   - List specific problems

2. WHY did it happen?
   - Root cause analysis

3. HOW to improve?
   - Concrete improvement actions

Output ONLY valid JSON:
{{
    "problems": ["<problem1>", "<problem2>"],
    "root_causes": ["<cause1>", "<cause2>"],
    "improvements": ["<action1>", "<action2>"],
    "focus_areas": ["<area1>", "<area2>"]
}}"""
)


ENHANCED_EXECUTION_WITH_REFLECTION = PromptTemplate(
    """You are solving a task with reflection feedback.

Original Question: {question}

Previous Attempt Issues:
{reflection_feedback}

Instructions:
- Address the identified issues
- Use tools when needed to get accurate information
- Be thorough and precise
- Respond in the SAME language as the question

Available Tools:
{tools}"""
)


ENHANCED_SELF_CONSISTENCY_PROMPT = PromptTemplate(
    """You are solving: {question}

Generate {num_paths} different reasoning paths.
Each path should use different approaches but reach a valid answer.

Output format:
Path 1: [reasoning and answer]
Path 2: [reasoning and answer]
...

Respond in the SAME language as the question."""
)


ENHANCED_VALIDATION = PromptTemplate(
    """You are a strict validator. Verify this answer.

Question: {question}
Answer: {answer}
Execution Steps: {steps}

Check:
1. Did it use appropriate tools?
2. Are tool results correctly interpreted?
3. Is the final answer accurate?
4. Is anything missing?

Output ONLY valid JSON:
{{
    "is_valid": <boolean>,
    "confidence": <float 0-1>,
    "issues_found": ["<issue1>", "<issue2>"],
    "used_tools": <boolean>,
    "tool_usage_appropriate": <boolean>
}}"""
)


def get_quality_evaluation_prompt(question: str, answer: str) -> str:
    return ENHANCED_QUALITY_EVALUATOR.format(question=question, answer=answer).totext()


def get_reflection_prompt(question: str, previous_answer: str, issues: str) -> str:
    return ENHANCED_REFLECTION.format(
        question=question, previous_answer=previous_answer, issues=issues
    ).totext()


def get_execution_with_reflection_prompt(
    question: str, reflection_feedback: str, tools: str
) -> str:
    return ENHANCED_EXECUTION_WITH_REFLECTION.format(
        question=question, reflection_feedback=reflection_feedback, tools=tools
    ).totext()


def get_validation_prompt(question: str, answer: str, steps: str) -> str:
    return ENHANCED_VALIDATION.format(question=question, answer=answer, steps=steps).totext()
