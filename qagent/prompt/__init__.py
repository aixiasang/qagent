from .react_prompts import (
    REACT_OPENAI_BASE,
    CLASSIC_REACT_FORMAT,
    CLASSIC_REACT_RULES,
    CLASSIC_REACT_TOOLS,
    CLASSIC_REACT_SYSTEM,
    REACT_ERROR_NO_TOOL,
    REACT_ERROR_INVALID_FORMAT,
    REACT_OBSERVATION_WRAPPER,
    REACT_STEP_OUTPUT,
    build_classic_react_system_prompt,
    get_react_openai_prompt,
    get_error_no_tool_prompt,
    get_error_invalid_format_prompt,
    get_observation_prompt,
    get_step_output_prompt,
)

from .enhanced_prompts import (
    ENHANCED_QUALITY_EVALUATOR,
    ENHANCED_REFLECTION,
    ENHANCED_EXECUTION_WITH_REFLECTION,
    ENHANCED_SELF_CONSISTENCY_PROMPT,
    ENHANCED_VALIDATION,
    get_quality_evaluation_prompt,
    get_reflection_prompt,
    get_execution_with_reflection_prompt,
    get_validation_prompt,
)

from .advanced_prompts import (
    PLAN_REACT_SYSTEM,
    REFLECTION_SYSTEM,
    SELF_CONSISTENCY_SYSTEM,
    build_plan_react_prompt,
    build_reflection_prompt,
    build_self_consistency_prompt,
)

from .memory_prompts import (
    AGENTIC_MEMORY_ANALYZE,
    AGENTIC_MEMORY_EVOLUTION,
    get_agentic_memory_analyze_prompt,
    get_agentic_memory_evolution_prompt,
)


REACT_PROMPTS = {
    "openai_base": REACT_OPENAI_BASE,
    "classic_format": CLASSIC_REACT_FORMAT,
    "classic_rules": CLASSIC_REACT_RULES,
    "classic_tools": CLASSIC_REACT_TOOLS,
    "classic_system": CLASSIC_REACT_SYSTEM,
    "error_no_tool": REACT_ERROR_NO_TOOL,
    "error_invalid_format": REACT_ERROR_INVALID_FORMAT,
    "observation_wrapper": REACT_OBSERVATION_WRAPPER,
    "step_output": REACT_STEP_OUTPUT,
}


ENHANCED_PROMPTS = {
    "quality_evaluator": ENHANCED_QUALITY_EVALUATOR,
    "reflection": ENHANCED_REFLECTION,
    "execution_with_reflection": ENHANCED_EXECUTION_WITH_REFLECTION,
    "self_consistency": ENHANCED_SELF_CONSISTENCY_PROMPT,
    "validation": ENHANCED_VALIDATION,
}


ADVANCED_PROMPTS = {
    "plan_react_system": PLAN_REACT_SYSTEM,
    "reflection_system": REFLECTION_SYSTEM,
    "self_consistency_system": SELF_CONSISTENCY_SYSTEM,
}


MEMORY_PROMPTS = {
    "analyze": AGENTIC_MEMORY_ANALYZE,
    "evolution": AGENTIC_MEMORY_EVOLUTION,
}


ALL_PROMPTS = {
    "react": REACT_PROMPTS,
    "enhanced": ENHANCED_PROMPTS,
    "advanced": ADVANCED_PROMPTS,
    "memory": MEMORY_PROMPTS,
}


__all__ = [
    "REACT_OPENAI_BASE",
    "CLASSIC_REACT_FORMAT",
    "CLASSIC_REACT_RULES",
    "CLASSIC_REACT_TOOLS",
    "CLASSIC_REACT_SYSTEM",
    "REACT_ERROR_NO_TOOL",
    "REACT_ERROR_INVALID_FORMAT",
    "REACT_OBSERVATION_WRAPPER",
    "REACT_STEP_OUTPUT",
    "build_classic_react_system_prompt",
    "get_react_openai_prompt",
    "get_error_no_tool_prompt",
    "get_error_invalid_format_prompt",
    "get_observation_prompt",
    "get_step_output_prompt",
    "ENHANCED_QUALITY_EVALUATOR",
    "ENHANCED_REFLECTION",
    "ENHANCED_EXECUTION_WITH_REFLECTION",
    "ENHANCED_SELF_CONSISTENCY_PROMPT",
    "ENHANCED_VALIDATION",
    "get_quality_evaluation_prompt",
    "get_reflection_prompt",
    "get_execution_with_reflection_prompt",
    "get_validation_prompt",
    "PLAN_REACT_SYSTEM",
    "REFLECTION_SYSTEM",
    "SELF_CONSISTENCY_SYSTEM",
    "build_plan_react_prompt",
    "build_reflection_prompt",
    "build_self_consistency_prompt",
    "AGENTIC_MEMORY_ANALYZE",
    "AGENTIC_MEMORY_EVOLUTION",
    "get_agentic_memory_analyze_prompt",
    "get_agentic_memory_evolution_prompt",
    "REACT_PROMPTS",
    "ENHANCED_PROMPTS",
    "ADVANCED_PROMPTS",
    "MEMORY_PROMPTS",
    "ALL_PROMPTS",
]
