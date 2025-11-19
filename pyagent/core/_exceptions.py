from typing import Any, Dict, Optional


class AgentError(Exception):
    def __init__(self, error_type: str, message: str, **kwargs: Any):
        self.error_type = error_type
        self.message = message
        self.extra = kwargs
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {"error": self.error_type, "message": self.message, **self.extra}


class ToolError(AgentError):
    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs: Any):
        super().__init__("ToolError", message, tool_name=tool_name, **kwargs)


class ToolNotFoundError(ToolError):
    def __init__(self, tool_name: str, **kwargs: Any):
        super().__init__(f"Tool '{tool_name}' not found", tool_name=tool_name, **kwargs)


class InvalidArgumentsError(ToolError):
    def __init__(
        self, message: str, tool_name: str, provided_args: list, **kwargs: Any
    ):
        super().__init__(
            message, tool_name=tool_name, provided_args=provided_args, **kwargs
        )


class MemoryError(AgentError):
    def __init__(self, message: str, **kwargs: Any):
        super().__init__("MemoryError", message, **kwargs)


class ModelError(AgentError):
    def __init__(self, message: str, **kwargs: Any):
        super().__init__("ModelError", message, **kwargs)


class HookError(AgentError):
    def __init__(self, message: str, hook_name: Optional[str] = None, **kwargs: Any):
        super().__init__("HookError", message, hook_name=hook_name, **kwargs)
