import inspect
import asyncio
import re
import json
import os
from typing import (
    Callable,
    Dict,
    List,
    Any,
    Optional,
    Set,
    Tuple,
    Union,
    get_type_hints,
    get_origin,
    get_args,
)
from dataclasses import dataclass
from ._exceptions import ToolError, ToolNotFoundError, InvalidArgumentsError
from ._trace import tool_span, custom_span, get_current_trace, SpanError


def format_mcp_result(content: list) -> list:
    import mcp.types

    contents = []
    for item in content:
        if isinstance(item, mcp.types.TextContent):
            contents.append({"type": "text", "text": item.text})
        elif isinstance(item, mcp.types.ImageContent):
            contents.append(
                {"type": "image/base64", "media_type": item.mimeType, "data": item.data}
            )
        elif isinstance(item, mcp.types.AudioContent):
            contents.append(
                {"type": "audio/base64", "media_type": item.mimeType, "data": item.data}
            )
        else:
            contents.append({"type": "unknown", "data": str(item)})
    return contents


TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    tuple: "array",
    set: "array",
    List: "array",
    Dict: "object",
    Set: "array",
    Tuple: "array",
    type(None): "null",
    None: "null",
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "tuple": "array",
    "set": "array",
    "List": "array",
    "Dict": "object",
    "Set": "array",
    "Tuple": "array",
    "None": "null",
}

JSON_TO_PYTHON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


@dataclass
class MCPServerConfig:
    name: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    prefix: Optional[str] = None
    transport: str = "stdio"
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class MCPClient:
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session = None
        self.tools_cache: Dict[str, Dict] = {}
        self._connected = False
        self._context = None

    async def connect(self):
        if self.config.transport == "stdio":
            await self._connect_stdio()
        elif self.config.transport == "sse":
            await self._connect_sse()
        else:
            raise ValueError(f"Unsupported transport: {self.config.transport}")

    async def _connect_stdio(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=self.config.command, args=self.config.args, env=self.config.env
        )

        self._context = stdio_client(server_params)
        read, write = await self._context.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()
        self._connected = True

    async def _connect_sse(self):
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        self._context = sse_client(self.config.url, headers=self.config.headers)
        read, write = await self._context.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()
        self._connected = True

    async def disconnect(self):
        if not self._connected:
            return

        errors = []

        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                errors.append(f"Session close error: {type(e).__name__}: {e}")
            finally:
                self.session = None

        if self._context:
            try:
                await self._context.__aexit__(None, None, None)
            except Exception as e:
                errors.append(f"Context close error: {type(e).__name__}: {e}")
            finally:
                self._context = None

        self.tools_cache.clear()
        self._connected = False

        if errors:
            import logging
            logging.warning(f"MCP disconnect warnings for {self.config.name}: {'; '.join(errors)}")

    async def list_tools(self) -> List[Dict]:
        result = await self.session.list_tools()
        tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema,
            }
            for tool in result.tools
        ]

        for tool in tools:
            self.tools_cache[tool["name"]] = tool
        return tools

    async def call_tool(self, name: str, arguments: Dict) -> Any:
        result = await self.session.call_tool(name, arguments)
        return result.content


class ToolKit:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict] = {}
        self._mcp_clients: Dict[str, "MCPClient"] = {}
        self._mcp_configs: Dict[str, "MCPServerConfig"] = {}

    def tool(self, func: Callable) -> Callable:
        self.register(func)
        return func

    def register(self, func: Callable, name: Optional[str] = None):
        func_name = name or func.__name__
        self._tools[func_name] = func
        self._schemas[func_name] = self._parse_function(func)
        return func

    def register_all(self, *funcs: Callable):
        for func in funcs:
            self.register(func)

    async def execute(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise ToolNotFoundError(tool_name=name)

        current_trace = get_current_trace()
        disabled = current_trace is None

        func = self._tools[name]

        with tool_span(
            tool_name=name,
            input_args=kwargs if not disabled else None,
            disabled=disabled,
        ) as span:
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(**kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: func(**kwargs))

                if not disabled:
                    span.span_data.output = result

                return result
            except TypeError as e:
                if not disabled:
                    span.span_data.error = str(e)
                    span.set_error(
                        SpanError(
                            message=f"Invalid arguments for tool {name}",
                            data={"tool_name": name, "error_type": "TypeError"},
                        )
                    )
                raise InvalidArgumentsError(
                    message=str(e), tool_name=name, provided_args=list(kwargs.keys())
                )
            except ToolError:
                if not disabled:
                    span.span_data.error = "ToolError"
                raise
            except Exception as e:
                if not disabled:
                    span.span_data.error = str(e)
                    span.set_error(
                        SpanError(
                            message=f"Tool execution failed: {name}",
                            data={"tool_name": name, "error_type": type(e).__name__},
                        )
                    )
                raise ToolError(message=str(e), tool_name=name, error_detail=type(e).__name__)

    async def execute_many(self, calls: List[Dict]) -> List[Any]:
        current_trace = get_current_trace()
        disabled = current_trace is None

        tool_names = []
        for call in calls:
            name = call.get("name") or call.get("function", {}).get("name")
            if name:
                tool_names.append(name)

        with custom_span(
            name="batch_tool_execution",
            data=(
                {
                    "tool_count": len(calls),
                    "tool_names": tool_names,
                }
                if not disabled
                else None
            ),
            disabled=disabled,
        ) as batch_span:
            tasks = []
            for call in calls:
                name = call.get("name") or call.get("function", {}).get("name")
                arguments = call.get("arguments") or call.get("function", {}).get("arguments", {})

                if isinstance(arguments, str):
                    import json

                    arguments = json.loads(arguments)

                tasks.append(self.execute(name, **arguments))

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                if not disabled:
                    success_count = sum(1 for r in results if not isinstance(r, Exception))
                    error_count = len(results) - success_count
                    batch_span.span_data.data["success_count"] = success_count
                    batch_span.span_data.data["error_count"] = error_count

                return results
            except Exception as e:
                if not disabled:
                    batch_span.set_error(
                        SpanError(
                            message=f"Batch tool execution failed",
                            data={"error_type": type(e).__name__},
                        )
                    )
                raise

    def to_openai_tools(self) -> List[Dict]:
        tools = []
        for name, schema in self._schemas.items():
            function_schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema["description"],
                    "parameters": {
                        "type": "object",
                        "properties": schema["parameters"].get("properties", {}),
                    },
                },
            }

            if "required" in schema["parameters"]:
                function_schema["function"]["parameters"]["required"] = schema["parameters"][
                    "required"
                ]

            tools.append(function_schema)
        return tools

    def get_tool_schema(self, name: str) -> Optional[Dict]:
        return self._schemas.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())


    async def add_mcp_stdio_server(
        self,
        name: str,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
        prefix: Optional[str] = None,
        auto_register: bool = True,
    ):
        config = MCPServerConfig(
            name=name,
            command=command,
            args=args,
            env=env,
            prefix=prefix or name,
            transport="stdio",
        )
        await self._add_mcp_server(config, auto_register)

    async def add_mcp_sse_server(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        prefix: Optional[str] = None,
        auto_register: bool = True,
    ):
        config = MCPServerConfig(
            name=name, url=url, headers=headers, prefix=prefix or name, transport="sse"
        )
        await self._add_mcp_server(config, auto_register)

    async def _add_mcp_server(self, config: "MCPServerConfig", auto_register: bool = True):
        client = MCPClient(config)
        await client.connect()

        self._mcp_clients[config.name] = client
        self._mcp_configs[config.name] = config

        if auto_register:
            await self._register_mcp_tools(config.name)

    async def _register_mcp_tools(self, server_name: str):
        if server_name not in self._mcp_clients:
            raise ValueError(f"MCP server '{server_name}' not found")

        client = self._mcp_clients[server_name]
        config = self._mcp_configs[server_name]
        tools = await client.list_tools()

        for tool in tools:
            tool_name = f"{config.prefix}__{tool['name']}"
            self._register_mcp_tool(tool_name, client, tool)

    def _register_mcp_tool(self, tool_name: str, client: "MCPClient", tool_schema: Dict):
        original_name = tool_schema["name"]
        description = tool_schema.get("description", "")
        input_schema = tool_schema.get("inputSchema", {})

        async def mcp_tool_wrapper(**kwargs):
            try:
                result = await client.call_tool(original_name, kwargs)
                return format_mcp_result(result)
            except Exception as e:
                return {
                    "error": type(e).__name__,
                    "message": str(e),
                    "tool_name": original_name,
                    "mcp_tool": True,
                }

        mcp_tool_wrapper.__name__ = tool_name
        mcp_tool_wrapper.__doc__ = description

        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        sig_params = []
        for param_name, param_schema in properties.items():
            param_type = JSON_TO_PYTHON_TYPE_MAP.get(param_schema.get("type", "string"), str)
            default = inspect.Parameter.empty if param_name in required else None
            sig_params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=param_type,
                )
            )

        mcp_tool_wrapper.__signature__ = inspect.Signature(sig_params)

        self.register(mcp_tool_wrapper, tool_name)

    async def remove_mcp_server(self, name: str):
        if name in self._mcp_clients:
            client = self._mcp_clients.pop(name)
            await client.disconnect()

        if name in self._mcp_configs:
            self._mcp_configs.pop(name)

    async def disconnect_all_mcp_servers(self):
        for client in self._mcp_clients.values():
            await client.disconnect()
        self._mcp_clients.clear()
        self._mcp_configs.clear()

    def _parse_function(self, func: Callable) -> Dict:
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        try:
            type_hints = get_type_hints(func)
        except:
            type_hints = {}

        doc_info = self._parse_docstring(doc)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self" or param_name == "cls":
                continue

            param_type = type_hints.get(param_name, str)
            json_type = self._python_type_to_json_type(param_type)

            param_desc = doc_info["params"].get(param_name, "")

            param_schema = {"type": json_type, "description": param_desc}

            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            else:
                if param.default is not None:
                    param_schema["default"] = param.default

            properties[param_name] = param_schema

        parameters = {"type": "object", "properties": properties}

        if required:
            parameters["required"] = required

        return {
            "name": func.__name__,
            "description": doc_info["description"],
            "parameters": parameters,
        }

    def _parse_docstring(self, docstring: str) -> Dict:
        if not docstring:
            return {"description": "", "params": {}}

        lines = docstring.strip().split("\n")

        description_lines = []
        params = {}

        in_args_section = False
        current_param = None

        for line in lines:
            stripped = line.strip()

            if stripped.lower().startswith("args:"):
                in_args_section = True
                continue

            if in_args_section:
                if stripped.lower().startswith(
                    (
                        "returns:",
                        "return:",
                        "raises:",
                        "raise:",
                        "yields:",
                        "yield:",
                        "examples:",
                        "example:",
                        "note:",
                        "notes:",
                        "warning:",
                        "warnings:",
                    )
                ):
                    in_args_section = False
                    continue

                param_match = re.match(r"^(\w+)\s*\([^)]+\):\s*(.+)$", stripped)
                if param_match:
                    current_param = param_match.group(1)
                    param_desc = param_match.group(2)
                    params[current_param] = param_desc
                elif current_param and stripped:
                    params[current_param] += " " + stripped
            else:
                if stripped and not stripped.lower().startswith(
                    ("args:", "returns:", "return:", "raises:", "raise:")
                ):
                    description_lines.append(stripped)

        description = " ".join(description_lines).strip() or "No description provided"

        return {"description": description, "params": params}

    def _python_type_to_json_type(self, python_type) -> str:
        if python_type in TYPE_MAP:
            return TYPE_MAP[python_type]

        type_str = str(python_type)
        for key, value in TYPE_MAP.items():
            if isinstance(key, str) and key in type_str.lower():
                return value

        origin = get_origin(python_type)
        if origin is not None:
            if origin in (list, List, set, Set, tuple, Tuple):
                return "array"
            elif origin in (dict, Dict):
                return "object"
            elif origin is Union:
                args = get_args(python_type)
                if len(args) > 0:
                    return self._python_type_to_json_type(args[0])

        if hasattr(python_type, "__origin__"):
            if python_type.__origin__ in (list, tuple, set):
                return "array"
            elif python_type.__origin__ is dict:
                return "object"

        return "string"


if __name__ == "__main__":
    registry = ToolKit()

    @registry.tool
    def add_numbers(a: int, b: int) -> int:
        """
        Add two numbers together.

        Args:
            a (int): First number
            b (int): Second number

        Returns:
            int: Sum of the two numbers
        """
        return a + b

    @registry.tool
    async def fetch_data(url: str, timeout: int = 30) -> dict:
        """
        Fetch data from a URL (simulated).

        Args:
            url (str): The URL to fetch
            timeout (int): Timeout in seconds

        Returns:
            dict: Response data
        """
        await asyncio.sleep(0.1)
        return {"url": url, "timeout": timeout, "status": "success"}

    @registry.tool
    def calculate(expression: str) -> float:
        """
        Calculate a mathematical expression.

        Args:
            expression (str): Mathematical expression to evaluate

        Returns:
            float: Result of the calculation
        """
        try:
            return float(eval(expression))
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")

    async def main():
        print("=" * 70)
        print("Tool Registry Robustness Test Suite")
        print("=" * 70)
        print()

        print("=" * 70)
        print("Part 1: Regular Tools Robustness Tests")
        print("=" * 70)
        print()

        print("Test 1.1: Normal Tool Execution")
        print("-" * 70)
        result = await registry.execute("add_numbers", a=10, b=20)
        print(f"add_numbers(10, 20) = {result}")
        print()

        print("Test 1.2: Async Tool Execution")
        print("-" * 70)
        result = await registry.execute("fetch_data", url="https://example.com")
        print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print()

        print("Test 1.3: Tool with Optional Parameters")
        print("-" * 70)
        result = await registry.execute("fetch_data", url="https://api.com", timeout=60)
        print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print()

        print("Test 1.4: Missing Required Arguments")
        print("-" * 70)
        result = await registry.execute("add_numbers", a=5)
        print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print()

        print("Test 1.5: Tool Not Found")
        print("-" * 70)
        result = await registry.execute("nonexistent_tool", param="value")
        print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print()

        print("Test 1.6: Tool Internal Exception")
        print("-" * 70)
        result = await registry.execute("calculate", expression="1/0")
        print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print()

        print("Test 1.7: Wrong Argument Names")
        print("-" * 70)
        result = await registry.execute("add_numbers", x=1, y=2)
        print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print()

        print("Test 1.8: List All Tools")
        print("-" * 70)
        all_tools = registry.list_tools()
        regular_tools = [t for t in all_tools if "__" not in t]
        print(f"Regular tools: {regular_tools}")
        print()

        print("Test 1.9: Get Tool Schema")
        print("-" * 70)
        schema = registry.get_tool_schema("add_numbers")
        print(f"Schema: {json.dumps(schema, indent=2, ensure_ascii=False)}")
        print()

        print("Test 1.10: Batch Execution")
        print("-" * 70)
        results = await registry.execute_many(
            [
                {"name": "add_numbers", "arguments": {"a": 1, "b": 2}},
                {"name": "add_numbers", "arguments": {"a": 10}},
                {"name": "nonexistent", "arguments": {}},
                {"name": "calculate", "arguments": {"expression": "2+3"}},
            ]
        )
        for i, result in enumerate(results):
            if isinstance(result, dict):
                print(f"Result {i+1}: {json.dumps(result, ensure_ascii=False)}")
            else:
                print(f"Result {i+1}: {result}")
        print()

        print("=" * 70)
        print("Part 2: MCP Tools Integration Tests")
        print("=" * 70)
        print()

        print("Test 2.1: Add Sequential Thinking Server (stdio)")
        print("=" * 70)
        try:
            await registry.add_mcp_stdio_server(
                name="sequential_thinking",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
                env={},
                prefix="thinking",
            )
            print(f"[OK] Sequential Thinking server connected (stdio)")
            print(f"[OK] Registered tools: {[t for t in registry.list_tools() if 'thinking' in t]}")
            print()
        except Exception as e:
            print(f"[FAIL] {e}")
            print()

        print("=" * 70)
        print("Test 2.2: Add Ali WebSearch Server (SSE)")
        print("=" * 70)
        ali_api_key = os.getenv("ali_api_key")
        if ali_api_key:
            try:
                await registry.add_mcp_sse_server(
                    name="ali_web_search",
                    url="https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
                    headers={"Authorization": f"Bearer {ali_api_key}"},
                    prefix="ali",
                )
                print(f"[OK] Ali WebSearch server connected (SSE)")
                print(f"[OK] Registered tools: {[t for t in registry.list_tools() if 'ali' in t]}")
                print()
            except Exception as e:
                print(f"[FAIL] {e}")
                print()
        else:
            print("[SKIP] ali_api_key not set")
            print()

        print("=" * 70)
        print("Test 2.3: List All MCP Tools")
        print("=" * 70)
        all_tools = registry.list_tools()
        print(f"Total tools registered: {len(all_tools)}")
        for tool_name in all_tools:
            print(f"  - {tool_name}")
        print()

        print("=" * 70)
        print("Test 2.4: Get MCP Tool Schemas (OpenAI format)")
        print("=" * 70)
        openai_tools = registry.to_openai_tools()
        for tool in openai_tools[:2]:
            print(json.dumps(tool, indent=2, ensure_ascii=False))
            print()
        if len(openai_tools) > 2:
            print(f"... and {len(openai_tools) - 2} more tools")
            print()

        print("=" * 70)
        print("Test 2.5: Execute Sequential Thinking Tool")
        print("=" * 70)
        thinking_tools = [t for t in registry.list_tools() if "thinking" in t]
        saved_thinking_tools = thinking_tools.copy()
        if thinking_tools:
            try:
                tool_name = thinking_tools[0]
                print(f"Calling tool: {tool_name}")
                result = await registry.execute(
                    tool_name,
                    thought="Let me think about how MCP works",
                    nextThoughtNeeded=True,
                    thoughtNumber=1,
                    totalThoughts=3,
                )
                print(f"[OK] Tool executed successfully")
                print(f"Result: {result}")
                print()
            except Exception as e:
                print(f"[FAIL] Execution failed: {e}")
                import traceback

                traceback.print_exc()
                print()
        else:
            print("[SKIP] Tool not available")
            print()

        print("=" * 70)
        print("Test 2.6: Execute Ali WebSearch Tool")
        print("=" * 70)
        ali_tools = [t for t in registry.list_tools() if "ali" in t]
        if ali_tools and ali_api_key:
            try:
                tool_name = ali_tools[0]
                schema = registry.get_tool_schema(tool_name)
                required_params = schema["parameters"].get("required", [])

                if "query" in required_params:
                    result = await registry.execute(tool_name, query="MCP protocol")
                    print(f"[OK] Execution result: {result}")
                    print()
                else:
                    print(f"[OK] Tool available but skipping execution (unknown params)")
                    print()
            except Exception as e:
                print(f"[FAIL] Execution failed: {e}")
                print()
        else:
            print("[SKIP] Tool not available or API key not set")
            print()

        print("=" * 70)
        print("Test 2.7: OpenAI Format Export (All Tools)")
        print("=" * 70)
        all_openai_tools = registry.to_openai_tools()
        print(f"Total OpenAI function schemas: {len(all_openai_tools)}")
        print(
            f"Regular tools: {len([t for t in all_openai_tools if '__' not in t['function']['name']])}"
        )
        print(f"MCP tools: {len([t for t in all_openai_tools if '__' in t['function']['name']])}")
        print()

        print("=" * 70)
        print("Test 2.8: MCP Invalid Arguments Handling")
        print("=" * 70)
        if saved_thinking_tools:
            result = await registry.execute(saved_thinking_tools[0], invalid_param="test")
            print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
            print()
        else:
            print("[SKIP] No thinking tools available")
            print()

        print("=" * 70)
        print("Cleanup: Disconnect MCP servers")
        print("=" * 70)
        await registry.disconnect_all_mcp_servers()
        print("All MCP servers disconnected")
        print()

    asyncio.run(main())
