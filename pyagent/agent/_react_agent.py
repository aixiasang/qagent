import asyncio
import json
import re
from typing import Optional, Union, AsyncGenerator, Dict, List

from ..core import Agent, BaseAgent, Memory, ChatResponse, Chater, ChaterPool, ToolCall, ToolKit, Speaker,with_reply_hooks,with_observe_hooks
from ..prompt import (
    get_react_openai_prompt,
    build_classic_react_system_prompt,
    get_error_no_tool_prompt,
    get_error_invalid_format_prompt,
    get_observation_prompt,
    get_step_output_prompt,
)


class ReActTracker:
    def __init__(self):
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.iteration = 0
    
    def record_thought(self, thought: str):
        self.thoughts.append({
            "iteration": self.iteration,
            "content": thought
        })
    
    def record_action(self, tool_calls: List[ToolCall]):
        self.actions.append({
            "iteration": self.iteration,
            "tools": [{"name": tc.fn_name, "args": tc.get_args_dict()} for tc in tool_calls]
        })
    
    def record_observation(self, results: List[ChatResponse]):
        observations = []
        for r in results:
            if r.tool_result:
                observations.append({
                    "tool": r.tool_result.fn_name,
                    "output": r.tool_result.fn_output
                })
        self.observations.append({
            "iteration": self.iteration,
            "results": observations
        })
    
    def next_iteration(self):
        self.iteration += 1
    
    def get_full_trace(self) -> Dict:
        return {
            "total_iterations": self.iteration,
            "thoughts": self.thoughts,
            "actions": self.actions,
            "observations": self.observations
        }
    
    def clear(self):
        self.thoughts.clear()
        self.actions.clear()
        self.observations.clear()
        self.iteration = 0


class ReActAgent(Agent):
    def __init__(
        self,
        name: str,
        chater: Union[Chater, ChaterPool],
        memory: Memory,
        tools: Optional[ToolKit] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        tool_timeout: Optional[int] = None,
        enable_logging: bool = False,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        speaker: Optional[Speaker] = None,
        track_reasoning: bool = True,
    ):
        final_prompt = system_prompt or get_react_openai_prompt()
        
        super().__init__(
            name=name,
            chater=chater,
            memory=memory,
            tools=tools,
            system_prompt=final_prompt,
            max_iterations=max_iterations,
            tool_timeout=tool_timeout,
            enable_logging=enable_logging,
            log_file=log_file,
            log_level=log_level,
            speaker=speaker,
        )
        
        self.track_reasoning = track_reasoning
        self.tracker = ReActTracker()
    
    @with_reply_hooks
    async def reply(
        self, user_message: str, stream: bool = False, auto_speak: bool = False
    ) -> Union[ChatResponse, AsyncGenerator[ChatResponse, None]]:
        self.logger.info(f"ReAct Agent received: {user_message[:50]}...")
        
        if self.track_reasoning:
            self.tracker.clear()
        
        user_msg = ChatResponse(role="user", content=user_message)
        self.memory.add(user_msg)
        
        tools = self.tools.to_openai_tools() if self.tools else None
        tool_choice = "auto" if self.tools else None
        
        for iteration in range(self.max_iterations):
            if self.track_reasoning:
                self.tracker.iteration = iteration
            
            self.logger.debug(f"ReAct iteration {iteration + 1}/{self.max_iterations}")
            history = self._build_history()
            
            if not stream:
                response = await self.chater.chat(
                    messages=history, tools=tools, tool_choice=tool_choice, stream=False
                )
                
                if self.track_reasoning and response.reasoning_content:
                    self.tracker.record_thought(response.reasoning_content)
                
                self.memory.add(response)
                
                tool_calls_list = (
                    response.tool_calls
                    if response.tool_calls
                    else ([response.tool_call] if response.tool_call else [])
                )
                
                if tool_calls_list:
                    if self.track_reasoning:
                        self.tracker.record_action(tool_calls_list)
                    
                    yield response
                    self.logger.info(f"Executing {len(tool_calls_list)} tools")
                    
                    tool_results = await self._handle_tool_calls_concurrent(tool_calls_list)
                    
                    if self.track_reasoning:
                        self.tracker.record_observation(tool_results)
                        self.tracker.next_iteration()
                    
                    self.memory.add(tool_results)
                else:
                    self.logger.info("ReAct completed")
                    response = self._run_post_reply_hooks(response, user_message)
                    yield response
                    
                    if auto_speak:
                        self.speak(response, stream=False)
                    
                    self._broadcast_to_audience(response)
                    break
            
            else:
                response = ChatResponse()
                last_msg = None
                first_chunk = True
                
                async for msg in await self.chater.chat(
                    messages=history, stream=True, tools=tools, tool_choice=tool_choice
                ):
                    if msg.reasoning_content:
                        response.reasoning_content += msg.reasoning_content
                        if self.track_reasoning and iteration == 0:
                            self.tracker.record_thought(msg.reasoning_content)
                    
                    if msg.content:
                        response.content += msg.content
                    
                    if msg.tool_call:
                        response.tool_call = msg.tool_call
                    
                    if msg.tool_calls:
                        response.tool_calls = msg.tool_calls
                    
                    last_msg = msg
                    
                    if auto_speak:
                        if first_chunk and (msg.content or msg.reasoning_content):
                            print("Agent: ", end="", flush=True)
                            first_chunk = False
                        self._output_chunk(msg)
                    
                    yield msg
                
                if last_msg:
                    response.usage = last_msg.usage
                    response.id = last_msg.id
                    response.created = last_msg.created
                    response.role = last_msg.role
                
                self.memory.add(response)
                
                tool_calls_list = (
                    response.tool_calls
                    if response.tool_calls
                    else ([response.tool_call] if response.tool_call else [])
                )
                
                if tool_calls_list:
                    if self.track_reasoning:
                        self.tracker.record_action(tool_calls_list)
                    
                    self.logger.info(f"Executing {len(tool_calls_list)} tools")
                    tool_results = await self._handle_tool_calls_concurrent(tool_calls_list)
                    
                    if self.track_reasoning:
                        self.tracker.record_observation(tool_results)
                        self.tracker.next_iteration()
                    
                    self.memory.add(tool_results)
                else:
                    self.logger.info("ReAct completed")
                    final_response = self._run_post_reply_hooks(response, user_message)
                    if final_response != response and hasattr(self.memory, 'messages'):
                        if len(self.memory.messages) > 0:
                            self.memory.messages[-1] = final_response
                    
                    if auto_speak:
                        print()
                    
                    self._broadcast_to_audience(final_response)
                    break
    
    def get_reasoning_trace(self) -> Dict:
        if not self.track_reasoning:
            return {"error": "Reasoning tracking is disabled"}
        return self.tracker.get_full_trace()
    
    def _output_chunk(self, msg: ChatResponse):
        if msg.content:
            print(msg.content, end="", flush=True)
        if msg.reasoning_content:
            print(msg.reasoning_content, end="", flush=True)


class ClassicReActAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        chater: Union[Chater, ChaterPool],
        memory: Memory,
        tools: Optional[ToolKit] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        tool_timeout: Optional[int] = None,
        enable_logging: bool = False,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        speaker: Optional[Speaker] = None,
    ):
        super().__init__(name, speaker=speaker)
        self.chater = chater
        self.memory = memory
        self.tools = tools
        self.max_iterations = max_iterations
        self.tool_timeout = tool_timeout
        self._audience: Optional[list["BaseAgent"]] = None
        
        self.system_prompt = system_prompt or self._build_system_prompt_from_template()
        
        from core._utils import AgentLogger
        self.logger = AgentLogger(
            name=f"ClassicReActAgent.{name}",
            level=log_level,
            enabled=enable_logging,
            log_file=log_file
        )
        
        self.react_history = []
    
    def _build_system_prompt_from_template(self) -> str:
        if not self.tools:
            return build_classic_react_system_prompt("No tools available")
        
        tools_desc = []
        for tool_name in self.tools.list_tools():
            schema = self.tools.get_tool_schema(tool_name)
            params = schema.get("parameters", {}).get("properties", {})
            params_str = json.dumps(params, ensure_ascii=False, indent=2)
            tool_desc = f"- {tool_name}: {schema.get('description', 'No description')}\n  Parameters: {params_str}"
            tools_desc.append(tool_desc)
        
        tools_text = "\n".join(tools_desc)
        return build_classic_react_system_prompt(tools_text)
    
    def _build_history(self) -> list[dict]:
        system_msg = {"role": "system", "content": self.system_prompt}
        memory_msgs = self.memory.to_openai()
        return [system_msg] + memory_msgs
    
    @with_observe_hooks
    def observe(self, msg: Union[ChatResponse, list[ChatResponse]]) -> None:
        if isinstance(msg, list):
            for m in msg:
                self.memory.add(m)
        else:
            self.memory.add(msg)
    
    def _broadcast_to_audience(self, msg: ChatResponse) -> None:
        if self._audience:
            for agent in self._audience:
                agent.observe(msg)
    
    @with_reply_hooks
    async def reply(
        self, user_message: str, stream: bool = False, auto_speak: bool = False
    ) -> Union[ChatResponse, AsyncGenerator[ChatResponse, None]]:
        self.logger.info(f"Classic ReAct received: {user_message[:50]}...")
        
        user_msg = ChatResponse(role="user", content=user_message)
        self.memory.add(user_msg)
        self.react_history.clear()
        
        context = user_message
        
        for iteration in range(self.max_iterations):
            self.logger.debug(f"Classic ReAct iteration {iteration + 1}/{self.max_iterations}")
            
            history = self._build_history()
            
            response = await self.chater.chat(messages=history, stream=False)
            
            self.memory.add(response)
            content = response.content if isinstance(response.content, str) else str(response.content)
            
            self.react_history.append({
                "iteration": iteration,
                "response": content
            })
            
            thought = self._extract_tag_content(content, "Thought")
            action = self._extract_tag_content(content, "Action")
            action_input = self._extract_tag_content(content, "ActionInput")
            
            if "<FinalAnswer>" in content:
                if iteration == 0 and self.tools and len(self.tools.list_tools()) > 0:
                    self.logger.warning("Model tried to answer without using tools, forcing tool use")
                    force_msg = ChatResponse(
                        role="user",
                        content=get_error_no_tool_prompt()
                    )
                    self.memory.add(force_msg)
                    continue
                
                final_answer = self._extract_final_answer(content)
                
                final_response = ChatResponse(
                    role="assistant",
                    content=final_answer,
                    id=response.id,
                    created=response.created,
                    usage=response.usage
                )
                
                final_response = self._run_post_reply_hooks(final_response, user_message)
                
                if auto_speak:
                    self.speak(final_response, stream=False)
                
                self._broadcast_to_audience(final_response)
                
                yield final_response
                break
            
            if action and action_input:
                self.logger.info(f"Executing action: {action}")
                
                try:
                    args_dict = json.loads(action_input)
                except json.JSONDecodeError:
                    args_dict = {}
                
                try:
                    if self.tool_timeout:
                        output = await asyncio.wait_for(
                            self.tools.execute(name=action.strip(), **args_dict),
                            timeout=self.tool_timeout
                        )
                    else:
                        output = await self.tools.execute(name=action.strip(), **args_dict)
                except asyncio.TimeoutError:
                    output = f"Tool execution timeout after {self.tool_timeout}s"
                except Exception as e:
                    output = f"Error: {str(e)}"
                
                observation_msg = ChatResponse(
                    role="user",
                    content=get_observation_prompt(output)
                )
                self.memory.add(observation_msg)
                
                yield ChatResponse(
                    role="assistant",
                    content=get_step_output_prompt(thought, action, action_input, output)
                )
            else:
                self.logger.warning("No valid action found")
                if iteration == 0 and self.tools and len(self.tools.list_tools()) > 0:
                    error_msg = ChatResponse(
                        role="user",
                        content=get_error_invalid_format_prompt()
                    )
                    self.memory.add(error_msg)
                    continue
                else:
                    final_response = ChatResponse(
                        role="assistant",
                        content=content
                    )
                    final_response = self._run_post_reply_hooks(final_response, user_message)
                    yield final_response
                    break
    
    def _extract_tag_content(self, text: str, tag: str) -> str:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        pattern_open = f"<{tag}>(.*?)(?=<|$)"
        match_open = re.search(pattern_open, text, re.DOTALL | re.IGNORECASE)
        if match_open:
            return match_open.group(1).strip()
        
        return ""
    
    def _extract_final_answer(self, text: str) -> str:
        answer = self._extract_tag_content(text, "FinalAnswer")
        if answer:
            return answer
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if "<FinalAnswer>" in line:
                return '\n'.join(lines[i+1:]).replace("</FinalAnswer>", "").strip()
        
        return text
    
    def _run_post_reply_hooks(self, response: ChatResponse, user_message: str) -> ChatResponse:
        from copy import deepcopy
        import logging
        
        current_response = response
        
        for hook in self._post_reply_hooks.values():
            try:
                result = hook(deepcopy(current_response))
                if result is not None:
                    current_response = result
            except Exception as e:
                logging.warning(f"Post-reply hook failed: {e}")
        
        for hook in self.__class__._class_hooks_post_reply.values():
            try:
                result = hook(self, deepcopy(current_response))
                if result is not None:
                    current_response = result
            except Exception as e:
                logging.warning(f"Class post-reply hook failed: {e}")
        
        return current_response
    
    def get_react_history(self) -> List[Dict]:
        return self.react_history
    
    def clear_memory(self):
        self.memory.clear()
        self.react_history.clear()


if __name__ == "__main__":
    from core._model import get_chater_cfg, ChaterPool
    from datetime import datetime
    
    async def get_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    async def calculate(expression: str) -> float:
        try:
            return float(eval(expression))
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")
    
    async def search_info(query: str) -> str:
        return f"Search results for '{query}': This is a simulated search result."
    
    async def ainput(prompt: str = "") -> str:
        return await asyncio.to_thread(input, prompt)
    
    async def test_react_agent():
        print("="*60)
        print("Testing ReActAgent (OpenAI Tool Calling)")
        print("="*60)
        
        tools = ToolKit()
        tools.register(get_time, "get_time")
        tools.register(calculate, "calculate")
        tools.register(search_info, "search_info")
        
        agent = ReActAgent(
            name="ReActBot",
            chater=ChaterPool([get_chater_cfg("siliconflow"), get_chater_cfg("zhipuai")]),
            memory=Memory(max_messages=None),
            tools=tools,
            max_iterations=5,
            enable_logging=False,
            track_reasoning=True
        )
        
        test_queries = [
            "现在几点了？",
            "What is 123 * 456?",
            "搜索一下人工智能的最新进展"
        ]
        
        for query in test_queries:
            print(f"\nUser: {query}")
            print("Agent: ", end="", flush=True)
            
            async for response in agent.reply(query, stream=True):
                if response.content:
                    print(response.content, end="", flush=True)
                if response.reasoning_content:
                    print(f"[Thinking: {response.reasoning_content}]", end="", flush=True)
            print()
            
            trace = agent.get_reasoning_trace()
            print(f"Iterations: {trace['total_iterations']}")
        
        print("\n" + "="*60)
    
    async def test_classic_react_agent():
        print("\n" + "="*60)
        print("Testing ClassicReActAgent (Classic Format)")
        print("="*60)
        
        tools = ToolKit()
        tools.register(get_time, "get_time")
        tools.register(calculate, "calculate")
        tools.register(search_info, "search_info")
        
        agent = ClassicReActAgent(
            name="ClassicReActBot",
            chater=ChaterPool([get_chater_cfg("siliconflow"), get_chater_cfg("zhipuai")]),
            memory=Memory(max_messages=None),
            tools=tools,
            max_iterations=5,
            enable_logging=False
        )
        
        test_queries = [
            "现在几点？",
            "Calculate 999 + 1"
        ]
        
        for query in test_queries:
            print(f"\nUser: {query}")
            
            async for response in agent.reply(query):
                print(f"Agent: {response.content}")
            
            print(f"Total steps: {len(agent.get_react_history())}")
        
        print("\n" + "="*60)
    
    async def main():
        await test_react_agent()
        await test_classic_react_agent()
    
    asyncio.run(main())
