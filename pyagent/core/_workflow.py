import uuid
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import (
    TypeVar,
    Generic,
    Dict,
    Any,
    List,
    Optional,
    Callable,
    Union,
    AsyncGenerator,
    Tuple,
    Type,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from ._agent import Agent
    from ._model import ChatResponse

from ._trace import custom_span, get_current_trace


class GraphConstants(str, Enum):
    START = "__start__"
    END = "__end__"


START = GraphConstants.START
END = GraphConstants.END

StateType = TypeVar("StateType", bound=Dict[str, Any])


@dataclass
class NodeResult:
    node_id: str
    output: Dict[str, Any]
    duration: float = 0.0
    error: Optional[str] = None


@dataclass
class WorkflowResult(Generic[StateType]):
    success: bool
    state: StateType
    history: List[NodeResult] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "state": dict(self.state) if self.state else {},
            "history": [
                {
                    "node_id": h.node_id,
                    "output": h.output,
                    "duration": h.duration,
                    "error": h.error,
                }
                for h in self.history
            ],
            "error": self.error,
        }


@dataclass
class Edge:
    source: str
    target: str


@dataclass
class ConditionalEdge:
    source: str
    condition: Callable[[Dict[str, Any]], str]
    routes: Dict[str, str]


class Reducer(ABC):
    @abstractmethod
    def reduce(self, current: Any, update: Any) -> Any:
        pass


class ReplaceReducer(Reducer):
    def reduce(self, current: Any, update: Any) -> Any:
        return update


class AppendReducer(Reducer):
    def reduce(self, current: Any, update: Any) -> Any:
        if current is None:
            current = []
        if isinstance(update, list):
            return current + update
        return current + [update]


class MergeReducer(Reducer):
    def reduce(self, current: Any, update: Any) -> Any:
        if current is None:
            current = {}
        if isinstance(update, dict):
            return {**current, **update}
        return update


class AddReducer(Reducer):
    def reduce(self, current: Any, update: Any) -> Any:
        if current is None:
            return update
        return current + update


DEFAULT_REDUCER = ReplaceReducer()


class Channel:
    def __init__(self, default: Any = None, reducer: Optional[Reducer] = None):
        self.default = default
        self.reducer = reducer or DEFAULT_REDUCER

    def apply(self, current: Any, update: Any) -> Any:
        return self.reducer.reduce(current, update)


NodeFunc = Callable[[Dict[str, Any]], Union[Dict[str, Any], None]]
AsyncNodeFunc = Callable[[Dict[str, Any]], Union[Dict[str, Any], None]]


class StateGraph(Generic[StateType]):
    def __init__(
        self,
        state_schema: Optional[Type[StateType]] = None,
        channels: Optional[Dict[str, Channel]] = None,
    ):
        self._state_schema = state_schema
        self._channels: Dict[str, Channel] = channels or {}
        self._nodes: Dict[str, Union[NodeFunc, AsyncNodeFunc, "Agent"]] = {}
        self._edges: List[Edge] = []
        self._conditional_edges: List[ConditionalEdge] = []
        self._entry_point: Optional[str] = None
        self._compiled: bool = False
        self._graph_id: str = str(uuid.uuid4())[:8]

    def add_node(
        self,
        name: str,
        node: Union[NodeFunc, AsyncNodeFunc, "Agent"],
    ) -> "StateGraph[StateType]":
        if name in (START.value, END.value):
            raise ValueError(f"Cannot use reserved name: {name}")
        self._nodes[name] = node
        self._compiled = False
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph[StateType]":
        if source == START.value:
            self._entry_point = target
        self._edges.append(Edge(source=source, target=target))
        self._compiled = False
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition: Callable[[Dict[str, Any]], str],
        routes: Optional[Dict[str, str]] = None,
    ) -> "StateGraph[StateType]":
        if source == START.value and self._entry_point is None:
            self._entry_point = "__conditional_start__"
        self._conditional_edges.append(
            ConditionalEdge(
                source=source,
                condition=condition,
                routes=routes or {},
            )
        )
        self._compiled = False
        return self

    def set_entry_point(self, node_id: str) -> "StateGraph[StateType]":
        self._entry_point = node_id
        return self

    def set_finish_point(self, node_id: str) -> "StateGraph[StateType]":
        self.add_edge(node_id, END.value)
        return self

    def compile(self) -> "CompiledGraph[StateType]":
        if not self._entry_point and self._entry_point != "__conditional_start__":
            for edge in self._edges:
                if edge.source == START.value:
                    self._entry_point = edge.target
                    break

        if not self._entry_point:
            raise ValueError("No entry point defined. Use add_edge(START, 'node') or set_entry_point()")

        self._compiled = True
        return CompiledGraph(self)

    def _get_next_nodes(self, current: str, state: Dict[str, Any]) -> List[str]:
        next_nodes = []

        for edge in self._edges:
            if edge.source == current:
                next_nodes.append(edge.target)

        for cond_edge in self._conditional_edges:
            if cond_edge.source == current:
                result = cond_edge.condition(state)
                if cond_edge.routes:
                    target = cond_edge.routes.get(result, result)
                else:
                    target = result
                if target and target != END.value:
                    next_nodes.append(target)
                elif target == END.value:
                    next_nodes.append(END.value)

        return next_nodes

    def _get_nodes(self) -> Dict[str, Any]:
        return self._nodes

    def get_graph_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self._nodes.keys()),
            "edges": [{"source": e.source, "target": e.target} for e in self._edges],
            "conditional_edges": [
                {"source": ce.source, "routes": ce.routes}
                for ce in self._conditional_edges
            ],
            "entry_point": self._entry_point,
        }


class CompiledGraph(Generic[StateType]):
    def __init__(self, graph: StateGraph[StateType]):
        self._graph = graph
        self._max_iterations = 100

    def with_config(self, max_iterations: int = 100) -> "CompiledGraph[StateType]":
        self._max_iterations = max_iterations
        return self

    def _init_state(self, input_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state: Dict[str, Any] = {}

        for key, channel in self._graph._channels.items():
            state[key] = channel.default

        if input_state:
            for key, value in input_state.items():
                state[key] = value

        return state

    def _apply_update(self, state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        if update is None:
            return state

        new_state = dict(state)
        for key, value in update.items():
            if key in self._graph._channels:
                channel = self._graph._channels[key]
                new_state[key] = channel.apply(state.get(key), value)
            else:
                new_state[key] = value

        return new_state

    async def _execute_node(
        self,
        node_id: str,
        state: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        import time

        node = self._graph._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node not found: {node_id}")

        start_time = time.time()

        if hasattr(node, "reply"):
            from ._model import ChatResponse

            input_msg = state.get("messages", [])
            if isinstance(input_msg, list) and input_msg:
                msg_content = input_msg[-1].get("content", "") if isinstance(input_msg[-1], dict) else str(input_msg[-1])
            else:
                msg_content = state.get("input", "")

            last_response = None
            async for response in node.reply(msg_content):
                last_response = response

            if last_response:
                update = {"output": last_response.content, "last_response": last_response}
                if "messages" in state:
                    update["messages"] = [{"role": "assistant", "content": last_response.content}]
            else:
                update = {}
        else:
            result = node(state)
            if asyncio.iscoroutine(result):
                update = await result
            else:
                update = result

        duration = time.time() - start_time
        return update or {}, duration

    async def ainvoke(
        self,
        input_state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> StateType:
        result = await self._run(input_state, config)
        return result.state

    async def _run(
        self,
        input_state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult[StateType]:
        state = self._init_state(input_state)
        history: List[NodeResult] = []
        current_node = self._graph._entry_point

        if current_node == "__conditional_start__":
            for cond_edge in self._graph._conditional_edges:
                if cond_edge.source == START.value:
                    result = cond_edge.condition(state)
                    current_node = cond_edge.routes.get(result, result)
                    break

        iteration = 0
        trace = get_current_trace()

        try:
            while current_node and current_node != END.value and iteration < self._max_iterations:
                iteration += 1

                if trace:
                    with custom_span(
                        name=f"workflow_node:{current_node}",
                        data={"node_id": current_node, "iteration": iteration},
                    ):
                        update, duration = await self._execute_node(current_node, state)
                else:
                    update, duration = await self._execute_node(current_node, state)

                state = self._apply_update(state, update)

                history.append(
                    NodeResult(
                        node_id=current_node,
                        output=update,
                        duration=duration,
                    )
                )

                next_nodes = self._graph._get_next_nodes(current_node, state)

                if not next_nodes:
                    break
                elif len(next_nodes) == 1:
                    current_node = next_nodes[0]
                else:
                    parallel_results = await asyncio.gather(
                        *[self._execute_node(n, state) for n in next_nodes if n != END.value]
                    )
                    for i, (update, duration) in enumerate(parallel_results):
                        state = self._apply_update(state, update)
                        history.append(
                            NodeResult(
                                node_id=next_nodes[i],
                                output=update,
                                duration=duration,
                            )
                        )
                    current_node = END.value

            return WorkflowResult(success=True, state=state, history=history)

        except Exception as e:
            return WorkflowResult(
                success=False,
                state=state,
                history=history,
                error=str(e),
            )

    def invoke(
        self,
        input_state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> StateType:
        return asyncio.run(self.ainvoke(input_state, config))

    async def astream(
        self,
        input_state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        state = self._init_state(input_state)
        current_node = self._graph._entry_point

        if current_node == "__conditional_start__":
            for cond_edge in self._graph._conditional_edges:
                if cond_edge.source == START.value:
                    result = cond_edge.condition(state)
                    current_node = cond_edge.routes.get(result, result)
                    break

        iteration = 0

        while current_node and current_node != END.value and iteration < self._max_iterations:
            iteration += 1

            update, _ = await self._execute_node(current_node, state)
            state = self._apply_update(state, update)

            yield current_node, dict(state)

            next_nodes = self._graph._get_next_nodes(current_node, state)

            if not next_nodes:
                break
            elif len(next_nodes) == 1:
                current_node = next_nodes[0]
            else:
                for node_id in next_nodes:
                    if node_id != END.value:
                        update, _ = await self._execute_node(node_id, state)
                        state = self._apply_update(state, update)
                        yield node_id, dict(state)
                current_node = END.value


def create_react_graph(
    agent: "Agent",
    tools_node: Optional[Callable] = None,
    max_iterations: int = 10,
) -> CompiledGraph:
    from ._model import ChatResponse

    graph = StateGraph()

    graph._channels = {
        "messages": Channel(default=[], reducer=AppendReducer()),
        "output": Channel(default=""),
    }

    async def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            content = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
        else:
            content = state.get("input", "")

        last_response = None
        async for response in agent.reply(content):
            last_response = response

        if last_response:
            return {
                "messages": [{"role": "assistant", "content": last_response.content}],
                "output": last_response.content,
                "tool_calls": last_response.tool_calls if hasattr(last_response, "tool_calls") else None,
                "last_response": last_response,
            }
        return {}

    async def default_tools_node(state: Dict[str, Any]) -> Dict[str, Any]:
        tool_calls = state.get("tool_calls")
        if not tool_calls or not agent.tools:
            return {}

        results = await agent.tools.execute_many(tool_calls)
        tool_messages = []
        for tc, result in zip(tool_calls, results):
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.call_id,
                "content": str(result),
            })

        return {"messages": tool_messages}

    def should_continue(state: Dict[str, Any]) -> str:
        tool_calls = state.get("tool_calls")
        if tool_calls:
            return "tools"
        return END.value

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node or default_tools_node)

    graph.add_edge(START.value, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END.value: END.value})
    graph.add_edge("tools", "agent")

    return graph.compile().with_config(max_iterations=max_iterations)


def create_sequential_graph(
    agents: List["Agent"],
) -> CompiledGraph:
    graph = StateGraph()

    graph._channels = {
        "messages": Channel(default=[], reducer=AppendReducer()),
        "output": Channel(default=""),
    }

    for i, agent in enumerate(agents):
        node_name = f"agent_{i}"

        async def make_node(a):
            async def node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
                messages = state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    content = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
                else:
                    content = state.get("input", "")

                last_response = None
                async for response in a.reply(content):
                    last_response = response

                if last_response:
                    return {
                        "messages": [{"role": "assistant", "content": last_response.content}],
                        "output": last_response.content,
                    }
                return {}
            return node_fn

        node_fn = asyncio.get_event_loop().run_until_complete(make_node(agent))
        graph.add_node(node_name, node_fn)

        if i == 0:
            graph.add_edge(START.value, node_name)
        else:
            graph.add_edge(f"agent_{i-1}", node_name)

        if i == len(agents) - 1:
            graph.add_edge(node_name, END.value)

    return graph.compile()


def create_parallel_graph(
    agents: List["Agent"],
    aggregator: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
) -> CompiledGraph:
    graph = StateGraph()

    graph._channels = {
        "messages": Channel(default=[], reducer=AppendReducer()),
        "results": Channel(default=[], reducer=AppendReducer()),
        "output": Channel(default=""),
    }

    for i, agent in enumerate(agents):
        node_name = f"agent_{i}"

        async def make_node(a, idx):
            async def node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
                content = state.get("input", "")

                last_response = None
                async for response in a.reply(content):
                    last_response = response

                if last_response:
                    return {
                        "results": [{"agent_idx": idx, "content": last_response.content}],
                    }
                return {}
            return node_fn

        node_fn = asyncio.get_event_loop().run_until_complete(make_node(agent, i))
        graph.add_node(node_name, node_fn)
        graph.add_edge(START.value, node_name)

    def default_aggregator(state: Dict[str, Any]) -> Dict[str, Any]:
        results = state.get("results", [])
        combined = "\n".join([r.get("content", "") for r in results])
        return {"output": combined}

    graph.add_node("aggregate", aggregator or default_aggregator)

    for i in range(len(agents)):
        graph.add_edge(f"agent_{i}", "aggregate")

    graph.add_edge("aggregate", END.value)

    return graph.compile()


class WorkflowBuilder:
    def __init__(self, name: str = "workflow"):
        self._name = name
        self._graph: StateGraph = StateGraph()
        self._graph._channels = {
            "messages": Channel(default=[], reducer=AppendReducer()),
            "output": Channel(default=""),
            "input": Channel(default=""),
        }

    def add_agent(
        self,
        name: str,
        agent: "Agent",
        input_key: str = "input",
        output_key: str = "output",
    ) -> "WorkflowBuilder":
        async def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            content = state.get(input_key, "")
            if not content and state.get("messages"):
                messages = state["messages"]
                if messages:
                    last = messages[-1]
                    content = last.get("content", "") if isinstance(last, dict) else str(last)

            last_response = None
            async for response in agent.reply(content):
                last_response = response

            if last_response:
                return {
                    output_key: last_response.content,
                    "messages": [{"role": "assistant", "content": last_response.content}],
                }
            return {}

        self._graph.add_node(name, agent_node)
        return self

    def add_function(
        self,
        name: str,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> "WorkflowBuilder":
        self._graph.add_node(name, fn)
        return self

    def add_router(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], str],
        routes: Dict[str, str],
    ) -> "WorkflowBuilder":
        def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"_route": condition(state)}

        self._graph.add_node(name, router_node)
        self._graph.add_conditional_edges(
            name,
            lambda s: s.get("_route", END.value),
            routes,
        )
        return self

    def add_branch(
        self,
        source: str,
        condition: Callable[[Dict[str, Any]], str],
        routes: Dict[str, str],
    ) -> "WorkflowBuilder":
        self._graph.add_conditional_edges(source, condition, routes)
        return self

    def chain(self, *node_names: str) -> "WorkflowBuilder":
        for i in range(len(node_names) - 1):
            self._graph.add_edge(node_names[i], node_names[i + 1])
        return self

    def set_entry(self, node_name: str) -> "WorkflowBuilder":
        self._graph.add_edge(START.value, node_name)
        return self

    def set_exit(self, node_name: str) -> "WorkflowBuilder":
        self._graph.add_edge(node_name, END.value)
        return self

    def add_channel(
        self,
        name: str,
        default: Any = None,
        reducer: Optional[Reducer] = None,
    ) -> "WorkflowBuilder":
        self._graph._channels[name] = Channel(default=default, reducer=reducer)
        return self

    def build(self) -> CompiledGraph:
        return self._graph.compile()


if __name__ == "__main__":

    async def test_basic_workflow():
        def step1(state):
            return {"value": state.get("input", 0) + 1}

        def step2(state):
            return {"value": state["value"] * 2}

        def step3(state):
            return {"output": f"Result: {state['value']}"}

        graph = StateGraph()
        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_node("step3", step3)

        graph.add_edge(START.value, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.add_edge("step3", END.value)

        compiled = graph.compile()
        result = await compiled.ainvoke({"input": 5})

        print("Basic workflow result:", result)
        assert result["value"] == 12
        assert result["output"] == "Result: 12"

    async def test_conditional_workflow():
        def classifier(state):
            return {"category": "positive" if state.get("score", 0) > 5 else "negative"}

        def positive_handler(state):
            return {"result": "Great score!"}

        def negative_handler(state):
            return {"result": "Need improvement"}

        def route_by_category(state):
            return state.get("category", "negative")

        graph = StateGraph()
        graph.add_node("classify", classifier)
        graph.add_node("positive", positive_handler)
        graph.add_node("negative", negative_handler)

        graph.add_edge(START.value, "classify")
        graph.add_conditional_edges(
            "classify",
            route_by_category,
            {"positive": "positive", "negative": "negative"},
        )
        graph.add_edge("positive", END.value)
        graph.add_edge("negative", END.value)

        compiled = graph.compile()

        result1 = await compiled.ainvoke({"score": 8})
        print("Conditional (positive):", result1)
        assert result1["result"] == "Great score!"

        result2 = await compiled.ainvoke({"score": 3})
        print("Conditional (negative):", result2)
        assert result2["result"] == "Need improvement"

    async def test_loop_workflow():
        def increment(state):
            return {"counter": state.get("counter", 0) + 1}

        def should_continue(state):
            if state.get("counter", 0) >= 3:
                return END.value
            return "increment"

        graph = StateGraph()
        graph.add_node("increment", increment)

        graph.add_edge(START.value, "increment")
        graph.add_conditional_edges("increment", should_continue, {"increment": "increment"})

        compiled = graph.compile().with_config(max_iterations=10)
        result = await compiled.ainvoke({"counter": 0})

        print("Loop workflow result:", result)
        assert result["counter"] == 3

    async def test_reducer():
        graph = StateGraph()
        graph._channels = {
            "items": Channel(default=[], reducer=AppendReducer()),
            "total": Channel(default=0, reducer=AddReducer()),
        }

        def add_item(state):
            return {"items": ["item1"], "total": 10}

        def add_more(state):
            return {"items": ["item2", "item3"], "total": 20}

        graph.add_node("add_item", add_item)
        graph.add_node("add_more", add_more)

        graph.add_edge(START.value, "add_item")
        graph.add_edge("add_item", "add_more")
        graph.add_edge("add_more", END.value)

        compiled = graph.compile()
        result = await compiled.ainvoke({})

        print("Reducer result:", result)
        assert result["items"] == ["item1", "item2", "item3"]
        assert result["total"] == 30

    async def test_stream():
        def step1(state):
            return {"step": 1, "value": "first"}

        def step2(state):
            return {"step": 2, "value": "second"}

        def step3(state):
            return {"step": 3, "value": "third"}

        graph = StateGraph()
        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_node("step3", step3)

        graph.add_edge(START.value, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.add_edge("step3", END.value)

        compiled = graph.compile()

        print("Streaming:")
        async for node_id, state in compiled.astream({}):
            print(f"  Node: {node_id}, State: {state}")

    async def test_builder():
        def process(state):
            return {"output": f"Processed: {state.get('input', '')}"}

        def validate(state):
            return "success" if len(state.get("output", "")) > 5 else "retry"

        wf = (
            WorkflowBuilder("test")
            .add_function("process", process)
            .add_branch("process", validate, {"success": END.value, "retry": "process"})
            .set_entry("process")
            .build()
        )

        result = await wf.ainvoke({"input": "hello"})
        print("Builder result:", result)

    async def main():
        print("=" * 50)
        print("Testing Basic Workflow")
        print("=" * 50)
        await test_basic_workflow()

        print("\n" + "=" * 50)
        print("Testing Conditional Workflow")
        print("=" * 50)
        await test_conditional_workflow()

        print("\n" + "=" * 50)
        print("Testing Loop Workflow")
        print("=" * 50)
        await test_loop_workflow()

        print("\n" + "=" * 50)
        print("Testing Reducer")
        print("=" * 50)
        await test_reducer()

        print("\n" + "=" * 50)
        print("Testing Stream")
        print("=" * 50)
        await test_stream()

        print("\n" + "=" * 50)
        print("Testing Builder")
        print("=" * 50)
        await test_builder()

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    asyncio.run(main())
