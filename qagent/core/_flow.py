import uuid
import asyncio
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Callable,
    Union,
    Any,
    AsyncGenerator,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from ._agent import Agent
    from ._model import ChatResponse

from ._trace import custom_span, get_current_trace


class FlowEnd:
    pass


END = FlowEnd()


@dataclass
class FlowContext:
    messages: Dict[str, "ChatResponse"] = field(default_factory=dict)
    history: List[tuple] = field(default_factory=list)
    current_node: str = ""
    loop_count: int = 0
    input_message: str = ""

    def last_response(self) -> Optional["ChatResponse"]:
        if not self.history:
            return None
        return self.history[-1][1]

    def response_from(self, node: str) -> Optional["ChatResponse"]:
        return self.messages.get(node)

    def all_responses(self) -> List["ChatResponse"]:
        return [r for _, r in self.history]


@dataclass
class Route:
    condition: Optional[Callable[["ChatResponse"], bool]] = None
    target: Union[str, FlowEnd] = ""
    is_default: bool = False


@dataclass
class NodeDef:
    name: str
    agent: Union["Agent", "Flow", List["Agent"]]
    is_parallel: bool = False


class WhenClause:
    def __init__(self, route_builder: "RouteBuilder", condition: Callable):
        self._route_builder = route_builder
        self._condition = condition

    def to(self, target: Union[str, FlowEnd]) -> "RouteBuilder":
        self._route_builder._routes.append(
            Route(condition=self._condition, target=target)
        )
        return self._route_builder


class DefaultClause:
    def __init__(self, route_builder: "RouteBuilder"):
        self._route_builder = route_builder

    def to(self, target: Union[str, FlowEnd]) -> "Flow":
        self._route_builder._routes.append(
            Route(condition=None, target=target, is_default=True)
        )
        self._route_builder._finalize()
        return self._route_builder._flow


class RouteBuilder:
    def __init__(self, flow: "Flow", from_node: str):
        self._flow = flow
        self._from_node = from_node
        self._routes: List[Route] = []

    def to(self, target: Union[str, FlowEnd]) -> "Flow":
        self._routes.append(Route(target=target))
        self._flow._routing[self._from_node] = self._routes
        return self._flow

    def when(self, condition: Callable[["ChatResponse"], bool]) -> WhenClause:
        return WhenClause(self, condition)

    def default(self) -> DefaultClause:
        return DefaultClause(self)

    def done(self) -> "Flow":
        self._finalize()
        return self._flow

    def _finalize(self):
        if self._routes:
            self._flow._routing[self._from_node] = self._routes


class Flow:
    def __init__(self, name: str = "flow"):
        self._name = name
        self._flow_id = str(uuid.uuid4())[:8]
        self._nodes: Dict[str, NodeDef] = {}
        self._node_order: List[str] = []
        self._routing: Dict[str, List[Route]] = {}
        self._entry_point: Optional[str] = None
        self._max_loops = 50
        self._current_route_builder: Optional[RouteBuilder] = None

    def add(
        self,
        name_or_agent: Union[str, "Agent"],
        agent: Optional["Agent"] = None,
    ) -> "Flow":
        self._finalize_route_builder()

        if agent is None:
            if hasattr(name_or_agent, "reply"):
                name = getattr(name_or_agent, "name", f"node_{len(self._nodes)}")
                agent = name_or_agent
            else:
                raise ValueError("Must provide agent")
        else:
            name = name_or_agent

        self._nodes[name] = NodeDef(name=name, agent=agent, is_parallel=False)
        self._node_order.append(name)

        if self._entry_point is None:
            self._entry_point = name

        return self

    def parallel(
        self,
        name: str,
        agents: List["Agent"],
    ) -> "Flow":
        self._finalize_route_builder()

        self._nodes[name] = NodeDef(name=name, agent=agents, is_parallel=True)
        self._node_order.append(name)

        if self._entry_point is None:
            self._entry_point = name

        return self

    def route(self, from_node: str) -> RouteBuilder:
        self._finalize_route_builder()
        self._current_route_builder = RouteBuilder(self, from_node)
        return self._current_route_builder

    def entry(self, node_name: str) -> "Flow":
        self._finalize_route_builder()
        self._entry_point = node_name
        return self

    def max_loops(self, n: int) -> "Flow":
        self._max_loops = n
        return self

    def _finalize_route_builder(self):
        if self._current_route_builder:
            self._current_route_builder._finalize()
            self._current_route_builder = None

    def _build_default_routes(self):
        for i, node_name in enumerate(self._node_order):
            if node_name not in self._routing:
                if i < len(self._node_order) - 1:
                    next_node = self._node_order[i + 1]
                    self._routing[node_name] = [Route(target=next_node)]
                else:
                    self._routing[node_name] = [Route(target=END)]

    def _get_next_node(
        self,
        current: str,
        response: "ChatResponse",
    ) -> Union[str, FlowEnd, None]:
        routes = self._routing.get(current, [])

        if not routes:
            idx = self._node_order.index(current) if current in self._node_order else -1
            if idx >= 0 and idx < len(self._node_order) - 1:
                return self._node_order[idx + 1]
            return END

        default_target = None

        for route in routes:
            if route.is_default:
                default_target = route.target
                continue

            if route.condition is None:
                return route.target

            try:
                if route.condition(response):
                    return route.target
            except Exception:
                continue

        return default_target if default_target is not None else END

    async def _execute_node(
        self,
        node_def: NodeDef,
        ctx: FlowContext,
    ) -> "ChatResponse":
        from ._model import ChatResponse

        last = ctx.last_response()
        input_msg = last.content if last else ctx.input_message

        if node_def.is_parallel:
            agents = node_def.agent
            tasks = [agent(input_msg) for agent in agents]
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]

            if valid_results:
                merged_content = "\n\n".join(
                    f"[{getattr(agents[i], 'name', f'Agent_{i}')}]\n{r.content}"
                    for i, r in enumerate(results) if r is not None
                )
                response = ChatResponse(role="assistant", content=merged_content)
            else:
                response = ChatResponse(role="assistant", content="")

        elif isinstance(node_def.agent, Flow):
            response = await node_def.agent.reply(input_msg)

        else:
            agent = node_def.agent
            response = await agent(input_msg)

        return response

    async def reply(
        self,
        message: str,
        stream: bool = False,
    ) -> "ChatResponse":
        from ._model import ChatResponse

        self._finalize_route_builder()
        self._build_default_routes()

        if not self._entry_point:
            raise ValueError("No entry point defined")

        ctx = FlowContext(input_message=message)
        current_node = self._entry_point
        iteration = 0

        trace = get_current_trace()

        while iteration < self._max_loops:
            iteration += 1
            ctx.loop_count = iteration

            node_def = self._nodes.get(current_node)
            if node_def is None:
                break

            ctx.current_node = current_node

            if trace:
                with custom_span(
                    name=f"flow:{self._name}:{current_node}",
                    data={"node": current_node, "iteration": iteration},
                ):
                    response = await self._execute_node(node_def, ctx)
            else:
                response = await self._execute_node(node_def, ctx)

            ctx.messages[current_node] = response
            ctx.history.append((current_node, response))

            next_node = self._get_next_node(current_node, response)

            if next_node is None or isinstance(next_node, FlowEnd):
                break

            current_node = next_node

        return ctx.last_response() or ChatResponse(role="assistant", content="")

    async def stream(
        self,
        message: str,
    ) -> AsyncGenerator["ChatResponse", None]:
        from ._model import ChatResponse

        self._finalize_route_builder()
        self._build_default_routes()

        if not self._entry_point:
            raise ValueError("No entry point defined")

        ctx = FlowContext(input_message=message)
        current_node = self._entry_point
        iteration = 0

        while iteration < self._max_loops:
            iteration += 1
            ctx.loop_count = iteration

            node_def = self._nodes.get(current_node)
            if node_def is None:
                break

            ctx.current_node = current_node

            if node_def.is_parallel:
                response = await self._execute_node(node_def, ctx)
                yield response
            elif isinstance(node_def.agent, Flow):
                response = await node_def.agent.reply(ctx.last_response().content if ctx.last_response() else message)
                yield response
            else:
                agent = node_def.agent
                last = ctx.last_response()
                input_msg = last.content if last else message
                response = await agent(input_msg, stream=True)
                yield response

            ctx.messages[current_node] = response
            ctx.history.append((current_node, response))

            next_node = self._get_next_node(current_node, response)

            if next_node is None or isinstance(next_node, FlowEnd):
                break

            current_node = next_node

    def __repr__(self) -> str:
        return f"Flow(name={self._name}, nodes={list(self._nodes.keys())})"


class SequentialFlow(Flow):
    def __init__(self, agents: List["Agent"], name: str = "sequential"):
        super().__init__(name)
        for agent in agents:
            agent_name = getattr(agent, "name", f"agent_{len(self._nodes)}")
            self.add(agent_name, agent)


class ParallelFlow(Flow):
    def __init__(
        self,
        agents: List["Agent"],
        name: str = "parallel",
    ):
        super().__init__(name)
        self.parallel("agents", agents)


def chain(*agents: "Agent") -> Flow:
    flow = Flow("chain")
    for agent in agents:
        flow.add(agent)
    return flow


if __name__ == "__main__":

    async def test_basic_flow():
        from ._model import ChatResponse

        class MockAgent:
            def __init__(self, name: str, response: str):
                self.name = name
                self._response = response

            async def reply(self, message: str, stream: bool = False):
                yield ChatResponse(role="assistant", content=f"{self.name}: {self._response}")

        agent_a = MockAgent("AgentA", "Step 1 done")
        agent_b = MockAgent("AgentB", "Step 2 done")
        agent_c = MockAgent("AgentC", "Step 3 done")

        flow = Flow("test").add("a", agent_a).add("b", agent_b).add("c", agent_c)

        result = await flow.reply("Start")
        print("Basic flow result:", result.content)
        assert "AgentC" in result.content

    async def test_conditional_flow():
        from ._model import ChatResponse

        class MockAgent:
            def __init__(self, name: str, response: str):
                self.name = name
                self._response = response

            async def reply(self, message: str, stream: bool = False):
                yield ChatResponse(role="assistant", content=self._response)

        planner = MockAgent("Planner", "Plan created")
        executor = MockAgent("Executor", "Executed")
        reviewer_approve = MockAgent("Reviewer", "APPROVED - looks good")

        flow = (
            Flow("review_flow")
            .add("plan", planner)
            .add("execute", executor)
            .add("review", reviewer_approve)
        )
        flow.route("plan").to("execute")
        flow.route("execute").to("review")
        flow.route("review").when(lambda r: "REJECTED" in r.content).to("plan").when(lambda r: "APPROVED" in r.content).to(END).default().to("plan")

        result = await flow.reply("Build app")
        print("Conditional flow result:", result.content)
        assert "APPROVED" in result.content

    async def test_loop_flow():
        from ._model import ChatResponse

        class CounterAgent:
            def __init__(self):
                self.name = "Counter"
                self.count = 0

            async def reply(self, message: str, stream: bool = False):
                self.count += 1
                status = "DONE" if self.count >= 3 else "CONTINUE"
                yield ChatResponse(role="assistant", content=f"Count: {self.count}, Status: {status}")

        counter = CounterAgent()

        flow = Flow("loop_flow").add("counter", counter).max_loops(10)
        flow.route("counter").when(lambda r: "DONE" in r.content).to(END).default().to("counter")

        result = await flow.reply("Start counting")
        print("Loop flow result:", result.content)
        assert "Count: 3" in result.content

    async def test_parallel_flow():
        from ._model import ChatResponse

        class MockAgent:
            def __init__(self, name: str, response: str):
                self.name = name
                self._response = response

            async def reply(self, message: str, stream: bool = False):
                yield ChatResponse(role="assistant", content=self._response)

        tech = MockAgent("Tech", "Technical analysis")
        biz = MockAgent("Business", "Business analysis")
        legal = MockAgent("Legal", "Legal analysis")

        flow = (
            Flow("parallel_test")
            .parallel("experts", [tech, biz, legal])
        )

        result = await flow.reply("Analyze this")
        print("Parallel flow result:", result.content)
        assert "Technical" in result.content
        assert "Business" in result.content
        assert "Legal" in result.content

    async def test_subflow():
        from ._model import ChatResponse

        class MockAgent:
            def __init__(self, name: str, response: str):
                self.name = name
                self._response = response

            async def reply(self, message: str, stream: bool = False):
                yield ChatResponse(role="assistant", content=f"{self.name}: {self._response}")

        inner_a = MockAgent("InnerA", "Inner step 1")
        inner_b = MockAgent("InnerB", "Inner step 2")
        outer_start = MockAgent("OuterStart", "Starting")
        outer_end = MockAgent("OuterEnd", "Finishing")

        inner_flow = Flow("inner").add("a", inner_a).add("b", inner_b)

        outer_flow = (
            Flow("outer")
            .add("start", outer_start)
            .add("inner", inner_flow)
            .add("end", outer_end)
        )

        result = await outer_flow.reply("Begin")
        print("Subflow result:", result.content)
        assert "OuterEnd" in result.content

    async def test_chain_helper():
        from ._model import ChatResponse

        class MockAgent:
            def __init__(self, name: str):
                self.name = name

            async def reply(self, message: str, stream: bool = False):
                yield ChatResponse(role="assistant", content=f"{self.name} processed")

        a = MockAgent("A")
        b = MockAgent("B")
        c = MockAgent("C")

        flow = chain(a, b, c)
        result = await flow.reply("Go")
        print("Chain helper result:", result.content)
        assert "C processed" in result.content

    async def main():
        print("=" * 50)
        print("Test: Basic Flow")
        print("=" * 50)
        await test_basic_flow()

        print("\n" + "=" * 50)
        print("Test: Conditional Flow")
        print("=" * 50)
        await test_conditional_flow()

        print("\n" + "=" * 50)
        print("Test: Loop Flow")
        print("=" * 50)
        await test_loop_flow()

        print("\n" + "=" * 50)
        print("Test: Parallel Flow")
        print("=" * 50)
        await test_parallel_flow()

        print("\n" + "=" * 50)
        print("Test: Subflow")
        print("=" * 50)
        await test_subflow()

        print("\n" + "=" * 50)
        print("Test: Chain Helper")
        print("=" * 50)
        await test_chain_helper()

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    asyncio.run(main())
