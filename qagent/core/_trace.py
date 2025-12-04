from __future__ import annotations

import abc
import contextvars
import json
import uuid
import threading
import atexit
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ._agent import BaseAgent

TSpanData = TypeVar("TSpanData", bound="SpanData")

_current_span: contextvars.ContextVar["Span[Any] | None"] = contextvars.ContextVar(
    "current_span", default=None
)
_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "current_trace", default=None
)


def gen_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex}"


def gen_span_id() -> str:
    return f"span_{uuid.uuid4().hex[:24]}"


def gen_group_id() -> str:
    return f"group_{uuid.uuid4().hex[:24]}"


def time_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SpanData(abc.ABC):
    @abc.abstractmethod
    def export(self) -> dict[str, Any]:
        pass

    @property
    @abc.abstractmethod
    def type(self) -> str:
        pass


class AgentSpanData(SpanData):
    __slots__ = (
        "agent_name",
        "agent_type",
        "agent_id",
        "tools",
        "user_input",
        "agent_output",
        "parent_agent_id",
        "handoff_from",
    )

    def __init__(
        self,
        agent_name: str,
        agent_type: str,
        tools: list[str] | None = None,
        user_input: str | None = None,
        agent_output: str | None = None,
        agent_id: str | None = None,
        parent_agent_id: str | None = None,
        handoff_from: str | None = None,
    ):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.tools = tools
        self.user_input = user_input
        self.agent_output = agent_output
        self.agent_id = agent_id
        self.parent_agent_id = parent_agent_id
        self.handoff_from = handoff_from

    @property
    def type(self) -> str:
        return "agent"

    def export(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.user_input:
            result["user_input"] = self.user_input
        if self.agent_output:
            result["agent_output"] = self.agent_output
        return result


class AgentStepSpanData(SpanData):
    __slots__ = ("step_index", "has_tool_calls", "tool_count", "status")

    def __init__(
        self,
        step_index: int | None = None,
        has_tool_calls: bool | None = None,
        tool_count: int | None = None,
        status: str | None = None,
    ):
        self.step_index = step_index
        self.has_tool_calls = has_tool_calls
        self.tool_count = tool_count
        self.status = status

    @property
    def type(self) -> str:
        return "agent_step"

    def export(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.step_index is not None:
            result["step_index"] = self.step_index
        if self.has_tool_calls is not None:
            result["has_tool_calls"] = self.has_tool_calls
        if self.tool_count is not None:
            result["tool_count"] = self.tool_count
        if self.status is not None:
            result["status"] = self.status
        return result


class GenerationSpanData(SpanData):
    __slots__ = ("model", "input_msgs", "output_msg", "usage", "params")

    def __init__(
        self,
        model: str | None = None,
        input_msgs: list[dict[str, Any]] | None = None,
        output_msg: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ):
        self.model = model
        self.input_msgs = input_msgs
        self.output_msg = output_msg
        self.usage = usage
        self.params = params

    @property
    def type(self) -> str:
        return "generation"

    def export(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.model:
            result["model"] = self.model
        if self.input_msgs is not None:
            result["input_msgs"] = self.input_msgs
        if self.output_msg is not None:
            result["output_msg"] = self.output_msg
        if self.usage is not None:
            result["usage"] = self.usage
        if self.params is not None:
            result["params"] = self.params
        return result


class ToolSpanData(SpanData):
    __slots__ = ("tool_name", "input_args", "output", "error")

    def __init__(
        self,
        tool_name: str,
        input_args: str | dict[str, Any] | None = None,
        output: Any | None = None,
        error: str | None = None,
    ):
        self.tool_name = tool_name
        self.input_args = input_args
        self.output = output
        self.error = error

    @property
    def type(self) -> str:
        return "tool"

    def export(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "tool_name": self.tool_name,
        }
        if self.input_args is not None:
            result["input_args"] = (
                self.input_args
                if isinstance(self.input_args, str)
                else json.dumps(self.input_args, ensure_ascii=False)
            )
        if self.output is not None:
            try:
                if isinstance(self.output, (str, int, float, bool, type(None), dict, list)):
                    result["output"] = self.output
                else:
                    result["output"] = str(self.output)
            except Exception:
                result["output"] = str(self.output)
        if self.error:
            result["error"] = self.error
        return result


class EmbedderSpanData(SpanData):
    __slots__ = ("model", "input_texts", "output_dimensions", "usage", "source", "error")

    def __init__(
        self,
        model: str | None = None,
        input_texts: list[str] | None = None,
        output_dimensions: int | None = None,
        usage: dict[str, Any] | None = None,
        source: str | None = None,
        error: str | None = None,
    ):
        self.model = model
        self.input_texts = input_texts
        self.output_dimensions = output_dimensions
        self.usage = usage
        self.source = source
        self.error = error

    @property
    def type(self) -> str:
        return "embedder"

    def export(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.model:
            result["model"] = self.model
        if self.input_texts is not None:
            result["input_count"] = len(self.input_texts)
            result["input_preview"] = [t[:100] for t in self.input_texts[:3]]
        if self.output_dimensions is not None:
            result["output_dimensions"] = self.output_dimensions
        if self.usage is not None:
            result["usage"] = self.usage
        if self.source:
            result["source"] = self.source
        if self.error:
            result["error"] = self.error
        return result


class CustomSpanData(SpanData):
    __slots__ = ("name", "data")

    def __init__(self, name: str, data: dict[str, Any] | None = None):
        self.name = name
        self.data = data or {}

    @property
    def type(self) -> str:
        return "custom"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "data": self.data,
        }


class SpanError:
    __slots__ = ("message", "data")

    def __init__(self, message: str, data: dict[str, Any] | None = None):
        self.message = message
        self.data = data

    def export(self) -> dict[str, Any]:
        return {"message": self.message, "data": self.data}


class TracingProcessor(abc.ABC):
    @abc.abstractmethod
    def on_trace_start(self, trace: "Trace") -> None:
        pass

    @abc.abstractmethod
    def on_trace_end(self, trace: "Trace") -> None:
        pass

    @abc.abstractmethod
    def on_span_start(self, span: "Span[Any]") -> None:
        pass

    @abc.abstractmethod
    def on_span_end(self, span: "Span[Any]") -> None:
        pass

    @abc.abstractmethod
    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


class ConsoleTracingProcessor(TracingProcessor):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._depth = 0

    def on_trace_start(self, trace: "Trace") -> None:
        if self.verbose:
            print(f"[Trace Start] {trace.trace_id} - {trace.name}")
        self._depth = 0

    def on_trace_end(self, trace: "Trace") -> None:
        if self.verbose:
            print(f"[Trace End] {trace.trace_id} - {trace.name}")

    def on_span_start(self, span: "Span[Any]") -> None:
        if self.verbose:
            indent = "  " * self._depth
            span_type = span.span_data.type
            span_info = self._get_span_info(span)
            print(f"{indent}[Span Start] {span_type} - {span_info}")
        self._depth += 1

    def on_span_end(self, span: "Span[Any]") -> None:
        self._depth = max(0, self._depth - 1)
        if self.verbose:
            indent = "  " * self._depth
            duration = self._calc_duration(span)
            print(f"{indent}[Span End] {span.span_data.type} ({duration}ms)")

    def _get_span_info(self, span: "Span[Any]") -> str:
        data = span.span_data
        if isinstance(data, AgentSpanData):
            return f"{data.agent_name}"
        elif isinstance(data, GenerationSpanData):
            return f"{data.model or 'unknown'}"
        elif isinstance(data, ToolSpanData):
            return f"{data.tool_name}"
        elif isinstance(data, CustomSpanData):
            return f"{data.name}"
        return ""

    def _calc_duration(self, span: "Span[Any]") -> int:
        if span.started_at and span.ended_at:
            start = datetime.fromisoformat(span.started_at)
            end = datetime.fromisoformat(span.ended_at)
            return int((end - start).total_seconds() * 1000)
        return 0

    def shutdown(self) -> None:
        pass


class MemoryTracingProcessor(TracingProcessor):
    def __init__(self):
        self.traces: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self.agents: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def on_trace_start(self, trace: "Trace") -> None:
        pass

    def on_trace_end(self, trace: "Trace") -> None:
        with self._lock:
            exported = trace.export()
            if exported:
                self.traces.append(exported)

    def on_span_start(self, span: "Span[Any]") -> None:
        pass

    def on_span_end(self, span: "Span[Any]") -> None:
        with self._lock:
            exported = span.export()
            if exported:
                self.spans.append(exported)

            span_data = span.span_data
            if isinstance(span_data, AgentSpanData) and span_data.agent_id:
                agent_id = span_data.agent_id
                if agent_id not in self.agents:
                    self.agents[agent_id] = {
                        "agent_name": span_data.agent_name,
                        "agent_type": span_data.agent_type,
                        "tools": span_data.tools,
                        "parent_agent_id": span_data.parent_agent_id,
                    }

    def shutdown(self) -> None:
        pass

    def get_traces(self) -> list[dict[str, Any]]:
        with self._lock:
            return self.traces.copy()

    def get_spans(self) -> list[dict[str, Any]]:
        with self._lock:
            return self.spans.copy()

    def get_agents(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return self.agents.copy()

    def clear(self) -> None:
        with self._lock:
            self.traces.clear()
            self.spans.clear()
            self.agents.clear()

    def export_to_json(self, filepath: str | Path) -> None:
        with self._lock:
            data = {
                "traces": self.traces,
                "spans": self.spans,
                "agents": self.agents,
            }
        Path(filepath).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class Trace(abc.ABC):
    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def group_id(self) -> str | None:
        pass

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False) -> None:
        pass

    @abc.abstractmethod
    def finish(self, reset_current: bool = False) -> None:
        pass

    @abc.abstractmethod
    def __enter__(self) -> "Trace":
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None:
        pass


class Span(abc.ABC, Generic[TSpanData]):
    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def span_id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def parent_id(self) -> str | None:
        pass

    @property
    @abc.abstractmethod
    def span_data(self) -> TSpanData:
        pass

    @property
    @abc.abstractmethod
    def started_at(self) -> str | None:
        pass

    @property
    @abc.abstractmethod
    def ended_at(self) -> str | None:
        pass

    @property
    @abc.abstractmethod
    def error(self) -> SpanError | None:
        pass

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False) -> None:
        pass

    @abc.abstractmethod
    def finish(self, reset_current: bool = False) -> None:
        pass

    @abc.abstractmethod
    def set_error(self, error: SpanError) -> None:
        pass

    @abc.abstractmethod
    def __enter__(self) -> "Span[TSpanData]":
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None:
        pass


class NoOpTrace(Trace):
    def __init__(self):
        self._prev_token: contextvars.Token[Trace | None] | None = None

    @property
    def trace_id(self) -> str:
        return "noop"

    @property
    def name(self) -> str:
        return "noop"

    @property
    def group_id(self) -> str | None:
        return None

    def start(self, mark_as_current: bool = False) -> None:
        if mark_as_current:
            self._prev_token = _current_trace.set(self)

    def finish(self, reset_current: bool = False) -> None:
        if reset_current and self._prev_token:
            _current_trace.reset(self._prev_token)
            self._prev_token = None

    def __enter__(self) -> "Trace":
        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish(reset_current=True)

    def export(self) -> dict[str, Any] | None:
        return None


class NoOpSpan(Span[TSpanData]):
    def __init__(self, span_data: TSpanData):
        self._span_data = span_data
        self._prev_token: contextvars.Token[Span[TSpanData] | None] | None = None

    @property
    def trace_id(self) -> str:
        return "noop"

    @property
    def span_id(self) -> str:
        return "noop"

    @property
    def parent_id(self) -> str | None:
        return None

    @property
    def span_data(self) -> TSpanData:
        return self._span_data

    @property
    def started_at(self) -> str | None:
        return None

    @property
    def ended_at(self) -> str | None:
        return None

    @property
    def error(self) -> SpanError | None:
        return None

    def start(self, mark_as_current: bool = False) -> None:
        if mark_as_current:
            self._prev_token = _current_span.set(self)

    def finish(self, reset_current: bool = False) -> None:
        if reset_current and self._prev_token:
            _current_span.reset(self._prev_token)
            self._prev_token = None

    def set_error(self, error: SpanError) -> None:
        pass

    def __enter__(self) -> "Span[TSpanData]":
        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish(reset_current=True)

    def export(self) -> dict[str, Any] | None:
        return None


class TraceImpl(Trace):
    __slots__ = (
        "_trace_id",
        "_name",
        "_group_id",
        "_metadata",
        "_processor",
        "_prev_token",
        "_started",
    )

    def __init__(
        self,
        name: str,
        trace_id: str | None,
        group_id: str | None,
        metadata: dict[str, Any] | None,
        processor: TracingProcessor,
    ):
        self._trace_id = trace_id or gen_trace_id()
        self._name = name
        self._group_id = group_id
        self._metadata = metadata or {}
        self._processor = processor
        self._prev_token: contextvars.Token[Trace | None] | None = None
        self._started = False

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def group_id(self) -> str | None:
        return self._group_id

    def start(self, mark_as_current: bool = False) -> None:
        if self._started:
            return
        self._started = True
        self._processor.on_trace_start(self)
        if mark_as_current:
            self._prev_token = _current_trace.set(self)

    def finish(self, reset_current: bool = False) -> None:
        if not self._started:
            return
        self._processor.on_trace_end(self)
        if reset_current and self._prev_token:
            _current_trace.reset(self._prev_token)
            self._prev_token = None

    def __enter__(self) -> "Trace":
        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish(reset_current=True)

    def export(self) -> dict[str, Any] | None:
        return {
            "object": "trace",
            "trace_id": self.trace_id,
            "name": self.name,
            "group_id": self.group_id,
            "metadata": self._metadata,
        }


class SpanImpl(Span[TSpanData]):
    __slots__ = (
        "_trace_id",
        "_span_id",
        "_parent_id",
        "_span_data",
        "_processor",
        "_started_at",
        "_ended_at",
        "_error",
        "_prev_token",
    )

    def __init__(
        self,
        trace_id: str,
        span_id: str | None,
        parent_id: str | None,
        span_data: TSpanData,
        processor: TracingProcessor,
    ):
        self._trace_id = trace_id
        self._span_id = span_id or gen_span_id()
        self._parent_id = parent_id
        self._span_data = span_data
        self._processor = processor
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._error: SpanError | None = None
        self._prev_token: contextvars.Token[Span[TSpanData] | None] | None = None

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def span_id(self) -> str:
        return self._span_id

    @property
    def parent_id(self) -> str | None:
        return self._parent_id

    @property
    def span_data(self) -> TSpanData:
        return self._span_data

    @property
    def started_at(self) -> str | None:
        return self._started_at

    @property
    def ended_at(self) -> str | None:
        return self._ended_at

    @property
    def error(self) -> SpanError | None:
        return self._error

    def start(self, mark_as_current: bool = False) -> None:
        if self._started_at:
            return
        self._started_at = time_iso()
        self._processor.on_span_start(self)
        if mark_as_current:
            self._prev_token = _current_span.set(self)

    def finish(self, reset_current: bool = False) -> None:
        if self._ended_at:
            return
        self._ended_at = time_iso()
        self._processor.on_span_end(self)
        if reset_current and self._prev_token:
            _current_span.reset(self._prev_token)
            self._prev_token = None

    def set_error(self, error: SpanError) -> None:
        self._error = error

    def __enter__(self) -> "Span[TSpanData]":
        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and exc_type is not GeneratorExit:
            self.set_error(SpanError(message=str(exc_val), data={"exc_type": exc_type.__name__}))
        self.finish(reset_current=True)

    def export(self) -> dict[str, Any] | None:
        return {
            "object": "span",
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "span_data": self.span_data.export(),
            "error": self.error.export() if self.error else None,
        }


class SynchronousMultiTracingProcessor(TracingProcessor):
    def __init__(self, processors: list[TracingProcessor] | None = None):
        self._processors: tuple[TracingProcessor, ...] = tuple(processors or [])
        self._lock = threading.Lock()

    def add_processor(self, processor: TracingProcessor) -> None:
        with self._lock:
            self._processors += (processor,)

    def set_processors(self, processors: list[TracingProcessor]) -> None:
        with self._lock:
            self._processors = tuple(processors)

    def on_trace_start(self, trace: Trace) -> None:
        for p in self._processors:
            try:
                p.on_trace_start(trace)
            except Exception:
                pass

    def on_trace_end(self, trace: Trace) -> None:
        for p in self._processors:
            try:
                p.on_trace_end(trace)
            except Exception:
                pass

    def on_span_start(self, span: Span[Any]) -> None:
        for p in self._processors:
            try:
                p.on_span_start(span)
            except Exception:
                pass

    def on_span_end(self, span: Span[Any]) -> None:
        for p in self._processors:
            try:
                p.on_span_end(span)
            except Exception:
                pass

    def shutdown(self) -> None:
        for p in self._processors:
            try:
                p.shutdown()
            except Exception:
                pass

    def force_flush(self) -> None:
        for p in self._processors:
            try:
                p.force_flush()
            except Exception:
                pass


class TraceProvider:
    def __init__(self, disabled: bool = False):
        self._processor = SynchronousMultiTracingProcessor()
        self._disabled = disabled

    def add_processor(self, processor: TracingProcessor) -> None:
        self._processor.add_processor(processor)

    def set_processors(self, processors: list[TracingProcessor]) -> None:
        self._processor.set_processors(processors)

    def set_disabled(self, disabled: bool) -> None:
        self._disabled = disabled

    def is_disabled(self) -> bool:
        return self._disabled

    def get_current_trace(self) -> Trace | None:
        return _current_trace.get()

    def get_current_span(self) -> Span[Any] | None:
        return _current_span.get()

    def create_trace(
        self,
        name: str,
        trace_id: str | None = None,
        group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        disabled: bool = False,
    ) -> Trace:
        if self._disabled or disabled:
            return NoOpTrace()
        return TraceImpl(name, trace_id, group_id, metadata, self._processor)

    def create_span(
        self,
        span_data: TSpanData,
        span_id: str | None = None,
        parent: Trace | Span[Any] | None = None,
        disabled: bool = False,
    ) -> Span[TSpanData]:
        if self._disabled or disabled:
            return NoOpSpan(span_data)

        if not parent:
            current_span = _current_span.get()
            current_trace = _current_trace.get()
            if not current_trace or isinstance(current_trace, NoOpTrace):
                return NoOpSpan(span_data)
            parent_id = (
                current_span.span_id
                if current_span and not isinstance(current_span, NoOpSpan)
                else None
            )
            trace_id = current_trace.trace_id
        elif isinstance(parent, Trace):
            if isinstance(parent, NoOpTrace):
                return NoOpSpan(span_data)
            trace_id = parent.trace_id
            parent_id = None
        elif isinstance(parent, Span):
            if isinstance(parent, NoOpSpan):
                return NoOpSpan(span_data)
            trace_id = parent.trace_id
            parent_id = parent.span_id
        else:
            return NoOpSpan(span_data)

        return SpanImpl(trace_id, span_id, parent_id, span_data, self._processor)

    def shutdown(self) -> None:
        self._processor.shutdown()

    def force_flush(self) -> None:
        self._processor.force_flush()


_global_trace_provider = TraceProvider()
_default_memory_processor = MemoryTracingProcessor()


def get_trace_provider() -> TraceProvider:
    return _global_trace_provider


def set_trace_provider(provider: TraceProvider) -> None:
    global _global_trace_provider
    _global_trace_provider = provider


def get_default_memory_processor() -> MemoryTracingProcessor:
    return _default_memory_processor


_global_trace_provider.add_processor(_default_memory_processor)


def trace(
    name: str,
    trace_id: str | None = None,
    group_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    disabled: bool = False,
) -> Trace:
    return get_trace_provider().create_trace(name, trace_id, group_id, metadata, disabled)


def agent_span(
    agent_name: str,
    agent_type: str,
    tools: list[str] | None = None,
    user_input: str | None = None,
    agent_output: str | None = None,
    agent_id: str | None = None,
    parent_agent_id: str | None = None,
    handoff_from: str | None = None,
    disabled: bool = False,
) -> Span[AgentSpanData]:
    return get_trace_provider().create_span(
        AgentSpanData(
            agent_name,
            agent_type,
            tools,
            user_input,
            agent_output,
            agent_id,
            parent_agent_id,
            handoff_from,
        ),
        disabled=disabled,
    )


def agent_step_span(
    step_index: int | None = None,
    has_tool_calls: bool | None = None,
    tool_count: int | None = None,
    status: str | None = None,
    disabled: bool = False,
) -> Span[AgentStepSpanData]:
    return get_trace_provider().create_span(
        AgentStepSpanData(step_index, has_tool_calls, tool_count, status),
        disabled=disabled,
    )


def generation_span(
    model: str | None = None,
    input_msgs: list[dict[str, Any]] | None = None,
    output_msg: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    disabled: bool = False,
) -> Span[GenerationSpanData]:
    return get_trace_provider().create_span(
        GenerationSpanData(model, input_msgs, output_msg, usage, params),
        disabled=disabled,
    )


def tool_span(
    tool_name: str,
    input_args: str | dict[str, Any] | None = None,
    output: Any | None = None,
    error: str | None = None,
    disabled: bool = False,
) -> Span[ToolSpanData]:
    return get_trace_provider().create_span(
        ToolSpanData(tool_name, input_args, output, error),
        disabled=disabled,
    )


def embedder_span(
    model: str | None = None,
    input_texts: list[str] | None = None,
    output_dimensions: int | None = None,
    usage: dict[str, Any] | None = None,
    source: str | None = None,
    error: str | None = None,
    disabled: bool = False,
) -> Span[EmbedderSpanData]:
    return get_trace_provider().create_span(
        EmbedderSpanData(model, input_texts, output_dimensions, usage, source, error),
        disabled=disabled,
    )


def custom_span(
    name: str,
    data: dict[str, Any] | None = None,
    disabled: bool = False,
) -> Span[CustomSpanData]:
    return get_trace_provider().create_span(
        CustomSpanData(name, data),
        disabled=disabled,
    )


def get_current_trace() -> Trace | None:
    return get_trace_provider().get_current_trace()


def get_current_span() -> Span[Any] | None:
    return get_trace_provider().get_current_span()


def export_traces(filepath: str | Path) -> None:
    _default_memory_processor.export_to_json(filepath)


def get_all_traces() -> list[dict[str, Any]]:
    return _default_memory_processor.get_traces()


def get_all_spans() -> list[dict[str, Any]]:
    return _default_memory_processor.get_spans()


def get_all_agents() -> dict[str, dict[str, Any]]:
    return _default_memory_processor.get_agents()


def clear_traces() -> None:
    _default_memory_processor.clear()


def enable_console_output(verbose: bool = True) -> None:
    console_processor = ConsoleTracingProcessor(verbose=verbose)
    get_trace_provider().add_processor(console_processor)


def disable_tracing() -> None:
    get_trace_provider().set_disabled(True)


def enable_tracing() -> None:
    get_trace_provider().set_disabled(False)


def shutdown_tracing() -> None:
    get_trace_provider().shutdown()


atexit.register(shutdown_tracing)
