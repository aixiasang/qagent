from typing import Optional, Union
from ._agent import Agent
from ._model import ChatResponse
from ._trace import custom_span, get_current_trace, get_current_span


class MsgHub:
    def __init__(
        self,
        participants: list[Agent],
        announcement: Optional[Union[ChatResponse, list[ChatResponse]]] = None,
    ):
        self.participants = participants
        self.announcement = announcement
        self._hub_span = None

    def __enter__(self) -> "MsgHub":
        for agent in self.participants:
            agent._audience = [a for a in self.participants if a != agent]

        current_trace = get_current_trace()
        if current_trace:
            current_span = get_current_span()
            if not (current_span and hasattr(current_span.span_data, "name")):
                self._hub_span = custom_span(
                    name="msghub_context",
                    data={
                        "participants": [a.name for a in self.participants],
                        "has_announcement": self.announcement is not None,
                    },
                )
                self._hub_span.__enter__()

        if self.announcement:
            self.broadcast(self.announcement)

        return self

    def __exit__(self, *args, **kwargs) -> None:
        if self._hub_span:
            self._hub_span.__exit__(*args)

        for agent in self.participants:
            agent._audience = None

    def add(self, agent: Union[Agent, list[Agent]]) -> None:
        if isinstance(agent, list):
            for a in agent:
                if a not in self.participants:
                    self.participants.append(a)
        else:
            if agent not in self.participants:
                self.participants.append(agent)

        self._reset_audience()

    def remove(self, agent: Union[Agent, list[Agent]]) -> None:
        if isinstance(agent, list):
            for a in agent:
                if a in self.participants:
                    self.participants.remove(a)
                    a._audience = None
        else:
            if agent in self.participants:
                self.participants.remove(agent)
                agent._audience = None

        self._reset_audience()

    def broadcast(self, msg: Union[ChatResponse, list[ChatResponse]]) -> None:
        for agent in self.participants:
            agent.observe(msg)

    def _reset_audience(self) -> None:
        for agent in self.participants:
            agent._audience = [a for a in self.participants if a != agent]


def msghub(
    participants: list[Agent],
    announcement: Optional[Union[ChatResponse, list[ChatResponse]]] = None,
) -> MsgHub:
    return MsgHub(participants, announcement)
