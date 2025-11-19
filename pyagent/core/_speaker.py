from abc import ABC, abstractmethod
from typing import Optional
from ._model import ChatResponse


class Speaker(ABC):
    @abstractmethod
    def speak_stream_start(self, agent_name: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def speak_chunk(self, chunk: ChatResponse) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def speak_stream_end(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def speak_complete(self, response: ChatResponse, agent_name: str) -> None:
        raise NotImplementedError


class ConsoleSpeaker(Speaker):
    def speak_stream_start(self, agent_name: str) -> None:
        print(f"{agent_name}: ", end="", flush=True)
    
    def speak_chunk(self, chunk: ChatResponse) -> None:
        if chunk.reasoning_content:
            print(chunk.reasoning_content, end="", flush=True)
        if chunk.content:
            print(chunk.content, end="", flush=True)
    
    def speak_stream_end(self) -> None:
        print()
    
    def speak_complete(self, response: ChatResponse, agent_name: str) -> None:
        if response.reasoning_content:
            print(f"[Thinking: {response.reasoning_content}]")
        if response.content:
            print(f"[{agent_name}] {response.content}")


class SilentSpeaker(Speaker):
    def speak_stream_start(self, agent_name: str) -> None:
        pass
    
    def speak_chunk(self, chunk: ChatResponse) -> None:
        pass
    
    def speak_stream_end(self) -> None:
        pass
    
    def speak_complete(self, response: ChatResponse, agent_name: str) -> None:
        pass



