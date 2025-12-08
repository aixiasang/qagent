"""
Tests for ReActAgent type imports and initialization.

This test verifies that the LogConfig TYPE_CHECKING import fix works correctly.
"""

import pytest
from typing import get_type_hints
from unittest.mock import Mock, AsyncMock


class TestReActAgentTypeImports:
    """Test type imports in ReActAgent."""

    def test_import_react_agent(self):
        """Test that ReActAgent can be imported without errors."""
        from qagent.agent._react_agent import ReActAgent
        assert ReActAgent is not None

    def test_import_classic_react_agent(self):
        """Test that ClassicReActAgent can be imported without errors."""
        from qagent.agent._react_agent import ClassicReActAgent
        assert ClassicReActAgent is not None

    def test_type_checking_import_exists(self):
        """Verify TYPE_CHECKING import is present."""
        import qagent.agent._react_agent as module
        
        # Check that TYPE_CHECKING is used
        source = open(module.__file__, 'r', encoding='utf-8').read()
        assert 'TYPE_CHECKING' in source
        assert 'if TYPE_CHECKING:' in source
        assert 'from ..core._agent import LogConfig' in source

    def test_react_agent_init_signature(self):
        """Test ReActAgent __init__ accepts log_config parameter."""
        from qagent.agent._react_agent import ReActAgent
        import inspect
        
        sig = inspect.signature(ReActAgent.__init__)
        params = list(sig.parameters.keys())
        
        assert 'log_config' in params

    def test_react_agent_init_with_none_log_config(self):
        """Test ReActAgent can be instantiated with log_config=None."""
        from qagent.agent._react_agent import ReActAgent
        from qagent.core._model import ChaterCfg, ClientCfg, ChatCfg, Chater
        
        # Create a mock chater
        mock_chater = Mock(spec=Chater)
        mock_chater.chat = AsyncMock()
        
        # Should not raise any error
        agent = ReActAgent(
            chater=mock_chater,
            name="TestReAct",
            log_config=None,
        )
        
        assert agent.name == "TestReAct"


class TestReActStep:
    """Test ReActStep dataclass."""

    def test_react_step_creation(self):
        """Test ReActStep can be created."""
        from qagent.agent._react_agent import ReActStep
        
        step = ReActStep(
            iteration=0,
            thought="I need to search",
            action="search",
            action_input={"query": "test"},
            observation="Found results",
            is_final=False,
        )
        
        assert step.iteration == 0
        assert step.thought == "I need to search"
        assert step.action == "search"
        assert step.action_input == {"query": "test"}
        assert step.observation == "Found results"
        assert step.is_final is False

    def test_react_step_defaults(self):
        """Test ReActStep default values."""
        from qagent.agent._react_agent import ReActStep
        
        step = ReActStep(iteration=1)
        
        assert step.iteration == 1
        assert step.thought is None
        assert step.action is None
        assert step.action_input is None
        assert step.observation is None
        assert step.is_final is False


class TestReActTrace:
    """Test ReActTrace dataclass."""

    def test_react_trace_creation(self):
        """Test ReActTrace can be created."""
        from qagent.agent._react_agent import ReActTrace
        
        trace = ReActTrace()
        
        assert trace.steps == []
        assert trace.final_answer is None
        assert trace.total_iterations == 0

    def test_react_trace_add_step(self):
        """Test adding steps to ReActTrace."""
        from qagent.agent._react_agent import ReActTrace, ReActStep
        
        trace = ReActTrace()
        
        step1 = ReActStep(iteration=0, thought="First thought")
        trace.add_step(step1)
        
        assert len(trace.steps) == 1
        assert trace.total_iterations == 1
        
        step2 = ReActStep(iteration=1, thought="Second thought")
        trace.add_step(step2)
        
        assert len(trace.steps) == 2
        assert trace.total_iterations == 2

    def test_react_trace_to_dict(self):
        """Test ReActTrace serialization."""
        from qagent.agent._react_agent import ReActTrace, ReActStep
        
        trace = ReActTrace()
        trace.add_step(ReActStep(iteration=0, thought="Think"))
        trace.final_answer = "42"
        
        result = trace.to_dict()
        
        assert result["total_iterations"] == 1
        assert len(result["steps"]) == 1
        assert result["steps"][0]["thought"] == "Think"
        assert result["final_answer"] == "42"

    def test_react_trace_clear(self):
        """Test clearing ReActTrace."""
        from qagent.agent._react_agent import ReActTrace, ReActStep
        
        trace = ReActTrace()
        trace.add_step(ReActStep(iteration=0))
        trace.final_answer = "answer"
        
        trace.clear()
        
        assert trace.steps == []
        assert trace.final_answer is None
        assert trace.total_iterations == 0


class TestClassicReActAgent:
    """Test ClassicReActAgent specific functionality."""

    def test_classic_react_system_prompt(self):
        """Test CLASSIC_REACT_SYSTEM prompt template."""
        from qagent.agent._react_agent import CLASSIC_REACT_SYSTEM
        
        assert "<Thought>" in CLASSIC_REACT_SYSTEM
        assert "<Action>" in CLASSIC_REACT_SYSTEM
        assert "<ActionInput>" in CLASSIC_REACT_SYSTEM
        assert "<FinalAnswer>" in CLASSIC_REACT_SYSTEM
        assert "{tools}" in CLASSIC_REACT_SYSTEM

    def test_classic_react_agent_init(self):
        """Test ClassicReActAgent initialization."""
        from qagent.agent._react_agent import ClassicReActAgent
        from qagent.core._model import Chater
        from unittest.mock import Mock, AsyncMock
        
        mock_chater = Mock(spec=Chater)
        mock_chater.chat = AsyncMock()
        
        agent = ClassicReActAgent(
            chater=mock_chater,
            name="ClassicReAct",
        )
        
        assert agent.name == "ClassicReAct"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
