"""
Agent hooking system for non-invasive instrumentation.

This module provides mechanisms to hook into agent operations
without modifying the agent code itself.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar, ParamSpec
from weakref import WeakValueDictionary

from agenttrace.core.node import NodeType
from agenttrace.core.edge import Edge, EdgeType

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class HookContext:
    """Context information passed to hooks."""

    agent_id: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    parent_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Hook(ABC):
    """Base class for all hooks."""

    @abstractmethod
    def on_enter(self, ctx: HookContext, *args: Any, **kwargs: Any) -> None:
        """Called before the hooked operation."""
        pass

    @abstractmethod
    def on_exit(self, ctx: HookContext, result: Any, error: Exception | None, *args: Any, **kwargs: Any) -> None:
        """Called after the hooked operation."""
        pass


class AgentHook(Hook):
    """
    Hook for agent operations.

    Captures agent inputs, outputs, and intermediate decisions.
    """

    def __init__(
        self,
        tracer: "Tracer",  # Forward reference
        capture_inputs: bool = True,
        capture_outputs: bool = True,
        capture_errors: bool = True,
    ):
        self.tracer = tracer
        self.capture_inputs = capture_inputs
        self.capture_outputs = capture_outputs
        self.capture_errors = capture_errors
        self._node_stack: dict[int, list[str]] = {}  # thread_id -> stack of node_ids

    def _get_stack(self) -> list[str]:
        """Get the node stack for the current thread."""
        thread_id = threading.get_ident()
        if thread_id not in self._node_stack:
            self._node_stack[thread_id] = []
        return self._node_stack[thread_id]

    def on_enter(self, ctx: HookContext, *args: Any, **kwargs: Any) -> str | None:
        """Record the input to an agent operation."""
        if not self.capture_inputs:
            return None

        stack = self._get_stack()
        parent_id = stack[-1] if stack else ctx.parent_node_id

        # Create input node
        input_data = {
            "args": self._serialize_args(args),
            "kwargs": self._serialize_kwargs(kwargs),
        }

        node = self.tracer.record(
            node_type=NodeType.AGENT_INPUT,
            agent_id=ctx.agent_id,
            data=input_data,
            metadata={
                "operation": ctx.operation,
                "timestamp": ctx.timestamp.isoformat(),
            },
            parent_ids=[parent_id] if parent_id else [],
        )

        stack.append(node.id)
        return node.id

    def on_exit(self, ctx: HookContext, result: Any, error: Exception | None, *args: Any, **kwargs: Any) -> str | None:
        """Record the output from an agent operation."""
        stack = self._get_stack()
        input_node_id = stack.pop() if stack else None

        if error and self.capture_errors:
            # Record error node
            node = self.tracer.record(
                node_type=NodeType.ERROR,
                agent_id=ctx.agent_id,
                data={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                metadata={
                    "operation": ctx.operation,
                    "timestamp": datetime.now().isoformat(),
                },
                parent_ids=[input_node_id] if input_node_id else [],
            )
            return node.id

        if not self.capture_outputs:
            return None

        # Create output node
        node = self.tracer.record(
            node_type=NodeType.AGENT_OUTPUT,
            agent_id=ctx.agent_id,
            data=self._serialize_result(result),
            metadata={
                "operation": ctx.operation,
                "timestamp": datetime.now().isoformat(),
            },
            parent_ids=[input_node_id] if input_node_id else [],
        )

        # Create input->output edge
        if input_node_id:
            self.tracer.graph.add_edge(
                Edge(
                    source_id=input_node_id,
                    target_id=node.id,
                    type=EdgeType.INPUT_OUTPUT,
                )
            )

        return node.id

    def _serialize_args(self, args: tuple) -> list:
        """Serialize function arguments."""
        return [self._safe_serialize(arg) for arg in args]

    def _serialize_kwargs(self, kwargs: dict) -> dict:
        """Serialize keyword arguments."""
        return {k: self._safe_serialize(v) for k, v in kwargs.items()}

    def _serialize_result(self, result: Any) -> Any:
        """Serialize function result."""
        return self._safe_serialize(result)

    def _safe_serialize(self, value: Any) -> Any:
        """Safely serialize a value, handling non-serializable types."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._safe_serialize(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._safe_serialize(v) for k, v in value.items()}
        # For other types, try repr or str
        try:
            return repr(value)
        except Exception:
            return f"<{type(value).__name__}>"


class ToolHook(AgentHook):
    """
    Hook specifically for tool calls.

    Extends AgentHook with tool-specific tracking.
    """

    def on_enter(self, ctx: HookContext, *args: Any, **kwargs: Any) -> str | None:
        """Record tool invocation."""
        if not self.capture_inputs:
            return None

        stack = self._get_stack()
        parent_id = stack[-1] if stack else ctx.parent_node_id

        input_data = {
            "tool_name": ctx.operation,
            "args": self._serialize_args(args),
            "kwargs": self._serialize_kwargs(kwargs),
        }

        node = self.tracer.record(
            node_type=NodeType.TOOL_CALL,
            agent_id=ctx.agent_id,
            data=input_data,
            metadata={
                "tool_name": ctx.operation,
                "timestamp": ctx.timestamp.isoformat(),
            },
            parent_ids=[parent_id] if parent_id else [],
        )

        stack.append(node.id)
        return node.id

    def on_exit(self, ctx: HookContext, result: Any, error: Exception | None, *args: Any, **kwargs: Any) -> str | None:
        """Record tool result."""
        stack = self._get_stack()
        call_node_id = stack.pop() if stack else None

        if error and self.capture_errors:
            node = self.tracer.record(
                node_type=NodeType.ERROR,
                agent_id=ctx.agent_id,
                data={
                    "tool_name": ctx.operation,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                metadata={
                    "tool_name": ctx.operation,
                    "timestamp": datetime.now().isoformat(),
                },
                parent_ids=[call_node_id] if call_node_id else [],
            )
            return node.id

        if not self.capture_outputs:
            return None

        node = self.tracer.record(
            node_type=NodeType.TOOL_RESULT,
            agent_id=ctx.agent_id,
            data={
                "tool_name": ctx.operation,
                "result": self._serialize_result(result),
            },
            metadata={
                "tool_name": ctx.operation,
                "timestamp": datetime.now().isoformat(),
            },
            parent_ids=[call_node_id] if call_node_id else [],
        )

        if call_node_id:
            self.tracer.graph.add_edge(
                Edge(
                    source_id=call_node_id,
                    target_id=node.id,
                    type=EdgeType.TOOL_INVOCATION,
                )
            )

        return node.id


class HookManager:
    """
    Manages hooks for multiple agents and operations.

    Provides a central registry for all hooks and handles
    hook lifecycle management.
    """

    _instance: "HookManager | None" = None

    def __new__(cls) -> "HookManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._hooks: dict[str, list[Hook]] = {}  # agent_id -> hooks
        self._global_hooks: list[Hook] = []
        self._agents: WeakValueDictionary = WeakValueDictionary()
        self._initialized = True

    def register_hook(self, hook: Hook, agent_id: str | None = None) -> None:
        """
        Register a hook for an agent or globally.

        Args:
            hook: The hook to register
            agent_id: If provided, hook only applies to this agent.
                      If None, applies globally to all agents.
        """
        if agent_id is None:
            self._global_hooks.append(hook)
        else:
            if agent_id not in self._hooks:
                self._hooks[agent_id] = []
            self._hooks[agent_id].append(hook)

    def unregister_hook(self, hook: Hook, agent_id: str | None = None) -> None:
        """Unregister a hook."""
        if agent_id is None:
            if hook in self._global_hooks:
                self._global_hooks.remove(hook)
        else:
            if agent_id in self._hooks and hook in self._hooks[agent_id]:
                self._hooks[agent_id].remove(hook)

    def get_hooks(self, agent_id: str) -> list[Hook]:
        """Get all hooks applicable to an agent."""
        return self._global_hooks + self._hooks.get(agent_id, [])

    @contextmanager
    def hook_context(self, ctx: HookContext, *args: Any, **kwargs: Any):
        """
        Context manager for executing hooks around an operation.

        Usage:
            with hook_manager.hook_context(ctx, arg1, arg2) as node_ids:
                result = original_operation(arg1, arg2)
            # hooks automatically called on exit
        """
        hooks = self.get_hooks(ctx.agent_id)
        node_ids = []

        # Call on_enter for all hooks
        for hook in hooks:
            try:
                node_id = hook.on_enter(ctx, *args, **kwargs)
                if node_id:
                    node_ids.append(node_id)
            except Exception:
                pass  # Don't let hook errors affect the main operation

        error = None
        result = None
        try:
            yield node_ids
        except Exception as e:
            error = e
            raise
        finally:
            # Call on_exit for all hooks
            for hook in reversed(hooks):
                try:
                    hook.on_exit(ctx, result, error, *args, **kwargs)
                except Exception:
                    pass

    def clear(self) -> None:
        """Clear all registered hooks."""
        self._hooks.clear()
        self._global_hooks.clear()


# Alias for backward reference
from agenttrace.tracer import Tracer  # noqa: E402
