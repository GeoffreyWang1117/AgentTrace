"""
Decorators for easy agent and tool tracing.

These decorators provide a simple way to add tracing to
agent methods and tool functions without modifying their logic.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable, TypeVar, ParamSpec

from agenttrace.core.node import NodeType
from agenttrace.core.edge import Edge, EdgeType
from agenttrace.hooks.agent_hook import HookContext

P = ParamSpec("P")
T = TypeVar("T")


def _get_tracer():
    """Get the global tracer instance."""
    from agenttrace.tracer import get_tracer

    return get_tracer()


def trace_agent(
    agent_id: str | None = None,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to trace agent method calls.

    Usage:
        class MyAgent:
            @trace_agent(agent_id="my_agent")
            def process(self, input_data):
                return result

        # Or with auto-detected agent_id:
        class MyAgent:
            agent_id = "my_agent"

            @trace_agent()
            def process(self, input_data):
                return result

    Args:
        agent_id: Explicit agent ID, or auto-detect from self.agent_id
        capture_inputs: Whether to capture input arguments
        capture_outputs: Whether to capture return values
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = _get_tracer()
            if tracer is None:
                return func(*args, **kwargs)

            # Determine agent_id
            resolved_agent_id = agent_id
            if resolved_agent_id is None and args:
                # Try to get from self
                self = args[0]
                resolved_agent_id = getattr(self, "agent_id", None)
            if resolved_agent_id is None:
                resolved_agent_id = func.__qualname__

            _ctx = HookContext(
                agent_id=resolved_agent_id,
                operation=func.__name__,
                parent_node_id=tracer.current_node_id,
            )

            input_node_id = None
            if capture_inputs:
                input_data = _capture_inputs(args, kwargs, func)
                input_node = tracer.record(
                    node_type=NodeType.AGENT_INPUT,
                    agent_id=resolved_agent_id,
                    data=input_data,
                    metadata={"operation": func.__name__},
                    parent_ids=[tracer.current_node_id] if tracer.current_node_id else [],
                )
                input_node_id = input_node.id

            # Set current context
            prev_node_id = tracer.current_node_id
            tracer.current_node_id = input_node_id

            try:
                result = func(*args, **kwargs)

                if capture_outputs:
                    output_node = tracer.record(
                        node_type=NodeType.AGENT_OUTPUT,
                        agent_id=resolved_agent_id,
                        data=_safe_serialize(result),
                        metadata={"operation": func.__name__},
                        parent_ids=[input_node_id] if input_node_id else [],
                    )
                    if input_node_id:
                        tracer.graph.add_edge(
                            Edge(
                                source_id=input_node_id,
                                target_id=output_node.id,
                                type=EdgeType.INPUT_OUTPUT,
                            )
                        )
                    tracer.current_node_id = output_node.id

                return result

            except Exception as e:
                error_node = tracer.record(
                    node_type=NodeType.ERROR,
                    agent_id=resolved_agent_id,
                    data={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    metadata={"operation": func.__name__},
                    parent_ids=[input_node_id] if input_node_id else [],
                )
                if input_node_id:
                    tracer.graph.add_edge(
                        Edge(
                            source_id=input_node_id,
                            target_id=error_node.id,
                            type=EdgeType.ERROR_PROPAGATION,
                        )
                    )
                raise

            finally:
                tracer.current_node_id = prev_node_id

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = _get_tracer()
            if tracer is None:
                return await func(*args, **kwargs)

            resolved_agent_id = agent_id
            if resolved_agent_id is None and args:
                self = args[0]
                resolved_agent_id = getattr(self, "agent_id", None)
            if resolved_agent_id is None:
                resolved_agent_id = func.__qualname__

            input_node_id = None
            if capture_inputs:
                input_data = _capture_inputs(args, kwargs, func)
                input_node = tracer.record(
                    node_type=NodeType.AGENT_INPUT,
                    agent_id=resolved_agent_id,
                    data=input_data,
                    metadata={"operation": func.__name__},
                    parent_ids=[tracer.current_node_id] if tracer.current_node_id else [],
                )
                input_node_id = input_node.id

            prev_node_id = tracer.current_node_id
            tracer.current_node_id = input_node_id

            try:
                result = await func(*args, **kwargs)

                if capture_outputs:
                    output_node = tracer.record(
                        node_type=NodeType.AGENT_OUTPUT,
                        agent_id=resolved_agent_id,
                        data=_safe_serialize(result),
                        metadata={"operation": func.__name__},
                        parent_ids=[input_node_id] if input_node_id else [],
                    )
                    if input_node_id:
                        tracer.graph.add_edge(
                            Edge(
                                source_id=input_node_id,
                                target_id=output_node.id,
                                type=EdgeType.INPUT_OUTPUT,
                            )
                        )
                    tracer.current_node_id = output_node.id

                return result

            except Exception as e:
                error_node = tracer.record(
                    node_type=NodeType.ERROR,
                    agent_id=resolved_agent_id,
                    data={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    metadata={"operation": func.__name__},
                    parent_ids=[input_node_id] if input_node_id else [],
                )
                if input_node_id:
                    tracer.graph.add_edge(
                        Edge(
                            source_id=input_node_id,
                            target_id=error_node.id,
                            type=EdgeType.ERROR_PROPAGATION,
                        )
                    )
                raise

            finally:
                tracer.current_node_id = prev_node_id

        return async_wrapper if is_async else sync_wrapper

    return decorator


def trace_tool(
    tool_name: str | None = None,
    agent_id: str = "tools",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to trace tool function calls.

    Usage:
        @trace_tool(tool_name="web_search")
        def search_web(query: str) -> list[str]:
            return results

    Args:
        tool_name: Name of the tool, defaults to function name
        agent_id: Agent ID to associate with tool calls
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        resolved_name = tool_name or func.__name__
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = _get_tracer()
            if tracer is None:
                return func(*args, **kwargs)

            # Record tool call
            call_node = tracer.record(
                node_type=NodeType.TOOL_CALL,
                agent_id=agent_id,
                data={
                    "tool_name": resolved_name,
                    "args": [_safe_serialize(a) for a in args],
                    "kwargs": {k: _safe_serialize(v) for k, v in kwargs.items()},
                },
                metadata={"tool_name": resolved_name},
                parent_ids=[tracer.current_node_id] if tracer.current_node_id else [],
            )

            try:
                result = func(*args, **kwargs)

                # Record tool result
                result_node = tracer.record(
                    node_type=NodeType.TOOL_RESULT,
                    agent_id=agent_id,
                    data={
                        "tool_name": resolved_name,
                        "result": _safe_serialize(result),
                    },
                    metadata={"tool_name": resolved_name},
                    parent_ids=[call_node.id],
                )

                tracer.graph.add_edge(
                    Edge(
                        source_id=call_node.id,
                        target_id=result_node.id,
                        type=EdgeType.TOOL_INVOCATION,
                    )
                )

                return result

            except Exception as e:
                error_node = tracer.record(
                    node_type=NodeType.ERROR,
                    agent_id=agent_id,
                    data={
                        "tool_name": resolved_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    metadata={"tool_name": resolved_name},
                    parent_ids=[call_node.id],
                )

                tracer.graph.add_edge(
                    Edge(
                        source_id=call_node.id,
                        target_id=error_node.id,
                        type=EdgeType.ERROR_PROPAGATION,
                    )
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = _get_tracer()
            if tracer is None:
                return await func(*args, **kwargs)

            call_node = tracer.record(
                node_type=NodeType.TOOL_CALL,
                agent_id=agent_id,
                data={
                    "tool_name": resolved_name,
                    "args": [_safe_serialize(a) for a in args],
                    "kwargs": {k: _safe_serialize(v) for k, v in kwargs.items()},
                },
                metadata={"tool_name": resolved_name},
                parent_ids=[tracer.current_node_id] if tracer.current_node_id else [],
            )

            try:
                result = await func(*args, **kwargs)

                result_node = tracer.record(
                    node_type=NodeType.TOOL_RESULT,
                    agent_id=agent_id,
                    data={
                        "tool_name": resolved_name,
                        "result": _safe_serialize(result),
                    },
                    metadata={"tool_name": resolved_name},
                    parent_ids=[call_node.id],
                )

                tracer.graph.add_edge(
                    Edge(
                        source_id=call_node.id,
                        target_id=result_node.id,
                        type=EdgeType.TOOL_INVOCATION,
                    )
                )

                return result

            except Exception as e:
                error_node = tracer.record(
                    node_type=NodeType.ERROR,
                    agent_id=agent_id,
                    data={
                        "tool_name": resolved_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    metadata={"tool_name": resolved_name},
                    parent_ids=[call_node.id],
                )

                tracer.graph.add_edge(
                    Edge(
                        source_id=call_node.id,
                        target_id=error_node.id,
                        type=EdgeType.ERROR_PROPAGATION,
                    )
                )
                raise

        return async_wrapper if is_async else sync_wrapper

    return decorator


def trace_state(
    state_name: str,
    agent_id: str = "state",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to trace state access (reads and writes).

    Usage:
        class StateManager:
            @trace_state("user_preferences")
            def get_preferences(self, user_id: str) -> dict:
                return self._preferences[user_id]

            @trace_state("user_preferences")
            def set_preferences(self, user_id: str, prefs: dict) -> None:
                self._preferences[user_id] = prefs

    Args:
        state_name: Name of the state being accessed
        agent_id: Agent ID to associate with state access
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Heuristic: if function name starts with get/read, it's a read
        is_read = func.__name__.startswith(("get", "read", "load", "fetch"))

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = _get_tracer()
            if tracer is None:
                return func(*args, **kwargs)

            node_type = NodeType.STATE_READ if is_read else NodeType.STATE_WRITE

            # For writes, capture the value being written
            if not is_read:
                _pre_node = tracer.record(
                    node_type=node_type,
                    agent_id=agent_id,
                    data={
                        "state_name": state_name,
                        "operation": func.__name__,
                        "args": [_safe_serialize(a) for a in (args[1:] if args else [])],
                        "kwargs": {k: _safe_serialize(v) for k, v in kwargs.items()},
                    },
                    metadata={"state_name": state_name},
                    parent_ids=[tracer.current_node_id] if tracer.current_node_id else [],
                )

            result = func(*args, **kwargs)

            # For reads, capture the value being read
            if is_read:
                tracer.record(
                    node_type=node_type,
                    agent_id=agent_id,
                    data={
                        "state_name": state_name,
                        "operation": func.__name__,
                        "value": _safe_serialize(result),
                    },
                    metadata={"state_name": state_name},
                    parent_ids=[tracer.current_node_id] if tracer.current_node_id else [],
                )

            return result

        return wrapper

    return decorator


def _capture_inputs(args: tuple, kwargs: dict, func: Callable) -> dict:
    """Capture and format function inputs."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Skip 'self' for methods
    if params and params[0] == "self":
        args = args[1:]
        params = params[1:]

    result = {}
    for i, arg in enumerate(args):
        if i < len(params):
            result[params[i]] = _safe_serialize(arg)
        else:
            result[f"arg_{i}"] = _safe_serialize(arg)

    for k, v in kwargs.items():
        result[k] = _safe_serialize(v)

    return result


def _safe_serialize(value: Any) -> Any:
    """Safely serialize a value for storage."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value[:100]]  # Limit length
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in list(value.items())[:100]}
    try:
        return repr(value)[:500]  # Limit repr length
    except Exception:
        return f"<{type(value).__name__}>"
