"""Decoupled Event Bus and Approval Delegation Protocols (`events.py`).

The UI layer does not manage agent state directly; it subscribes to events
and responds through approval delegates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
from typing import Any, Callable, Dict, List, Optional, Protocol


class EventType(str, Enum):
    """Event types emitted during the agent lifecycle."""

    STEP_STARTED = "step_started"
    THINKING_DELTA = "thinking_delta"
    TEXT_DELTA = "text_delta"
    TOOL_CALL_PREPARING = "tool_call_preparing"
    TOOL_CALL_DETECTED = "tool_call_detected"
    TOOL_EXECUTION_STARTED = "tool_execution_started"
    TOOL_HEARTBEAT = "tool_heartbeat"
    TOOL_EXECUTION_TIMEOUT = "tool_execution_timeout"
    TOOL_EXECUTION_CANCELLED = "tool_execution_cancelled"
    TOOL_EXECUTION_COMPLETED = "tool_execution_completed"
    CONTEXT_COMPACTED = "context_compacted"
    MODEL_SWITCHED = "model_switched"
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_PROGRESS = "subagent_progress"
    SUBAGENT_HEARTBEAT = "subagent_heartbeat"
    SUBAGENT_COMPLETED = "subagent_completed"
    PLAN_UPDATED = "plan_updated"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    TURN_COMPLETED = "turn_completed"


@dataclass
class AgentEvent:
    """Agent event implementation."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalRequest:
    """Approval request implementation."""

    tool_name: str
    arguments: Dict[str, Any]
    description: str
    diff_preview: Optional[str] = None
    caller_role: str = "main"
    caller_agent: str = "orchestrator"
    risk_level: str = "standard"


class ApprovalDelegate(Protocol):
    """Approval delegate implementation."""

    def request_approval(self, request: ApprovalRequest) -> bool:
        ...

    def ask_user_question(self, question: str, options: Optional[List[str]] = None) -> str:
        ...


EventListener = Callable[[AgentEvent], None]


class EventBus:
    """EventBus: Pub/Sub."""

    def __init__(self) -> None:
        self._listeners: Dict[Optional[EventType], List[EventListener]] = {}
        self._lock = threading.Lock()

    def subscribe(
        self,
        listener: EventListener,
        event_type: Optional[EventType] = None,
    ) -> None:
        """Subscribe."""
        with self._lock:
            bucket = self._listeners.setdefault(event_type, [])
            if listener not in bucket:
                bucket.append(listener)

    def unsubscribe(
        self,
        listener: EventListener,
        event_type: Optional[EventType] = None,
    ) -> None:
        """Unsubscribe."""
        with self._lock:
            bucket = self._listeners.get(event_type, [])
            if listener in bucket:
                bucket.remove(listener)

    def emit(self, event_type: EventType, **data: Any) -> None:
        """Emit."""
        event = AgentEvent(type=event_type, data=data)
        with self._lock:
            type_listeners = list(self._listeners.get(event_type, []))
            global_listeners = list(self._listeners.get(None, []))

        for listener in type_listeners:
            listener(event)
        for listener in global_listeners:
            listener(event)
