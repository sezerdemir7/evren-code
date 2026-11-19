"""Sprint Events module."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class SprintEventType(str, Enum):
    """Sprint event type implementation."""
    SPRINT_CREATED = "sprint_created"
    SPRINT_PLANNED = "sprint_planned"
    SPRINT_APPROVED = "sprint_approved"
    SPRINT_STARTED = "sprint_started"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    CORRECTION_STARTED = "correction_started"
    ACCEPTANCE_EVALUATED = "acceptance_evaluated"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    SPRINT_COMPLETED = "sprint_completed"
    SPRINT_FAILED = "sprint_failed"
    SPRINT_CANCELLED = "sprint_cancelled"
    SPRINT_PAUSED = "sprint_paused"
    SPRINT_RESUMED = "sprint_resumed"


@dataclass
class SprintEvent:
    """Sprint event implementation."""
    event_type: SprintEventType
    sprint_id: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "sprint_id": self.sprint_id,
            "message": self.message,
            "details": dict(self.details),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SprintEvent:
        return cls(
            event_type=SprintEventType(data["event_type"]),
            sprint_id=str(data.get("sprint_id", "")),
            message=str(data.get("message", "")),
            details=dict(data.get("details", {})),
            timestamp=float(data.get("timestamp", time.time())),
        )
