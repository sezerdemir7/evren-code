"""Models module."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class AgentRole(str, Enum):
    """Agent role implementation."""
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    TESTER = "tester"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    PLANNER = "planner"


class AgentStatus(str, Enum):
    """Agent status implementation."""
    IDLE = "idle"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Task status implementation."""
    PENDING = "pending"
    READY = "ready"
    ASSIGNED = "assigned"
    RUNNING = "running"
    WAITING_FOR_DEPENDENCY = "waiting_for_dependency"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class MessageType(str, Enum):
    """Message type implementation."""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    REVIEW_REQUEST = "review_request"
    REVIEW_FEEDBACK = "review_feedback"
    QUERY = "query"
    INFORMATION = "information"
    ERROR = "error"
    SYSTEM_ALERT = "system_alert"


@dataclass
class AgentTaskResult:
    """Agent task result implementation."""
    task_id: str
    agent_id: str
    role: AgentRole
    success: bool
    output: str
    modified_files: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
    tool_calls_count: int = 0
    review_approved: Optional[bool] = None
    review_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "role": self.role.value if isinstance(self.role, AgentRole) else str(self.role),
            "success": self.success,
            "output": self.output,
            "modified_files": list(self.modified_files),
            "artifacts": dict(self.artifacts),
            "error": self.error,
            "execution_time_seconds": self.execution_time_seconds,
            "tool_calls_count": self.tool_calls_count,
            "review_approved": self.review_approved,
            "review_notes": self.review_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentTaskResult:
        return cls(
            task_id=str(data.get("task_id", "")),
            agent_id=str(data.get("agent_id", "")),
            role=AgentRole(data.get("role", "developer")),
            success=bool(data.get("success", False)),
            output=str(data.get("output", "")),
            modified_files=list(data.get("modified_files", [])),
            artifacts=dict(data.get("artifacts", {})),
            error=data.get("error"),
            execution_time_seconds=float(data.get("execution_time_seconds", 0.0)),
            tool_calls_count=int(data.get("tool_calls_count", 0)),
            review_approved=data.get("review_approved"),
            review_notes=str(data.get("review_notes", "")),
        )


@dataclass
class EngineeringTask:
    """Engineering task implementation."""
    task_id: str
    title: str
    description: str
    role: AgentRole
    dependencies: List[str] = field(default_factory=list)
    assigned_files: List[str] = field(default_factory=list)
    read_only_files: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    domain: str = "general"
    provides_contracts: List[str] = field(default_factory=list)
    consumes_contracts: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    result: Optional[AgentTaskResult] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancel_event: Optional[Any] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "role": self.role.value if isinstance(self.role, AgentRole) else str(self.role),
            "dependencies": list(self.dependencies),
            "assigned_files": list(self.assigned_files),
            "read_only_files": list(self.read_only_files),
            "acceptance_criteria": list(self.acceptance_criteria),
            "domain": self.domain,
            "provides_contracts": list(self.provides_contracts),
            "consumes_contracts": list(self.consumes_contracts),
            "status": self.status.value if isinstance(self.status, TaskStatus) else str(self.status),
            "assigned_agent_id": self.assigned_agent_id,
            "result": self.result.to_dict() if self.result else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "metadata": {k: v for k, v in self.metadata.items() if not isinstance(v, threading.Event)},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EngineeringTask:
        res_data = data.get("result")
        return cls(
            task_id=str(data.get("task_id", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            role=AgentRole(data.get("role", "developer")),
            dependencies=list(data.get("dependencies", [])),
            assigned_files=list(data.get("assigned_files", [])),
            read_only_files=list(data.get("read_only_files", [])),
            acceptance_criteria=list(data.get("acceptance_criteria", [])),
            domain=str(data.get("domain", "general")),
            provides_contracts=list(data.get("provides_contracts", [])),
            consumes_contracts=list(data.get("consumes_contracts", [])),
            status=TaskStatus(data.get("status", "pending")),
            assigned_agent_id=data.get("assigned_agent_id"),
            result=AgentTaskResult.from_dict(res_data) if res_data else None,
            created_at=float(data.get("created_at", time.time())),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            retry_count=int(data.get("retry_count", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AgentMessage:
    """Agent message implementation."""
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    sender_id: str = "coordinator"
    recipient_id: str = "broadcast"
    message_type: MessageType = MessageType.INFORMATION
    correlation_id: str = ""
    task_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value if isinstance(self.message_type, MessageType) else str(self.message_type),
            "correlation_id": self.correlation_id,
            "task_id": self.task_id,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentMessage:
        return cls(
            message_id=str(data.get("message_id", "")),
            sender_id=str(data.get("sender_id", "coordinator")),
            recipient_id=str(data.get("recipient_id", "broadcast")),
            message_type=MessageType(data.get("message_type", "information")),
            correlation_id=str(data.get("correlation_id", "")),
            task_id=data.get("task_id"),
            payload=dict(data.get("payload", {})),
            timestamp=float(data.get("timestamp", time.time())),
        )


@dataclass
class FileLock:
    """File lock implementation."""
    path: str
    owner_task_id: str
    owner_agent_id: str
    lock_type: str = "exclusive_write"
    acquired_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "owner_task_id": self.owner_task_id,
            "owner_agent_id": self.owner_agent_id,
            "lock_type": self.lock_type,
            "acquired_at": self.acquired_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileLock:
        return cls(
            path=str(data.get("path", "")),
            owner_task_id=str(data.get("owner_task_id", "")),
            owner_agent_id=str(data.get("owner_agent_id", "")),
            lock_type=str(data.get("lock_type", "exclusive_write")),
            acquired_at=float(data.get("acquired_at", time.time())),
        )


@dataclass
class ConflictReport:
    """Conflict report implementation."""
    conflict_id: str
    task_a_id: str
    task_b_id: str
    conflicting_files: List[str]
    resolution_strategy: str  # "sequentialize", "isolate_subpaths", "reject"
    resolved: bool = False
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "task_a_id": self.task_a_id,
            "task_b_id": self.task_b_id,
            "conflicting_files": list(self.conflicting_files),
            "resolution_strategy": self.resolution_strategy,
            "resolved": self.resolved,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConflictReport:
        return cls(
            conflict_id=str(data.get("conflict_id", "")),
            task_a_id=str(data.get("task_a_id", "")),
            task_b_id=str(data.get("task_b_id", "")),
            conflicting_files=list(data.get("conflicting_files", [])),
            resolution_strategy=str(data.get("resolution_strategy", "sequentialize")),
            resolved=bool(data.get("resolved", False)),
            details=str(data.get("details", "")),
        )


@dataclass
class MultiAgentRunConfig:
    """Multi agent run config implementation."""
    enabled: bool = False
    auto_decompose: bool = True
    max_parallel_agents: int = 4
    task_timeout_seconds: float = 300.0
    max_retry_attempts: int = 2
    auto_review: bool = True
    verify_with_engine: bool = True
    heartbeat_interval_seconds: float = 10.0
    recover_on_timeout: bool = True
    persistence_dir: str = ".evren/multi_agent"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MultiAgentRunConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            auto_decompose=bool(data.get("auto_decompose", True)),
            max_parallel_agents=int(data.get("max_parallel_agents", 4)),
            task_timeout_seconds=float(data.get("task_timeout_seconds", 300.0)),
            max_retry_attempts=int(data.get("max_retry_attempts", 2)),
            auto_review=bool(data.get("auto_review", True)),
            verify_with_engine=bool(data.get("verify_with_engine", True)),
            heartbeat_interval_seconds=float(data.get("heartbeat_interval_seconds", 10.0)),
            recover_on_timeout=bool(data.get("recover_on_timeout", True)),
            persistence_dir=str(data.get("persistence_dir", ".evren/multi_agent")),
        )


@dataclass
class OrchestrationReport:
    """Orchestration report implementation."""
    run_id: str
    goal: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    blocked_tasks: int
    total_execution_time_seconds: float
    modified_files: List[str]
    task_results: Dict[str, AgentTaskResult]
    conflicts_detected: List[ConflictReport] = field(default_factory=list)
    success: bool = False
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "blocked_tasks": self.blocked_tasks,
            "total_execution_time_seconds": self.total_execution_time_seconds,
            "modified_files": list(self.modified_files),
            "task_results": {tid: res.to_dict() for tid, res in self.task_results.items()},
            "conflicts_detected": [c.to_dict() for c in self.conflicts_detected],
            "success": self.success,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrchestrationReport:
        t_results = {}
        for tid, rdata in data.get("task_results", {}).items():
            t_results[tid] = AgentTaskResult.from_dict(rdata)
        conflicts = [ConflictReport.from_dict(c) for c in data.get("conflicts_detected", [])]
        return cls(
            run_id=str(data.get("run_id", "")),
            goal=str(data.get("goal", "")),
            total_tasks=int(data.get("total_tasks", 0)),
            completed_tasks=int(data.get("completed_tasks", 0)),
            failed_tasks=int(data.get("failed_tasks", 0)),
            blocked_tasks=int(data.get("blocked_tasks", 0)),
            total_execution_time_seconds=float(data.get("total_execution_time_seconds", 0.0)),
            modified_files=list(data.get("modified_files", [])),
            task_results=t_results,
            conflicts_detected=conflicts,
            success=bool(data.get("success", False)),
            summary=str(data.get("summary", "")),
        )
