"""Models module."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from evren_code.core.multi_agent.models import (
    AgentRole,
    AgentTaskResult,
    EngineeringTask,
    TaskStatus,
)


class SprintStatus(str, Enum):
    """Sprint status implementation."""
    DRAFT = "draft"
    PLANNED = "planned"
    APPROVED = "approved"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    REQUIRES_USER_APPROVAL = "requires_user_approval"


class EpicStatus(str, Enum):
    """Epik tamamlanma durumu."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class AcceptanceMethod(str, Enum):
    """Acceptance method implementation."""
    TEST_RESULT = "test_result"
    FILE_EXISTS = "file_exists"
    SYMBOL_EXISTS = "symbol_exists"
    API_CONTRACT = "api_contract"
    BUILD_SUCCESS = "build_success"
    REVIEW_APPROVAL = "review_approval"
    MANUAL_APPROVAL = "manual_approval"
    CUSTOM_VALIDATOR = "custom_validator"


class AcceptanceStatus(str, Enum):
    """Kabul kriteri durumu."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    REQUIRES_USER_INPUT = "requires_user_input"


class ApprovalDecision(str, Enum):
    """Approval decision implementation."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


@dataclass
class AcceptanceCriterion:
    """Acceptance criterion implementation."""
    id: str
    description: str
    method: AcceptanceMethod
    required_evidence: str
    status: AcceptanceStatus = AcceptanceStatus.PENDING
    associated_task_ids: List[str] = field(default_factory=list)
    verification_result: str = ""
    evidence_data: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "method": self.method.value if isinstance(self.method, AcceptanceMethod) else str(self.method),
            "required_evidence": self.required_evidence,
            "status": self.status.value if isinstance(self.status, AcceptanceStatus) else str(self.status),
            "associated_task_ids": list(self.associated_task_ids),
            "verification_result": self.verification_result,
            "evidence_data": dict(self.evidence_data),
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AcceptanceCriterion:
        return cls(
            id=str(data["id"]),
            description=str(data.get("description", "")),
            method=AcceptanceMethod(data.get("method", "test_result")),
            required_evidence=str(data.get("required_evidence", "")),
            status=AcceptanceStatus(data.get("status", "pending")),
            associated_task_ids=list(data.get("associated_task_ids", [])),
            verification_result=str(data.get("verification_result", "")),
            evidence_data=dict(data.get("evidence_data", {})),
            evaluated_at=data.get("evaluated_at"),
        )


@dataclass
class SprintEpic:
    """Sprint epic implementation."""
    id: str
    title: str
    description: str
    task_ids: List[str] = field(default_factory=list)
    status: EpicStatus = EpicStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "task_ids": list(self.task_ids),
            "status": self.status.value if isinstance(self.status, EpicStatus) else str(self.status),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SprintEpic:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            task_ids=list(data.get("task_ids", [])),
            status=EpicStatus(data.get("status", "pending")),
        )


@dataclass
class SprintTask:
    """Sprint task implementation."""
    task_id: str
    epic_id: str
    title: str
    description: str
    role: AgentRole
    dependencies: List[str] = field(default_factory=list)
    assigned_files: List[str] = field(default_factory=list)
    read_only_files: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    estimated_complexity: str = "medium"
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    result: Optional[AgentTaskResult] = None
    origin_task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_engineering_task(self) -> EngineeringTask:
        """To engineering task."""
        meta = dict(self.metadata)
        meta["epic_id"] = self.epic_id
        meta["estimated_complexity"] = self.estimated_complexity
        meta["priority"] = self.priority
        if self.origin_task_id:
            meta["origin_task_id"] = self.origin_task_id
        return EngineeringTask(
            task_id=self.task_id,
            title=self.title,
            description=self.description,
            role=self.role,
            dependencies=list(self.dependencies),
            assigned_files=list(self.assigned_files),
            read_only_files=list(self.read_only_files),
            acceptance_criteria=list(self.acceptance_criteria),
            status=self.status,
            assigned_agent_id=self.assigned_agent_id,
            result=self.result,
            metadata=meta,
        )

    @classmethod
    def from_engineering_task(cls, task: EngineeringTask, epic_id: str = "") -> SprintTask:
        eid = epic_id or str(task.metadata.get("epic_id", "epic_general"))
        orig_id = task.metadata.get("origin_task_id")
        return cls(
            task_id=task.task_id,
            epic_id=eid,
            title=task.title,
            description=task.description,
            role=task.role,
            dependencies=list(task.dependencies),
            assigned_files=list(task.assigned_files),
            read_only_files=list(task.read_only_files),
            acceptance_criteria=list(task.acceptance_criteria),
            estimated_complexity=str(task.metadata.get("estimated_complexity", "medium")),
            priority=int(task.metadata.get("priority", 1)),
            status=task.status,
            assigned_agent_id=task.assigned_agent_id,
            result=task.result,
            origin_task_id=str(orig_id) if orig_id else None,
            metadata=dict(task.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "epic_id": self.epic_id,
            "title": self.title,
            "description": self.description,
            "role": self.role.value if isinstance(self.role, AgentRole) else str(self.role),
            "dependencies": list(self.dependencies),
            "assigned_files": list(self.assigned_files),
            "read_only_files": list(self.read_only_files),
            "acceptance_criteria": list(self.acceptance_criteria),
            "estimated_complexity": self.estimated_complexity,
            "priority": self.priority,
            "status": self.status.value if isinstance(self.status, TaskStatus) else str(self.status),
            "assigned_agent_id": self.assigned_agent_id,
            "result": self.result.to_dict() if self.result else None,
            "origin_task_id": self.origin_task_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SprintTask:
        res_data = data.get("result")
        return cls(
            task_id=str(data["task_id"]),
            epic_id=str(data.get("epic_id", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            role=AgentRole(data.get("role", "developer")),
            dependencies=list(data.get("dependencies", [])),
            assigned_files=list(data.get("assigned_files", [])),
            read_only_files=list(data.get("read_only_files", [])),
            acceptance_criteria=list(data.get("acceptance_criteria", [])),
            estimated_complexity=str(data.get("estimated_complexity", "medium")),
            priority=int(data.get("priority", 1)),
            status=TaskStatus(data.get("status", "pending")),
            assigned_agent_id=data.get("assigned_agent_id"),
            result=AgentTaskResult.from_dict(res_data) if res_data else None,
            origin_task_id=data.get("origin_task_id"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class HumanApprovalRequest:
    """Human approval request implementation."""
    approval_id: str
    title: str
    description: str
    operation_type: str
    task_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    decision: ApprovalDecision = ApprovalDecision.PENDING
    decision_notes: str = ""
    requested_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "title": self.title,
            "description": self.description,
            "operation_type": self.operation_type,
            "task_id": self.task_id,
            "details": dict(self.details),
            "decision": self.decision.value if isinstance(self.decision, ApprovalDecision) else str(self.decision),
            "decision_notes": self.decision_notes,
            "requested_at": self.requested_at,
            "decided_at": self.decided_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HumanApprovalRequest:
        return cls(
            approval_id=str(data["approval_id"]),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            operation_type=str(data.get("operation_type", "general")),
            task_id=data.get("task_id"),
            details=dict(data.get("details", {})),
            decision=ApprovalDecision(data.get("decision", "pending")),
            decision_notes=str(data.get("decision_notes", "")),
            requested_at=float(data.get("requested_at", time.time())),
            decided_at=data.get("decided_at"),
        )


@dataclass
class SprintConfig:
    """Sprint config implementation."""
    enabled: bool = False
    max_iterations: int = 5
    max_tasks_per_sprint: int = 30
    max_parallel_agents: int = 3
    max_retries_per_task: int = 2
    stop_on_critical_failure: bool = True
    require_reviewer_approval: bool = True
    require_regression_check: bool = True
    require_acceptance_criteria: bool = True
    checkpoint_interval: int = 1
    resume_interrupted_sprints: bool = True
    storage_dir: str = ".evren/sprints"
    sprint_timeout_seconds: float = 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SprintConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            max_iterations=int(data.get("max_iterations", 5)),
            max_tasks_per_sprint=int(data.get("max_tasks_per_sprint", 30)),
            max_parallel_agents=int(data.get("max_parallel_agents", 3)),
            max_retries_per_task=int(data.get("max_retries_per_task", 2)),
            stop_on_critical_failure=bool(data.get("stop_on_critical_failure", True)),
            require_reviewer_approval=bool(data.get("require_reviewer_approval", True)),
            require_regression_check=bool(data.get("require_regression_check", True)),
            require_acceptance_criteria=bool(data.get("require_acceptance_criteria", True)),
            checkpoint_interval=int(data.get("checkpoint_interval", 1)),
            resume_interrupted_sprints=bool(data.get("resume_interrupted_sprints", True)),
            storage_dir=str(data.get("storage_dir", ".evren/sprints")),
            sprint_timeout_seconds=float(data.get("sprint_timeout_seconds", 3600.0)),
        )


@dataclass
class SprintMetadata:
    """Sprint metadata implementation."""
    sprint_id: str
    title: str
    objective: str
    description: str = ""
    scope: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sprint_id": self.sprint_id,
            "title": self.title,
            "objective": self.objective,
            "description": self.description,
            "scope": list(self.scope),
            "out_of_scope": list(self.out_of_scope),
            "assumptions": list(self.assumptions),
            "risks": list(self.risks),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SprintMetadata:
        return cls(
            sprint_id=str(data["sprint_id"]),
            title=str(data.get("title", "")),
            objective=str(data.get("objective", "")),
            description=str(data.get("description", "")),
            scope=list(data.get("scope", [])),
            out_of_scope=list(data.get("out_of_scope", [])),
            assumptions=list(data.get("assumptions", [])),
            risks=list(data.get("risks", [])),
            created_at=float(data.get("created_at", time.time())),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class Sprint:
    """Sprint implementation."""
    metadata: SprintMetadata
    config: SprintConfig
    epics: Dict[str, SprintEpic] = field(default_factory=dict)
    tasks: Dict[str, SprintTask] = field(default_factory=dict)
    acceptance_criteria: Dict[str, AcceptanceCriterion] = field(default_factory=dict)
    approval_requests: Dict[str, HumanApprovalRequest] = field(default_factory=dict)
    status: SprintStatus = SprintStatus.DRAFT
    current_iteration: int = 0
    total_iterations: int = 0
    report: Optional[Dict[str, Any]] = None

    @property
    def id(self) -> str:
        return self.metadata.sprint_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "config": self.config.to_dict(),
            "epics": {eid: e.to_dict() for eid, e in self.epics.items()},
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "acceptance_criteria": {aid: a.to_dict() for aid, a in self.acceptance_criteria.items()},
            "approval_requests": {apid: ap.to_dict() for apid, ap in self.approval_requests.items()},
            "status": self.status.value if isinstance(self.status, SprintStatus) else str(self.status),
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "report": self.report,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Sprint:
        meta = SprintMetadata.from_dict(data["metadata"])
        cfg = SprintConfig.from_dict(data.get("config", {}))
        epics = {eid: SprintEpic.from_dict(ed) for eid, ed in data.get("epics", {}).items()}
        tasks = {tid: SprintTask.from_dict(td) for tid, td in data.get("tasks", {}).items()}
        acc = {aid: AcceptanceCriterion.from_dict(ad) for aid, ad in data.get("acceptance_criteria", {}).items()}
        appr = {apid: HumanApprovalRequest.from_dict(apd) for apid, apd in data.get("approval_requests", {}).items()}
        return cls(
            metadata=meta,
            config=cfg,
            epics=epics,
            tasks=tasks,
            acceptance_criteria=acc,
            approval_requests=appr,
            status=SprintStatus(data.get("status", "draft")),
            current_iteration=int(data.get("current_iteration", 0)),
            total_iterations=int(data.get("total_iterations", 0)),
            report=data.get("report"),
        )
