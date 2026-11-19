"""SprintManager module."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from evren_code.config import AppConfig
from evren_code.core.events import EventBus
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.core.multi_agent.coordinator import MultiAgentCoordinator
from evren_code.core.repo_intelligence.engine import RepoIntelligenceEngine
from evren_code.core.security import ToolExecutor
from evren_code.core.sprint.acceptance import AcceptanceCriteriaEngine
from evren_code.core.sprint.completion import SprintCompletionEngine
from evren_code.core.sprint.events import SprintEvent, SprintEventType
from evren_code.core.sprint.exceptions import (
    ApprovalRequiredError,
    SprintError,
    SprintExecutionError,
    SprintPlanningError,
)
from evren_code.core.sprint.execution_engine import SprintExecutionEngine
from evren_code.core.sprint.iteration_controller import IterationController
from evren_code.core.sprint.models import (
    AcceptanceCriterion,
    ApprovalDecision,
    HumanApprovalRequest,
    Sprint,
    SprintConfig,
    SprintStatus,
    SprintTask,
)
from evren_code.core.sprint.planner import SprintPlanner
from evren_code.core.sprint.progress_tracker import ProgressTracker
from evren_code.core.sprint.recovery import SprintPersistence
from evren_code.core.sprint.reporting import SprintReporter
from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine
from evren_code.core.verification.engine import VerificationEngine
from evren_code.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class SprintManager:
    """Sprint manager implementation."""

    def __init__(
        self,
        config: AppConfig,
        llm_client: EvrenLLMClient,
        tool_executor: ToolExecutor,
        registry: ToolRegistry,
        coordinator: Optional[MultiAgentCoordinator] = None,
        repo_intelligence: Optional[RepoIntelligenceEngine] = None,
        symbol_engine: Optional[CodebaseSymbolEngine] = None,
        verification_engine: Optional[VerificationEngine] = None,
        event_bus: Optional[EventBus] = None,
        sprint_config: Optional[SprintConfig] = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.registry = registry
        self.repo_intelligence = repo_intelligence
        self.symbol_engine = symbol_engine
        self.verification_engine = verification_engine
        self.event_bus = event_bus or EventBus()
        self.sprint_config = sprint_config or SprintConfig()

        self.workspace_root = Path(config.workspace_root).resolve()
        self.persistence = SprintPersistence(
            workspace_root=self.workspace_root,
            base_dir=self.sprint_config.storage_dir,
        )

        self.planner = SprintPlanner(
            llm_client=llm_client,
            repo_intelligence=repo_intelligence,
            symbol_engine=symbol_engine,
        )

        self.acceptance_engine = AcceptanceCriteriaEngine(
            workspace_root=self.workspace_root,
            tool_executor=tool_executor,
            verification_engine=verification_engine,
            symbol_engine=symbol_engine,
        )

        self.completion_engine = SprintCompletionEngine(
            acceptance_engine=self.acceptance_engine,
            verification_engine=verification_engine,
        )

        self.coordinator = coordinator or MultiAgentCoordinator(
            config=config,
            llm_client=llm_client,
            tool_executor=tool_executor,
            registry=registry,
            repo_intelligence=repo_intelligence,
            symbol_engine=symbol_engine,
            verification_engine=verification_engine,
            event_bus=self.event_bus,
        )

        self.progress_tracker = ProgressTracker()
        self.iteration_controller = IterationController()
        self.reporter = SprintReporter()

        self.execution_engine = SprintExecutionEngine(
            coordinator=self.coordinator,
            acceptance_engine=self.acceptance_engine,
            completion_engine=self.completion_engine,
            persistence=self.persistence,
            planner=self.planner,
            iteration_controller=self.iteration_controller,
            progress_tracker=self.progress_tracker,
            reporter=self.reporter,
            event_bus=self.event_bus,
        )

        self.active_sprint: Optional[Sprint] = None
        self._execution_lock = threading.RLock()
        self._running_sprints: Set[str] = set()

    def create_and_plan(
        self,
        objective: str,
        title: Optional[str] = None,
        scope: Optional[List[str]] = None,
        out_of_scope: Optional[List[str]] = None,
        config: Optional[SprintConfig] = None,
    ) -> Sprint:
        """Create and plan."""
        cfg = config or self.sprint_config
        sprint = self.planner.plan_sprint(
            objective=objective,
            config=cfg,
            title=title,
            scope=scope,
            out_of_scope=out_of_scope,
        )
        self.persistence.save_sprint(sprint)
        self.active_sprint = sprint
        logger.info(f"Sprint '{sprint.id}' başarıyla planlandı. Durum: {sprint.status.value}")
        return sprint

    def approve_sprint(self, sprint_id: str, notes: str = "") -> bool:
        """Approve sprint."""
        sprint = self.get_sprint(sprint_id)
        if not sprint:
            raise SprintError(f"Sprint bulunamadı: {sprint_id}")

        if sprint.status not in (SprintStatus.PLANNED, SprintStatus.DRAFT):
            logger.warning(f"Sprint zaten onaylanmış veya farklı bir durumda: {sprint.status.value}")

        sprint.status = SprintStatus.APPROVED
        self.persistence.save_sprint(sprint)
        self.active_sprint = sprint
        logger.info(f"Sprint '{sprint_id}' kullanıcı tarafından onaylandı.")
        return True

    def start_sprint(
        self,
        sprint_id: Optional[str] = None,
        on_update: Optional[Callable[[SprintTask], None]] = None,
    ) -> Sprint:
        """Start sprint."""
        with self._execution_lock:
            sid = sprint_id or (self.active_sprint.id if self.active_sprint else None)
            if not sid:
                raise SprintError("Başlatılacak aktif bir sprint belirtilmedi.")

            sprint = self.get_sprint(sid)
            if not sprint:
                raise SprintError(f"Sprint bulunamadı: {sid}")

            if sprint.id in self._running_sprints or sprint.status == SprintStatus.RUNNING:
                raise SprintExecutionError(f"Sprint '{sprint.id}' zaten aktif olarak yürütülmektedir.")

            if sprint.status == SprintStatus.DRAFT:
                raise SprintExecutionError("DRAFT durumundaki sprint çalıştırılamaz; önce planlanmalı ve onaylanmalıdır.")
            if sprint.status == SprintStatus.PLANNED:
                raise SprintExecutionError("PLANNED durumundaki sprint doğrudan çalıştırılamaz; önce kullanıcı onayı (/sprint approve) verilmelidir.")
            if sprint.status in (SprintStatus.COMPLETED, SprintStatus.COMPLETED_WITH_WARNINGS, SprintStatus.FAILED, SprintStatus.CANCELLED):
                raise SprintExecutionError(f"Tamamlanmış veya iptal edilmiş sprint ({sprint.status.value}) yeniden başlatılamaz.")

            if sprint.status not in (SprintStatus.APPROVED, SprintStatus.PAUSED, SprintStatus.REQUIRES_USER_APPROVAL):
                raise SprintExecutionError(f"Geçersiz sprint durum geçişi: '{sprint.status.value}' durumundan RUNNING durumuna geçilemez.")

            self._running_sprints.add(sprint.id)
            self.execution_engine.request_resume(sprint.id)
            self.active_sprint = sprint

        try:
            res_sprint = self.execution_engine.run_sprint(sprint, on_update=on_update)
            return res_sprint
        finally:
            with self._execution_lock:
                self._running_sprints.discard(sprint.id)

    def pause_sprint(self, sprint_id: Optional[str] = None) -> bool:
        """Pause sprint."""
        with self._execution_lock:
            sid = sprint_id or (self.active_sprint.id if self.active_sprint else None)
            if not sid:
                return False

            sprint = self.get_sprint(sid)
            if not sprint:
                return False

            if sprint.status == SprintStatus.RUNNING:
                sprint.status = SprintStatus.PAUSED
                self.persistence.save_sprint(sprint)
                self.execution_engine.request_pause(sprint.id)
                logger.info(f"Sprint '{sid}' duraklatıldı.")
                return True
            return False

    def resume_sprint(
        self,
        sprint_id: Optional[str] = None,
        on_update: Optional[Callable[[SprintTask], None]] = None,
    ) -> Sprint:
        """Resume sprint."""
        with self._execution_lock:
            sid = sprint_id or (self.active_sprint.id if self.active_sprint else None)
            if not sid:
                raise SprintError("Devam ettirilecek sprint belirtilmedi.")

            sprint = self.get_sprint(sid)
            if not sprint:
                raise SprintError(f"Sprint bulunamadı: {sid}")

            if sprint.status != SprintStatus.PAUSED and sprint.status != SprintStatus.RUNNING:
                raise SprintExecutionError(f"Yalnızca PAUSED durumundaki sprintler devam ettirilebilir (Mevcut durum: {sprint.status.value}).")

            if sprint.id in self._running_sprints:
                raise SprintExecutionError(f"Sprint '{sprint.id}' zaten aktif olarak yürütülmektedir.")

            # Kurtarma senkronizasyonu
            sprint = self.persistence.reconcile_on_crash_recovery(sprint)
            sprint.status = SprintStatus.APPROVED
            self._running_sprints.add(sprint.id)
            self.execution_engine.request_resume(sprint.id)
            self.active_sprint = sprint

        try:
            res_sprint = self.execution_engine.run_sprint(sprint, on_update=on_update)
            return res_sprint
        finally:
            with self._execution_lock:
                self._running_sprints.discard(sprint.id)

    def cancel_sprint(self, sprint_id: Optional[str] = None) -> bool:
        """Sprinti iptal eder."""
        with self._execution_lock:
            sid = sprint_id or (self.active_sprint.id if self.active_sprint else None)
            if not sid:
                return False

            sprint = self.get_sprint(sid)
            if not sprint:
                return False

            sprint.status = SprintStatus.CANCELLED
            sprint.metadata.completed_at = time.time()
            self.persistence.save_sprint(sprint)
            self.execution_engine.request_cancel(sprint.id)
            logger.info(f"Sprint '{sid}' iptal edildi.")
            return True

    def request_approval(
        self,
        sprint_id: str,
        title: str,
        description: str,
        operation_type: str,
        task_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> HumanApprovalRequest:
        """Request approval."""
        sprint = self.get_sprint(sprint_id)
        if not sprint:
            raise SprintError(f"Sprint bulunamadı: {sprint_id}")

        apid = f"appr_{uuid.uuid4().hex[:6]}"
        req = HumanApprovalRequest(
            approval_id=apid,
            title=title,
            description=description,
            operation_type=operation_type,
            task_id=task_id,
            details=dict(details or {}),
            decision=ApprovalDecision.PENDING,
        )
        sprint.approval_requests[apid] = req
        self.persistence.save_sprint(sprint)
        return req

    def submit_approval(
        self,
        sprint_id: str,
        approval_id: str,
        decision: ApprovalDecision,
        notes: str = "",
    ) -> bool:
        """Submit approval."""
        sprint = self.get_sprint(sprint_id)
        if not sprint:
            raise SprintError(f"Sprint bulunamadı: {sprint_id}")

        req = sprint.approval_requests.get(approval_id)
        if not req:
            raise SprintError(f"Onay talebi bulunamadı: {approval_id}")

        req.decision = decision
        req.decision_notes = notes
        req.decided_at = time.time()
        self.persistence.save_sprint(sprint)
        logger.info(f"Sprint '{sprint_id}' için onay '{approval_id}' -> {decision.value} olarak güncellendi.")
        return True

    def get_sprint(self, sprint_id: str) -> Optional[Sprint]:
        """Get sprint."""
        if self.active_sprint and self.active_sprint.id == sprint_id:
            return self.active_sprint
        sprint = self.persistence.load_sprint(sprint_id)
        if sprint:
            self.active_sprint = sprint
        return sprint

    def list_sprints(self) -> List[Dict[str, Any]]:
        """List sprints."""
        return self.persistence.list_sprints()

    def get_status_dashboard(self, sprint_id: Optional[str] = None) -> str:
        """Get status dashboard."""
        sid = sprint_id or (self.active_sprint.id if self.active_sprint else None)
        if not sid:
            return "Aktif sprint bulunmuyor."
        sprint = self.get_sprint(sid)
        if not sprint:
            return f"Sprint bulunamadı: {sid}"
        return self.progress_tracker.render_ascii_dashboard(sprint)

    def get_markdown_report(self, sprint_id: Optional[str] = None) -> str:
        """Get markdown report."""
        sid = sprint_id or (self.active_sprint.id if self.active_sprint else None)
        if not sid:
            return "Aktif sprint bulunmuyor."
        sprint = self.get_sprint(sid)
        if not sprint:
            return f"Sprint bulunamadı: {sid}"
        return self.reporter.generate_markdown_report(sprint)
