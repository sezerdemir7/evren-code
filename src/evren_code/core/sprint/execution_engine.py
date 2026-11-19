"""SprintExecutionEngine module."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Callable, Dict, List, Optional, Set

from evren_code.core.events import EventBus
from evren_code.core.multi_agent.coordinator import MultiAgentCoordinator
from evren_code.core.multi_agent.models import EngineeringTask, TaskStatus
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.sprint.acceptance import AcceptanceCriteriaEngine
from evren_code.core.sprint.completion import SprintCompletionEngine
from evren_code.core.sprint.events import SprintEvent, SprintEventType
from evren_code.core.sprint.exceptions import (
    ApprovalRequiredError,
    SprintExecutionError,
)
from evren_code.core.sprint.iteration_controller import IterationController
from evren_code.core.sprint.models import (
    AcceptanceStatus,
    ApprovalDecision,
    Sprint,
    SprintStatus,
    SprintTask,
)
from evren_code.core.sprint.planner import SprintPlanner
from evren_code.core.sprint.progress_tracker import ProgressTracker
from evren_code.core.sprint.recovery import SprintPersistence
from evren_code.core.sprint.reporting import SprintReporter

logger = logging.getLogger(__name__)


class SprintExecutionEngine:
    """Sprint execution engine implementation."""

    def __init__(
        self,
        coordinator: MultiAgentCoordinator,
        acceptance_engine: AcceptanceCriteriaEngine,
        completion_engine: SprintCompletionEngine,
        persistence: SprintPersistence,
        planner: SprintPlanner,
        iteration_controller: Optional[IterationController] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        reporter: Optional[SprintReporter] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.coordinator = coordinator
        self.acceptance_engine = acceptance_engine
        self.completion_engine = completion_engine
        self.persistence = persistence
        self.planner = planner
        self.iteration_controller = iteration_controller or IterationController()
        self.progress_tracker = progress_tracker or ProgressTracker()
        self.reporter = reporter or SprintReporter()
        self.event_bus = event_bus or EventBus()
        self._paused_sprints: Set[str] = set()
        self._cancelled_sprints: Set[str] = set()

    def request_pause(self, sprint_id: str) -> None:
        """Request pause."""
        self._paused_sprints.add(sprint_id)

    def request_cancel(self, sprint_id: str) -> None:
        """Request cancel."""
        self._cancelled_sprints.add(sprint_id)

    def request_resume(self, sprint_id: str) -> None:
        """Request resume."""
        self._paused_sprints.discard(sprint_id)

    def _snapshot_file(self, file_rel_path: str) -> Dict[str, Any]:
        """Snapshot file."""
        target = (self.coordinator.config.workspace_root / file_rel_path).resolve()
        if target.is_file():
            try:
                content = target.read_bytes()
                h = hashlib.sha256(content).hexdigest()
                return {"exists": True, "size": len(content), "hash": h}
            except Exception:
                return {"exists": True, "size": 0, "hash": ""}
        return {"exists": False}

    def run_sprint(
        self,
        sprint: Sprint,
        on_update: Optional[Callable[[SprintTask], None]] = None,
    ) -> Sprint:
        """Run sprint."""
        start_time = time.time()
        sprint.metadata.started_at = sprint.metadata.started_at or start_time

        if sprint.id in self._paused_sprints or sprint.status == SprintStatus.PAUSED:
            sprint.status = SprintStatus.PAUSED
            self.persistence.save_sprint(sprint)
            return sprint

        if sprint.id in self._cancelled_sprints or sprint.status == SprintStatus.CANCELLED:
            sprint.status = SprintStatus.CANCELLED
            self.persistence.save_sprint(sprint)
            return sprint

        if sprint.status in (SprintStatus.COMPLETED, SprintStatus.CANCELLED):
            logger.info(f"Sprint '{sprint.id}' zaten {sprint.status.value} durumunda.")
            return sprint

        sprint.status = SprintStatus.RUNNING
        self._emit_event(sprint, SprintEventType.SPRINT_STARTED, {"status": sprint.status.value}, "Sprint başlatıldı")
        self.persistence.save_sprint(sprint)

        try:
            while True:
                if sprint.id in self._paused_sprints or sprint.status == SprintStatus.PAUSED:
                    sprint.status = SprintStatus.PAUSED
                    logger.info(f"Sprint '{sprint.id}' duraklatıldı. Yürütme döngüsünden çıkılıyor.")
                    break

                if sprint.id in self._cancelled_sprints or sprint.status == SprintStatus.CANCELLED:
                    sprint.status = SprintStatus.CANCELLED
                    sprint.metadata.completed_at = time.time()
                    logger.info(f"Sprint '{sprint.id}' iptal edildi. Yürütme döngüsünden çıkılıyor.")
                    break

                if sprint.metadata.started_at and getattr(sprint.config, "sprint_timeout_seconds", None):
                    elapsed = time.time() - sprint.metadata.started_at
                    if elapsed > sprint.config.sprint_timeout_seconds:
                        sprint.status = SprintStatus.FAILED
                        sprint.metadata.completed_at = time.time()
                        self._emit_event(
                            sprint,
                            SprintEventType.SPRINT_FAILED,
                            {"reason": f"Sprint toplam duvar saati zaman aşımına ({sprint.config.sprint_timeout_seconds}s) ulaştı."},
                            "Sprint zaman aşımı",
                        )
                        break

                iteration = self.iteration_controller.start_next_iteration(sprint)
                self._emit_event(
                    sprint,
                    SprintEventType.ITERATION_STARTED,
                    {"iteration": iteration, "max_iterations": sprint.config.max_iterations},
                    f"İterasyon {iteration} başlatıldı",
                )

                active_tasks = [
                    t for t in sprint.tasks.values()
                    if t.status in (TaskStatus.PENDING, TaskStatus.FAILED)
                ]

                if active_tasks:
                    batch_graph = TaskGraph()
                    for t in active_tasks:
                        t.status = TaskStatus.PENDING
                        snapshots = {}
                        for af in t.assigned_files:
                            snapshots[af] = self._snapshot_file(af)
                        t.metadata["file_snapshots"] = snapshots
                        batch_graph.add_task(t.to_engineering_task())

                    self.persistence.save_sprint(sprint)

                    def _task_step(eng_task: EngineeringTask) -> None:
                        sp_task = sprint.tasks.get(eng_task.task_id)
                        if sp_task:
                            sp_task.status = eng_task.status
                            sp_task.assigned_agent_id = eng_task.assigned_agent_id
                            sp_task.result = eng_task.result
                            if on_update:
                                on_update(sp_task)
                            event_type = (
                                SprintEventType.TASK_COMPLETED
                                if sp_task.status == TaskStatus.COMPLETED
                                else (
                                    SprintEventType.TASK_FAILED
                                    if sp_task.status == TaskStatus.FAILED
                                    else SprintEventType.TASK_STARTED
                                )
                            )
                            self._emit_event(
                                sprint,
                                event_type,
                                {"task_id": sp_task.task_id, "status": sp_task.status.value},
                                f"Görev güncellendi: {sp_task.task_id}",
                            )
                            self.persistence.save_sprint(sprint)

                    logger.info(f"Sprint '{sprint.id}': {len(active_tasks)} görev koordine ediliyor.")
                    coord_report = self.coordinator.run_plan(
                        goal=sprint.metadata.objective,
                        graph=batch_graph,
                        on_update=_task_step,
                        run_id=f"{sprint.id}_iter_{iteration}",
                    )

                    for t in active_tasks:
                        eng_t = batch_graph.get_task(t.task_id)
                        if eng_t:
                            t.status = eng_t.status
                            t.result = eng_t.result

                if sprint.id in self._paused_sprints or sprint.status == SprintStatus.PAUSED:
                    sprint.status = SprintStatus.PAUSED
                    break
                if sprint.id in self._cancelled_sprints or sprint.status == SprintStatus.CANCELLED:
                    sprint.status = SprintStatus.CANCELLED
                    sprint.metadata.completed_at = time.time()
                    break

                self.acceptance_engine.evaluate_all(sprint)
                self._emit_event(
                    sprint,
                    SprintEventType.ACCEPTANCE_EVALUATED,
                    {
                        cid: c.status.value
                        for cid, c in sprint.acceptance_criteria.items()
                    },
                    "Kabul kriterleri değerlendirildi",
                )

                is_done, final_status, reasons = self.completion_engine.evaluate_sprint_completion(sprint)

                failed_tasks = [t for t in sprint.tasks.values() if t.status == TaskStatus.FAILED]
                failed_criteria = [
                    c for c in sprint.acceptance_criteria.values()
                    if c.status in (AcceptanceStatus.FAILED, AcceptanceStatus.PENDING)
                ]

                self.iteration_controller.record_iteration_result(
                    sprint=sprint,
                    completed_task_ids=[t.task_id for t in sprint.tasks.values() if t.status == TaskStatus.COMPLETED],
                    failed_task_ids=[t.task_id for t in failed_tasks],
                    passed_crit_ids=[c.id for c in sprint.acceptance_criteria.values() if c.status == AcceptanceStatus.PASSED],
                    failed_crit_ids=[c.id for c in failed_criteria],
                )

                if is_done:
                    sprint.status = final_status
                    sprint.metadata.completed_at = time.time()
                    self._emit_event(
                        sprint,
                        SprintEventType.SPRINT_COMPLETED,
                        {"status": sprint.status.value, "reasons": reasons},
                        "Sprint başarıyla tamamlandı",
                    )
                    break

                if final_status == SprintStatus.REQUIRES_USER_APPROVAL:
                    sprint.status = SprintStatus.REQUIRES_USER_APPROVAL
                    self._emit_event(
                        sprint,
                        SprintEventType.APPROVAL_REQUESTED,
                        {"reasons": reasons},
                        "İnsan onayı talep edildi",
                    )
                    break

                can_continue, cont_reason = self.iteration_controller.should_continue(
                    sprint, failed_tasks, failed_criteria
                )

                if not can_continue:
                    sprint.status = SprintStatus.FAILED
                    sprint.metadata.completed_at = time.time()
                    self._emit_event(
                        sprint,
                        SprintEventType.SPRINT_FAILED,
                        {"reason": cont_reason, "blockers": reasons},
                        "Sprint başarısız oldu",
                    )
                    break

                logger.info(f"Sprint '{sprint.id}': Düzeltme planı hazırlanıyor. Sebep: {cont_reason}")
                new_tasks = self.planner.plan_remediation(
                    sprint=sprint,
                    failed_tasks=failed_tasks,
                    failed_criteria=failed_criteria,
                    iteration=iteration,
                )

                for nt in new_tasks:
                    sprint.tasks[nt.task_id] = nt
                    if nt.epic_id in sprint.epics and nt.task_id not in sprint.epics[nt.epic_id].task_ids:
                        sprint.epics[nt.epic_id].task_ids.append(nt.task_id)

                self.persistence.save_sprint(sprint)

        except Exception as e:
            logger.error(f"Sprint yürütülürken beklenmeyen istisna: {e}", exc_info=True)
            sprint.status = SprintStatus.FAILED
            sprint.metadata.completed_at = time.time()
            self._emit_event(sprint, SprintEventType.SPRINT_FAILED, {"error": str(e)}, "Sprint hatayla durdu")
            raise SprintExecutionError(f"Sprint yürütmesi kesintiye uğradı: {e}")
        finally:
            exec_time = time.time() - start_time
            sprint.report = self.reporter.generate_report(sprint, exec_time)
            self.persistence.save_sprint(sprint)

        return sprint

    def _emit_event(self, sprint: Sprint, event_type: SprintEventType, details: Dict, message: str = "") -> None:
        """Emit event."""
        ev = SprintEvent(
            event_type=event_type,
            sprint_id=sprint.id,
            message=message or event_type.value,
            details=details,
        )
        self.persistence.log_event(sprint.id, ev)
        try:
            self.event_bus.publish(event_type.value, ev.to_dict())
        except Exception:
            pass
