"""Multi-Agent Scheduler module."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Callable, Dict, List, Optional, Set

from evren_code.core.events import EventBus
from evren_code.core.multi_agent.agent_registry import AgentRegistry
from evren_code.core.multi_agent.communication import AgentMessageBus
from evren_code.core.multi_agent.context_sharing import ContextSharingEngine
from evren_code.core.multi_agent.file_ownership import FileOwnershipManager
from evren_code.core.multi_agent.models import (
    AgentMessage,
    AgentRole,
    AgentTaskResult,
    EngineeringTask,
    MessageType,
    MultiAgentRunConfig,
    TaskStatus,
)
from evren_code.core.multi_agent.result_aggregator import ResultAggregator
from evren_code.core.multi_agent.task_graph import TaskGraph


class MultiAgentScheduler:
    """Multi agent scheduler implementation."""

    def __init__(
        self,
        task_graph: TaskGraph,
        agent_registry: AgentRegistry,
        file_ownership: FileOwnershipManager,
        context_sharing: ContextSharingEngine,
        result_aggregator: ResultAggregator,
        message_bus: Optional[AgentMessageBus] = None,
        config: Optional[MultiAgentRunConfig] = None,
        event_bus: Optional[EventBus] = None,
        shadow_git: Optional[Any] = None,
    ) -> None:
        self.task_graph = task_graph
        self.agent_registry = agent_registry
        self.file_ownership = file_ownership
        self.context_sharing = context_sharing
        self.result_aggregator = result_aggregator
        self.message_bus = message_bus or AgentMessageBus()
        self.config = config or MultiAgentRunConfig()
        self.event_bus = event_bus or EventBus()
        self.shadow_git = shadow_git

        self._cancelled = threading.Event()
        self._running_tasks: Set[str] = set()
        self._lock = threading.RLock()
        self._cancel_events: Dict[str, threading.Event] = {}
        self._orphaned_futures: Dict[concurrent.futures.Future, str] = {}
        self._orphaned_timestamps: Dict[concurrent.futures.Future, float] = {}
        self._task_epochs: Dict[str, int] = {}
        self.late_ignored_count: int = 0
        self.recovered_timeout_tasks: int = 0

    def cancel(self) -> None:
        """Cancel."""
        self._cancelled.set()
        with self._lock:
            for cev in self._cancel_events.values():
                cev.set()
            for task in self.task_graph.tasks.values():
                if not task.is_terminal:
                    self.task_graph.mark_task_status(task.task_id, TaskStatus.CANCELLED)

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def execute_all(
        self,
        on_task_update: Optional[Callable[[EngineeringTask], None]] = None,
        poll_interval: float = 0.05,
    ) -> bool:
        """Execute all."""
        valid, errors = self.task_graph.validate_dag()
        if not valid:
            raise ValueError(f"Çizge doğrulanamadı: {'; '.join(errors)}")

        max_workers = max(1, self.config.max_parallel_agents)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        future_map: Dict[concurrent.futures.Future, str] = {}

        try:
            while not self._cancelled.is_set() and not self.task_graph.is_finished():
                ready_tasks = self.task_graph.get_ready_tasks()

                for task in ready_tasks:
                    with self._lock:
                        if len(self._running_tasks) >= max_workers:
                            break
                        if task.task_id in self._running_tasks:
                            continue

                        agent = self.agent_registry.get_or_create_agent(task.role, agent_id=task.assigned_agent_id)
                        task.assigned_agent_id = agent.agent_id

                        acquired, err = self.file_ownership.acquire_locks(
                            task_id=task.task_id,
                            agent_id=agent.agent_id,
                            write_files=task.assigned_files,
                            read_files=task.read_only_files,
                        )
                        if not acquired:
                            agent.status = TaskStatus.IDLE if hasattr(TaskStatus, "IDLE") else agent.status
                            from evren_code.core.multi_agent.models import AgentStatus
                            agent.status = AgentStatus.IDLE
                            continue

                        task.status = TaskStatus.RUNNING
                        task.started_at = time.time()
                        self._running_tasks.add(task.task_id)
                        attempt_epoch = self._task_epochs.get(task.task_id, 0) + 1
                        self._task_epochs[task.task_id] = attempt_epoch

                    ctx = self.context_sharing.prepare_task_context(task, self.task_graph)

                    cancel_ev = threading.Event()
                    self._cancel_events[task.task_id] = cancel_ev
                    task.cancel_event = cancel_ev

                    future = executor.submit(self._run_single_task, agent, task, ctx)
                    future_map[future] = (task.task_id, attempt_epoch)

                    if on_task_update:
                        on_task_update(task)

                now_check = time.time()
                orphan_grace = float(
                    getattr(
                        self.config,
                        "orphan_lock_grace_seconds",
                        max(1.5, self.config.task_timeout_seconds * 2.0),
                    )
                )
                has_waiting_ready = any(
                    t.status in (TaskStatus.READY, TaskStatus.PENDING) and t.task_id not in self._running_tasks
                    for t in self.task_graph.tasks.values()
                )
                for of in list(self._orphaned_futures.keys()):
                    orphan_tid = self._orphaned_futures.get(of)
                    orphan_ts = self._orphaned_timestamps.get(of, now_check)
                    is_stale_livelock = (
                        not of.done()
                        and len(self._running_tasks) == 0
                        and len(future_map) == 0
                        and has_waiting_ready
                        and (now_check - orphan_ts) >= orphan_grace
                    )
                    if of.done() or is_stale_livelock:
                        otid = self._orphaned_futures.pop(of, None)
                        self._orphaned_timestamps.pop(of, None)
                        if otid:
                            if otid not in self._running_tasks:
                                self.file_ownership.release_locks(otid)
                                self._cancel_events.pop(otid, None)

                now = time.time()
                timeout_futures = []
                for f, (tid, f_epoch) in list(future_map.items()):
                    task = self.task_graph.get_task(tid)
                    if task and task.started_at:
                        if (now - task.started_at) > self.config.task_timeout_seconds:
                            timeout_futures.append((f, tid, f_epoch))

                for f, tid, f_epoch in timeout_futures:
                    future_map.pop(f, None)
                    cev = self._cancel_events.get(tid)
                    if cev:
                        cev.set()
                    try:
                        f.cancel()
                    except Exception:
                        pass
                    with self._lock:
                        self._running_tasks.discard(tid)
                        self._task_epochs[tid] = self._task_epochs.get(tid, f_epoch) + 1

                    task = self.task_graph.get_task(tid)
                    allow_timeout_recovery = bool(
                        getattr(self.config, "recover_on_timeout", False)
                        or (task and getattr(task, "metadata", {}).get("recover_on_timeout", False))
                    )
                    if task:
                        if (
                            allow_timeout_recovery
                            and task.retry_count < self.config.max_retry_attempts
                            and not self._cancelled.is_set()
                        ):
                            task.retry_count += 1
                            self.recovered_timeout_tasks += 1
                            task.description += (
                                f"\n\n[ZAMAN AŞIMI KURTARMA VE TEKRAR DENEME #{task.retry_count}]: "
                                f"Önceki deneme {self.config.task_timeout_seconds}s sınırını aştı."
                            )
                            task.status = TaskStatus.READY
                            task.started_at = None
                            task.completed_at = None
                            self.file_ownership.release_locks(tid)
                            self._cancel_events.pop(tid, None)
                            if on_task_update:
                                on_task_update(task)
                        else:
                            task.status = TaskStatus.FAILED
                            task.completed_at = time.time()
                            task.result = AgentTaskResult(
                                task_id=task.task_id,
                                agent_id=task.assigned_agent_id or "unknown",
                                role=task.role,
                                success=False,
                                output=f"Görev zaman aşımına uğradı ({self.config.task_timeout_seconds}s sınırını aştı).",
                                error=f"Task timeout after {self.config.task_timeout_seconds}s",
                                execution_time_seconds=now - (task.started_at or now),
                            )
                            self.task_graph.mark_task_status(task.task_id, task.status)
                            if on_task_update:
                                on_task_update(task)

                    if f.done():
                        if tid not in self._running_tasks:
                            self.file_ownership.release_locks(tid)
                            self._cancel_events.pop(tid, None)
                    else:
                        if not (task and task.status == TaskStatus.READY):
                            self._orphaned_futures[f] = tid
                            self._orphaned_timestamps[f] = time.time()

                        def _make_cleanup(target_future, task_id=tid, post_timeout_epoch=self._task_epochs.get(tid, 0)):
                            def _cleanup(done_fut):
                                with self._lock:
                                    if not done_fut.cancelled():
                                        self.late_ignored_count += 1
                                    self._orphaned_futures.pop(target_future, None)
                                    self._orphaned_timestamps.pop(target_future, None)
                                    if (
                                        task_id not in self._running_tasks
                                        and self._task_epochs.get(task_id, 0) == post_timeout_epoch
                                    ):
                                        self.file_ownership.release_locks(task_id)
                                        self._cancel_events.pop(task_id, None)
                            return _cleanup

                        f.add_done_callback(_make_cleanup(f, tid))

                done_futures = [f for f in future_map if f.done()]
                for f in done_futures:
                    tid, f_epoch = future_map.pop(f)
                    with self._lock:
                        self._running_tasks.discard(tid)
                        current_epoch = self._task_epochs.get(tid, f_epoch)

                    task = self.task_graph.get_task(tid)
                    if not task:
                        continue

                    if f_epoch != current_epoch or task.is_terminal or self._cancelled.is_set():
                        with self._lock:
                            self.late_ignored_count += 1
                        continue

                    try:
                        result: AgentTaskResult = f.result()
                        task.result = result
                        task.completed_at = time.time()

                        is_failed = False
                        if result.success:
                            if result.review_approved is False:
                                is_failed = True
                            else:
                                task.status = TaskStatus.COMPLETED
                        else:
                            is_failed = True

                        if is_failed:
                            if task.retry_count < self.config.max_retry_attempts and not self._cancelled.is_set():
                                task.retry_count += 1
                                feedback_notes = []
                                if result.review_notes:
                                    feedback_notes.append(f"Reviewer Geri Bildirimi: {result.review_notes}")
                                if result.error:
                                    feedback_notes.append(f"Hata: {result.error}")
                                if not result.modified_files and task.role == AgentRole.DEVELOPER:
                                    feedback_notes.append(
                                        "UYARI: Önceki adımda hiçbir dosya düzenlenmedi! Lütfen `edit_file` veya `write_file` araçlarıyla kod değişikliklerini fiilen uygula."
                                    )

                                feedback_str = "\n".join(feedback_notes)
                                task.description += f"\n\n[DÜZELTME VE TEKRAR DENEME #{task.retry_count}]:\n{feedback_str}"
                                task.status = TaskStatus.READY
                                task.started_at = None
                                task.completed_at = None
                            else:
                                task.status = TaskStatus.FAILED

                    except Exception as exc:
                        if task.retry_count < self.config.max_retry_attempts and not self._cancelled.is_set():
                            task.retry_count += 1
                            task.description += f"\n\n[DÜZELTME VE TEKRAR DENEME #{task.retry_count}]:\nİstisna Hatası: {exc}"
                            task.status = TaskStatus.READY
                            task.started_at = None
                            task.completed_at = None
                        else:
                            task.status = TaskStatus.FAILED
                            task.result = AgentTaskResult(
                                task_id=task.task_id,
                                agent_id=task.assigned_agent_id or "unknown",
                                role=task.role,
                                success=False,
                                output=f"Görev istisna ile çöktü: {exc}",
                                error=str(exc),
                            )

                    self.file_ownership.release_locks(task.task_id)
                    self._cancel_events.pop(task.task_id, None)
                    self.task_graph.mark_task_status(task.task_id, task.status)

                    if on_task_update:
                        on_task_update(task)

                if not done_futures and not self.task_graph.is_finished():
                    time.sleep(poll_interval)

        finally:
            for cev in self._cancel_events.values():
                cev.set()
            all_active = list(future_map.keys()) + list(self._orphaned_futures.keys())
            if all_active:
                concurrent.futures.wait(all_active, timeout=0.5)
            for f, (tid, _epoch) in list(future_map.items()):
                if f.done():
                    self.file_ownership.release_locks(tid)
            for f, tid in list(self._orphaned_futures.items()):
                if f.done():
                    self.file_ownership.release_locks(tid)
            executor.shutdown(wait=False, cancel_futures=True)

        return self.task_graph.all_completed()

    def _run_single_task(
        self,
        agent: Any,
        task: EngineeringTask,
        ctx: Dict[str, Any],
    ) -> AgentTaskResult:
        """Run single task."""
        self.message_bus.send_message(
            AgentMessage(
                sender_id="scheduler",
                recipient_id=agent.agent_id,
                message_type=MessageType.TASK_ASSIGNMENT,
                task_id=task.task_id,
                payload={"title": task.title},
            )
        )
        wt_path = None
        if self.shadow_git and getattr(task, "metadata", {}).get("use_worktree", False):
            try:
                wt_path = self.shadow_git.create_worktree(task.task_id)
                if wt_path:
                    ctx["worktree_path"] = str(wt_path)
            except Exception:
                wt_path = None

        try:
            result = agent.execute_task(task, context_data=ctx)

            if result.success and self.config.auto_review:
                result = self.result_aggregator.review_task_result(task, result)
        finally:
            if wt_path and self.shadow_git:
                try:
                    self.shadow_git.remove_worktree(task.task_id)
                except Exception:
                    pass

        self.message_bus.send_message(
            AgentMessage(
                sender_id=agent.agent_id,
                recipient_id="scheduler",
                message_type=MessageType.TASK_RESULT,
                task_id=task.task_id,
                payload={"success": result.success, "modified_files": result.modified_files},
            )
        )
        return result
