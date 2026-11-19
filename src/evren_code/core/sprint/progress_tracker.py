"""ProgressTracker module."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evren_code.core.multi_agent.models import TaskStatus
from evren_code.core.sprint.models import (
    AcceptanceStatus,
    ApprovalDecision,
    EpicStatus,
    Sprint,
    SprintStatus,
)


@dataclass
class SprintProgressMetrics:
    """Sprint progress metrics implementation."""
    total_epics: int = 0
    completed_epics: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    pending_tasks: int = 0
    total_criteria: int = 0
    passed_criteria: int = 0
    failed_criteria: int = 0
    pending_criteria: int = 0
    pending_approvals: int = 0
    completion_percentage: float = 0.0
    iteration: int = 0
    max_iterations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_epics": self.total_epics,
            "completed_epics": self.completed_epics,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "pending_tasks": self.pending_tasks,
            "total_criteria": self.total_criteria,
            "passed_criteria": self.passed_criteria,
            "failed_criteria": self.failed_criteria,
            "pending_criteria": self.pending_criteria,
            "pending_approvals": self.pending_approvals,
            "completion_percentage": round(self.completion_percentage, 1),
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
        }


class ProgressTracker:
    """Progress tracker implementation."""

    def compute_metrics(self, sprint: Sprint) -> SprintProgressMetrics:
        """Deterministik metrikleri hesaplar."""
        for epic in sprint.epics.values():
            if not epic.task_ids:
                epic.status = EpicStatus.COMPLETED
                continue
            all_done = all(
                sprint.tasks.get(tid) and sprint.tasks[tid].status == TaskStatus.COMPLETED
                for tid in epic.task_ids
            )
            any_failed = any(
                sprint.tasks.get(tid) and sprint.tasks[tid].status == TaskStatus.FAILED
                for tid in epic.task_ids
            )
            any_in_progress = any(
                sprint.tasks.get(tid) and sprint.tasks[tid].status == TaskStatus.RUNNING
                for tid in epic.task_ids
            )
            if all_done:
                epic.status = EpicStatus.COMPLETED
            elif any_failed:
                epic.status = EpicStatus.FAILED
            elif any_in_progress:
                epic.status = EpicStatus.IN_PROGRESS
            else:
                epic.status = EpicStatus.PENDING

        total_epics = len(sprint.epics)
        completed_epics = sum(1 for e in sprint.epics.values() if e.status == EpicStatus.COMPLETED)

        total_tasks = len(sprint.tasks)
        completed_tasks = sum(1 for t in sprint.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for t in sprint.tasks.values() if t.status == TaskStatus.FAILED)
        pending_tasks = sum(1 for t in sprint.tasks.values() if t.status == TaskStatus.PENDING)

        total_criteria = len(sprint.acceptance_criteria)
        passed_criteria = sum(1 for c in sprint.acceptance_criteria.values() if c.status == AcceptanceStatus.PASSED)
        failed_criteria = sum(1 for c in sprint.acceptance_criteria.values() if c.status == AcceptanceStatus.FAILED)
        pending_criteria = sum(1 for c in sprint.acceptance_criteria.values() if c.status in (AcceptanceStatus.PENDING, AcceptanceStatus.IN_PROGRESS))

        pending_approvals = sum(
            1 for a in sprint.approval_requests.values() if a.decision == ApprovalDecision.PENDING
        )

        pct_tasks = (completed_tasks / total_tasks * 100.0) if total_tasks > 0 else 100.0
        pct_crit = (passed_criteria / total_criteria * 100.0) if total_criteria > 0 else 100.0
        overall_pct = (pct_tasks * 0.6) + (pct_crit * 0.4)

        return SprintProgressMetrics(
            total_epics=total_epics,
            completed_epics=completed_epics,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            pending_tasks=pending_tasks,
            total_criteria=total_criteria,
            passed_criteria=passed_criteria,
            failed_criteria=failed_criteria,
            pending_criteria=pending_criteria,
            pending_approvals=pending_approvals,
            completion_percentage=overall_pct,
            iteration=sprint.current_iteration,
            max_iterations=sprint.config.max_iterations,
        )

    def render_ascii_dashboard(self, sprint: Sprint) -> str:
        """Render ascii dashboard."""
        metrics = self.compute_metrics(sprint)
        status_val = sprint.status.value.upper()
        bar_len = 20
        filled_len = int(bar_len * metrics.completion_percentage / 100.0)
        progress_bar = "█" * filled_len + "░" * (bar_len - filled_len)

        lines = [
            "=" * 70,
            f" EVREN SPRINT: {sprint.metadata.title}",
            f" ID: {sprint.id} | Durum: [{status_val}] | İterasyon: {sprint.current_iteration}/{sprint.config.max_iterations}",
            f" İlerleme: [{progress_bar}] {metrics.completion_percentage:.1f}%",
            "=" * 70,
            "",
            " [EPİKLER]",
        ]

        for eid, ep in sprint.epics.items():
            status_icon = "✓" if ep.status == EpicStatus.COMPLETED else ("✗" if ep.status == EpicStatus.FAILED else "○")
            lines.append(f"  {status_icon} [{ep.status.value.upper()}] {ep.title} ({len(ep.task_ids)} görev)")

        lines.extend(["", " [GÖREVLER]"])
        for tid, t in sprint.tasks.items():
            s_icon = "✓" if t.status == TaskStatus.COMPLETED else ("✗" if t.status == TaskStatus.FAILED else "○")
            agent_str = f"@{t.assigned_agent_id}" if t.assigned_agent_id else f"[{t.role.value}]"
            lines.append(f"  {s_icon} {t.task_id} ({t.title}) - {agent_str} [{t.status.value}]")

        lines.extend(["", " [KABUL KRİTERLERİ]"])
        for cid, c in sprint.acceptance_criteria.items():
            c_icon = "✓" if c.status == AcceptanceStatus.PASSED else ("✗" if c.status == AcceptanceStatus.FAILED else "?")
            lines.append(f"  {c_icon} {c.id}: {c.description} [{c.method.value}] -> {c.status.value}")

        if metrics.pending_approvals > 0:
            lines.extend(["", f" ⚠ BEKLEYEN İNSAN ONAYLARI ({metrics.pending_approvals}):"])
            for apid, req in sprint.approval_requests.items():
                if req.decision == ApprovalDecision.PENDING:
                    lines.append(f"    - {apid}: {req.title} ({req.operation_type})")

        lines.append("=" * 70)
        return "\n".join(lines)
