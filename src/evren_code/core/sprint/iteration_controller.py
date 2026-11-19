"""IterationController module."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from evren_code.core.multi_agent.models import TaskStatus
from evren_code.core.sprint.models import (
    AcceptanceCriterion,
    AcceptanceStatus,
    Sprint,
    SprintStatus,
    SprintTask,
)

logger = logging.getLogger(__name__)


class IterationController:
    """IterationController: repetition breaker."""

    def __init__(self) -> None:
        # task_id -> list of error hashes
        self._failure_signatures: Dict[str, List[str]] = {}
        self._iteration_history: List[Dict[str, Any]] = []

    def start_next_iteration(self, sprint: Sprint) -> int:
        """Start next iteration."""
        sprint.current_iteration += 1
        sprint.total_iterations = max(sprint.total_iterations, sprint.current_iteration)
        logger.info(f"Sprint '{sprint.id}' {sprint.current_iteration}. iterasyon başlatıldı.")
        return sprint.current_iteration

    def should_continue(
        self,
        sprint: Sprint,
        failed_tasks: List[SprintTask],
        failed_criteria: List[AcceptanceCriterion],
    ) -> Tuple[bool, str]:
        """Should continue."""
        if not failed_tasks and not failed_criteria:
            return False, "Tüm görevler ve kabul kriterleri başarıyla tamamlandı."

        if sprint.current_iteration >= sprint.config.max_iterations:
            return False, f"Maksimum iterasyon sınırına ({sprint.config.max_iterations}) ulaşıldı."

        if len(sprint.tasks) >= sprint.config.max_tasks_per_sprint:
            return False, f"Sprint maksimum görev sınırına ({sprint.config.max_tasks_per_sprint}) ulaştı."

        if sprint.metadata.started_at and getattr(sprint.config, "sprint_timeout_seconds", None):
            elapsed = time.time() - sprint.metadata.started_at
            if elapsed > sprint.config.sprint_timeout_seconds:
                return False, f"Sprint toplam duvar saati zaman aşımına ({sprint.config.sprint_timeout_seconds}s) ulaştı."

        for t in failed_tasks:
            err_msg = ""
            if t.result and t.result.error:
                err_msg = t.result.error
            elif t.result and t.result.output:
                err_msg = t.result.output[-300:]

            sig = hashlib.md5(err_msg.encode("utf-8")).hexdigest()
            effective_key = t.origin_task_id or t.task_id
            history = self._failure_signatures.setdefault(effective_key, [])
            history.append(sig)

            identical_failures = history.count(sig)
            if identical_failures > sprint.config.max_retries_per_task:
                return (
                    False,
                    f"Tekrarlayan hata tespit edildi: '{effective_key}' kök görevine ait hatalar "
                    f"aynı imza ile {identical_failures} kez başarısız oldu. Otonom döngü kırıldı.",
                )

        if sprint.config.stop_on_critical_failure:
            for t in failed_tasks:
                effective_key = t.origin_task_id or t.task_id
                if t.priority <= 1 and len(self._failure_signatures.get(effective_key, [])) > sprint.config.max_retries_per_task:
                    return False, f"Kritik öncelikli (P1) görev '{effective_key}' onarılamadı; kritik hata durdurma politikası devrede."

        return True, "Yeni iyileştirme ve düzeltme iterasyonu yürütülebilir."

    def record_iteration_result(
        self,
        sprint: Sprint,
        completed_task_ids: List[str],
        failed_task_ids: List[str],
        passed_crit_ids: List[str],
        failed_crit_ids: List[str],
    ) -> None:
        """Record iteration result."""
        entry = {
            "iteration": sprint.current_iteration,
            "completed_tasks": list(completed_task_ids),
            "failed_tasks": list(failed_task_ids),
            "passed_criteria": list(passed_crit_ids),
            "failed_criteria": list(failed_crit_ids),
        }
        self._iteration_history.append(entry)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._iteration_history)
