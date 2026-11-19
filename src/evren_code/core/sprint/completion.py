"""SprintCompletionEngine module."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from evren_code.core.multi_agent.models import AgentRole, TaskStatus
from evren_code.core.sprint.acceptance import AcceptanceCriteriaEngine
from evren_code.core.sprint.models import (
    AcceptanceStatus,
    ApprovalDecision,
    Sprint,
    SprintStatus,
)
from evren_code.core.verification.engine import VerificationEngine

logger = logging.getLogger(__name__)


class SprintCompletionEngine:
    """Sprint completion engine implementation."""

    def __init__(
        self,
        acceptance_engine: AcceptanceCriteriaEngine,
        verification_engine: Optional[VerificationEngine] = None,
    ) -> None:
        self.acceptance_engine = acceptance_engine
        self.verification_engine = verification_engine

    def evaluate_sprint_completion(self, sprint: Sprint) -> Tuple[bool, SprintStatus, List[str]]:
        """Evaluate sprint completion."""
        blockers: List[str] = []
        warnings: List[str] = []

        pending_approvals = [
            req for req in sprint.approval_requests.values()
            if req.decision == ApprovalDecision.PENDING
        ]
        if pending_approvals:
            req_titles = [r.title for r in pending_approvals]
            return False, SprintStatus.REQUIRES_USER_APPROVAL, [
                f"Kritik işlem için insan onayı bekleniyor: {', '.join(req_titles)}"
            ]

        failed_tasks = [t for t in sprint.tasks.values() if t.status == TaskStatus.FAILED]
        pending_tasks = [
            t for t in sprint.tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        ]

        if failed_tasks:
            blockers.append(f"{len(failed_tasks)} adet görev başarısız durumda.")
        if pending_tasks:
            blockers.append(f"{len(pending_tasks)} adet görev henüz tamamlanmadı.")

        self.acceptance_engine.evaluate_all(sprint)
        failed_criteria = [
            c for c in sprint.acceptance_criteria.values()
            if c.status == AcceptanceStatus.FAILED
        ]
        pending_criteria = [
            c for c in sprint.acceptance_criteria.values()
            if c.status in (AcceptanceStatus.PENDING, AcceptanceStatus.IN_PROGRESS, AcceptanceStatus.REQUIRES_USER_INPUT)
        ]

        if failed_criteria:
            crit_ids = [c.id for c in failed_criteria]
            blockers.append(f"Kabul kriterleri sağlanamadı: {', '.join(crit_ids)}")
        if pending_criteria:
            crit_ids = [c.id for c in pending_criteria]
            blockers.append(f"Kabul kriterleri henüz doğrulanmadı: {', '.join(crit_ids)}")

        if sprint.config.require_reviewer_approval:
            reviewer_tasks = [t for t in sprint.tasks.values() if t.role == AgentRole.REVIEWER]
            if not reviewer_tasks:
                blockers.append("Sprint yapılandırmasında reviewer onayı zorunlu ancak reviewer görevi tanımlanmamış.")
            else:
                for rt in reviewer_tasks:
                    if rt.status != TaskStatus.COMPLETED or not rt.result:
                        blockers.append(f"Reviewer görevi '{rt.task_id}' tamamlanmadı.")
                    elif "[APPROVED]" not in rt.result.output:
                        blockers.append(f"Reviewer görevi '{rt.task_id}' onay vermedi ([APPROVED] bulunamadı).")

        if sprint.config.require_regression_check and self.verification_engine:
            try:
                test_res = self.verification_engine.run_all_tests()
                if not test_res.passed:
                    blockers.append(f"Regresyon testleri başarısız: {test_res.failure_summary}")
            except Exception as ve:
                logger.warning(f"Regresyon doğrulama çalıştırılamadı: {ve}")
                warnings.append(f"Regresyon testi çalıştırılamadı: {ve}")

        # Nihai Karar
        if blockers:
            return False, SprintStatus.FAILED, blockers

        if warnings:
            return True, SprintStatus.COMPLETED_WITH_WARNINGS, warnings

        return True, SprintStatus.COMPLETED, ["Tüm mühendislik hedefleri, görevler ve kabul kriterleri başarıyla sağlandı."]
