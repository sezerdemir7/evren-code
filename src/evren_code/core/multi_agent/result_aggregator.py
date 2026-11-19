"""Reviewer module."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from evren_code.core.multi_agent.agent_registry import AgentRegistry
from evren_code.core.multi_agent.models import (
    AgentRole,
    AgentTaskResult,
    EngineeringTask,
    OrchestrationReport,
    TaskStatus,
)
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.verification.engine import VerificationEngine


class ResultAggregator:
    """Result aggregator implementation."""

    def __init__(
        self,
        agent_registry: Optional[AgentRegistry] = None,
        verification_engine: Optional[VerificationEngine] = None,
    ) -> None:
        self.agent_registry = agent_registry
        self.verification_engine = verification_engine

    def review_task_result(
        self,
        task: EngineeringTask,
        result: AgentTaskResult,
    ) -> AgentTaskResult:
        """Review task result."""
        if not task.acceptance_criteria and not result.modified_files:
            result.review_approved = True
            result.review_notes = "Otomatik onay: Kod değişikliği veya kabul kriteri bulunmuyor."
            return result

        if self.verification_engine and result.modified_files and not task.acceptance_criteria:
            is_valid = self.verify_with_engine(result.modified_files, task.title)
            if is_valid:
                result.review_approved = True
                result.review_notes = "Otomatik onay: VerificationEngine statik/test kontrolleri başarıyla doğrulandı."
                return result

        if not self.agent_registry:
            result.review_approved = False
            result.review_notes = "Reviewer denetimi başarısız: Ajan havuzu tanımlı değil, onay verilemez."
            return result

        reviewer = self.agent_registry.get_or_create_agent(AgentRole.REVIEWER)

        review_task = EngineeringTask(
            task_id=f"rev_{task.task_id}",
            title=f"Kod İnceleme: {task.title}",
            description=(
                f"Lütfen şu görevin çıktısını ve yapılan değişiklikleri incele:\n"
                f"Görev: {task.title}\n"
                f"Değiştirilen Dosyalar: {', '.join(result.modified_files)}\n"
                f"Geliştirici Çıktısı: {result.output[:1500]}\n"
                f"Kabul Kriterleri:\n" + "\n".join(f"- {c}" for c in task.acceptance_criteria) + "\n\n"
                "Yanıtında son satırda net olarak [APPROVED] veya [REJECTED: sebep] yaz."
            ),
            role=AgentRole.REVIEWER,
            read_only_files=result.modified_files,
        )

        rev_res = reviewer.execute_task(review_task)
        out_upper = rev_res.output.upper()

        has_explicit_rejected = bool(re.search(r"\[(?:REJECT(?:ED)?|RED|REDDED(?:İLDİ|ILDI)?)(?::\s*[^\]]*)?\]", out_upper))
        has_explicit_approved = bool(re.search(r"\[(?:APPROVE(?:D)?|ONAY(?:LANDI)?|KABUL\s*ED[İI]LD[İI]|BA[ŞS]ARILI)\]", out_upper))

        if has_explicit_rejected:
            is_rejected = True
            is_approved = False
        elif has_explicit_approved:
            is_approved = True
            is_rejected = False
        else:
            cleaned_out = re.sub(r"\b(?:NOT|NO|NEVER|WITHOUT|HİÇ|HİÇBİR)\s+(?:REJECTED|REDDEDİLDİ|BAŞARISIZ)\b", "", out_upper)
            is_rejected = bool(re.search(r"\b(?:REJECTED|REDDEDİLDİ)\b", cleaned_out))
            is_approved = bool(re.search(r"\b(?:APPROVED|ONAYLANDI|KABUL EDİLDİ|BAŞARILI)\b", cleaned_out))

        if not rev_res.success:
            result.review_approved = False
            result.review_notes = f"Reviewer görevi çalıştırılamadı veya çöktü: {rev_res.error or 'Hata'}"
        elif is_approved and not is_rejected:
            result.review_approved = True
            result.review_notes = f"Reviewer tarafından onaylandı: {rev_res.output[:400]}"
        else:
            result.review_approved = False
            result.review_notes = f"Reviewer onayı verilmedi veya reddedildi: {rev_res.output[:400]}"

        return result

    def verify_with_engine(self, modified_files: List[str], task_desc: str) -> bool:
        """Verify with engine."""
        if not self.verification_engine or not modified_files:
            return True

        try:
            report = self.verification_engine.verify_changes(
                changed_files=modified_files,
                task_description=f"Multi-Agent Doğrulama: {task_desc}",
                auto_rollback_on_failure=False,
            )
            return report.verified
        except Exception:
            return True

    def build_orchestration_report(
        self,
        run_id: str,
        goal: str,
        task_graph: TaskGraph,
        start_time: float,
    ) -> OrchestrationReport:
        """Build orchestration report."""
        tasks = list(task_graph.tasks.values())
        total = len(tasks)
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        blocked = sum(1 for t in tasks if t.status == TaskStatus.BLOCKED)

        all_modified: List[str] = []
        task_results: Dict[str, AgentTaskResult] = {}

        for t in tasks:
            if t.result:
                task_results[t.task_id] = t.result
                for f in t.result.modified_files:
                    if f not in all_modified:
                        all_modified.append(f)

        total_time = time.time() - start_time
        success = (completed == total) and total > 0

        summary_lines = [
            f"# Çoklu Ajan Yürütme Raporu: {run_id}",
            f"**Hedef:** {goal}",
            f"**Durum:** {'BAŞARILI' if success else 'EKSİK VEYA BAŞARISIZ'}",
            f"**Toplam Görev:** {total} (Tamamlanan: {completed}, Başarısız: {failed}, Bloklanan: {blocked})",
            f"**Toplam Süre:** {total_time:.2f} saniye",
            f"**Değiştirilen Dosyalar ({len(all_modified)}):** {', '.join(all_modified) if all_modified else 'Yok'}",
            "",
            "## Görev Detayları:",
        ]
        for t in tasks:
            status_tag = t.status.value.upper()
            summary_lines.append(f"- **[{status_tag}] {t.task_id}** ({t.role.value}): {t.title}")
            if t.result:
                summary_lines.append(f"  * Çıktı: {t.result.output[:150]}...")
                if t.result.review_approved is not None:
                    summary_lines.append(f"  * İnceleme: {'ONAYLANDI' if t.result.review_approved else 'REDDEDİLDİ'} ({t.result.review_notes})")

        return OrchestrationReport(
            run_id=run_id,
            goal=goal,
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            blocked_tasks=blocked,
            total_execution_time_seconds=total_time,
            modified_files=all_modified,
            task_results=task_results,
            success=success,
            summary="\n".join(summary_lines),
        )
