"""SprintReporter module."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set

from evren_code.core.multi_agent.models import TaskStatus
from evren_code.core.sprint.models import (
    AcceptanceStatus,
    ApprovalDecision,
    Sprint,
    SprintStatus,
)


class SprintReporter:
    """Sprint reporter implementation."""

    def generate_report(self, sprint: Sprint, execution_time: float = 0.0) -> Dict[str, Any]:
        """Generate report."""
        completed_tasks = [t for t in sprint.tasks.values() if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in sprint.tasks.values() if t.status == TaskStatus.FAILED]

        passed_criteria = [c for c in sprint.acceptance_criteria.values() if c.status == AcceptanceStatus.PASSED]
        failed_criteria = [c for c in sprint.acceptance_criteria.values() if c.status == AcceptanceStatus.FAILED]

        modified_files: Set[str] = set()
        for t in sprint.tasks.values():
            if t.result and t.result.modified_files:
                modified_files.update(t.result.modified_files)
            elif t.assigned_files and t.status == TaskStatus.COMPLETED:
                modified_files.update(t.assigned_files)

        report = {
            "sprint_id": sprint.id,
            "title": sprint.metadata.title,
            "objective": sprint.metadata.objective,
            "status": sprint.status.value,
            "total_iterations": sprint.total_iterations,
            "current_iteration": sprint.current_iteration,
            "execution_time_seconds": round(execution_time, 2),
            "epics_count": len(sprint.epics),
            "tasks": {
                "total": len(sprint.tasks),
                "completed": len(completed_tasks),
                "failed": len(failed_tasks),
                "items": [t.to_dict() for t in sprint.tasks.values()],
            },
            "acceptance_criteria": {
                "total": len(sprint.acceptance_criteria),
                "passed": len(passed_criteria),
                "failed": len(failed_criteria),
                "items": [c.to_dict() for c in sprint.acceptance_criteria.values()],
            },
            "modified_files": sorted(list(modified_files)),
            "approvals": [a.to_dict() for a in sprint.approval_requests.values()],
            "generated_at": time.time(),
        }
        return report

    def generate_markdown_report(self, sprint: Sprint, execution_time: float = 0.0) -> str:
        """Generate markdown report."""
        rep = self.generate_report(sprint, execution_time)
        status_badge = f"**{sprint.status.value.upper()}**"

        lines = [
            f"# EVREN Agent — Sprint Teslimat Raporu",
            f"",
            f"**Sprint ID:** `{sprint.id}`  ",
            f"**Başlık:** {sprint.metadata.title}  ",
            f"**Son Durum:** {status_badge}  ",
            f"**Toplam İterasyon:** {sprint.total_iterations} / {sprint.config.max_iterations}  ",
            f"**Çalışma Süresi:** {rep['execution_time_seconds']} saniye  ",
            f"",
            f"---",
            f"",
            f"## 1. Hedef ve Kapsam",
            f"",
            f"**Hedef:** {sprint.metadata.objective}  ",
            f"",
            f"### Kapsam İçi Bileşenler:",
        ]

        if sprint.metadata.scope:
            for sc in sprint.metadata.scope:
                lines.append(f"- {sc}")
        else:
            lines.append("- Belirtilen tüm mühendislik hedefleri.")

        lines.extend([
            f"",
            f"### Kapsam Dışı Bileşenler:",
        ])
        if sprint.metadata.out_of_scope:
            for osc in sprint.metadata.out_of_scope:
                lines.append(f"- {osc}")
        else:
            lines.append("- Kapsam dışı istisna bulunmamaktadır.")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 2. Epikler ve Görev Özeti",
            f"",
            f"Toplam {rep['epics_count']} epik, {rep['tasks']['total']} görev planlandı. "
            f"Tamamlanan: **{rep['tasks']['completed']}**, Başarısız: **{rep['tasks']['failed']}**.",
            f"",
            f"| Görev ID | Başlık | Rol | Ajan | Durum | Değişen Dosyalar |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for t in sprint.tasks.values():
            agent_label = t.assigned_agent_id or "-"
            mod_count = len(t.assigned_files)
            lines.append(
                f"| `{t.task_id}` | {t.title} | `{t.role.value}` | `{agent_label}` | **{t.status.value}** | {mod_count} dosya |"
            )

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Kabul Kriterleri ve Somut Kanıtlar",
            f"",
            f"Toplam {rep['acceptance_criteria']['total']} kriterden "
            f"**{rep['acceptance_criteria']['passed']}** adedi doğrulandı, "
            f"**{rep['acceptance_criteria']['failed']}** adedi başarısız oldu.",
            f"",
            f"| Kriter ID | Açıklama | Doğrulama Yöntemi | Durum | Kanıt / Sonuç |",
            f"| :--- | :--- | :--- | :--- | :--- |",
        ])

        for c in sprint.acceptance_criteria.values():
            res_snippet = c.verification_result.replace("\n", " ")[:60] if c.verification_result else "-"
            lines.append(
                f"| `{c.id}` | {c.description} | `{c.method.value}` | **{c.status.value}** | {res_snippet} |"
            )

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 4. Değiştirilen / Üretilen Dosyalar",
            f"",
        ])

        if rep["modified_files"]:
            for f in rep["modified_files"]:
                lines.append(f"- `{f}`")
        else:
            lines.append("*Herhangi bir dosya değişikliği kaydedilmedi.*")

        if sprint.approval_requests:
            lines.extend([
                f"",
                f"---",
                f"",
                f"## 5. İnsan Onay Kayıtları",
                f"",
                f"| Onay ID | Başlık | Tür | Karar | Açıklama |",
                f"| :--- | :--- | :--- | :--- | :--- |",
            ])
            for a in sprint.approval_requests.values():
                lines.append(
                    f"| `{a.approval_id}` | {a.title} | `{a.operation_type}` | **{a.decision.value}** | {a.decision_notes or '-'} |"
                )

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 6. Sonuç ve Değerlendirme",
            f"",
        ])

        if sprint.status == SprintStatus.COMPLETED:
            lines.append("✓ **Sprint Başarıyla Tamamlandı:** Tüm kabul kriterleri ve mühendislik hedefleri eksiksiz sağlandı.")
        elif sprint.status == SprintStatus.COMPLETED_WITH_WARNINGS:
            lines.append("⚠ **Sprint Uyarılarla Tamamlandı:** Temel hedefler sağlandı ancak bazı ikincil kontrollerde uyarılar oluştu.")
        elif sprint.status == SprintStatus.REQUIRES_USER_APPROVAL:
            lines.append("⏸ **İnsan Onayı Bekleniyor:** Kritik işlemler için kullanıcı onayı gerekmektedir.")
        else:
            lines.append("✗ **Sprint Başarısız Oldu:** Belirlenen iterasyon sınırları dahilinde hedefler veya kabul kriterleri doğrulanamadı.")

        return "\n".join(lines)
