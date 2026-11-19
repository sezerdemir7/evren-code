"""AcceptanceCriteriaEngine module."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evren_code.core.multi_agent.models import AgentRole, TaskStatus
from evren_code.core.security import ToolExecutor
from evren_code.core.sprint.exceptions import AcceptanceVerificationError
from evren_code.core.sprint.models import (
    AcceptanceCriterion,
    AcceptanceMethod,
    AcceptanceStatus,
    ApprovalDecision,
    Sprint,
)
from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine
from evren_code.core.verification.engine import VerificationEngine

logger = logging.getLogger(__name__)


class AcceptanceCriteriaEngine:
    """Acceptance criteria engine implementation."""

    def __init__(
        self,
        workspace_root: Path,
        tool_executor: Optional[ToolExecutor] = None,
        verification_engine: Optional[VerificationEngine] = None,
        symbol_engine: Optional[CodebaseSymbolEngine] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.tool_executor = tool_executor
        self.verification_engine = verification_engine
        self.symbol_engine = symbol_engine

    def evaluate_criterion(self, criterion: AcceptanceCriterion, sprint: Sprint) -> AcceptanceCriterion:
        """Evaluate criterion."""
        criterion.evaluated_at = time.time()
        method = criterion.method

        try:
            if method == AcceptanceMethod.FILE_EXISTS:
                self._verify_file_exists(criterion)
            elif method == AcceptanceMethod.SYMBOL_EXISTS:
                self._verify_symbol_exists(criterion)
            elif method == AcceptanceMethod.BUILD_SUCCESS:
                self._verify_build_success(criterion, sprint)
            elif method == AcceptanceMethod.TEST_RESULT:
                self._verify_test_result(criterion, sprint)
            elif method == AcceptanceMethod.REVIEW_APPROVAL:
                self._verify_review_approval(criterion, sprint)
            elif method == AcceptanceMethod.MANUAL_APPROVAL:
                self._verify_manual_approval(criterion, sprint)
            else:
                self._verify_generic(criterion, sprint)
        except Exception as e:
            logger.error(f"Kabul kriteri '{criterion.id}' değerlendirilirken hata: {e}")
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = f"Doğrulama istisnası: {str(e)}"

        return criterion

    def evaluate_all(self, sprint: Sprint) -> Dict[str, AcceptanceCriterion]:
        """Evaluate all."""
        for crit in sprint.acceptance_criteria.values():
            self.evaluate_criterion(crit, sprint)
        return sprint.acceptance_criteria

    def all_mandatory_passed(self, sprint: Sprint) -> bool:
        """All mandatory passed."""
        if not sprint.acceptance_criteria:
            return True
        return all(c.status == AcceptanceStatus.PASSED for c in sprint.acceptance_criteria.values())

    def _verify_file_exists(self, criterion: AcceptanceCriterion) -> None:
        """Verify file exists."""
        target_path_str = criterion.required_evidence.strip()
        if not target_path_str:
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = "Gerekli kanıt dosya yolu belirtilmemiş."
            return

        file_path = (self.workspace_root / target_path_str).resolve()
        try:
            file_path.relative_to(self.workspace_root)
        except ValueError:
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = f"Güvenlik ihlali: Dosya yolu çalışma alanı dışında: {target_path_str}"
            return

        if file_path.exists() and file_path.is_file():
            size = file_path.stat().st_size
            criterion.status = AcceptanceStatus.PASSED
            criterion.verification_result = f"Dosya diskte mevcut ({size} bayt): {target_path_str}"
            criterion.evidence_data = {"path": str(file_path), "size": size, "exists": True}
        else:
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = f"Dosya diskte bulunamadı: {target_path_str}"
            criterion.evidence_data = {"path": str(file_path), "exists": False}

    def _verify_symbol_exists(self, criterion: AcceptanceCriterion) -> None:
        """Verify symbol exists."""
        symbol_spec = criterion.required_evidence.strip()
        if not symbol_spec:
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = "Hedef sembol adı veya 'dosya:sembol' belirtilmemiş."
            return

        file_hint = ""
        symbol_name = symbol_spec
        if ":" in symbol_spec:
            file_hint, symbol_name = symbol_spec.split(":", 1)
            file_hint = file_hint.strip()
            symbol_name = symbol_name.strip()

        found = False
        details = {}

        if self.symbol_engine:
            try:
                if file_hint:
                    symbols = self.symbol_engine.get_file_symbols(file_hint)
                    for s in symbols:
                        if getattr(s, "name", "") == symbol_name:
                            found = True
                            details = {"symbol": symbol_name, "file": file_hint, "kind": getattr(s, "kind", "")}
                            break
                if not found:
                    all_syms = self.symbol_engine.find_symbols_by_name(symbol_name)
                    if all_syms:
                        found = True
                        s_first = all_syms[0]
                        details = {
                            "symbol": symbol_name,
                            "file": getattr(s_first, "file_path", ""),
                            "kind": getattr(s_first, "kind", ""),
                        }
            except Exception as se:
                logger.debug(f"Sembol motoru sorgusu hatası: {se}")

        if not found and file_hint:
            target_file = (self.workspace_root / file_hint).resolve()
            if target_file.exists() and target_file.is_file():
                try:
                    content = target_file.read_text(encoding="utf-8", errors="replace")
                    if (
                        f"def {symbol_name}" in content
                        or f"class {symbol_name}" in content
                        or f"function {symbol_name}" in content
                    ):
                        found = True
                        details = {"symbol": symbol_name, "file": file_hint, "method": "text_search"}
                except Exception:
                    pass

        if found:
            criterion.status = AcceptanceStatus.PASSED
            criterion.verification_result = f"Sembol başarıyla doğrulandı: {symbol_name}"
            criterion.evidence_data = details
        else:
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = f"Sembol kod tabanında bulunamadı: {symbol_spec}"
            criterion.evidence_data = {"symbol": symbol_spec, "found": False}

    def _verify_build_success(self, criterion: AcceptanceCriterion, sprint: Optional[Sprint] = None) -> None:
        """Verify build success."""
        cmd = criterion.required_evidence.strip()
        if cmd and self.tool_executor:
            try:
                res = self.tool_executor.execute("run_command", {"command": cmd, "cwd": "."})
                output_str = res if isinstance(res, str) else getattr(res, "output", str(res))

                m = re.search(r"(?:Çıkış Kodu \(Exit Code\)|Exit Code):\s*(-?\d+)", output_str, re.IGNORECASE)
                if m:
                    exit_code = int(m.group(1))
                    if exit_code == 0:
                        criterion.status = AcceptanceStatus.PASSED
                        criterion.verification_result = f"Derleme başarılı. Çıktı: {output_str[:200]}"
                        criterion.evidence_data = {"command": cmd, "exit_code": 0, "output": output_str[:500]}
                    else:
                        criterion.status = AcceptanceStatus.FAILED
                        criterion.verification_result = f"Derleme komutu başarısız oldu (çıkış kodu {exit_code}): {output_str[:200]}"
                        criterion.evidence_data = {"command": cmd, "exit_code": exit_code, "output": output_str[:500]}
                    return

                if output_str.startswith("[GÜVENLİK ENGELİ") or output_str.startswith("Hata:"):
                    criterion.status = AcceptanceStatus.FAILED
                    criterion.verification_result = f"Derleme komut yürütme engeli/hatası: {output_str[:200]}"
                    criterion.evidence_data = {"command": cmd, "error": output_str[:500]}
                    return

                criterion.status = AcceptanceStatus.FAILED
                criterion.verification_result = f"Derleme çıktısında çıkış kodu doğrulanamadı: {output_str[:200]}"
                criterion.evidence_data = {"command": cmd, "output": output_str[:500]}
                return
            except Exception as e:
                criterion.status = AcceptanceStatus.FAILED
                criterion.verification_result = f"Derleme komut yürütme hatası: {e}"
                criterion.evidence_data = {"command": cmd, "error": str(e)}
                return

        if self.verification_engine:
            try:
                syntax_report = self.verification_engine.syntax_checker.check_workspace(self.workspace_root)
                if not syntax_report.has_errors:
                    criterion.status = AcceptanceStatus.PASSED
                    criterion.verification_result = "Tüm çalışma alanı sözdizimi doğrulaması başarılı."
                    criterion.evidence_data = {"syntax_errors": 0}
                else:
                    criterion.status = AcceptanceStatus.FAILED
                    criterion.verification_result = f"Sözdizimi hataları tespit edildi: {len(syntax_report.errors)} adet."
                    criterion.evidence_data = {"errors": [str(e) for e in syntax_report.errors[:5]]}
                return
            except Exception as ve:
                logger.debug(f"Syntax verification error: {ve}")
                criterion.status = AcceptanceStatus.FAILED
                criterion.verification_result = f"Sözdizimi motoru hatası: {ve}"
                criterion.evidence_data = {"error": str(ve)}
                return

        if sprint and criterion.associated_task_ids:
            tasks_to_check = [sprint.tasks[tid] for tid in criterion.associated_task_ids if tid in sprint.tasks]
            if tasks_to_check and all(t.status == TaskStatus.COMPLETED and t.result and t.result.success for t in tasks_to_check):
                criterion.status = AcceptanceStatus.PASSED
                criterion.verification_result = f"Derleme kontrolü: İlişkili görevler ({', '.join(criterion.associated_task_ids)}) hatasız tamamlandı."
                criterion.evidence_data = {"tasks": criterion.associated_task_ids}
                return
            else:
                criterion.status = AcceptanceStatus.FAILED
                criterion.verification_result = "Derleme kanıtı eksik: İlişkili görevler henüz tamamlanmadı veya başarısız."
                criterion.evidence_data = {"error": "Associated tasks not completed"}
                return

        criterion.status = AcceptanceStatus.FAILED
        criterion.verification_result = "Derleme kanıtı doğrulanamadı (Doğrulama aracı veya komut yürütücü tanımlı değil)."
        criterion.evidence_data = {"error": "Missing verification engine or tool executor"}

    def _verify_test_result(self, criterion: AcceptanceCriterion, sprint: Optional[Sprint] = None) -> None:
        """Verify test result."""
        cmd = criterion.required_evidence.strip()
        if cmd and self.tool_executor:
            try:
                res = self.tool_executor.execute("run_command", {"command": cmd, "cwd": "."})
                output_str = res if isinstance(res, str) else getattr(res, "output", str(res))

                m = re.search(r"(?:Çıkış Kodu \(Exit Code\)|Exit Code):\s*(-?\d+)", output_str, re.IGNORECASE)
                if m:
                    exit_code = int(m.group(1))
                    if exit_code == 0:
                        criterion.status = AcceptanceStatus.PASSED
                        criterion.verification_result = f"Test komutu başarılı: {output_str[:200]}"
                        criterion.evidence_data = {"command": cmd, "exit_code": 0, "output": output_str[:500]}
                    else:
                        criterion.status = AcceptanceStatus.FAILED
                        criterion.verification_result = f"Testler başarısız (çıkış kodu {exit_code}): {output_str[:300]}"
                        criterion.evidence_data = {"command": cmd, "exit_code": exit_code, "output": output_str[:500]}
                    return

                if output_str.startswith("[GÜVENLİK ENGELİ") or output_str.startswith("Hata:"):
                    criterion.status = AcceptanceStatus.FAILED
                    criterion.verification_result = f"Test komut yürütme engeli/hatası: {output_str[:300]}"
                    criterion.evidence_data = {"command": cmd, "error": output_str[:500]}
                    return

                criterion.status = AcceptanceStatus.FAILED
                criterion.verification_result = f"Test komut çıktısında çıkış kodu doğrulanamadı: {output_str[:300]}"
                criterion.evidence_data = {"command": cmd, "output": output_str[:500]}
                return
            except Exception as e:
                criterion.status = AcceptanceStatus.FAILED
                criterion.verification_result = f"Test çalıştırma hatası: {e}"
                criterion.evidence_data = {"command": cmd, "error": str(e)}
                return

        if self.verification_engine:
            try:
                test_report = self.verification_engine.run_all_tests()
                if test_report.passed:
                    criterion.status = AcceptanceStatus.PASSED
                    criterion.verification_result = f"Tüm testler başarılı ({test_report.total_tests} test geçti)."
                    criterion.evidence_data = {"total": test_report.total_tests, "passed": test_report.total_tests}
                else:
                    criterion.status = AcceptanceStatus.FAILED
                    criterion.verification_result = f"Test paketinde başarısızlıklar var: {test_report.failure_summary}"
                    criterion.evidence_data = {"failures": test_report.failures}
                return
            except Exception as ve:
                logger.debug(f"Test verification error: {ve}")
                criterion.status = AcceptanceStatus.FAILED
                criterion.verification_result = f"Test motoru yürütme hatası: {ve}"
                criterion.evidence_data = {"error": str(ve)}
                return

        if sprint and criterion.associated_task_ids:
            tasks_to_check = [sprint.tasks[tid] for tid in criterion.associated_task_ids if tid in sprint.tasks]
            if tasks_to_check and all(t.status == TaskStatus.COMPLETED and t.result and t.result.success for t in tasks_to_check):
                criterion.status = AcceptanceStatus.PASSED
                criterion.verification_result = f"Test kontrolü: İlişkili görevler ({', '.join(criterion.associated_task_ids)}) hatasız tamamlandı."
                criterion.evidence_data = {"tasks": criterion.associated_task_ids}
                return
            else:
                criterion.status = AcceptanceStatus.FAILED
                criterion.verification_result = "Test kanıtı eksik: İlişkili görevler henüz tamamlanmadı veya başarısız."
                criterion.evidence_data = {"error": "Associated tasks not completed"}
                return

        criterion.status = AcceptanceStatus.FAILED
        criterion.verification_result = "Test sonucu doğrulanamadı (Test motoru veya komut yürütücü tanımlı değil)."
        criterion.evidence_data = {"error": "Missing verification engine or tool executor"}
        criterion.evidence_data = {"error": "Missing verification engine or tool executor"}

    def _verify_review_approval(self, criterion: AcceptanceCriterion, sprint: Sprint) -> None:
        """Verify review approval."""
        relevant_tasks = [
            sprint.tasks[tid]
            for tid in criterion.associated_task_ids
            if tid in sprint.tasks
        ]
        if not relevant_tasks:
            relevant_tasks = [t for t in sprint.tasks.values() if t.role == AgentRole.REVIEWER]

        if not relevant_tasks:
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = "Gözden geçirme onayı başarısız: Sprint içinde ilgili Reviewer görevi bulunmuyor."
            criterion.evidence_data = {"error": "No reviewer tasks found in sprint"}
            return

        all_approved = True
        reasons = []

        for t in relevant_tasks:
            if t.status != TaskStatus.COMPLETED or not t.result:
                all_approved = False
                reasons.append(f"Görev '{t.task_id}' henüz tamamlanmadı.")
                continue

            out = t.result.output
            if "[APPROVED]" in out:
                continue
            elif "[REJECTED" in out:
                all_approved = False
                reasons.append(f"Görev '{t.task_id}' gözden geçirme tarafından reddedildi.")
            else:
                all_approved = False
                reasons.append(f"Görev '{t.task_id}' çıktısında açık [APPROVED] onayı yok.")

        if all_approved:
            criterion.status = AcceptanceStatus.PASSED
            criterion.verification_result = "Tüm ilgili kod gözden geçirme görevleri [APPROVED] kararı aldı."
            criterion.evidence_data = {"approved_tasks": [t.task_id for t in relevant_tasks]}
        else:
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = f"Gözden geçirme onayı eksik veya ret var: {'; '.join(reasons)}"
            criterion.evidence_data = {"reasons": reasons}

    def _verify_manual_approval(self, criterion: AcceptanceCriterion, sprint: Sprint) -> None:
        """Verify manual approval."""
        matching_reqs = [
            req for req in sprint.approval_requests.values()
            if req.task_id in criterion.associated_task_ids or req.approval_id == criterion.id
        ]

        if not matching_reqs:
            criterion.status = AcceptanceStatus.REQUIRES_USER_INPUT
            criterion.verification_result = "Kullanıcı onayı bekleniyor (Henüz onay talebi yanıtlanmadı)."
            return

        approved_count = sum(1 for r in matching_reqs if r.decision == ApprovalDecision.APPROVED)
        rejected_count = sum(1 for r in matching_reqs if r.decision == ApprovalDecision.REJECTED)

        if rejected_count > 0:
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = "Kullanıcı manuel onayı reddetti."
            criterion.evidence_data = {"decision": "rejected"}
        elif approved_count == len(matching_reqs):
            criterion.status = AcceptanceStatus.PASSED
            criterion.verification_result = "Kullanıcı manuel onayı onayladı."
            criterion.evidence_data = {"decision": "approved"}
        else:
            criterion.status = AcceptanceStatus.REQUIRES_USER_INPUT
            criterion.verification_result = "Kullanıcı manuel onayı bekleniyor."

    def _verify_generic(self, criterion: AcceptanceCriterion, sprint: Sprint) -> None:
        """Verify generic."""
        if not criterion.associated_task_ids:
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = "Genel kriter doğrulanamadı: İlişkili görev veya somut kanıt bulunmuyor."
            criterion.evidence_data = {"error": "No associated task IDs or concrete evidence"}
            return

        all_completed = all(
            sprint.tasks.get(tid) and sprint.tasks[tid].status == TaskStatus.COMPLETED
            for tid in criterion.associated_task_ids
        )
        if all_completed:
            criterion.status = AcceptanceStatus.PASSED
            criterion.verification_result = "İlişkili tüm görevler başarıyla tamamlandı."
        else:
            criterion.status = AcceptanceStatus.FAILED
            criterion.verification_result = "İlişkili bazı görevler henüz tamamlanmadı veya başarısız."
