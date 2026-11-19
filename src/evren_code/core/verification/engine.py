"""Engine module."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from evren_code.core.repo_intelligence.engine import RepoIntelligenceEngine
from evren_code.core.security import ToolExecutor
from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine
from evren_code.core.verification.classifier import FailureClassifier
from evren_code.core.verification.corrector import CourseCorrectionEngine
from evren_code.core.verification.history import VerificationHistoryStore
from evren_code.core.verification.models import (
    CorrectionAttempt,
    FailureCategory,
    FailureClassification,
    RegressionReport,
    TestExecutionResult,
    TestFailure,
    VerificationReport,
    VerificationResult,
    VerificationStep,
    VerificationStatus,
)
from evren_code.core.verification.regression import RegressionDetector
from evren_code.core.verification.retry import RetryController
from evren_code.core.verification.test_selector import TestSelectionEngine
from evren_code.core.verification.validator import ChangeValidator


class VerificationEngine:
    """Verification engine implementation."""

    EXIT_CODE_RE = re.compile(r"Çıkış Kodu \(Exit Code\):\s*(-?\d+)")
    STDOUT_SPLIT_RE = re.compile(r"--- STDOUT ---\n(.*?)(?=(?:--- STDERR ---|\Z))", re.DOTALL)
    STDERR_SPLIT_RE = re.compile(r"--- STDERR ---\n(.*)", re.DOTALL)

    def __init__(
        self,
        workspace_root: Path,
        tool_executor: ToolExecutor,
        repo_intelligence: Optional[RepoIntelligenceEngine] = None,
        symbol_engine: Optional[CodebaseSymbolEngine] = None,
        history_store: Optional[VerificationHistoryStore] = None,
        max_attempts: int = 3,
        max_cycles: int = 4,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.tool_executor = tool_executor
        self.repo_intelligence = repo_intelligence
        self.symbol_engine = symbol_engine
        self.history_store = history_store or VerificationHistoryStore(self.workspace_root)
        self.max_attempts = max_attempts
        self.max_cycles = max_cycles

        # Alt motorlar
        self.test_selector = TestSelectionEngine(
            workspace_root=self.workspace_root,
            repo_intelligence=self.repo_intelligence,
            symbol_engine=self.symbol_engine,
        )
        self.validator = ChangeValidator(workspace_root=self.workspace_root)
        self.corrector = CourseCorrectionEngine(
            workspace_root=self.workspace_root,
            symbol_engine=self.symbol_engine,
            validator=self.validator,
        )
        self.retry_controller = RetryController(
            max_attempts=self.max_attempts,
            max_cycles=self.max_cycles,
        )

        self.last_report: Optional[VerificationReport] = None
        self.baseline_result: Optional[VerificationResult] = None

    def capture_baseline(self, is_trust_mode: bool = False) -> VerificationResult:
        """Capture baseline (baseline)."""
        steps = self.test_selector.select_tests_for_changes([])
        res = self.execute_steps(steps, is_trust_mode=is_trust_mode)
        self.baseline_result = res
        return res

    def execute_steps(
        self,
        steps: List[VerificationStep],
        is_trust_mode: bool = False,
        changed_files: Optional[List[str]] = None,
    ) -> VerificationResult:
        """Execute steps."""
        t0 = time.perf_counter()
        results: List[TestExecutionResult] = []
        overall = VerificationStatus.VERIFIED

        baseline_fails = set()
        if self.baseline_result:
            for s in self.baseline_result.steps:
                for f in s.failures:
                    baseline_fails.add(f.test_name)

        classifier = FailureClassifier(
            changed_files=changed_files or [],
            baseline_failures=baseline_fails,
        )

        for step in steps:
            step_t0 = time.perf_counter()
            exec_str = self.tool_executor.execute(
                tool_name="run_command",
                arguments={"command": step.command, "cwd": step.cwd},
                caller_role="tester",
                caller_agent="verification_engine",
                is_trust_mode=is_trust_mode,
            )
            step_duration = time.perf_counter() - step_t0

            exit_code, stdout, stderr, status = self._parse_execution_output(exec_str)

            exec_res = TestExecutionResult(
                command=step.command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=step_duration,
                timestamp=datetime.now(timezone.utc).isoformat(),
                step_name=step.name,
                target_type=step.target_type,
                status=status,
            )

            from evren_code.core.verification.test_reporter import TestOutputParser
            parsed_m = TestOutputParser.parse_output(
                f"{stdout}\n{stderr}\n{exec_str}",
                exit_code=exit_code,
                workspace_root=self.workspace_root,
            )

            if exit_code != 0 or status in (VerificationStatus.FAILED, VerificationStatus.BLOCKED):
                failures = classifier.parse_and_classify(exec_res)
                exec_res.failures = failures
                exec_res.failed_count = max(parsed_m.tests_failed, len(failures) if failures else 1)
                exec_res.passed_count = parsed_m.tests_passed
                if status == VerificationStatus.BLOCKED:
                    overall = VerificationStatus.BLOCKED
                elif overall != VerificationStatus.BLOCKED:
                    overall = VerificationStatus.FAILED
            else:
                exec_res.passed_count = max(1, parsed_m.tests_passed)
                exec_res.failed_count = parsed_m.tests_failed

            results.append(exec_res)

        total_duration = time.perf_counter() - t0
        return VerificationResult(
            steps=results,
            overall_status=overall,
            total_duration_seconds=total_duration,
        )

    def _parse_execution_output(self, raw_str: str) -> Tuple[int, str, str, VerificationStatus]:
        """Parses exit_code, stdout, stderr, and verification status from tool execution output."""
        if "[GÜVENLİK ENGELİ" in raw_str or "[KULLANICI REDDİ" in raw_str or "[SECURITY POLICY BLOCK" in raw_str or "[USER REJECTION" in raw_str:
            return -1, "", raw_str, VerificationStatus.BLOCKED

        if raw_str.startswith("Hata:") or raw_str.startswith("Error:") or raw_str.startswith("Güvenlik Hatası:") or raw_str.startswith("Security Error:"):
            return -1, "", raw_str, VerificationStatus.FAILED

        m_code = self.EXIT_CODE_RE.search(raw_str)
        exit_code = int(m_code.group(1)) if m_code else (0 if ("Başarılı" in raw_str or "Success" in raw_str) else 0)

        m_out = self.STDOUT_SPLIT_RE.search(raw_str)
        stdout = m_out.group(1).strip() if m_out else raw_str

        m_err = self.STDERR_SPLIT_RE.search(raw_str)
        stderr = m_err.group(1).strip() if m_err else ""

        status = VerificationStatus.VERIFIED if exit_code == 0 else VerificationStatus.FAILED
        return exit_code, stdout, stderr, status

    def _take_snapshot(self, file_paths: Set[str]) -> Dict[str, Optional[str]]:
        """Take snapshot."""
        snap: Dict[str, Optional[str]] = {}
        for f in file_paths:
            try:
                p = (self.workspace_root / f).resolve()
                if p.is_file():
                    snap[f] = p.read_text(encoding="utf-8", errors="replace")
                else:
                    snap[f] = None
            except Exception:
                snap[f] = None
        return snap

    def _rollback_snapshot(
        self,
        snapshots: Dict[str, Optional[str]],
        created_or_modified_files: Set[str],
    ) -> Tuple[bool, List[str]]:
        """Rollback snapshot."""
        errors: List[str] = []
        all_files = set(snapshots.keys()) | set(created_or_modified_files)

        for rel_f in all_files:
            try:
                p = (self.workspace_root / rel_f).resolve()
                orig_cnt = snapshots.get(rel_f)
                if orig_cnt is None:
                    if p.is_file() or p.is_symlink():
                        p.unlink(missing_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(orig_cnt, encoding="utf-8")
            except Exception as exc:
                errors.append(f"'{rel_f}' geri alınamadı: {exc}")

        return (len(errors) == 0, errors)

    def run_verification(
        self,
        changed_files: List[str],
        task_description: str = "",
        baseline_result: Optional[VerificationResult] = None,
        orchestrator_correction_handler: Optional[Callable[[Dict[str, Any]], Tuple[bool, List[str]]]] = None,
        is_trust_mode: bool = False,
    ) -> VerificationReport:
        """Run verification."""
        start_time = time.perf_counter()
        self.retry_controller.reset()

        base_res = baseline_result or self.baseline_result
        regression_detector = RegressionDetector(baseline_result=base_res)

        impact_summary: List[str] = []
        if self.symbol_engine:
            for cf in changed_files:
                try:
                    syms = self.symbol_engine.graph.get_file_symbols(cf)
                    for s in syms[:3]:
                        imp = self.symbol_engine.impact_analyzer.analyze_symbol(s.name)
                        impact_summary.append(
                            f"Sembol `{s.name}` -> {len(imp.direct_callers)} doğrudan çağıran, "
                            f"{len(imp.related_test_files)} ilgili test"
                        )
                except Exception:
                    pass

        steps = self.test_selector.select_tests_for_changes(changed_files)
        verification_res = self.execute_steps(steps, is_trust_mode=is_trust_mode, changed_files=changed_files)

        # Regresyon analizi
        reg_report = regression_detector.detect_regressions(verification_res)

        all_classifications: List[FailureClassification] = []
        for step_res in verification_res.steps:
            for fail in step_res.failures:
                if fail.classification:
                    all_classifications.append(fail.classification)

        correction_attempts: List[CorrectionAttempt] = []

        while True:
            self.retry_controller.record_cycle()
            should_retry, reason = self.retry_controller.should_retry(
                status=verification_res.overall_status,
                classifications=all_classifications,
                regression=reg_report,
            )

            if not should_retry or not orchestrator_correction_handler:
                break

            target_failure: Optional[TestFailure] = None
            for step_res in verification_res.steps:
                for f in step_res.failures:
                    if f.classification and f.classification.category == FailureCategory.AGENT_INTRODUCED:
                        target_failure = f
                        break
                    elif not target_failure:
                        target_failure = f
                if target_failure and target_failure.classification and target_failure.classification.category == FailureCategory.AGENT_INTRODUCED:
                    break

            if not target_failure:
                break

            attempt_idx = self.retry_controller.current_attempt + 1
            plan = self.corrector.formulate_correction_plan(
                failure=target_failure,
                changed_files=changed_files,
                attempt_number=attempt_idx,
            )

            target_file_set = set(changed_files + ([target_failure.file_path] if target_failure.file_path else []))
            snapshots = self._take_snapshot(target_file_set)

            success, modified_files = orchestrator_correction_handler(plan)

            validation_passed = True
            validation_violations: List[str] = []
            if success and modified_files:
                for mf in modified_files:
                    p = (self.workspace_root / mf).resolve()
                    new_cnt = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
                    old_cnt = snapshots.get(mf) or ""
                    outcome = self.validator.validate_patch(mf, old_cnt, new_cnt)
                    if not outcome.is_valid:
                        validation_passed = False
                        validation_violations.extend(outcome.violations)

                if not validation_passed:
                    rollback_ok, rollback_errors = self._rollback_snapshot(snapshots, set(modified_files))
                    success = False
                    if not rollback_ok:
                        reason = f"KRİTİK ROLLBACK HATASI: {'; '.join(rollback_errors)} | İhlal: {'; '.join(validation_violations)}"
                    else:
                        reason = f"ChangeValidator Güvenlik Engeli: {'; '.join(validation_violations)}"

            attempt = CorrectionAttempt(
                attempt_number=attempt_idx,
                failure_addressed=target_failure.test_name,
                plan_description=plan.get("plan_description", ""),
                modified_files=modified_files,
                success=success,
                notes=reason,
            )
            self.retry_controller.record_attempt(attempt)
            correction_attempts.append(attempt)

            if success and modified_files:
                self.corrector.post_correction_update(modified_files)
                new_changed = list(set(changed_files + modified_files))
                steps = self.test_selector.select_tests_for_changes(new_changed)
                verification_res = self.execute_steps(steps, is_trust_mode=is_trust_mode, changed_files=new_changed)
                reg_report = regression_detector.detect_regressions(verification_res)

                all_classifications = []
                for s_res in verification_res.steps:
                    for f in s_res.failures:
                        if f.classification:
                            all_classifications.append(f.classification)
            else:
                break

        total_duration = time.perf_counter() - start_time
        final_status = verification_res.overall_status

        report = VerificationReport(
            task_description=task_description or "Otonom Görev Doğrulaması",
            implementation_status="SUCCESS" if final_status == VerificationStatus.VERIFIED else (
                "BLOCKED" if final_status == VerificationStatus.BLOCKED else "FAILED"
            ),
            changed_files=changed_files,
            impact_summary=impact_summary,
            step_results=verification_res.steps,
            correction_attempts=correction_attempts,
            failure_analysis=[c.to_dict() for c in all_classifications],
            regression_analysis=reg_report,
            final_status=final_status,
            total_duration_seconds=total_duration,
        )

        # Depoya kaydet
        self.history_store.save_report(report)
        self.last_report = report
        return report

    def format_report_markdown(self, report: VerificationReport) -> str:
        """Generates a token-efficient markdown verification report."""
        status_icons = {
            VerificationStatus.VERIFIED: "✅ DOĞRULANDI (VERIFIED)",
            VerificationStatus.PARTIALLY_VERIFIED: "⚠️ KISMİ DOĞRULAMA (PARTIALLY_VERIFIED)",
            VerificationStatus.FAILED: "❌ BAŞARISIZ (FAILED)",
            VerificationStatus.BLOCKED: "🛑 GÜVENLİK ENGELİ (BLOCKED)",
            VerificationStatus.NOT_RUN: "⚪ ÇALIŞTIRILMADI (NOT_RUN)",
        }

        lines = [
            f"# EVREN Agent Doğrulama Raporu: {status_icons.get(report.final_status, report.final_status.value)}",
            f"- **Task / Görev:** {report.task_description}",
            f"- **Duration / Süre:** {report.total_duration_seconds:.2f}s",
            f"- **Modified Files / Değiştirilen Dosyalar ({len(report.changed_files)}):** {', '.join(report.changed_files) if report.changed_files else 'None'}",
        ]

        if report.impact_summary:
            lines.append("\n### Symbol Impact Summary / Sembol Etki Özeti:")
            for imp in report.impact_summary[:5]:
                lines.append(f"- {imp}")

        lines.append(f"\n### Executed Verification Steps / Yürütülen Doğrulama Adımları ({len(report.step_results)}):")
        for step in report.step_results:
            icon = "✅" if step.exit_code == 0 else "❌"
            lines.append(
                f"- {icon} `{step.step_name or step.command}` -> Exit: {step.exit_code} ({step.duration_seconds:.2f}s)"
            )
            if step.failures:
                for f in step.failures[:3]:
                    cat_str = f"[{f.classification.category.value}] " if f.classification else ""
                    lines.append(f"  * {cat_str}`{f.test_name}`: {f.failure_message[:150]}")

        if report.correction_attempts:
            lines.append(f"\n### Autonomous Remediation Attempts / Otonom Düzeltme Girişimleri ({len(report.correction_attempts)}):")
            for ca in report.correction_attempts:
                c_icon = "✅" if ca.success else "❌"
                lines.append(
                    f"- {c_icon} Attempt #{ca.attempt_number}: {ca.failure_addressed} -> {ca.plan_description}"
                )

        if report.regression_analysis:
            reg = report.regression_analysis
            lines.append("\n### Regression Status / Regresyon Durumu:")
            if reg.is_baseline_missing:
                lines.append("- *No baseline recorded for regression comparison.*")
            elif reg.is_regression_detected:
                lines.append(f"- ⚠️ **{len(reg.new_regressions)} new regressions detected!**")
                for nr in reg.new_regressions[:5]:
                    lines.append(f"  * `{nr}`")
            else:
                lines.append("- ✅ No new regressions detected.")

            if reg.resolved_failures:
                lines.append(f"- 🎉 {len(reg.resolved_failures)} previous failures resolved.")

        return "\n".join(lines)
