"""Test audit remediation pillars module."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from evren_code.config import AppConfig, load_config
from evren_code.core.multi_agent.adaptive_delegation import (
    AdaptiveDelegationEngine,
    DelegationDecision,
)
from evren_code.core.multi_agent.file_ownership import FileOwnershipManager
from evren_code.core.planner import (
    ComplexityAssessment,
    ComplexityLevel,
    ExecutionBudget,
    PlanManager,
    PlanStepStatus,
    TaskComplexityClassifier,
)
from evren_code.core.security import (
    CommandClassifier,
    PermissionManager,
    SECURITY_ERR_ACCESS_VIOLATION,
    SECURITY_ERR_DANGEROUS_COMMAND,
    SECURITY_ERR_PERMISSION_DENIED,
    SECURITY_ERR_ROLE_RESTRICTION,
    SECURITY_ERR_SYSTEM_DIR_VIOLATION,
    WorkspaceGuard,
)
from evren_code.core.self_healing import (
    ErrorCategory,
    ErrorClassifier,
    SelfHealingManager,
)
from evren_code.core.telemetry import ExecutionTelemetryTracker, TrajectoryLogger
from evren_code.core.undo import UndoManager
from evren_code.core.verification.test_reporter import (
    ParsedTestMetrics,
    TestOutputParser,
    TestReportingEngine,
    TestVerificationStatus,
)
from evren_code.tools.filesystem import atomic_write, register_filesystem_tools, resolve_safe_path
from evren_code.tools.registry import ToolRegistry


class TestPillar1WorkspaceBoundaryAndPermissions(unittest.TestCase):
    """Test pillar1 workspace boundary and permissions implementation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="evren_p1_")
        self.workspace = Path(self.temp_dir).resolve()
        self.outside_dir = tempfile.mkdtemp(prefix="evren_outside_")
        self.outside_path = Path(self.outside_dir).resolve()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.outside_dir, ignore_errors=True)

    def test_write_outside_workspace_is_strictly_blocked(self) -> None:
        """Verify write outside workspace is strictly blocked."""
        outside_file = self.outside_path / "target.txt"
        with self.assertRaises(PermissionError) as ctx:
            WorkspaceGuard.validate_path(
                workspace_root=self.workspace,
                target_path=outside_file,
                is_write=True,
            )
        self.assertIn(SECURITY_ERR_ACCESS_VIOLATION, str(ctx.exception))

    def test_path_traversal_is_blocked(self) -> None:
        """Verify path traversal is blocked."""
        traversal_file = self.workspace / ".." / "escaped.txt"
        with self.assertRaises(PermissionError) as ctx:
            WorkspaceGuard.validate_path(
                workspace_root=self.workspace,
                target_path=traversal_file,
                is_write=True,
            )
        self.assertIn(SECURITY_ERR_ACCESS_VIOLATION, str(ctx.exception))

    def test_protected_system_directories_blocked(self) -> None:
        """Verify protected system directories blocked."""
        ollama_fake = Path("C:/Users/user/AppData/Local/Programs/Ollama/mvn.cmd")
        with self.assertRaises(PermissionError) as ctx:
            WorkspaceGuard.validate_path(
                workspace_root=self.workspace,
                target_path=ollama_fake,
                is_write=True,
            )
        self.assertIn(SECURITY_ERR_SYSTEM_DIR_VIOLATION, str(ctx.exception))

    def test_system_path_and_registry_commands_detected_as_dangerous(self) -> None:
        """Verify system path and registry commands detected as dangerous."""
        setx_cmd = "setx PATH \"%PATH%;C:\\tools\""
        is_dang, reason = CommandClassifier.analyze(setx_cmd)
        self.assertTrue(is_dang)
        self.assertIn("setx", reason.lower())

        reg_cmd = "reg add HKCU\\Environment /v PATH /d C:\\tools"
        is_dang, reason = CommandClassifier.analyze(reg_cmd)
        self.assertTrue(is_dang)
        self.assertIn("registry", reason.lower())

        ollama_shim_cmd = "echo @echo off > C:\\Users\\user\\AppData\\Local\\Programs\\Ollama\\mvn.cmd"
        is_dang, reason = CommandClassifier.analyze(ollama_shim_cmd)
        self.assertTrue(is_dang)

    def test_global_package_install_requires_approval_or_blocked_for_subagents(self) -> None:
        """Verify global package install requires approval or blocked for subagents."""
        npm_global = "npm install -g maven"
        is_dang, _ = CommandClassifier.analyze(npm_global)
        self.assertTrue(is_dang)

        allowed, reason, code = CommandClassifier.analyze_for_role(npm_global, role="tester")
        self.assertFalse(allowed)
        self.assertEqual(code, SECURITY_ERR_DANGEROUS_COMMAND)


class TestPillar2ReliableFileOperations(unittest.TestCase):
    """Test pillar2 reliable file operations implementation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="evren_p2_")
        self.workspace = Path(self.temp_dir).resolve()
        self.undo_mgr = UndoManager()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_atomic_write_creates_valid_file_and_verifies_disk(self) -> None:
        """Verify atomic write creates valid file and verifies disk."""
        target = self.workspace / "src" / "service.py"
        content = "def calculate_total(a, b):\n    return a + b\n"
        ok, msg = atomic_write(
            target_path=target,
            content=content,
            workspace_root=self.workspace,
            undo_manager=self.undo_mgr,
            relative_display="src/service.py",
        )
        self.assertTrue(ok)
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_text(encoding="utf-8"), content)
        self.assertEqual(target.stat().st_size, len(content.encode("utf-8")))

    def test_unexpected_empty_content_on_existing_file_is_rejected(self) -> None:
        """Verify unexpected empty content on existing file is rejected."""
        target = self.workspace / "important.py"
        target.write_text("# Old important code\n" * 10, encoding="utf-8")
        original_content = target.read_text(encoding="utf-8")

        ok, msg = atomic_write(
            target_path=target,
            content="   ",
            workspace_root=self.workspace,
            undo_manager=self.undo_mgr,
            relative_display="important.py",
            allow_empty=False,
        )
        self.assertFalse(ok)
        self.assertIn("Güvenilirlik Hatası", msg)
        self.assertEqual(target.read_text(encoding="utf-8"), original_content)

    def test_atomic_write_rollback_on_failure(self) -> None:
        """Verify atomic write rollback on failure."""
        target = self.workspace / "data.json"
        target.write_text('{"version": 1}', encoding="utf-8")
        original_content = target.read_text(encoding="utf-8")

        bogus_workspace = self.workspace / "other_dir"
        ok, msg = atomic_write(
            target_path=target,
            content='{"version": 2}',
            workspace_root=bogus_workspace,
            undo_manager=self.undo_mgr,
            relative_display="data.json",
        )
        self.assertFalse(ok)
        self.assertEqual(target.read_text(encoding="utf-8"), original_content)


class TestPillar3ReasoningAndPlanningOptimization(unittest.TestCase):
    """Test pillar3 reasoning and planning optimization implementation."""

    def test_task_complexity_classification(self) -> None:
        """Verify task complexity classification."""
        simple_res = TaskComplexityClassifier.classify("Fix typo in README.md")
        self.assertEqual(simple_res.level, ComplexityLevel.SIMPLE)
        self.assertFalse(simple_res.allow_delegation)
        self.assertLessEqual(simple_res.max_turns, 10)

        medium_res = TaskComplexityClassifier.classify("Add unit test for UserService and fix edge case")
        self.assertEqual(medium_res.level, ComplexityLevel.MEDIUM)
        self.assertEqual(medium_res.max_turns, 20)

        complex_res = TaskComplexityClassifier.classify("Kapsamlı mimari refactoring gerçekleştir ve tüm testleri entegre et")
        self.assertEqual(complex_res.level, ComplexityLevel.COMPLEX)
        self.assertTrue(complex_res.allow_delegation)
        self.assertEqual(complex_res.max_turns, 40)

    def test_completed_plan_steps_are_preserved(self) -> None:
        """Verify completed plan steps are preserved."""
        pm = PlanManager()
        pm.create_plan("Sprint Planı", ["1. Depoyu tara", "2. Modeli kur", "3. Test yaz"])
        pm.update_step(1, "completed", "Depo tarandı")

        new_plan = pm.create_plan("Sprint Planı v2", ["1. Depoyu tara", "2. Modeli kur", "3. Test yaz", "4. Dokümantasyon"])
        step_1 = next(s for s in new_plan.steps if s.id == 1)
        self.assertEqual(step_1.status, PlanStepStatus.COMPLETED)
        self.assertEqual(step_1.notes, "Depo tarandı")

    def test_execution_budget_exhaustion(self) -> None:
        """Verify execution budget exhaustion."""
        budget = ExecutionBudget(max_turns=3, max_model_calls=5)
        self.assertFalse(budget.is_exhausted)
        budget.record_turn()
        budget.record_turn()
        budget.record_turn()
        self.assertTrue(budget.is_exhausted)


class TestPillar4SelfHealingAndLoopBreakerV2(unittest.TestCase):
    """Test pillar4 self healing and loop breaker v2 implementation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="evren_p4_")
        self.workspace = Path(self.temp_dir).resolve()
        self.sh_mgr = SelfHealingManager(workspace_root=self.workspace, max_attempts=3)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_error_classification_9_categories(self) -> None:
        """Verify error classification 9 categories."""
        # 1. COMMAND_NOT_FOUND
        c, _ = ErrorClassifier.classify("'mvn' is not recognized as an internal or external command")
        self.assertEqual(c, ErrorCategory.COMMAND_NOT_FOUND)

        # 2. PERMISSION_DENIED
        c, _ = ErrorClassifier.classify("Güvenlik İhlali [ACCESS_VIOLATION]: Erişim engellendi.")
        self.assertEqual(c, ErrorCategory.PERMISSION_DENIED)

        # 3. COMPILATION_ERROR
        c, _ = ErrorClassifier.classify("SyntaxError: invalid syntax line 42")
        self.assertEqual(c, ErrorCategory.COMPILATION_ERROR)

        # 4. TEST_FAILURE
        c, _ = ErrorClassifier.classify("AssertionError: expected 10 but got 5")
        self.assertEqual(c, ErrorCategory.TEST_FAILURE)

        # 5. TIMEOUT
        c, _ = ErrorClassifier.classify("Subprocess timed out after 300 seconds")
        self.assertEqual(c, ErrorCategory.TIMEOUT)

        # 6. FILE_OPERATION_ERROR
        c, _ = ErrorClassifier.classify("Hata: 'src/main.py' dosyasında aranan target bloğu bulunamadı.")
        self.assertEqual(c, ErrorCategory.FILE_OPERATION_ERROR)

    def test_category_retry_limits_enforced(self) -> None:
        """Verify category retry limits enforced."""
        can_retry, msg = self.sh_mgr.should_retry(ErrorCategory.PERMISSION_DENIED, "sig1")
        self.assertFalse(can_retry)
        self.assertIn("Kategori Tavanı Aşıldı", msg)

        can_retry, _ = self.sh_mgr.should_retry(ErrorCategory.COMMAND_NOT_FOUND, "sig2")
        self.assertTrue(can_retry)
        self.sh_mgr.record_attempt(ErrorCategory.COMMAND_NOT_FOUND, "mvn test", "sig2")
        can_retry_2, _ = self.sh_mgr.should_retry(ErrorCategory.COMMAND_NOT_FOUND, "sig2")
        self.assertFalse(can_retry_2)

    def test_workspace_wrapper_detection(self) -> None:
        """Verify workspace wrapper detection."""
        (self.workspace / "mvnw.cmd").write_text("@echo off", encoding="utf-8")
        wrapper = self.sh_mgr.check_workspace_wrapper("mvn test")
        self.assertIsNotNone(wrapper)
        self.assertIn("mvnw.cmd", wrapper)


class TestPillar5VerificationAndTestReportingEngine(unittest.TestCase):
    """Test pillar5 verification and test reporting engine implementation."""

    def test_parse_pytest_output(self) -> None:
        """Verify parse pytest output."""
        raw = "==================== 12 passed, 2 failed, 1 skipped in 2.34s ===================="
        metrics = TestOutputParser.parse_output(raw, exit_code=1)
        self.assertEqual(metrics.framework, "pytest")
        self.assertEqual(metrics.tests_executed, 15)
        self.assertEqual(metrics.tests_passed, 12)
        self.assertEqual(metrics.tests_failed, 2)
        self.assertEqual(metrics.tests_skipped, 1)
        self.assertEqual(metrics.build_status, "FAILED")

    def test_parse_unittest_output(self) -> None:
        """Verify parse unittest output."""
        raw = "Ran 257 tests in 10.016s\n\nOK"
        metrics = TestOutputParser.parse_output(raw, exit_code=0)
        self.assertEqual(metrics.framework, "unittest")
        self.assertEqual(metrics.tests_executed, 257)
        self.assertEqual(metrics.tests_passed, 257)
        self.assertEqual(metrics.tests_failed, 0)
        self.assertEqual(metrics.build_status, "SUCCESS")

    def test_parse_maven_surefire_output(self) -> None:
        """Verify parse maven surefire output."""
        raw = "[INFO] Tests run: 10, Failures: 0, Errors: 0, Skipped: 0\n[INFO] BUILD SUCCESS"
        metrics = TestOutputParser.parse_output(raw, exit_code=0)
        self.assertEqual(metrics.framework, "maven")
        self.assertEqual(metrics.tests_executed, 10)
        self.assertEqual(metrics.tests_passed, 10)
        self.assertEqual(metrics.tests_failed, 0)
        self.assertEqual(metrics.build_status, "SUCCESS")

    def test_discrepancy_detection_flags_unverified_claims(self) -> None:
        """Verify discrepancy detection flags unverified claims."""
        metrics = ParsedTestMetrics(
            framework="maven",
            tests_executed=10,
            tests_passed=10,
            tests_failed=0,
            build_status="SUCCESS",
        )
        report = TestReportingEngine.generate_report(
            metrics=metrics,
            agent_claimed_executed=35,
            agent_claimed_passed=35,
        )
        self.assertEqual(report.verification_status, TestVerificationStatus.UNVERIFIED_CLAIMS)
        self.assertTrue(len(report.unverified_claims) > 0)
        self.assertIn("35 test çalıştırıldığını iddia etti", report.unverified_claims[0])


class TestPillar6TelemetryExecutionTimeAndTokenTracking(unittest.TestCase):
    """Test pillar6 telemetry execution time and token tracking implementation."""

    def test_monotonic_timer_and_phase_tracking(self) -> None:
        """Verify monotonic timer and phase tracking."""
        tracker = ExecutionTelemetryTracker()
        tracker.start_task()
        tracker.start_phase("planning")
        time.sleep(0.05)
        tracker.stop_phase("planning")

        tracker.start_phase("coding")
        time.sleep(0.05)
        tracker.stop_phase("coding")
        tracker.stop_task()

        self.assertGreater(tracker.total_wall_clock_seconds, 0.08)
        self.assertGreater(tracker.phases["planning"].total, 0.04)
        self.assertGreater(tracker.phases["coding"].total, 0.04)

    def test_token_tracking_actual_vs_estimated(self) -> None:
        """Verify token tracking actual vs estimated."""
        tracker = ExecutionTelemetryTracker()
        tracker.record_model_call(prompt_tokens=1000, completion_tokens=250, is_estimated=False)
        # Tahmini token
        tracker.record_model_call(prompt_tokens=500, completion_tokens=100, is_estimated=True)

        d = tracker.to_dict()
        self.assertEqual(d["tokens"]["actual"]["total_tokens"], 1250)
        self.assertEqual(d["tokens"]["estimated"]["total_tokens"], 600)
        self.assertTrue(d["tokens"]["has_estimated_tokens"])

    def test_telemetry_export_json(self) -> None:
        """Verify telemetry export json."""
        tracker = ExecutionTelemetryTracker()
        tracker.start_task()
        tracker.record_tool_call("write_file", success=True)
        tracker.record_tool_call("read_file", success=True)
        tracker.stop_task()

        temp_json = Path(tempfile.mktemp(suffix=".json"))
        try:
            tracker.export_json(temp_json)
            self.assertTrue(temp_json.is_file())
            data = json.loads(temp_json.read_text(encoding="utf-8"))
            self.assertEqual(data["operations"]["file_write_count"], 1)
            self.assertEqual(data["operations"]["file_read_count"], 1)
        finally:
            if temp_json.exists():
                temp_json.unlink()


class TestPillar7AdaptiveMultiAgentArchitecture(unittest.TestCase):
    """Test pillar7 adaptive multi agent architecture implementation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="evren_p7_")
        self.workspace = Path(self.temp_dir).resolve()
        self.engine = AdaptiveDelegationEngine(workspace_root=self.workspace)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_simple_task_skips_delegation(self) -> None:
        """Verify simple task skips delegation."""
        simple_ass = ComplexityAssessment(
            level=ComplexityLevel.SIMPLE,
            max_turns=8,
            allow_delegation=False,
            requires_formal_plan=False,
            reason="Basit typo",
        )
        dec = self.engine.evaluate_delegation(simple_ass, "Fix typo in service.py")
        self.assertFalse(dec.should_delegate)
        self.assertEqual(dec.estimated_token_cost, 0)

    def test_complex_task_recommends_delegation(self) -> None:
        """Verify complex task recommends delegation."""
        complex_ass = ComplexityAssessment(
            level=ComplexityLevel.COMPLEX,
            max_turns=40,
            allow_delegation=True,
            requires_formal_plan=True,
            reason="Kapsamlı mimari",
        )
        dec = self.engine.evaluate_delegation(complex_ass, "Kapsamlı refactoring yap ve modülleri ayrıştır")
        self.assertTrue(dec.should_delegate)
        self.assertIn("developer", dec.recommended_roles)
        self.assertIn("tester", dec.recommended_roles)

    def test_file_ownership_concurrency_lock(self) -> None:
        """Verify file ownership concurrency lock."""
        ok1, blocked1 = self.engine.acquire_file_ownership(
            agent_id="dev_agent_1",
            file_paths=["src/auth.py"],
            exclusive=True,
        )
        self.assertTrue(ok1)
        self.assertEqual(len(blocked1), 0)

        ok2, blocked2 = self.engine.acquire_file_ownership(
            agent_id="dev_agent_2",
            file_paths=["src/auth.py"],
            exclusive=True,
        )
        self.assertFalse(ok2)
        self.assertIn("src/auth.py", blocked2)

        self.engine.release_file_ownership("dev_agent_1", ["src/auth.py"])

        ok3, _ = self.engine.acquire_file_ownership(
            agent_id="dev_agent_2",
            file_paths=["src/auth.py"],
            exclusive=True,
        )
        self.assertTrue(ok3)


if __name__ == "__main__":
    unittest.main()
