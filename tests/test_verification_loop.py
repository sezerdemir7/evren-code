"""Test verification loop module."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from evren_code.config import AppConfig
from evren_code.core.security import (
    PermissionManager,
    SECURITY_ERR_PERMISSION_DENIED,
    ToolExecutor,
)
from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine
from evren_code.core.verification.classifier import FailureClassifier
from evren_code.core.verification.corrector import CourseCorrectionEngine
from evren_code.core.verification.engine import VerificationEngine
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
    VerificationStatus,
    VerificationStep,
)
from evren_code.core.verification.regression import RegressionDetector
from evren_code.core.verification.retry import RetryController
from evren_code.core.verification.test_selector import TestSelectionEngine
from evren_code.core.verification.tools import register_verification_tools
from evren_code.core.verification.validator import ChangeValidator
from evren_code.tools.registry import ToolRegistry


class TestVerificationModels(unittest.TestCase):
    """Test verification models implementation."""

    def test_serialization_and_deserialization(self) -> None:
        fail = TestFailure(
            test_name="tests/test_auth.py::test_login",
            file_path="tests/test_auth.py",
            line_number=42,
            failure_message="AssertionError: 401 != 200",
            stack_trace="Traceback ... in test_login assert 401 == 200",
            classification=FailureClassification(
                category=FailureCategory.AGENT_INTRODUCED,
                confidence=0.92,
                evidence="Hata doğrudan değiştirilen auth dosyasından kaynaklanıyor",
                related_files=["src/auth.py"],
                suggested_action="src/auth.py içindeki login dönüşünü kontrol edin",
            ),
        )

        step_res = TestExecutionResult(
            command="pytest tests/test_auth.py",
            exit_code=1,
            stdout="1 failed in 0.05s",
            stderr="",
            duration_seconds=0.15,
            timestamp="2026-11-05T12:00:00Z",
            step_name="python:targeted_tests",
            failed_count=1,
            failures=[fail],
            status=VerificationStatus.FAILED,
        )

        report = VerificationReport(
            task_description="Giriş servisi güncellemesi",
            implementation_status="FAILED",
            changed_files=["src/auth.py"],
            impact_summary=["src/auth.py: 1 doğrudan çağıran"],
            step_results=[step_res],
            correction_attempts=[],
            failure_analysis=[fail.classification.to_dict()],
            regression_analysis=RegressionReport(
                new_regressions=["tests/test_auth.py::test_login"],
                is_regression_detected=True,
            ),
            final_status=VerificationStatus.FAILED,
            total_duration_seconds=0.25,
        )

        d = report.to_dict()
        restored = VerificationReport.from_dict(d)

        self.assertEqual(restored.task_description, "Giriş servisi güncellemesi")
        self.assertEqual(restored.final_status, VerificationStatus.FAILED)
        self.assertEqual(len(restored.step_results), 1)
        self.assertEqual(len(restored.step_results[0].failures), 1)
        self.assertEqual(
            restored.step_results[0].failures[0].classification.category,
            FailureCategory.AGENT_INTRODUCED,
        )
        self.assertTrue(restored.regression_analysis.is_regression_detected)


class TestTestSelectionEngine(unittest.TestCase):
    """Test test selection engine implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_select_directly_changed_test_file(self) -> None:
        selector = TestSelectionEngine(self.workspace_root)
        steps = selector.select_tests_for_changes(["tests/test_auth.py"])

        self.assertTrue(any(s.target_type == "test" for s in steps))
        self.assertTrue(any("tests/test_auth.py" in s.command for s in steps))

    def test_select_with_typecheck_and_build(self) -> None:
        (self.workspace_root / "tsconfig.json").write_text("{}", encoding="utf-8")
        selector = TestSelectionEngine(self.workspace_root)
        steps = selector.select_tests_for_changes(["src/index.ts"])

        self.assertTrue(any(s.target_type == "typecheck" for s in steps))
        self.assertTrue(any("tsc" in s.command for s in steps))

    def test_select_fallback_suite_when_no_direct_tests(self) -> None:
        selector = TestSelectionEngine(self.workspace_root)
        steps = selector.select_tests_for_changes(["src/utils.py"])

        self.assertTrue(len(steps) > 0)
        self.assertEqual(steps[-1].target_type, "test")
        self.assertTrue("unittest" in steps[-1].command or "pytest" in steps[-1].command)


class TestFailureClassifier(unittest.TestCase):
    """Test failure classifier implementation."""

    def test_classify_agent_introduced(self) -> None:
        classifier = FailureClassifier(changed_files=["src/auth/service.py"])
        fail = TestFailure(
            test_name="tests/test_auth.py::test_login",
            file_path="tests/test_auth.py",
            failure_message="AssertionError: False is not True",
            stack_trace="File 'src/auth/service.py', line 55, in authenticate\nassert token is not None",
        )
        clf = classifier.classify_failure(fail)

        self.assertEqual(clf.category, FailureCategory.AGENT_INTRODUCED)
        self.assertGreaterEqual(clf.confidence, 0.70)
        self.assertTrue(any("service.py" in f for f in clf.related_files))

    def test_classify_pre_existing(self) -> None:
        baseline = {"tests/test_legacy.py::test_old_bug"}
        classifier = FailureClassifier(
            changed_files=["src/auth.py"],
            baseline_failures=baseline,
        )
        fail = TestFailure(
            test_name="tests/test_legacy.py::test_old_bug",
            failure_message="Legacy issue",
        )
        clf = classifier.classify_failure(fail)

        self.assertEqual(clf.category, FailureCategory.PRE_EXISTING)
        self.assertGreaterEqual(clf.confidence, 0.90)

    def test_classify_environment(self) -> None:
        classifier = FailureClassifier(changed_files=["src/worker.py"])
        fail = TestFailure(
            test_name="test_worker",
            failure_message="ConnectionRefusedError: [Errno 111] Connection refused",
            stack_trace="socket.error: address already in use",
        )
        clf = classifier.classify_failure(fail)

        self.assertEqual(clf.category, FailureCategory.ENVIRONMENT)
        self.assertGreaterEqual(clf.confidence, 0.85)

    def test_classify_dependency(self) -> None:
        classifier = FailureClassifier(changed_files=["src/main.py"])
        fail = TestFailure(
            test_name="test_main",
            failure_message="ModuleNotFoundError: No module named 'fastapi'",
            stack_trace="ImportError: No module named 'fastapi'",
        )
        clf = classifier.classify_failure(fail)

        self.assertEqual(clf.category, FailureCategory.DEPENDENCY)
        self.assertGreaterEqual(clf.confidence, 0.85)

    def test_classify_test_configuration(self) -> None:
        classifier = FailureClassifier(changed_files=["src/main.py"])
        fail = TestFailure(
            test_name="pytest_run",
            failure_message="pytest: error: unrecognized arguments: --unknown-flag",
        )
        clf = classifier.classify_failure(fail)

        self.assertEqual(clf.category, FailureCategory.TEST_CONFIGURATION)
        self.assertGreaterEqual(clf.confidence, 0.80)

    def test_classify_unknown(self) -> None:
        classifier = FailureClassifier(changed_files=["src/main.py"])
        fail = TestFailure(
            test_name="test_weird",
            failure_message="Unexpected internal code 99",
        )
        clf = classifier.classify_failure(fail)

        self.assertEqual(clf.category, FailureCategory.UNKNOWN)


class TestRegressionDetector(unittest.TestCase):
    """Test regression detector implementation."""

    def test_detect_new_regressions_and_resolved(self) -> None:
        # Baseline: test_a failing, test_b passing
        step_base = TestExecutionResult(
            command="pytest",
            exit_code=1,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            timestamp="",
            failures=[TestFailure(test_name="test_a")],
        )
        baseline = VerificationResult(steps=[step_base])

        detector = RegressionDetector(baseline_result=baseline)

        # Current: test_a passed (resolved), but test_b failed (new regression)
        step_curr = TestExecutionResult(
            command="pytest",
            exit_code=1,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            timestamp="",
            failures=[TestFailure(test_name="test_b")],
        )
        curr = VerificationResult(steps=[step_curr])

        rep = detector.detect_regressions(curr)

        self.assertTrue(rep.is_regression_detected)
        self.assertEqual(rep.new_regressions, ["test_b"])
        self.assertEqual(rep.resolved_failures, ["test_a"])
        self.assertEqual(rep.remaining_failures, [])

    def test_detect_missing_baseline(self) -> None:
        detector = RegressionDetector(baseline_result=None)
        curr = VerificationResult(steps=[TestExecutionResult(
            command="pytest", exit_code=1, stdout="", stderr="", duration_seconds=0.1, timestamp="",
            failures=[TestFailure(test_name="test_x")],
        )])
        rep = detector.detect_regressions(curr)

        self.assertTrue(rep.is_baseline_missing)
        self.assertFalse(rep.is_regression_detected)
        self.assertEqual(rep.remaining_failures, ["test_x"])


class TestChangeValidator(unittest.TestCase):
    """Test change validator implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_reject_test_weakening_decorator(self) -> None:
        validator = ChangeValidator(self.workspace_root)
        patch = "+ @pytest.mark.skip(reason='temporarily passing')\n+ def test_login(): pass"
        outcome = validator.validate_patch(
            file_path="tests/test_auth.py",
            old_content="def test_login(): pass",
            new_content="@pytest.mark.skip\ndef test_login(): pass",
            diff_text=patch,
        )
        self.assertFalse(outcome.is_valid)
        self.assertTrue(any("zayıflatma" in v.lower() for v in outcome.violations))

    def test_reject_assert_deletion_in_test_file(self) -> None:
        validator = ChangeValidator(self.workspace_root)
        patch = "- assert response.status_code == 200"
        outcome = validator.validate_patch(
            file_path="tests/test_api.py",
            old_content="assert response.status_code == 200",
            new_content="",
            diff_text=patch,
        )
        self.assertFalse(outcome.is_valid)
        self.assertTrue(any("silme" in v.lower() for v in outcome.violations))

    def test_accept_valid_surgical_edit(self) -> None:
        validator = ChangeValidator(self.workspace_root)
        old_code = "def add(a, b):\n    return a - b"
        new_code = "def add(a, b):\n    return a + b"
        outcome = validator.validate_patch(
            file_path="src/calc.py",
            old_content=old_code,
            new_content=new_code,
        )
        self.assertTrue(outcome.is_valid)
        self.assertEqual(len(outcome.violations), 0)


class TestRetryController(unittest.TestCase):
    """Test retry controller implementation."""

    def test_stop_on_verified(self) -> None:
        rc = RetryController(max_attempts=3)
        retry, msg = rc.should_retry(VerificationStatus.VERIFIED, [])
        self.assertFalse(retry)
        self.assertIn("VERIFIED", msg)

    def test_stop_on_max_attempts(self) -> None:
        rc = RetryController(max_attempts=2)
        rc.record_attempt(CorrectionAttempt(1, "test_1", "plan 1"))
        rc.record_attempt(CorrectionAttempt(2, "test_1", "plan 2"))

        retry, msg = rc.should_retry(VerificationStatus.FAILED, [
            FailureClassification(FailureCategory.AGENT_INTRODUCED, 0.9, "error")
        ])
        self.assertFalse(retry)
        self.assertIn("sınırına", msg)

    def test_stop_on_environment_only(self) -> None:
        rc = RetryController(max_attempts=3)
        retry, msg = rc.should_retry(VerificationStatus.FAILED, [
            FailureClassification(FailureCategory.ENVIRONMENT, 0.9, "Connection refused")
        ])
        self.assertFalse(retry)
        self.assertIn("ENVIRONMENT", msg)

    def test_stop_on_repetition_loop(self) -> None:
        rc = RetryController(max_attempts=5)
        clfs = [FailureClassification(FailureCategory.AGENT_INTRODUCED, 0.9, "Same assertion error at line 40")]

        rc.should_retry(VerificationStatus.FAILED, clfs)
        rc.record_attempt(CorrectionAttempt(1, "t1", "p1"))
        rc.should_retry(VerificationStatus.FAILED, clfs)
        rc.record_attempt(CorrectionAttempt(2, "t1", "p2"))
        retry, msg = rc.should_retry(VerificationStatus.FAILED, clfs)

        self.assertFalse(retry)
        self.assertIn("kısırdöngü", msg.lower())


class TestCourseCorrectionEngine(unittest.TestCase):
    """Test course correction engine implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_formulate_correction_plan(self) -> None:
        corrector = CourseCorrectionEngine(self.workspace_root)
        fail = TestFailure(
            test_name="tests/test_auth.py::test_jwt",
            file_path="src/auth.py",
            line_number=20,
            failure_message="SignatureError: secret expired",
            classification=FailureClassification(
                category=FailureCategory.AGENT_INTRODUCED,
                confidence=0.88,
                evidence="Token expiration mismatch",
                suggested_action="Token TTL süresini doğrulayın",
            ),
        )

        plan = corrector.formulate_correction_plan(fail, changed_files=["src/auth.py"], attempt_number=1)

        self.assertEqual(plan["attempt_number"], 1)
        self.assertIn("OTONOM DÜZELTME DÖNGÜSÜ", plan["correction_prompt"])
        self.assertIn("TESTLERİ SİLMEYİN", plan["correction_prompt"])
        self.assertIn("Token TTL süresini doğrulayın", plan["correction_prompt"])


class TestVerificationHistoryStore(unittest.TestCase):
    """Test verification history store implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_retrieve_latest_and_history(self) -> None:
        store = VerificationHistoryStore(self.workspace_root)
        report = VerificationReport(
            task_description="Birim test koşumu",
            implementation_status="SUCCESS",
            final_status=VerificationStatus.VERIFIED,
            total_duration_seconds=1.2,
        )

        saved_path = store.save_report(report)
        self.assertTrue(saved_path.exists())
        self.assertTrue(store.latest_file.exists())

        latest = store.get_latest_report()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.task_description, "Birim test koşumu")
        self.assertEqual(latest.final_status, VerificationStatus.VERIFIED)

        history = store.list_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["final_status"], "VERIFIED")


class TestVerificationEngineAndGovernance(unittest.TestCase):
    """Test verification engine and governance implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir)
        self.registry = ToolRegistry()
        self.permission_manager = PermissionManager(self.workspace_root)
        self.approval_delegate = MagicMock()
        self.tool_executor = ToolExecutor(
            registry=self.registry,
            permission_manager=self.permission_manager,
            approval_delegate=self.approval_delegate,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_verification_executes_strictly_through_tool_executor(self) -> None:
        # Mock executor spy
        mock_executor = MagicMock()
        mock_executor.execute.return_value = "Çıkış Kodu (Exit Code): 0\n--- STDOUT ---\n1 passed in 0.01s"

        engine = VerificationEngine(
            workspace_root=self.workspace_root,
            tool_executor=mock_executor,
        )

        steps = [VerificationStep(name="test:unit", command="pytest tests/", cwd=".")]
        res = engine.execute_steps(steps)

        self.assertEqual(res.overall_status, VerificationStatus.VERIFIED)
        mock_executor.execute.assert_called_once_with(
            tool_name="run_command",
            arguments={"command": "pytest tests/", "cwd": "."},
            caller_role="tester",
            caller_agent="verification_engine",
            is_trust_mode=False,
        )

    def test_verification_handles_security_block(self) -> None:
        mock_executor = MagicMock()
        mock_executor.execute.return_value = "[GÜVENLİK ENGELİ / ERR_DANGEROUS]: Tehlikeli komut engellendi"

        engine = VerificationEngine(
            workspace_root=self.workspace_root,
            tool_executor=mock_executor,
        )

        steps = [VerificationStep(name="test:bad", command="rm -rf /", cwd=".")]
        res = engine.execute_steps(steps)

        self.assertEqual(res.overall_status, VerificationStatus.BLOCKED)
        self.assertEqual(res.steps[0].status, VerificationStatus.BLOCKED)

    def test_register_verification_tools(self) -> None:
        mock_executor = MagicMock()
        mock_executor.execute.return_value = "Çıkış Kodu (Exit Code): 0\n--- STDOUT ---\n2 passed in 0.02s"

        engine = VerificationEngine(
            workspace_root=self.workspace_root,
            tool_executor=mock_executor,
        )
        register_verification_tools(self.registry, engine)

        self.assertIn("run_verification", self.registry._tools)
        self.assertIn("get_verification_report", self.registry._tools)
        self.assertIn("get_verification_history", self.registry._tools)

        out = self.registry.execute("run_verification", {"changed_files": "tests/test_foo.py"})
        self.assertIn("EVREN Agent Doğrulama Raporu", out)
        self.assertIn("DOĞRULANDI", out)


if __name__ == "__main__":
    unittest.main()
