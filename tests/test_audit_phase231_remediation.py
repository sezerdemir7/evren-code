"""Test audit phase231 remediation module."""

from __future__ import annotations

import ast
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from evren_code.config import AppConfig
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.events import EventBus, EventType
from evren_code.core.security import (
    CommandClassifier,
    PermissionManager,
    SECURITY_ERR_ROLE_RESTRICTION,
    ToolExecutor,
)
from evren_code.core.verification.classifier import FailureClassifier
from evren_code.core.verification.engine import VerificationEngine
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
from evren_code.core.verification.validator import ChangeValidator
from evren_code.tools.registry import ToolRegistry


class TestAuditARetryPolicy(unittest.TestCase):
    """Test audit a retry policy implementation."""

    def test_agent_introduced_with_high_confidence_is_eligible(self) -> None:
        rc = RetryController(max_attempts=3, min_confidence=0.5)
        clfs = [
            FailureClassification(
                category=FailureCategory.AGENT_INTRODUCED,
                confidence=0.85,
                evidence="AssertionError in modified service.py",
            )
        ]
        retry, msg = rc.should_retry(VerificationStatus.FAILED, clfs)
        self.assertTrue(retry)
        self.assertIn("Düzeltme denemesi yapılabilir", msg)

    def test_agent_introduced_with_low_confidence_stops_retry(self) -> None:
        rc = RetryController(max_attempts=3, min_confidence=0.5)
        clfs = [
            FailureClassification(
                category=FailureCategory.AGENT_INTRODUCED,
                confidence=0.35,
                evidence="Weak match in service.py",
            )
        ]
        retry, msg = rc.should_retry(VerificationStatus.FAILED, clfs)
        self.assertFalse(retry)
        self.assertIn("güven skoru", msg.lower())

    def test_pre_existing_only_stops_retry(self) -> None:
        rc = RetryController(max_attempts=3)
        clfs = [
            FailureClassification(
                category=FailureCategory.PRE_EXISTING,
                confidence=0.95,
                evidence="Test failed in baseline run before edits",
            )
        ]
        retry, msg = rc.should_retry(VerificationStatus.FAILED, clfs)
        self.assertFalse(retry)
        self.assertIn("PRE_EXISTING", msg)

    def test_environment_only_stops_retry(self) -> None:
        rc = RetryController(max_attempts=3)
        clfs = [
            FailureClassification(
                category=FailureCategory.ENVIRONMENT,
                confidence=0.90,
                evidence="Port 5432 already in use",
            )
        ]
        retry, msg = rc.should_retry(VerificationStatus.FAILED, clfs)
        self.assertFalse(retry)
        self.assertIn("ENVIRONMENT", msg)

    def test_dependency_only_stops_retry(self) -> None:
        rc = RetryController(max_attempts=3)
        clfs = [
            FailureClassification(
                category=FailureCategory.DEPENDENCY,
                confidence=0.90,
                evidence="No module named 'requests'",
            )
        ]
        retry, msg = rc.should_retry(VerificationStatus.FAILED, clfs)
        self.assertFalse(retry)
        self.assertIn("DEPENDENCY", msg)

    def test_test_configuration_only_stops_retry(self) -> None:
        rc = RetryController(max_attempts=3)
        clfs = [
            FailureClassification(
                category=FailureCategory.TEST_CONFIGURATION,
                confidence=0.85,
                evidence="pytest: unrecognized arguments: --invalid",
            )
        ]
        retry, msg = rc.should_retry(VerificationStatus.FAILED, clfs)
        self.assertFalse(retry)
        self.assertIn("TEST_CONFIGURATION", msg)

    def test_unknown_only_stops_retry(self) -> None:
        rc = RetryController(max_attempts=3)
        clfs = [
            FailureClassification(
                category=FailureCategory.UNKNOWN,
                confidence=0.30,
                evidence="Mysterious exit code 42",
            )
        ]
        retry, msg = rc.should_retry(VerificationStatus.FAILED, clfs)
        self.assertFalse(retry)
        self.assertIn("UNKNOWN", msg)

    def test_max_cycles_enforced(self) -> None:
        rc = RetryController(max_attempts=10, max_cycles=2)
        clfs = [FailureClassification(FailureCategory.AGENT_INTRODUCED, 0.9, "error")]

        rc.record_cycle()
        retry, _ = rc.should_retry(VerificationStatus.FAILED, clfs)
        self.assertTrue(retry)

        rc.record_cycle()
        retry, msg = rc.should_retry(VerificationStatus.FAILED, clfs)
        self.assertFalse(retry)
        self.assertIn("döngü sınırına", msg.lower())


class TestAuditBSafeExecutionAndTesterRole(unittest.TestCase):
    """Test audit b safe execution and tester role implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tester_role_cannot_run_package_managers(self) -> None:
        forbidden_commands = [
            "pip install requests",
            "pip3 install --upgrade fastapi",
            "conda install numpy",
            "npm install lodash",
            "npm i express",
            "npm add react",
            "npm update",
            "yarn add axios",
            "pnpm install",
            "cargo install ripgrep",
            "go get github.com/gin-gonic/gin",
            "go install google.golang.org/protobuf/cmd/protoc-gen-go@latest",
        ]
        for cmd in forbidden_commands:
            allowed, reason, err_code = CommandClassifier.analyze_for_role(cmd, role="tester")
            self.assertFalse(allowed, f"Komuta izin verilmemeliydi: {cmd}")
            self.assertEqual(err_code, SECURITY_ERR_ROLE_RESTRICTION)
            self.assertTrue(
                any(w in reason.lower() for w in ("kurulum", "paket", "crate", "modül")),
                f"Hata mesajı kurulum yasağı içermeli: {reason}",
            )

    def test_test_selector_adds_offline_and_no_install_flags(self) -> None:
        selector = TestSelectionEngine(self.workspace_root)

        # 1. TypeScript
        (self.workspace_root / "tsconfig.json").write_text("{}", encoding="utf-8")
        steps = selector._create_build_or_lint_steps(["src/index.ts"])
        self.assertTrue(any("npx --no-install tsc --noEmit" in s.command for s in steps))

        # 2. Rust
        (self.workspace_root / "Cargo.toml").write_text("", encoding="utf-8")
        steps = selector._create_build_or_lint_steps(["src/main.rs"])
        self.assertTrue(any("cargo check --offline" in s.command for s in steps))

        rust_steps = selector._create_targeted_test_steps(["src/main_test.rs"], {})
        self.assertTrue(any("cargo test --offline" in s.command for s in rust_steps))

        # 3. Go
        (self.workspace_root / "go.mod").write_text("", encoding="utf-8")
        go_steps = selector._create_targeted_test_steps(["pkg/util_test.go"], {})
        self.assertTrue(any("go test -mod=readonly" in s.command for s in go_steps))


class TestAuditCTestIntegrity(unittest.TestCase):
    """Test audit c test integrity implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir)
        self.validator = ChangeValidator(self.workspace_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_detect_ast_assert_count_reduction(self) -> None:
        old_test = (
            "def test_calc():\n"
            "    assert add(1, 2) == 3\n"
            "    assert subtract(5, 2) == 3\n"
        )
        new_test = (
            "def test_calc():\n"
            "    assert add(1, 2) == 3\n"
        )
        outcome = self.validator.validate_patch("tests/test_calc.py", old_test, new_test)
        self.assertFalse(outcome.is_valid)
        self.assertTrue(any("assert sayısı azaltıldı" in v for v in outcome.violations))

    def test_detect_ast_test_function_deletion(self) -> None:
        old_test = (
            "def test_one():\n    assert True\n\n"
            "def test_two():\n    assert True\n"
        )
        new_test = (
            "def test_one():\n    assert True\n"
        )
        outcome = self.validator.validate_patch("tests/test_suite.py", old_test, new_test)
        self.assertFalse(outcome.is_valid)
        self.assertTrue(any("test fonksiyonu sayısı azaltıldı" in v for v in outcome.violations))

    def test_detect_trivial_assertion_bypass(self) -> None:
        old_test = "def test_foo():\n    assert calculate() == 100\n"
        new_test = "def test_foo():\n    assert True\n"
        outcome = self.validator.validate_patch("tests/test_foo.py", old_test, new_test)
        self.assertFalse(outcome.is_valid)
        self.assertTrue(any("assert True" in v or "önemsiz iddia" in v for v in outcome.violations))

    def test_detect_empty_assertion_error_catch_bypass(self) -> None:
        old_test = "def test_api():\n    assert client.get().status == 200\n"
        new_test = (
            "def test_api():\n"
            "    try:\n"
            "        assert client.get().status == 200\n"
            "    except AssertionError:\n"
            "        pass\n"
        )
        outcome = self.validator.validate_patch("tests/test_api.py", old_test, new_test)
        self.assertFalse(outcome.is_valid)
        self.assertTrue(any("AssertionError/Exception yutan" in v for v in outcome.violations))

    def test_detect_path_containment_violation(self) -> None:
        outcome = self.validator.validate_patch("../../../etc/passwd", "root", "hacked")
        self.assertFalse(outcome.is_valid)
        self.assertTrue(any("sınır ihlali" in v for v in outcome.violations))


class TestAuditDBaselineAndRegressionLifecycle(unittest.TestCase):
    """Test audit d baseline and regression lifecycle implementation."""

    def test_clean_baseline_and_regression_detection(self) -> None:
        step_base = TestExecutionResult(
            command="pytest",
            exit_code=0,
            stdout="5 passed in 0.1s",
            stderr="",
            duration_seconds=0.1,
            timestamp="",
            failures=[],
        )
        detector = RegressionDetector(baseline_result=VerificationResult(steps=[step_base]))

        # Post-change: 1 new regression
        step_post = TestExecutionResult(
            command="pytest",
            exit_code=1,
            stdout="FAILED test_auth.py::test_jwt",
            stderr="",
            duration_seconds=0.1,
            timestamp="",
            failures=[TestFailure(test_name="test_auth.py::test_jwt")],
        )
        rep = detector.detect_regressions(VerificationResult(steps=[step_post]))

        self.assertTrue(rep.is_regression_detected)
        self.assertEqual(rep.new_regressions, ["test_auth.py::test_jwt"])
        self.assertEqual(rep.resolved_failures, [])

    def test_pre_existing_failure_not_attributed_as_regression(self) -> None:
        step_base = TestExecutionResult(
            command="pytest",
            exit_code=1,
            stdout="FAILED test_old.py::test_legacy",
            stderr="",
            duration_seconds=0.1,
            timestamp="",
            failures=[TestFailure(test_name="test_old.py::test_legacy")],
        )
        detector = RegressionDetector(baseline_result=VerificationResult(steps=[step_base]))

        # Post-change: same legacy failure continues
        step_post = TestExecutionResult(
            command="pytest",
            exit_code=1,
            stdout="FAILED test_old.py::test_legacy",
            stderr="",
            duration_seconds=0.1,
            timestamp="",
            failures=[TestFailure(test_name="test_old.py::test_legacy")],
        )
        rep = detector.detect_regressions(VerificationResult(steps=[step_post]))

        self.assertFalse(rep.is_regression_detected)
        self.assertEqual(rep.new_regressions, [])
        self.assertEqual(rep.remaining_failures, ["test_old.py::test_legacy"])

    def test_regression_fixed_marked_as_resolved(self) -> None:
        step_base = TestExecutionResult(
            command="pytest",
            exit_code=1,
            stdout="FAILED test_bug.py::test_flaky",
            stderr="",
            duration_seconds=0.1,
            timestamp="",
            failures=[TestFailure(test_name="test_bug.py::test_flaky")],
        )
        detector = RegressionDetector(baseline_result=VerificationResult(steps=[step_base]))

        # Post-change: now passing
        step_post = TestExecutionResult(
            command="pytest",
            exit_code=0,
            stdout="1 passed in 0.05s",
            stderr="",
            duration_seconds=0.05,
            timestamp="",
            failures=[],
        )
        rep = detector.detect_regressions(VerificationResult(steps=[step_post]))

        self.assertFalse(rep.is_regression_detected)
        self.assertEqual(rep.resolved_failures, ["test_bug.py::test_flaky"])


class TestAuditEEndToEndCorrectionAndRollback(unittest.TestCase):
    """Test audit e end to end correction and rollback implementation."""

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

    def test_e2e_successful_course_correction(self) -> None:
        call_count = 0
        def fake_exec(tool_name, arguments, caller_role, caller_agent, is_trust_mode):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    "Çıkış Kodu (Exit Code): 1\n"
                    "--- STDOUT ---\n"
                    "FAILED tests/test_calc.py::test_add - AssertionError: 4 != 5\n"
                    "--- STDERR ---\n"
                )
            return "Çıkış Kodu (Exit Code): 0\n--- STDOUT ---\n1 passed in 0.01s"

        mock_executor = MagicMock()
        mock_executor.execute.side_effect = fake_exec

        engine = VerificationEngine(
            workspace_root=self.workspace_root,
            tool_executor=mock_executor,
            max_attempts=3,
        )

        test_file = self.workspace_root / "tests" / "test_calc.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("def test_add(): assert add(2, 2) == 4", encoding="utf-8")

        calc_file = self.workspace_root / "src" / "calc.py"
        calc_file.parent.mkdir(parents=True, exist_ok=True)
        calc_file.write_text("def add(a, b): return a + b + 1", encoding="utf-8")

        def fake_correction_handler(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
            calc_file.write_text("def add(a, b): return a + b", encoding="utf-8")
            return True, ["src/calc.py"]

        report = engine.run_verification(
            changed_files=["src/calc.py"],
            task_description="Toplama fonksiyonu testi",
            orchestrator_correction_handler=fake_correction_handler,
        )

        self.assertEqual(report.final_status, VerificationStatus.VERIFIED)
        self.assertEqual(report.implementation_status, "SUCCESS")
        self.assertEqual(len(report.correction_attempts), 1)
        self.assertTrue(report.correction_attempts[0].success)

    def test_e2e_test_weakening_attempt_is_rejected_and_rolled_back(self) -> None:
        mock_executor = MagicMock()
        mock_executor.execute.return_value = (
            "Çıkış Kodu (Exit Code): 1\n"
            "--- STDOUT ---\n"
            "FAILED tests/test_sec.py::test_auth - AssertionError\n"
            "--- STDERR ---\n"
        )

        engine = VerificationEngine(
            workspace_root=self.workspace_root,
            tool_executor=mock_executor,
            max_attempts=2,
        )

        test_file = self.workspace_root / "tests" / "test_sec.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        orig_test_content = (
            "def test_auth():\n"
            "    assert check_token('valid') is True\n"
        )
        test_file.write_text(orig_test_content, encoding="utf-8")

        def malicious_correction_handler(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
            test_file.write_text("@pytest.mark.skip\ndef test_auth(): pass", encoding="utf-8")
            return True, ["tests/test_sec.py"]

        report = engine.run_verification(
            changed_files=["tests/test_sec.py"],
            task_description="Güvenlik testi",
            orchestrator_correction_handler=malicious_correction_handler,
        )

        self.assertEqual(report.final_status, VerificationStatus.FAILED)
        self.assertFalse(report.correction_attempts[0].success)
        self.assertIn("ChangeValidator", report.correction_attempts[0].notes)

        self.assertEqual(test_file.read_text(encoding="utf-8"), orig_test_content)

    def test_cli_quick_shell_runs_strictly_through_tool_executor(self) -> None:
        from evren_code.cli import handle_slash_command
        mock_orchestrator = MagicMock()
        mock_orchestrator.tool_executor.execute.return_value = "Mocked command output"
        mock_orchestrator.trust_mode = False

        user_input = "!echo test"
        raw_cmd = user_input[1:].strip()
        mock_orchestrator.tool_executor.execute(
            tool_name="run_command",
            arguments={"command": raw_cmd, "cwd": "."},
            caller_role="main",
            caller_agent="cli_quick_shell",
            is_trust_mode=mock_orchestrator.trust_mode,
        )
        mock_orchestrator.tool_executor.execute.assert_called_with(
            tool_name="run_command",
            arguments={"command": "echo test", "cwd": "."},
            caller_role="main",
            caller_agent="cli_quick_shell",
            is_trust_mode=False,
        )


if __name__ == "__main__":
    unittest.main()
