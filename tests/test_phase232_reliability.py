"""Test phase232 reliability module."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from evren_code.config import AppConfig
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.security import (
    PermissionManager,
    SECURITY_ERR_PERMISSION_DENIED,
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


class TestOfflineDependencySafety(unittest.TestCase):
    """Test offline dependency safety implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_maven_commands_use_offline_flag(self) -> None:
        (self.workspace_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
        selector = TestSelectionEngine(self.workspace_root)

        b_steps = selector._create_build_or_lint_steps(["src/main/java/App.java"])
        self.assertTrue(any(s.command == "mvn compile -o -DskipTests" for s in b_steps))

        t_steps = selector._create_targeted_test_steps(["src/test/java/AppTest.java"], {})
        self.assertTrue(any("-o" in s.command and "mvn test" in s.command for s in t_steps))

        fb_step = selector._create_fallback_test_step(["src/main/java/App.java"])
        self.assertIsNotNone(fb_step)
        self.assertIn("-o", fb_step.command)

    def test_gradle_commands_use_offline_flag(self) -> None:
        (self.workspace_root / "build.gradle").write_text("", encoding="utf-8")
        selector = TestSelectionEngine(self.workspace_root)

        t_steps = selector._create_targeted_test_steps(["src/test/java/ServiceTest.java"], {})
        self.assertTrue(any("--offline" in s.command and "gradlew" in s.command for s in t_steps))

        fb_step = selector._create_fallback_test_step(["src/main/java/Service.java"])
        self.assertIsNotNone(fb_step)
        self.assertIn("--offline", fb_step.command)

    def test_go_commands_use_readonly_mod_safety(self) -> None:
        (self.workspace_root / "go.mod").write_text("module example.com/app", encoding="utf-8")
        selector = TestSelectionEngine(self.workspace_root)

        t_steps = selector._create_targeted_test_steps(["pkg/api/handler_test.go"], {})
        self.assertTrue(any("-mod=readonly" in s.command for s in t_steps))

        fb_step = selector._create_fallback_test_step(["main.go"])
        self.assertIsNotNone(fb_step)
        self.assertIn("-mod=readonly", fb_step.command)


class TestMultiLanguageTestIntegrity(unittest.TestCase):
    """Test multi language test integrity implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir)
        self.validator = ChangeValidator(self.workspace_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_java_disabled_and_ignore_annotations_detected(self) -> None:
        old_java = (
            "public class UserServiceTest {\n"
            "    @Test\n"
            "    public void testCreateUser() {\n"
            "        assertTrue(service.create());\n"
            "    }\n"
            "}\n"
        )
        # 1. @Disabled ekleme (JUnit 5)
        new_disabled = (
            "public class UserServiceTest {\n"
            "    @Disabled(\"Skip for now\")\n"
            "    @Test\n"
            "    public void testCreateUser() {\n"
            "        assertTrue(service.create());\n"
            "    }\n"
            "}\n"
        )
        outcome1 = self.validator.validate_patch("UserServiceTest.java", old_java, new_disabled)
        self.assertFalse(outcome1.is_valid)
        self.assertTrue(any("zayıflatma" in v for v in outcome1.violations))

        # 2. @Ignore ekleme (JUnit 4)
        new_ignore = (
            "public class UserServiceTest {\n"
            "    @Ignore\n"
            "    @Test\n"
            "    public void testCreateUser() {\n"
            "        assertTrue(service.create());\n"
            "    }\n"
            "}\n"
        )
        outcome2 = self.validator.validate_patch("UserServiceTest.java", old_java, new_ignore)
        self.assertFalse(outcome2.is_valid)

    def test_java_test_annotation_and_method_deletion_detected(self) -> None:
        old_java = (
            "public class CalcTest {\n"
            "    @Test\n"
            "    public void testAdd() {\n"
            "        assertEquals(4, 2 + 2);\n"
            "    }\n"
            "}\n"
        )
        new_no_annotation = (
            "public class CalcTest {\n"
            "    public void testAdd() {\n"
            "        assertEquals(4, 2 + 2);\n"
            "    }\n"
            "}\n"
        )
        outcome = self.validator.validate_patch("CalcTest.java", old_java, new_no_annotation)
        self.assertFalse(outcome.is_valid)
        self.assertTrue(any("silme tespit edildi" in v for v in outcome.violations))

    def test_jest_vitest_skip_and_xmethods_detected(self) -> None:
        old_js = (
            "describe('AuthService', () => {\n"
            "    it('should login successfully', () => {\n"
            "        expect(login()).toBe(true);\n"
            "    });\n"
            "});\n"
        )
        new_skip = (
            "describe('AuthService', () => {\n"
            "    it.skip('should login successfully', () => {\n"
            "        expect(login()).toBe(true);\n"
            "    });\n"
            "});\n"
        )
        outcome1 = self.validator.validate_patch("auth.test.ts", old_js, new_skip)
        self.assertFalse(outcome1.is_valid)

        new_xit = (
            "describe('AuthService', () => {\n"
            "    xit('should login successfully', () => {\n"
            "        expect(login()).toBe(true);\n"
            "    });\n"
            "});\n"
        )
        outcome2 = self.validator.validate_patch("auth.test.js", old_js, new_xit)
        self.assertFalse(outcome2.is_valid)

    def test_test_file_clearing_or_deletion_detected(self) -> None:
        old_ts = "test('it works', () => { expect(1).toBe(1); });"
        new_empty = "   \n\n  "
        outcome = self.validator.validate_patch("components.spec.tsx", old_ts, new_empty)
        self.assertFalse(outcome.is_valid)
        self.assertTrue(any("tamamen silinemez veya boşaltılamaz" in v for v in outcome.violations))

    def test_test_configuration_narrowing_detected(self) -> None:
        old_conf = "module.exports = { testMatch: ['**/*.test.js'] };"
        new_conf = (
            "module.exports = {\n"
            "    testMatch: ['**/*.test.js'],\n"
            "    testPathIgnorePatterns: ['/auth/'],\n"
            "};"
        )
        outcome = self.validator.validate_patch("jest.config.js", old_conf, new_conf)
        self.assertFalse(outcome.is_valid)
        self.assertTrue(any("daraltma tespit edildi" in v for v in outcome.violations))


class TestRollbackReliability(unittest.TestCase):
    """Test rollback reliability implementation."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir)
        self.tool_executor = MagicMock()
        self.engine = VerificationEngine(
            workspace_root=self.workspace_root,
            tool_executor=self.tool_executor,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_rollback_deletes_newly_created_files(self) -> None:
        snapshots = self.engine._take_snapshot({"src/new_helper.py"})
        self.assertIsNone(snapshots["src/new_helper.py"])

        new_f = self.workspace_root / "src" / "new_helper.py"
        new_f.parent.mkdir(parents=True, exist_ok=True)
        new_f.write_text("# new file", encoding="utf-8")
        self.assertTrue(new_f.exists())

        ok, errors = self.engine._rollback_snapshot(snapshots, {"src/new_helper.py"})
        self.assertTrue(ok)
        self.assertEqual(len(errors), 0)
        self.assertFalse(new_f.exists(), "Yeni oluşturulan dosya rollback sırasında silinmeliydi!")

    def test_rollback_restores_modified_file(self) -> None:
        target_f = self.workspace_root / "src" / "math.py"
        target_f.parent.mkdir(parents=True, exist_ok=True)
        target_f.write_text("def val(): return 42", encoding="utf-8")

        snapshots = self.engine._take_snapshot({"src/math.py"})

        target_f.write_text("def val(): return 0", encoding="utf-8")

        # Rollback
        ok, errors = self.engine._rollback_snapshot(snapshots, {"src/math.py"})
        self.assertTrue(ok)
        self.assertEqual(target_f.read_text(encoding="utf-8"), "def val(): return 42")

    def test_rollback_recreates_deleted_file(self) -> None:
        target_f = self.workspace_root / "src" / "important.py"
        target_f.parent.mkdir(parents=True, exist_ok=True)
        orig_content = "SECRET_KEY = 'orig'"
        target_f.write_text(orig_content, encoding="utf-8")

        snapshots = self.engine._take_snapshot({"src/important.py"})

        target_f.unlink()
        self.assertFalse(target_f.exists())

        # Rollback
        ok, errors = self.engine._rollback_snapshot(snapshots, {"src/important.py"})
        self.assertTrue(ok)
        self.assertTrue(target_f.exists())
        self.assertEqual(target_f.read_text(encoding="utf-8"), orig_content)

    def test_rollback_handles_multiple_files_atomically(self) -> None:
        f1 = self.workspace_root / "src" / "a.py"
        f2 = self.workspace_root / "src" / "b.py"
        f1.parent.mkdir(parents=True, exist_ok=True)
        f1.write_text("a = 1", encoding="utf-8")
        f2.write_text("b = 2", encoding="utf-8")

        snapshots = self.engine._take_snapshot({"src/a.py", "src/b.py", "src/c.py"})

        f1.write_text("a = 99", encoding="utf-8")
        f2.unlink()
        f3 = self.workspace_root / "src" / "c.py"
        f3.write_text("c = 3", encoding="utf-8")

        ok, errors = self.engine._rollback_snapshot(snapshots, {"src/a.py", "src/b.py", "src/c.py"})
        self.assertTrue(ok)
        self.assertEqual(f1.read_text(encoding="utf-8"), "a = 1")
        self.assertTrue(f2.exists())
        self.assertEqual(f2.read_text(encoding="utf-8"), "b = 2")
        self.assertFalse(f3.exists())

    def test_rollback_failure_reported_explicitly_never_success(self) -> None:
        snapshots = {"locked.py": "original_content"}
        locked_p = self.workspace_root / "locked.py"

        with patch.object(Path, "write_text", side_effect=PermissionError("Dosya kilitli")):
            ok, errors = self.engine._rollback_snapshot(snapshots, {"locked.py"})
            self.assertFalse(ok)
            self.assertTrue(len(errors) > 0)
            self.assertTrue(any("Dosya kilitli" in e for e in errors))


class TestEndToEndVerificationScenarios(unittest.TestCase):
    """4. End-to-End Verification Scenarios: Orchestrator + VerificationEngine entegrasyonu."""

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

    def test_correction_fails_stops_cleanly(self) -> None:
        mock_executor = MagicMock()
        mock_executor.execute.return_value = (
            "Çıkış Kodu (Exit Code): 1\n"
            "--- STDOUT ---\n"
            "FAILED tests/test_app.py::test_run - AssertionError\n"
            "--- STDERR ---\n"
        )
        engine = VerificationEngine(
            workspace_root=self.workspace_root,
            tool_executor=mock_executor,
            max_attempts=2,
        )

        test_f = self.workspace_root / "tests" / "test_app.py"
        test_f.parent.mkdir(parents=True, exist_ok=True)
        test_f.write_text("def test_run(): assert False", encoding="utf-8")

        def failing_handler(plan):
            return False, []

        report = engine.run_verification(
            changed_files=["tests/test_app.py"],
            task_description="Başarısız düzeltme testi",
            orchestrator_correction_handler=failing_handler,
        )

        self.assertEqual(report.final_status, VerificationStatus.FAILED)
        self.assertEqual(report.implementation_status, "FAILED")
        self.assertEqual(len(report.correction_attempts), 1)
        self.assertFalse(report.correction_attempts[0].success)

    def test_second_regression_detected_and_handled(self) -> None:
        step_count = 0
        def fake_exec(tool_name, arguments, caller_role, caller_agent, is_trust_mode):
            nonlocal step_count
            step_count += 1
            if step_count == 1:
                return (
                    "Çıkış Kodu (Exit Code): 1\n"
                    "--- STDOUT ---\n"
                    "FAILED tests/test_app.py::test_a - AssertionError\n"
                    "File 'src/app.py', line 10, in run\n"
                )
            return (
                "Çıkış Kodu (Exit Code): 1\n"
                "--- STDOUT ---\n"
                "FAILED tests/test_app.py::test_b - AssertionError\n"
            )

        mock_executor = MagicMock()
        mock_executor.execute.side_effect = fake_exec

        base_res = VerificationResult(steps=[TestExecutionResult(
            command="pytest", exit_code=0, stdout="1 passed in 0.01s", stderr="", duration_seconds=0.1, timestamp="",
            failures=[],
        )])

        engine = VerificationEngine(
            workspace_root=self.workspace_root,
            tool_executor=mock_executor,
            max_attempts=3,
        )
        engine.baseline_result = base_res

        def bad_correction_handler(plan):
            (self.workspace_root / "src" / "app.py").write_text("x = 2", encoding="utf-8")
            return True, ["src/app.py"]

        (self.workspace_root / "src" / "app.py").parent.mkdir(parents=True, exist_ok=True)
        (self.workspace_root / "src" / "app.py").write_text("x = 1", encoding="utf-8")

        report = engine.run_verification(
            changed_files=["src/app.py"],
            task_description="Regresyon testi",
            orchestrator_correction_handler=bad_correction_handler,
        )

        self.assertIsNotNone(report.regression_analysis)
        self.assertTrue(report.regression_analysis.is_regression_detected)
        self.assertIn("tests/test_app.py::test_b", report.regression_analysis.new_regressions)

    def test_tool_permission_denied_never_reported_as_success(self) -> None:
        mock_executor = MagicMock()
        mock_executor.execute.return_value = "[GÜVENLİK ENGELİ / ERR_PERMISSION_DENIED]: İşlem engellendi"

        engine = VerificationEngine(
            workspace_root=self.workspace_root,
            tool_executor=mock_executor,
        )

        report = engine.run_verification(
            changed_files=["src/main.py"],
            task_description="Yetki engeli testi",
        )

        self.assertEqual(report.final_status, VerificationStatus.BLOCKED)
        self.assertEqual(report.implementation_status, "BLOCKED")
        self.assertEqual(len(report.step_results), 1)
        self.assertEqual(report.step_results[0].status, VerificationStatus.BLOCKED)

    def test_max_retry_limit_cannot_be_exceeded(self) -> None:
        mock_executor = MagicMock()
        mock_executor.execute.return_value = "Çıkış Kodu (Exit Code): 1\n--- STDOUT ---\nFAILED tests/test_loop.py::test_fail - AssertionError\n"

        engine = VerificationEngine(
            workspace_root=self.workspace_root,
            tool_executor=mock_executor,
            max_attempts=2,
        )

        (self.workspace_root / "src" / "x.py").parent.mkdir(parents=True, exist_ok=True)
        (self.workspace_root / "src" / "x.py").write_text("x = 1", encoding="utf-8")

        attempts_called = 0
        def looping_handler(plan):
            nonlocal attempts_called
            attempts_called += 1
            (self.workspace_root / "src" / "x.py").write_text(f"x = {attempts_called}", encoding="utf-8")
            return True, ["src/x.py"]

        report = engine.run_verification(
            changed_files=["src/x.py"],
            task_description="Maksimum deneme testi",
            orchestrator_correction_handler=looping_handler,
        )

        self.assertLessEqual(len(report.correction_attempts), 2)
        self.assertEqual(report.final_status, VerificationStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
