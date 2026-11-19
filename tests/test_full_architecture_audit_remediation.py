"""Test full architecture audit remediation module."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.context import ContextManager
from evren_code.core.events import EventBus, EventType
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.core.planner import (
    ComplexityAssessment,
    ComplexityLevel,
    ExecutionBudget,
    ExecutionState,
    GoalGate,
    ScopeGuard,
    TaskComplexityClassifier,
    TaskContract,
    TaskContractEngine,
)
from evren_code.core.security import CommandClassifier, PermissionManager, ToolExecutor, WorkspaceGuard
from evren_code.tools.filesystem import atomic_write, register_filesystem_tools
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.shell import mask_sensitive_data, normalize_windows_command, register_shell_tools
from evren_code.tools.web import register_web_tools


class TestFullArchitectureAuditRemediation(unittest.TestCase):
    """Test full architecture audit remediation implementation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.config = load_config()
        self.config.workspace_root = self.workspace
        self.registry = ToolRegistry()
        self.event_bus = EventBus()

        register_filesystem_tools(self.registry, self.config)
        register_web_tools(self.registry, self.config)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_01_simple_task_classification_and_budget(self) -> None:
        assessment = TaskComplexityClassifier.classify("typo düzelt")
        self.assertEqual(assessment.level, ComplexityLevel.SIMPLE)
        self.assertLessEqual(assessment.max_turns, 8)
        self.assertFalse(assessment.allow_delegation)
        self.assertFalse(assessment.requires_formal_plan)

    def test_02_task_contract_engine_creation(self) -> None:
        contract = TaskContractEngine.create_contract("register sayfası ekle kurum ve bireysel koçluk için")
        self.assertIn("register sayfası ekle", contract.goal.lower())
        self.assertGreaterEqual(len(contract.acceptance_criteria), 2)
        self.assertIn("write_file", contract.required_actions)
        self.assertEqual(contract.state, ExecutionState.PLANNING)
        self.assertFalse(contract.is_completed())

    def test_03_task_contract_validates_forbidden_unsolicited_actions(self) -> None:
        contract = TaskContractEngine.create_contract("login sayfası oluştur")
        valid, reason = TaskContractEngine.validate_action(
            contract, "run_command", {"command": "curl.exe -X POST http://localhost:8080/api/v1/auth/register"}
        )
        self.assertFalse(valid)
        self.assertIn("Görev sözleşmesi ihlali", reason)

        valid_build, _ = TaskContractEngine.validate_action(
            contract, "run_command", {"command": "npm run build"}
        )
        self.assertTrue(valid_build)

    def test_04_goal_gate_dod_satisfaction_with_contract(self) -> None:
        contract = TaskContractEngine.create_contract("register sayfası ekle")
        modified_files = ["src/app/(auth)/register/page.tsx"]
        cmd_output = "Compiled successfully in 1200ms (exit code: 0)"

        res = GoalGate.evaluate(
            prompt="register sayfası ekle",
            modified_files=modified_files,
            recent_commands_output=cmd_output,
            contract=contract,
        )
        self.assertTrue(res.is_satisfied)
        self.assertEqual(res.suggested_action, "stop_and_report")
        self.assertTrue(contract.is_completed())

    def test_05_scope_guard_blocks_shell_and_http_scope_creep(self) -> None:
        prompt = "register sayfasını ekle"
        unsolicited_cmd = "curl.exe -X POST http://localhost:8080/api/v1/auth/register -d '{\"email\":\"test@test.com\"}'"
        blocked, reason = ScopeGuard.check_unsolicited_action(prompt, unsolicited_cmd)
        self.assertTrue(blocked)
        self.assertIn("Scope Creep Guard", reason)

        allowed_prompt = "register endpointini curl ile test et"
        blocked_allowed, _ = ScopeGuard.check_unsolicited_action(allowed_prompt, unsolicited_cmd)
        self.assertFalse(blocked_allowed)

    def test_06_context_isolation_prevents_prompt_bleeding(self) -> None:
        context = ContextManager(self.config, base_system_prompt="Sen yardımcısın.")
        context.add_user_message("login sayfasını incele")
        context.add_assistant_message(
            content="Login sayfasını inceledim. [Düşünce: Şimdi koç hesabı açmak için register endpointine curl atacağım.]"
        )
        self.assertEqual(len(context.raw_messages), 2)

        context.start_new_task("register sayfası ekle")
        raw_msgs = context.raw_messages
        self.assertEqual(len(raw_msgs), 1)
        self.assertEqual(raw_msgs[0]["content"], "register sayfası ekle")
        self.assertIn("Tamamlanan Görev Özeti", context.summary_memory)

    def test_07_tool_executor_truncates_oversized_output(self) -> None:
        perm_mgr = PermissionManager(self.workspace)
        delegate = MagicMock()
        delegate.request_approval.return_value = True

        mock_registry = MagicMock()
        giant_output = "A" * 30000
        mock_registry.execute.return_value = giant_output
        mock_registry.get.return_value = MagicMock(requires_approval=False)

        executor = ToolExecutor(
            registry=mock_registry,
            permission_manager=perm_mgr,
            approval_delegate=delegate,
            max_output_chars=10000,
        )
        result = executor.execute("dummy_tool", {})
        self.assertLessEqual(len(result), 10500)
        self.assertIn("Araç çıktısı bağlam taşmasını önlemek için kısaltıldı", result)

    def test_08_safe_http_request_tool_registration_and_execution(self) -> None:
        tool = self.registry.get("http_request")
        self.assertIsNotNone(tool)

        err = self.registry.execute("http_request", {"url": "ftp://example.com"})
        self.assertTrue("URL must start with" in err or "başlamalıdır" in err)

        with patch("httpx.Client.request") as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.reason_phrase = "OK"
            mock_resp.headers = {"content-type": "application/json"}
            mock_resp.text = '{"status":"ok"}'
            mock_req.return_value = mock_resp

            res = self.registry.execute("http_request", {"url": "http://localhost:8080/api/health", "method": "GET"})
            self.assertIn("HTTP 200", res)
            self.assertIn('{"status":"ok"}', res)

    def test_09_stream_options_in_llm_client_payload(self) -> None:
        client = EvrenLLMClient(self.config, self.event_bus)
        mock_post = MagicMock()

        with patch.object(client, "_send_streaming") as mock_stream:
            mock_stream.return_value = MagicMock(content="tamam")
            client.chat(messages=[{"role": "user", "content": "test"}], stream=True)
            call_args = mock_stream.call_args
            payload = call_args[0][1]
            self.assertIn("stream_options", payload)
            self.assertEqual(payload["stream_options"], {"include_usage": True})

    def test_10_cancellation_architecture_and_execution_state(self) -> None:
        orchestrator = AgentOrchestrator(self.config)
        self.assertEqual(orchestrator.execution_state, ExecutionState.IDLE)

        orchestrator.cancel()
        self.assertEqual(orchestrator.execution_state, ExecutionState.CANCELLED)
        self.assertTrue(orchestrator._cancel_requested)

    def test_11_execution_budget_exceeded(self) -> None:
        budget = ExecutionBudget(max_turns=3, max_model_calls=5)
        self.assertFalse(budget.is_exceeded())
        budget.record_turn()
        budget.record_turn()
        budget.record_turn()
        self.assertTrue(budget.is_exceeded())

    def test_12_workspace_guard_blocks_path_traversal_and_system_dirs(self) -> None:
        outside_path = Path("C:/Windows/System32/calc.exe")
        with self.assertRaises(PermissionError):
            WorkspaceGuard.validate_path(self.workspace, outside_path, is_write=True)

        traversal_path = self.workspace / ".." / "secret.env"
        with self.assertRaises(PermissionError):
            WorkspaceGuard.validate_path(self.workspace, traversal_path, is_write=True)

    def test_13_atomic_write_creates_file_and_protects_integrity(self) -> None:
        target = self.workspace / "src" / "index.ts"
        content = "console.log('hello world');"
        ok, msg = atomic_write(target, content, workspace_root=self.workspace)
        self.assertTrue(ok)
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_text(encoding="utf-8"), content)

    def test_14_binary_class_file_protection(self) -> None:
        class_file = self.workspace / "App.class"
        class_file.write_bytes(b"\xca\xfe\xba\xbe\x00\x00")
        res = self.registry.execute("read_file", {"path": "App.class"})
        self.assertIn("ikili (binary) dosyad", res.lower())

    # Senaryo 15: Java controller eksik anotasyon import tespiti
    def test_15_java_annotation_missing_import_check(self) -> None:
        bad_controller = (
            "package com.example;\n\n"
            "public class UserController {\n"
            "    @PutMapping(\"/update\")\n"
            "    public String update() { return \"ok\"; }\n"
            "}\n"
        )
        res = self.registry.execute("write_file", {"path": "UserController.java", "content": bad_controller})
        self.assertIn("Java Eksik Import", res)
        self.assertIn("@PutMapping", res)

    def test_16_fuzzy_line_matching_with_blank_lines(self) -> None:
        target = self.workspace / "config.ts"
        target.write_text("export const port = 3000;\n\nexport const host = 'localhost';\n", encoding="utf-8")

        res = self.registry.execute(
            "edit_file",
            {
                "path": "config.ts",
                "target": "export const port = 3000;\nexport const host = 'localhost';",
                "replacement": "export const port = 8080;\nexport const host = '0.0.0.0';",
            },
        )
        self.assertIn("güncellendi", res.lower())
        self.assertIn("8080", target.read_text(encoding="utf-8"))

    def test_17_shell_cmd_wrapper_for_dir_s_b(self) -> None:
        norm = normalize_windows_command("dir /s /b *.java", use_cmd=False)
        self.assertTrue(norm.startswith('cmd.exe /c "dir /s /b'))

    def test_18_shell_mask_preserves_xsrf_token_prefix(self) -> None:
        raw = "Set-Cookie: XSRF-TOKEN=secret1234567890secret1234567890; Path=/"
        masked = mask_sensitive_data(raw)
        self.assertNotIn("Bearer", masked)
        self.assertIn("XSRF-TOKEN=***MASKED_TOKEN***", masked)

    def test_19_shell_invoke_webrequest_preservation(self) -> None:
        cmd = "Invoke-WebRequest -Uri 'http://localhost:8080' -Method GET"
        norm = normalize_windows_command(cmd, use_cmd=False)
        self.assertIn("Invoke-WebRequest", norm)
        self.assertNotIn("Invoke-RestMethod", norm)

    def test_20_urgent_keyword_budget_clamping(self) -> None:
        prompt = "register sayfasını hemen hızlı bir şekilde ekle bakalım"
        assessment = TaskComplexityClassifier.classify(prompt)
        self.assertTrue(assessment.is_urgent)
        self.assertTrue(assessment.strict_scope_enforcement)
        self.assertLessEqual(assessment.max_turns, 25)

    def test_21_agent_turn_test_reporting_no_name_error_on_re(self) -> None:
        from evren_code.core.verification.test_reporter import ParsedTestMetrics
        orchestrator = AgentOrchestrator(self.config)

        mock_response = MagicMock()
        mock_response.content = "Yapılan değişiklikler tamamlandı. 12 test başarıyla geçti."
        mock_response.tool_calls = None
        mock_response.thinking = ""
        mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        mock_response.repetition_detected = False

        orchestrator.llm_client.chat = MagicMock(return_value=mock_response)

        with patch.object(orchestrator.self_healing_manager, "detect_verification_task") as mock_det:
            mock_task = MagicMock()
            mock_task.command = "npm run build"
            mock_task.description = "build check"
            mock_det.return_value = mock_task

            with patch.object(orchestrator.self_healing_manager, "execute_verification") as mock_exec:
                mock_exec.return_value = (True, "npm run build", "Compiled successfully in 800ms")

                result = orchestrator.run_turn("login sol panelini güncelle")
                self.assertIn("12 test başarıyla geçti", result)
                self.assertEqual(orchestrator.execution_state, ExecutionState.COMPLETED)


if __name__ == "__main__":
    unittest.main()
