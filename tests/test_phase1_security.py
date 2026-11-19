"""Test phase1 security module."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from evren_code.config import AppConfig, load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.events import ApprovalDelegate, ApprovalRequest, EventBus, EventType
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem
from evren_code.core.security import (
    CommandClassifier,
    PermissionManager,
    PermissionMode,
    SECURITY_ERR_ACCESS_VIOLATION,
    SECURITY_ERR_APPROVAL_REJECTED,
    SECURITY_ERR_DANGEROUS_COMMAND,
    SECURITY_ERR_PERMISSION_DENIED,
    SECURITY_ERR_ROLE_RESTRICTION,
    ToolExecutor,
)
from evren_code.core.self_healing import SelfHealingManager, VerificationTask
from evren_code.core.subagents import SubagentManager
from evren_code.core.undo import UndoManager
from evren_code.tools.filesystem import register_filesystem_tools
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.shell import mask_sensitive_data, register_shell_tools


class RecordingApprovalDelegate:
    """Recording approval delegate implementation."""

    def __init__(self, approve_all: bool = False) -> None:
        self.approve_all = approve_all
        self.requests: List[ApprovalRequest] = []

    def request_approval(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self.approve_all

    def ask_user_question(self, question: str, options: Optional[List[str]] = None) -> str:
        return ""


class TestPhase1SecurityAndGovernance(unittest.TestCase):
    """Test phase1 security and governance implementation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="evren_phase1_test_")
        self.workspace = Path(self.temp_dir).resolve()
        self.config = load_config(workspace_root=self.workspace)
        self.event_bus = EventBus()
        self.registry = ToolRegistry(require_approval_for=self.config.tools.require_approval_for)
        self.undo_manager = UndoManager()
        register_filesystem_tools(self.registry, self.config, self.undo_manager)
        register_shell_tools(self.registry, self.config)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_subagent_cannot_bypass_permission_for_dangerous_command(self) -> None:
        """Verify subagent cannot bypass permission for dangerous command."""
        pm = PermissionManager(self.workspace)
        delegate = RecordingApprovalDelegate(approve_all=True)
        executor = ToolExecutor(
            registry=self.registry,
            permission_manager=pm,
            approval_delegate=delegate,
            event_bus=self.event_bus,
        )

        result = executor.execute(
            tool_name="run_command",
            arguments={"command": "rm -rf /"},
            caller_role="tester",
            caller_agent="subagent_tester",
            is_trust_mode=True,
        )

        self.assertIn(f"[GÜVENLİK ENGELİ / {SECURITY_ERR_DANGEROUS_COMMAND}]", result)
        self.assertIn("yüksek riskli komut engellendi", result)
        self.assertEqual(len(delegate.requests), 0)

    def test_subagent_tester_unauthorized_git_push_blocked(self) -> None:
        """Verify subagent tester unauthorized git push blocked."""
        pm = PermissionManager(self.workspace)
        delegate = RecordingApprovalDelegate(approve_all=True)
        executor = ToolExecutor(
            registry=self.registry,
            permission_manager=pm,
            approval_delegate=delegate,
            event_bus=self.event_bus,
        )

        result = executor.execute(
            tool_name="run_command",
            arguments={"command": "git push origin main"},
            caller_role="tester",
            caller_agent="subagent_tester",
        )

        self.assertIn(f"[GÜVENLİK ENGELİ / {SECURITY_ERR_ROLE_RESTRICTION}]", result)
        self.assertIn("Tester yetki aşımı", result)

    def test_subagent_researcher_cannot_run_shell_or_modify_files(self) -> None:
        """Verify subagent researcher cannot run shell or modify files."""
        pm = PermissionManager(self.workspace)
        delegate = RecordingApprovalDelegate(approve_all=True)
        executor = ToolExecutor(
            registry=self.registry,
            permission_manager=pm,
            approval_delegate=delegate,
            event_bus=self.event_bus,
        )

        # 1. run_command engellenmeli
        res_cmd = executor.execute(
            tool_name="run_command",
            arguments={"command": "ls -la"},
            caller_role="researcher",
            caller_agent="subagent_researcher",
        )
        self.assertIn(f"[GÜVENLİK ENGELİ / {SECURITY_ERR_ROLE_RESTRICTION}]", res_cmd)

        # 2. write_file engellenmeli
        res_write = executor.execute(
            tool_name="write_file",
            arguments={"path": "hacked.py", "content": "print(1)"},
            caller_role="researcher",
            caller_agent="subagent_researcher",
        )
        self.assertIn(f"[GÜVENLİK ENGELİ / {SECURITY_ERR_ROLE_RESTRICTION}]", res_write)

    def test_subagent_approval_gate_identifies_caller_and_enforces_rejection(self) -> None:
        """Verify subagent approval gate identifies caller and enforces rejection."""
        pm = PermissionManager(self.workspace)
        delegate = RecordingApprovalDelegate(approve_all=False)  # Reddet
        executor = ToolExecutor(
            registry=self.registry,
            permission_manager=pm,
            approval_delegate=delegate,
            event_bus=self.event_bus,
        )

        result = executor.execute(
            tool_name="write_file",
            arguments={"path": "subagent_file.txt", "content": "deneme"},
            caller_role="coder",
            caller_agent="subagent_coder_42",
        )

        self.assertIn(f"[KULLANICI REDDİ / {SECURITY_ERR_APPROVAL_REJECTED}]", result)
        self.assertEqual(len(delegate.requests), 1)
        req = delegate.requests[0]
        self.assertEqual(req.caller_role, "coder")
        self.assertEqual(req.caller_agent, "subagent_coder_42")
        self.assertEqual(req.tool_name, "write_file")
        self.assertFalse((self.workspace / "subagent_file.txt").exists())

    def test_subagent_manager_execution_passes_through_tool_executor(self) -> None:
        """Verify subagent manager execution passes through tool executor."""
        delegate = RecordingApprovalDelegate(approve_all=False)
        sub_mgr = SubagentManager(
            config=self.config,
            registry=self.registry,
            llm_client=MagicMock(),
            event_bus=self.event_bus,
            approval_delegate=delegate,
        )

        mock_response = ParsedAssistantResponse(
            content="",
            tool_calls=[
                ToolCallItem(
                    id="call_bad",
                    name="write_file",
                    arguments={"path": "bad.txt", "content": "malicious"},
                )
            ],
            thinking="",
        )
        sub_mgr.llm_client.chat = MagicMock(side_effect=[
            mock_response,
            ParsedAssistantResponse(content="Araştırma tamamlandı.", tool_calls=[]),
        ])

        output = sub_mgr.delegate("researcher", "Kod tabanını incele")
        self.assertIn("Araştırma tamamlandı", output)
        self.assertFalse((self.workspace / "bad.txt").exists())

    def test_mask_sensitive_data_in_shell_output(self) -> None:
        """Verify mask sensitive data in shell output."""
        test_key = "evren_llm_sec_9876543210abcdef1234"
        os.environ["EVREN_API_KEY"] = test_key

        raw_output = (
            f"Starting server with EVREN_API_KEY={test_key}\n"
            "Authorization: Bearer my_secret_token_1234567890abcdef\n"
            "User password: password=SuperSecretPassword123"
        )
        masked = mask_sensitive_data(raw_output)

        self.assertNotIn(test_key, masked)
        self.assertIn("***MASKED_EVREN_API_KEY***", masked)
        self.assertNotIn("my_secret_token_1234567890abcdef", masked)
        self.assertIn("Bearer ***MASKED_TOKEN***", masked)
        self.assertNotIn("SuperSecretPassword123", masked)

    def test_write_multiple_files_prevalidation_blocks_path_traversal(self) -> None:
        """Verify write multiple files prevalidation blocks path traversal."""
        file1 = self.workspace / "valid1.txt"
        escape_file = self.workspace.parent / "escape_attempt.txt"

        payload = [
            {"path": "valid1.txt", "content": "gecerli icerik"},
            {"path": "../escape_attempt.txt", "content": "kacak icerik"},
        ]

        result = self.registry.execute("write_multiple_files", {"files": payload})

        self.assertIn("Güvenlik Hatası", result)
        self.assertFalse(file1.exists(), "İlk dosya geçerli olsa bile listedeki bir hata yüzünden yazılmamalıdır (Atomik Koruma)!")
        self.assertFalse(escape_file.exists())

    def test_write_multiple_files_preview_builder(self) -> None:
        """Verify write multiple files preview builder."""
        existing_file = self.workspace / "mevcut.txt"
        existing_file.write_text("eski satir 1\neski satir 2\n", encoding="utf-8")

        tool_def = self.registry.get("write_multiple_files")
        self.assertIsNotNone(tool_def)
        self.assertIsNotNone(tool_def.preview_builder)

        payload = {
            "files": [
                {"path": "mevcut.txt", "content": "yeni satir 1\neski satir 2\n"},
                {"path": "yeni_dosya.txt", "content": "tamamen yeni icerik\n"},
            ]
        }
        diff = tool_def.preview_builder(payload)
        self.assertIsNotNone(diff)
        self.assertIn("--- a/mevcut.txt", diff)
        self.assertIn("+++ b/mevcut.txt", diff)
        self.assertIn("-eski satir 1", diff)
        self.assertIn("+yeni satir 1", diff)
        self.assertIn("+++ b/yeni_dosya.txt", diff)

    def test_self_healing_auto_detect_setting_respected(self) -> None:
        """Verify self healing auto detect setting respected."""
        pkg = self.workspace / "package.json"
        pkg.write_text(json.dumps({"scripts": {"test": "jest"}}), encoding="utf-8")

        # 1. auto_detect = True
        sh_true = SelfHealingManager(self.workspace, enabled=True, auto_detect=True)
        detected_true = sh_true.detect_verification_task()
        self.assertIsNotNone(detected_true)
        self.assertIn("npm test", detected_true.command)

        sh_false = SelfHealingManager(self.workspace, enabled=True, auto_detect=False)
        detected_false = sh_false.detect_verification_task()
        self.assertIsNone(detected_false)

        custom_task = VerificationTask("python my_custom_test.py", self.workspace, "Özel Test")
        sh_false._active_task = custom_task
        self.assertEqual(sh_false.detect_verification_task(), custom_task)

    def test_permission_manager_respects_config_require_approval_for(self) -> None:
        """Verify permission manager respects config require approval for."""
        pm = PermissionManager(
            workspace_root=self.workspace,
            require_approval_for=["search_code", "read_file"],
        )

        dec = pm.check("search_code", {"query": "test"})
        self.assertTrue(dec.requires_approval)
        self.assertIn("onay gerektiriyor", dec.reason)

    def test_default_mode_requires_approval_for_modifying_tools_even_if_not_in_config_list(self) -> None:
        """Verify default mode requires approval for modifying tools even if not in config list."""
        pm = PermissionManager(
            workspace_root=self.workspace,
            initial_mode=PermissionMode.DEFAULT,
            require_approval_for=["read_file"],
        )
        dec = pm.check("write_file", {"path": "foo.py", "content": "1"})
        self.assertTrue(dec.requires_approval)
        self.assertTrue(dec.allowed)

        dec_edit = pm.check("edit_file", {"path": "foo.py", "target": "1", "replacement": "2"})
        self.assertTrue(dec_edit.requires_approval)

    def test_encoded_powershell_detected_and_dangerous(self) -> None:
        """Verify encoded powershell detected and dangerous."""
        cmds = [
            "powershell -EncodedCommand JABzACAAPQAgAC4ALgAuAA==",
            "powershell.exe -enc JABzACAAPQAgAC4ALgAuAA==",
            "pwsh -e JABzACAAPQAgAC4ALgAuAA==",
            "powershell -encodedcommand abcdef==",
        ]
        for c in cmds:
            is_danger, reason = CommandClassifier.analyze(c)
            self.assertTrue(is_danger, f"'{c}' tehlikeli olarak algılanmalıydı!")
            self.assertIn("kodlanmış PowerShell", reason)

    def test_tester_role_blocks_inline_python_and_node_scripts(self) -> None:
        """Verify tester role blocks inline python and node scripts."""
        pm = PermissionManager(self.workspace)
        delegate = RecordingApprovalDelegate(approve_all=True)
        executor = ToolExecutor(
            registry=self.registry,
            permission_manager=pm,
            approval_delegate=delegate,
            event_bus=self.event_bus,
        )

        # python -c engellenmeli
        res_py = executor.execute(
            tool_name="run_command",
            arguments={"command": 'python -c "import shutil; shutil.rmtree(\'.\')"', "cwd": "."},
            caller_role="tester",
            caller_agent="self_healing",
        )
        self.assertIn(f"[GÜVENLİK ENGELİ / {SECURITY_ERR_ROLE_RESTRICTION}]", res_py)
        self.assertIn("Tester yetki aşımı", res_py)

        # node -e engellenmeli
        res_node = executor.execute(
            tool_name="run_command",
            arguments={"command": 'node -e "process.exit(1)"', "cwd": "."},
            caller_role="tester",
            caller_agent="subagent_tester",
        )
        self.assertIn(f"[GÜVENLİK ENGELİ / {SECURITY_ERR_ROLE_RESTRICTION}]", res_node)

    def test_tester_role_allows_legitimate_test_commands(self) -> None:
        """Verify tester role allows legitimate test commands."""
        clean_cmds = [
            "pytest",
            "python -m unittest discover",
            "npm test",
            "npm run build && npm test",
            "npx tsc --noEmit",
            "cargo test",
            "git status",
            "git diff HEAD~1",
        ]
        for cmd in clean_cmds:
            allowed, reason, code = CommandClassifier.analyze_for_role(cmd, "tester")
            self.assertTrue(allowed, f"'{cmd}' tester için izinli olmalıydı! Sebep: {reason}")

    def test_self_healing_blocked_in_plan_mode(self) -> None:
        """Verify self healing blocked in plan mode."""
        pm = PermissionManager(self.workspace, initial_mode=PermissionMode.PLAN)
        delegate = RecordingApprovalDelegate(approve_all=True)
        executor = ToolExecutor(
            registry=self.registry,
            permission_manager=pm,
            approval_delegate=delegate,
            event_bus=self.event_bus,
        )

        res = executor.execute(
            tool_name="run_command",
            arguments={"command": "npm test", "cwd": "."},
            caller_role="tester",
            caller_agent="self_healing",
        )
        self.assertIn(f"[GÜVENLİK ENGELİ / {SECURITY_ERR_PERMISSION_DENIED}]", res)
        self.assertIn("Plan Modu Aktif", res)

    def test_parallel_prefetch_skips_tools_requiring_approval(self) -> None:
        """Verify parallel prefetch skips tools requiring approval."""
        # read_file onay gerektirenler listesine eklensin
        self.registry = ToolRegistry(require_approval_for=["read_file"])
        register_filesystem_tools(self.registry, self.config, self.undo_manager)
        orch = AgentOrchestrator(config=self.config)
        orch.registry = self.registry
        orch.permission_manager = PermissionManager(self.workspace, require_approval_for=["read_file"])
        orch.tool_executor = ToolExecutor(self.registry, orch.permission_manager, orch.approval_delegate)

        f1 = self.workspace / "a.txt"
        f2 = self.workspace / "b.txt"
        f1.write_text("aaa", encoding="utf-8")
        f2.write_text("bbb", encoding="utf-8")

        # Mock LLM calls: 2 parallel read_file calls
        mock_response = ParsedAssistantResponse(
            content="",
            tool_calls=[
                ToolCallItem(id="c1", name="read_file", arguments={"path": "a.txt"}),
                ToolCallItem(id="c2", name="read_file", arguments={"path": "b.txt"}),
            ],
            thinking="",
        )
        orch.llm_client.chat = MagicMock(side_effect=[
            mock_response,
            ParsedAssistantResponse(content="Tamamlandı.", tool_calls=[]),
        ])

        # Approval delegate reddetsin
        rec_del = RecordingApprovalDelegate(approve_all=False)
        orch.approval_delegate = rec_del
        orch.tool_executor.approval_delegate = rec_del

        orch.run_turn("Dosyaları oku")
        self.assertGreater(len(rec_del.requests), 0)

    def test_write_multiple_files_reports_partial_rollback_failure(self) -> None:
        """Verify write multiple files reports partial rollback failure."""
        class FailingUndoManager(UndoManager):
            def undo_last(self) -> str:
                return "Hata: 'locked.txt' geri alınırken sorun oluştu: [Errno 13] Permission denied"

        fail_undo = FailingUndoManager()
        reg = ToolRegistry()
        register_filesystem_tools(reg, self.config, undo_manager=fail_undo)

        payload = [
            {"path": "good.txt", "content": "first"},
            {"path": "invalid_dir/bad.txt", "content": "second"},
        ]
        (self.workspace / "invalid_dir").write_text("I am a file, not a directory!", encoding="utf-8")

        res = reg.execute("write_multiple_files", {"files": payload})
        self.assertIn("Kritik Hata", res)
        self.assertIn("KISMİ BAŞARISIZLIK", res)
        self.assertIn("locked.txt", res)

    def test_secret_masking_pem_db_url_and_json_secrets(self) -> None:
        """Verify secret masking pem db url and json secrets."""
        raw = (
            "Server connection string: postgresql://db_user:SuperSecretPassword123@db.prod.internal:5432/myapp\n"
            'Configuration payload: {"service": "auth", "client_secret": "my_client_secret_xyz123", "active": true}\n'
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0YqP4w7Z1n9j3V8x...\n"
            "-----END RSA PRIVATE KEY-----\n"
            "Done."
        )
        masked = mask_sensitive_data(raw)

        self.assertNotIn("SuperSecretPassword123", masked)
        self.assertIn("postgresql://db_user:***MASKED_PASSWORD***@db.prod.internal:5432/myapp", masked)
        self.assertNotIn("my_client_secret_xyz123", masked)
        self.assertIn('"client_secret": "***MASKED_SECRET***"', masked)
        self.assertNotIn("MIIEowIBAAKCAQEA0YqP4w7Z1n9j3V8x", masked)
        self.assertIn("***MASKED_PRIVATE_KEY_BLOCK***", masked)

    def test_cli_slash_commands_and_run_governed_by_tool_executor(self) -> None:
        """Verify cli slash commands and run governed by tool executor."""
        from evren_code.cli import handle_slash_command
        from evren_code.core.evren_auth import EvrenAuthManager
        from evren_code.core.session import SessionManager
        from evren_code.ui.console import ConsoleUI

        orch = AgentOrchestrator(config=self.config)
        orch.tool_executor.execute = MagicMock(return_value="[MOCK_EXECUTED]")
        ui = ConsoleUI(self.event_bus)

        handled = handle_slash_command(
            "/run pytest -v",
            orch,
            EvrenAuthManager(self.config),
            SessionManager(self.config),
            ui,
        )
        self.assertTrue(handled)
        orch.tool_executor.execute.assert_called_with(
            tool_name="run_command",
            arguments={"command": "pytest -v", "cwd": "."},
            caller_role="main",
            caller_agent="cli_user",
            is_trust_mode=orch.trust_mode,
        )


if __name__ == "__main__":
    unittest.main()
