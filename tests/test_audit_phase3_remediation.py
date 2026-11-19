"""Test audit phase3 remediation module."""

import json
import tempfile
import unittest
from pathlib import Path

from evren_code.config import AppConfig, load_config
from evren_code.core.agent import AgentOrchestrator, DefaultAutoApproveDelegate
from evren_code.core.context import ContextManager
from evren_code.core.events import ApprovalRequest, EventBus
from evren_code.core.file_tracker import FileStateTracker
from evren_code.core.parser import StatefulJsonRepair, ToolCallItem, parse_assistant_message, repair_and_parse_json
from evren_code.core.self_healing import SelfHealingManager, VerificationTask
from evren_code.tools.filesystem import register_filesystem_tools
from evren_code.tools.registry import ToolRegistry


class TestAuditPhase3Remediation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name).resolve()
        self.config = load_config(workspace_root=self.workspace_root)
        self.context = ContextManager(self.config, base_system_prompt="Test Sistem Promptu")
        self.file_tracker = FileStateTracker()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_conservative_token_estimation(self):
        """Verify conservative token estimation."""
        turkish_code = """
        public class KullaniciServisi {
            // Türkçe karakterler ve yoğun semboller
            public String kullaniciKaydet(String ad, String soyad) {
                if (ad == null || soyad == null) {
                    throw new IllegalArgumentException("Ad ve soyad boş olamaz!");
                }
                return "Başarıyla kaydedildi: " + ad + " " + soyad;
            }
        }
        """
        tok_est = ContextManager._estimate_text_tokens(turkish_code)
        raw_len = len(turkish_code)
        naive_estimate = raw_len // 3.5
        self.assertGreater(tok_est, naive_estimate)
        self.assertLessEqual(tok_est, int(raw_len / 2.0) + 2)

    def test_fresh_output_protection_in_eviction(self):
        """Verify fresh output protection in eviction."""
        self.context.add_user_message("Eski görev")
        self.context.add_assistant_message(
            content="Eski araç çağırıyorum",
            tool_calls=[ToolCallItem(id="c1", name="read_file", arguments={"path": "old.txt"})],
        )
        old_output = "Dosya: old.txt\n" + ("x" * 2500)
        self.context.add_tool_result("c1", "read_file", old_output)

        self.context.add_assistant_message(
            content="Tur 2 aracı",
            tool_calls=[ToolCallItem(id="c2", name="read_file", arguments={"path": "mid.txt"})],
        )
        mid_output = "Dosya: mid.txt\n" + ("y" * 2500)
        self.context.add_tool_result("c2", "read_file", mid_output)

        # Tur 3: En taze tur (son 1. tur)
        self.context.add_assistant_message(
            content="Tur 3 aracı (en taze)",
            tool_calls=[ToolCallItem(id="c3", name="run_command", arguments={"command": "mvn test"})],
        )
        fresh_output = "Test Sonucu: BUILD SUCCESS\n" + ("z" * 2500)
        self.context.add_tool_result("c3", "run_command", fresh_output)

        evicted = self.context.evict_large_tool_results(max_chars_per_result=500, file_tracker=self.file_tracker)
        self.assertGreaterEqual(evicted, 1)

        msgs = self.context.raw_messages
        self.assertIn("bağlam tasarrufu için kırpıldı", str(msgs[2].get("content")))

        self.assertEqual(msgs[4].get("content"), mid_output)
        self.assertEqual(msgs[6].get("content"), fresh_output)

    def test_file_tracker_invalidation_on_eviction(self):
        """Verify file tracker invalidation on eviction."""
        sample_file = self.workspace_root / "src" / "App.java"
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text("public class App { int a = 10; }\n" + ("// dolgu satırı\n" * 100), encoding="utf-8")

        self.file_tracker.record_read(str(sample_file))
        is_unchanged, stub = self.file_tracker.check_file(str(sample_file))
        self.assertTrue(is_unchanged)
        self.assertIn("[Dosya Değişmedi / File Unchanged]", stub)

        self.context.add_assistant_message(
            content="Okuma",
            tool_calls=[ToolCallItem(id="c1", name="read_file", arguments={"path": "src/App.java"})],
        )
        big_content = f"Dosya: src/App.java (Satır 1-100)\n" + ("x" * 2000)
        self.context.add_tool_result("c1", "read_file", big_content)

        self.context.add_assistant_message("Tur 2")
        self.context.add_assistant_message("Tur 3")

        self.context.evict_large_tool_results(max_chars_per_result=400, file_tracker=self.file_tracker)

        is_unchanged_after, stub_after = self.file_tracker.check_file(str(sample_file))
        self.assertFalse(is_unchanged_after)
        self.assertIsNone(stub_after)

    def test_default_auto_approve_delegate_fail_safe(self):
        """Verify default auto approve delegate fail safe."""
        delegate = DefaultAutoApproveDelegate()

        safe_req = ApprovalRequest(
            tool_name="read_file",
            arguments={"path": "README.md"},
            description="Dosya okuma",
            risk_level="standard",
        )
        self.assertTrue(delegate.request_approval(safe_req))

        danger_req = ApprovalRequest(
            tool_name="run_command",
            arguments={"command": "rm -rf /"},
            description="Kök dizini silme",
            risk_level="standard",
        )
        self.assertFalse(delegate.request_approval(danger_req))

        win_danger_req = ApprovalRequest(
            tool_name="run_command",
            arguments={"command": "del /f /s /q C:\\Windows"},
            description="Windows silme",
            risk_level="standard",
        )
        self.assertFalse(delegate.request_approval(win_danger_req))

        git_reset_req = ApprovalRequest(
            tool_name="git_reset_hard",
            arguments={},
            description="Tüm geçmişi ez",
            risk_level="standard",
        )
        self.assertFalse(delegate.request_approval(git_reset_req))

        crit_req = ApprovalRequest(
            tool_name="run_command",
            arguments={"command": "echo test"},
            description="Kritik işlem",
            risk_level="critical",
        )
        self.assertFalse(delegate.request_approval(crit_req))

    def test_wrapper_aware_self_healing(self):
        """Verify wrapper aware self healing."""
        sh_mgr = SelfHealingManager(self.workspace_root)

        # 1. Projede mvnw.cmd varsa
        mvnw_cmd_file = self.workspace_root / "mvnw.cmd"
        mvnw_cmd_file.write_text("@echo off\necho wrapper", encoding="utf-8")

        res_cmd = sh_mgr.check_workspace_wrapper("mvn test-compile -q", cwd=self.workspace_root)
        self.assertEqual(res_cmd, "mvnw.cmd test-compile -q")

        # 2. Projede gradlew.bat varsa
        mvnw_cmd_file.unlink()
        gradlew_file = self.workspace_root / "gradlew.bat"
        gradlew_file.write_text("@echo off\necho gradle wrapper", encoding="utf-8")

        res_gradle = sh_mgr.check_workspace_wrapper("gradle compileJava -q", cwd=self.workspace_root)
        self.assertEqual(res_gradle, "gradlew.bat compileJava -q")

    def test_truncated_tool_call_protection(self):
        """Verify truncated tool call protection."""
        raw_cut_json = '{"name": "write_file", "arguments": {"path": "src/Broken.java", "content": "public class Broken { int x ='
        parsed = repair_and_parse_json(raw_cut_json)
        self.assertTrue(parsed.get("_is_truncated"))

        asst_msg = {
            "role": "assistant",
            "content": f"<tool_call>{raw_cut_json}</tool_call>",
        }
        resp = parse_assistant_message(asst_msg)
        self.assertEqual(len(resp.tool_calls), 1)
        tc = resp.tool_calls[0]
        self.assertTrue(tc.is_truncated)
        self.assertEqual(tc.name, "write_file")


if __name__ == "__main__":
    unittest.main()
