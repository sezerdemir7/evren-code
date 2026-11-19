"""Test audit remediation rabbit hole module."""

from pathlib import Path
import tempfile
import unittest

from evren_code.config import load_config
from evren_code.core.planner import (
    ComplexityAssessment,
    ComplexityLevel,
    GoalGate,
    ScopeGuard,
    TaskComplexityClassifier,
)
from evren_code.core.security import CommandClassifier, PermissionManager
from evren_code.tools.filesystem import register_filesystem_tools
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.shell import mask_sensitive_data


class TestRabbitHoleRemediation(unittest.TestCase):
    """Test rabbit hole remediation implementation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.registry = ToolRegistry()
        self.config = load_config()
        self.config.workspace_root = self.workspace
        register_filesystem_tools(self.registry, self.config)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_mask_sensitive_data_preserves_token_prefix(self) -> None:
        """Verify mask sensitive data preserves token prefix."""
        raw_header = "Set-Cookie: XSRF-TOKEN=abcdef1234567890abcdef1234567890; Path=/; SameSite=Lax"
        masked = mask_sensitive_data(raw_header)
        self.assertNotIn("Bearer", masked)
        self.assertIn("XSRF-TOKEN=***MASKED_TOKEN***", masked)

        bearer_header = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        masked_bearer = mask_sensitive_data(bearer_header)
        self.assertIn("Bearer ***MASKED_TOKEN***", masked_bearer)

    def test_read_file_blocks_binary_class_files(self) -> None:
        """Verify read file blocks binary class files."""
        class_file = self.workspace / "SecurityConfig.class"
        class_file.write_bytes(b"\xca\xfe\xba\xbe\x00\x00\x00\x34")

        read_fn = self.registry.get("read_file").handler
        result = read_fn(path="SecurityConfig.class")

        self.assertTrue("Hata:" in result or "Error:" in result)
        self.assertIn("derlenmiş bir ikili (binary) dosyadır", result)

    def test_command_classifier_blocks_binary_inspection(self) -> None:
        """Verify command classifier blocks binary inspection."""
        cmd1 = "[IO.File]::ReadAllBytes('backend/target/classes/SecurityConfig.class')"
        is_dang1, reason1 = CommandClassifier.analyze(cmd1)
        self.assertTrue(is_dang1)
        self.assertIn("ikili (binary/.class) dosya", reason1)

        cmd2 = "javap -c SecurityConfig.class"
        is_dang2, reason2 = CommandClassifier.analyze(cmd2)
        self.assertTrue(is_dang2)
        self.assertIn("Java bayt kodu", reason2)

    def test_task_complexity_urgent_keywords(self) -> None:
        """Verify task complexity urgent keywords."""
        prompt = "register sayfası eksik burada hemen bitane register sayfası ekle hemen hızlı bir şekilde yap bunu"
        assessment = TaskComplexityClassifier.classify(prompt)

        self.assertTrue(assessment.is_urgent)
        self.assertLessEqual(assessment.max_turns, 12)
        self.assertFalse(assessment.allow_delegation)

    def test_scope_guard_blocks_unsolicited_live_api_actions(self) -> None:
        """Verify scope guard blocks unsolicited live api actions."""
        prompt = "register sayfası eksik burada hemen bitane register sayfası ekle hemen hızlı bir şekilde yap bunu"
        unsolicited_cmd = "curl.exe -X POST http://localhost:8080/api/v1/auth/register -d @data.json"

        is_blocked, reason = ScopeGuard.check_unsolicited_action(prompt, unsolicited_cmd)
        self.assertTrue(is_blocked)
        self.assertIn("Kapsam Aşımı Engeli", reason)

        prompt_with_demo = "Bana test için bir demo hesap aç ve register et"
        is_blocked2, _ = ScopeGuard.check_unsolicited_action(prompt_with_demo, unsolicited_cmd)
        self.assertFalse(is_blocked2)

    def test_permission_manager_enforces_scope_guard(self) -> None:
        """Verify permission manager enforces scope guard."""
        pm = PermissionManager(workspace_root=self.workspace)
        pm.current_prompt = "register sayfası eksik burada hemen bitane register sayfası ekle"

        decision = pm.check(
            tool_name="run_command",
            arguments={"command": "curl.exe -X POST http://localhost:8080/api/v1/workspaces -d @ws.json"},
            is_trust_mode=True,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("Kapsam Aşımı Engeli", decision.reason)

    def test_goal_gate_detects_completion(self) -> None:
        """Verify goal gate detects completion."""
        prompt = "register sayfası eksik hemen bir register sayfası ekle"
        modified = ["src/app/(auth)/register/page.tsx", "src/features/auth/schemas.ts"]
        build_output = "Compiled successfully in 1.4s (314 modules)\nRoute (app) /register"

        res = GoalGate.evaluate(
            prompt=prompt,
            modified_files=modified,
            recent_commands_output=build_output,
        )
        self.assertTrue(res.is_satisfied)
        self.assertEqual(res.suggested_action, "stop_and_report")
        self.assertIn("Definition of Done", res.reason)

    def test_goal_gate_does_not_trigger_prematurely(self) -> None:
        """Verify goal gate does not trigger prematurely."""
        prompt = "register sayfası eksik hemen bir register sayfası ekle"
        modified = ["README.md"]
        build_output = "Failed to compile"

        res = GoalGate.evaluate(
            prompt=prompt,
            modified_files=modified,
            recent_commands_output=build_output,
        )
        self.assertFalse(res.is_satisfied)

    def test_normalize_windows_command_preserves_invoke_webrequest(self) -> None:
        """Verify normalize windows command preserves invoke webrequest."""
        from evren_code.tools.shell import normalize_windows_command
        cmd = "Invoke-WebRequest -Uri http://localhost:8080/api/v1/auth/login -Method Post"
        norm = normalize_windows_command(cmd, use_cmd=False)
        self.assertIn("Invoke-WebRequest", norm)
        self.assertIn("-UseBasicParsing", norm)
        self.assertNotIn("Invoke-RestMethod", norm)

    def test_normalize_windows_command_wraps_dir_flags(self) -> None:
        """Verify normalize windows command wraps dir flags."""
        from evren_code.tools.shell import normalize_windows_command
        cmd = "dir /s /b backend\\src\\main"
        norm = normalize_windows_command(cmd, use_cmd=False)
        self.assertIn('cmd.exe /c "dir /s /b backend\\src\\main"', norm)

    def test_edit_file_fuzzy_blank_line_tolerance(self) -> None:
        """Verify edit file fuzzy blank line tolerance."""
        test_file = self.workspace / "api.ts"
        test_file.write_text(
            "export const getCatalog = async () => {\n"
            "  const res = await api.get('/catalog');\n"
            "\n"
            "  return res.data;\n"
            "};\n",
            encoding="utf-8",
        )

        edit_fn = self.registry.get("edit_file").handler
        target_without_blank = (
            "export const getCatalog = async () => {\n"
            "  const res = await api.get('/catalog');\n"
            "  return res.data;\n"
            "};"
        )
        replacement = (
            "export const getCatalog = async () => {\n"
            "  const res = await api.get('/catalog/v2');\n"
            "  return res.data;\n"
            "};"
        )
        res = edit_fn(path="api.ts", target=target_without_blank, replacement=replacement)
        self.assertIn("Başarılı:", res)
        self.assertIn("/catalog/v2", test_file.read_text(encoding="utf-8"))

    def test_java_inline_diagnostics_catches_missing_annotation_import(self) -> None:
        """Verify java inline diagnostics catches missing annotation import."""
        write_fn = self.registry.get("write_file").handler
        java_code = (
            "package com.example.controller;\n"
            "@RestController\n"
            "public class AuthController {\n"
            "    @PutMapping(\"/profile\")\n"
            "    public void update() {}\n"
            "}\n"
        )
        res = write_fn(path="AuthController.java", content=java_code)
        self.assertIn("Java Eksik Import", res)
        self.assertIn("@PutMapping", res)


if __name__ == "__main__":
    unittest.main()
