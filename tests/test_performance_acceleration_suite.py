"""Unit tests verifying performance acceleration, stale syntax gate clearing,
path newline auto-sanitization, and smart command timeouts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import load_config
from evren_code.core.context import ToolCallItem
from evren_code.core.turn_pipeline.context import TurnContext
from evren_code.core.turn_pipeline.gatekeeper import TurnGatekeeper
from evren_code.core.turn_pipeline.tool_runner import TurnToolRunner
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.filesystem import register_filesystem_tools
from evren_code.tools.shell import calculate_effective_timeout


class TestPerformanceAccelerationSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.config = load_config(self.workspace)
        self.config.workspace_root = self.workspace
        self.config.tools.command_timeout_seconds = 60
        self.config.tools.build_command_timeout_seconds = 90
        self.registry = ToolRegistry()
        register_filesystem_tools(self.registry, self.config)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_syntax_gate_clears_stale_errors_on_disk_validity(self) -> None:
        """Verify syntax gate clears stale errors on disk validity."""
        clean_file = self.workspace / "clean_code.py"
        clean_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

        mock_agent = MagicMock()
        mock_agent.workspace = self.workspace
        mock_agent.event_bus = MagicMock()

        gatekeeper = TurnGatekeeper(mock_agent)
        turn_ctx = TurnContext(user_input="test task")
        turn_ctx.files_with_syntax_errors["clean_code.py"] = "Old stale syntax error from step 10"

        mock_response = MagicMock()
        mock_response.content = "Final Delivery Report"

        passed = gatekeeper._check_syntax_error_gate(turn_ctx, mock_response)
        self.assertTrue(passed)
        self.assertEqual(len(turn_ctx.files_with_syntax_errors), 0)

    def test_path_sanitization_removes_newlines_and_spaces(self) -> None:
        """Verify path sanitization removes newlines and spaces."""
        write_fn = self.registry.get("write_file").handler
        read_fn = self.registry.get("read_file").handler
        edit_fn = self.registry.get("edit_file").handler

        # 1. write_file with trailing newline
        res = write_fn(path="test_file.txt\n", content="Line 1\nLine 2\n")
        self.assertIn("test_file.txt", res)
        self.assertTrue((self.workspace / "test_file.txt").exists())

        # 2. read_file with newline and quotes
        read_res = read_fn(path=' "test_file.txt"\n ')
        self.assertIn("Line 1", read_res)

        # 3. edit_file with newline
        edit_res = edit_fn(path="\ntest_file.txt\n", target="Line 1", replacement="Line One")
        self.assertIn("Success", edit_res)
        self.assertIn("Line One", (self.workspace / "test_file.txt").read_text(encoding="utf-8"))

    def test_write_multiple_files_accepts_single_dict_and_strips_newlines(self) -> None:
        """Verify write multiple files accepts single dict and strips newlines."""
        wmf = self.registry.get("write_multiple_files").handler

        res = wmf(files={"path": "single_batch.txt\n", "content": "batch content"})
        self.assertIn("Success", res)
        self.assertTrue((self.workspace / "single_batch.txt").exists())

        res2 = wmf(files=[{"path": "\nbatch2.txt\n", "content": "batch 2 content"}])
        self.assertIn("Success", res2)
        self.assertTrue((self.workspace / "batch2.txt").exists())

    def test_tool_runner_clears_syntax_errors_on_tsc_exit_zero(self) -> None:
        """Verify tool runner clears syntax errors on tsc exit zero."""
        mock_agent = MagicMock()
        mock_agent.workspace = self.workspace
        mock_agent.tools = {}
        mock_agent.event_bus = MagicMock()
        mock_agent.context = MagicMock()

        mock_loop_guard = MagicMock()
        mock_loop_guard.check_muted_tool.return_value = (False, False, "")
        mock_loop_guard.check_consecutive_tool_call_loop.return_value = (False, "")
        mock_loop_guard.inspect_and_guard_tool_call.return_value = (False, None, None)
        runner = TurnToolRunner(mock_agent, mock_loop_guard)
        turn_ctx = TurnContext(user_input="test compile")
        turn_ctx.target_spec = MagicMock()
        turn_ctx.files_with_syntax_errors["src/lib/i18n.ts"] = "TS2307 Module not found"

        tc = ToolCallItem(
            id="call_tsc",
            name="run_command",
            arguments={"command": "npx tsc --noEmit"},
            from_xml=False,
        )

        mock_response = MagicMock()
        mock_response.finish_reason = "stop"
        mock_response.is_truncated = False
        mock_response.thinking = "Thinking about tsc compilation"
        mock_response.tool_calls = [tc]

        # mock tool execution to return Exit Code: 0
        mock_agent.tool_executor.execute.return_value = "Exit Code / Çıkış Kodu (Exit Code): 0\nAll declarations valid."

        runner.handle_tool_calls(turn_ctx=turn_ctx, response=mock_response)

        # files_with_syntax_errors must be empty now!
        self.assertEqual(len(turn_ctx.files_with_syntax_errors), 0)

    def test_build_timeout_configured_to_90_seconds(self) -> None:
        """Verify build timeout configured to 90 seconds."""
        timeout = calculate_effective_timeout("npm run build", self.config)
        self.assertEqual(timeout, 90)


if __name__ == "__main__":
    unittest.main()
