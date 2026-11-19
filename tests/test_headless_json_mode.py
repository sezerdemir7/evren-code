"""Test headless json mode module."""

import io
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evren_code.cli import build_arg_parser, main
from evren_code.config import load_config
from evren_code.core.events import EventBus
from evren_code.core.llm_client import EvrenLLMClient


class TestHeadlessJsonMode(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(__file__).resolve().parent.parent

    def test_arg_parser_supports_headless_json_role_mcp_flags(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args([
            "--headless",
            "--json",
            "--role", "architect",
            "--mcp",
            "projedeki", "testleri", "calistir"
        ])
        self.assertTrue(args.headless)
        self.assertTrue(args.json_output)
        self.assertEqual(args.role, "architect")
        self.assertTrue(args.mcp)
        self.assertEqual(args.prompt, ["projedeki", "testleri", "calistir"])

    def test_headless_without_prompt_returns_error_code(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            ret = main(["--headless"])
            self.assertEqual(ret, 1)
            self.assertIn("görev (prompt) belirtilmelidir", mock_stderr.getvalue())

    def test_json_without_prompt_returns_json_error(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            ret = main(["--json"])
            self.assertEqual(ret, 1)
            raw = mock_stdout.getvalue()
            data = json.loads(raw)
            self.assertEqual(data["status"], "error")
            self.assertIn("prompt", data["error"].lower())

    @patch("evren_code.cli.EvrenAuthManager")
    @patch("evren_code.cli.AgentOrchestrator")
    def test_headless_json_success_flow(self, mock_orch_cls, mock_auth_cls) -> None:
        mock_auth = mock_auth_cls.return_value
        mock_auth.load_stored_key.return_value = True

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_turn.return_value = "İşlem başarıyla tamamlandı."
        mock_orch.undo_manager._stack = [
            {"path": str(self.workspace / "src" / "sample.py")},
            {"path": str(self.workspace / "README.md")},
        ]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            ret = main([
                "--headless",
                "--json",
                "--role", "coder",
                "-C", str(self.workspace),
                "hızlı", "bir", "görev"
            ])
            self.assertEqual(ret, 0)
            raw = mock_stdout.getvalue()
            data = json.loads(raw)

            self.assertEqual(data["status"], "success")
            self.assertEqual(data["task"], "hızlı bir görev")
            self.assertEqual(data["role"], "coder")
            self.assertEqual(data["summary"], "İşlem başarıyla tamamlandı.")
            self.assertIn("src/sample.py", data["modified_files"])
            self.assertIn("README.md", data["modified_files"])

    @patch("evren_code.cli.run_onboarding_flow")
    @patch("evren_code.mcp_server.run_mcp_server")
    def test_mcp_cli_invocation(self, mock_run_mcp, mock_onboarding) -> None:
        mock_run_mcp.return_value = 0
        ret = main(["mcp"])
        self.assertEqual(ret, 0)
        mock_run_mcp.assert_called_once()

    @patch("evren_code.cli.EvrenAuthManager")
    @patch("evren_code.cli.EvrenLLMClient")
    def test_direct_llm_ask_mode(self, mock_llm_cls, mock_auth_cls) -> None:
        mock_auth = mock_auth_cls.return_value
        mock_auth.load_stored_key.return_value = True

        mock_llm = mock_llm_cls.return_value
        mock_resp = MagicMock()
        mock_resp.content = "Doğrudan LLM yanıtı."
        mock_resp.thinking = "Model muhakemesi..."
        mock_resp.usage = {"input": 5, "output": 15}
        mock_resp.finish_reason = "stop"
        mock_llm.chat.return_value = mock_resp
        mock_llm.active_model.id = "deepseek-v4.1-flash"

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            ret = main([
                "ask",
                "--json",
                "-m", "deepseek-v4.1-flash",
                "hızlı", "bir", "soru"
            ])
            self.assertEqual(ret, 0)
            data = json.loads(mock_stdout.getvalue())
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["content"], "Doğrudan LLM yanıtı.")
            self.assertEqual(data["thinking"], "Model muhakemesi...")
            self.assertEqual(data["model"], "deepseek-v4.1-flash")

    def test_agent_help_hidden_command(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            ret = main(["--agent-help"])
            self.assertEqual(ret, 0)
            out = mock_stdout.getvalue()
            self.assertIn("Agent-to-Agent Protocol", out)
            self.assertIn("deepseek-v4.1-flash", out)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout_json:
            ret = main(["instructions", "--json"])
            self.assertEqual(ret, 0)
            data = json.loads(mock_stdout_json.getvalue())
            self.assertEqual(data["name"], "evren-code")
            self.assertIn("available_models", data)

    def test_role_to_model_mapping(self) -> None:
        config = load_config(workspace_root=self.workspace)
        client = EvrenLLMClient(config=config, event_bus=EventBus())

        client.set_model("architect")
        self.assertEqual(client.active_model.id, "deepseek-v4.1-flash")

        client.set_model("coder")
        self.assertEqual(client.active_model.id, "qwen3.8-flash-next")

        client.set_model("vision")
        self.assertEqual(client.active_model.id, "qwen3-vl-30b")

        client.set_model("compactor")
        self.assertEqual(client.active_model.id, "gemma-4-31b")


if __name__ == "__main__":
    unittest.main()
