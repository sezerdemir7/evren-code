"""Test mcp server module."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evren_code.mcp_server import create_mcp_server


class TestMCPServer(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_mcp_server()
        self.workspace = Path(__file__).resolve().parent.parent

    def test_mcp_server_registers_expected_tools(self) -> None:
        tools = self.server._tool_manager._tools
        self.assertIn("evren_task", tools)
        self.assertIn("evren_architect", tools)
        self.assertIn("evren_code_review", tools)
        self.assertIn("evren_models", tools)
        self.assertIn("evren_llm_call", tools)

    @patch("evren_code.mcp_server._create_llm_client_for_mcp")
    def test_evren_llm_call_tool_execution(self, mock_create_client) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Python kodu: def add(a, b): return a + b"
        mock_resp.thinking = "Toplama fonksiyonu yazılmalı..."
        mock_resp.usage = {"input_tokens": 10, "output_tokens": 20}
        mock_resp.finish_reason = "stop"
        mock_client.chat.return_value = mock_resp
        mock_client.active_model.id = "deepseek-v4.1-flash"
        mock_create_client.return_value = mock_client

        tool_fn = self.server._tool_manager._tools["evren_llm_call"].fn
        raw = tool_fn(prompt="Toplama fonksiyonu yaz", model="deepseek-v4.1-flash")
        data = json.loads(raw)

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model"], "deepseek-v4.1-flash")
        self.assertIn("Toplama fonksiyonu", data["thinking"])
        self.assertIn("def add", data["content"])
        mock_client.close.assert_called_once()

    def test_evren_models_tool_returns_valid_catalog_json(self) -> None:
        tool_fn = self.server._tool_manager._tools["evren_models"].fn
        raw = tool_fn()
        data = json.loads(raw)
        self.assertIn("default_model", data)
        self.assertIn("roles", data)
        self.assertIn("catalog", data)
        self.assertIn("deepseek-v4.1-flash", data["catalog"])
        self.assertIn("architect", data["roles"])

    @patch("evren_code.mcp_server._create_orchestrator_for_workspace")
    def test_evren_task_tool_execution(self, mock_create_orch) -> None:
        mock_orch = MagicMock()
        mock_orch.run_turn.return_value = "Refactor başarıyla yapıldı."
        mock_orch.llm_client.active_model.id = "deepseek-v4.1-flash"
        mock_orch.undo_manager._stack = [
            {"path": str(self.workspace / "src" / "worker.py")},
        ]
        mock_create_orch.return_value = mock_orch

        tool_fn = self.server._tool_manager._tools["evren_task"].fn
        raw = tool_fn(task="refactor auth", workspace=str(self.workspace), role="architect")
        data = json.loads(raw)

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["role"], "architect")
        self.assertEqual(data["summary"], "Refactor başarıyla yapıldı.")
        self.assertIn("src/worker.py", data["modified_files"])
        mock_orch.close.assert_called_once()

    @patch("evren_code.mcp_server._create_orchestrator_for_workspace")
    def test_evren_architect_tool_execution(self, mock_create_orch) -> None:
        mock_orch = MagicMock()
        mock_orch.run_turn.return_value = "Mimari Plan:\n1. Adım\n2. Adım"
        mock_create_orch.return_value = mock_orch

        tool_fn = self.server._tool_manager._tools["evren_architect"].fn
        out = tool_fn(task="Cache katmanı tasarla", workspace=str(self.workspace))

        self.assertIn("Mimari Plan", out)
        mock_create_orch.assert_called_with(self.workspace, role_or_model="architect")
        mock_orch.close.assert_called_once()

    @patch("evren_code.mcp_server._create_orchestrator_for_workspace")
    def test_evren_code_review_tool_execution(self, mock_create_orch) -> None:
        mock_orch = MagicMock()
        mock_orch.run_turn.return_value = "Kod inceleme sonucu: Sorun bulunamadı."
        mock_create_orch.return_value = mock_orch

        tool_fn = self.server._tool_manager._tools["evren_code_review"].fn
        out = tool_fn(path="src/evren_code", workspace=str(self.workspace))

        self.assertIn("Sorun bulunamadı", out)
        mock_create_orch.assert_called_with(self.workspace, role_or_model="reviewer")
        mock_orch.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
