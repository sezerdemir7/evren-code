"""Test architecture remediation package b module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.context import ContextManager, ensure_tool_call_invariants


class TestArchitectureRemediationPackageB(unittest.TestCase):

    def test_ensure_tool_call_invariants_removes_orphan_leading_tools(self) -> None:
        """Verify ensure tool call invariants removes orphan leading tools."""
        raw_msgs = [
            {"role": "tool", "tool_call_id": "orphan_call_1", "content": "res1"},
            {"role": "user", "content": "Merhaba"},
            {"role": "assistant", "content": "Size nasıl yardımcı olabilirim?"},
        ]
        sanitized = ensure_tool_call_invariants(raw_msgs)
        self.assertEqual(len(sanitized), 2)
        self.assertEqual(sanitized[0]["role"], "user")
        self.assertEqual(sanitized[1]["role"], "assistant")

    def test_ensure_tool_call_invariants_synthesizes_missing_tool_responses(self) -> None:
        """Verify ensure tool call invariants synthesizes missing tool responses."""
        raw_msgs = [
            {"role": "user", "content": "İki araç çalıştır"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_abc", "function": {"name": "read_file", "arguments": "{}"}},
                    {"id": "call_xyz", "function": {"name": "write_file", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "dosya içeriği"},
        ]

        sanitized = ensure_tool_call_invariants(raw_msgs)
        # sanitized: user, assistant, tool(call_abc), tool(call_xyz sentetik)
        self.assertEqual(len(sanitized), 4)
        self.assertEqual(sanitized[1]["role"], "assistant")
        self.assertEqual(sanitized[2]["tool_call_id"], "call_abc")
        self.assertEqual(sanitized[3]["tool_call_id"], "call_xyz")
        self.assertTrue("archived" in sanitized[3]["content"].lower() or "arşivlendi" in sanitized[3]["content"].lower())

    def test_context_compact_preserves_atomic_tool_pairing(self) -> None:
        """Verify context compact preserves atomic tool pairing."""
        cfg = load_config()
        ctx = ContextManager(cfg, "Sen bir asistansın.")

        ctx.add_user_message("Adım 1")
        ctx.add_assistant_message(
            content=None,
            tool_calls=[{"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}],
        )
        ctx.add_tool_result("call_1", "test_tool", "Sonuç 1")
        ctx.add_user_message("Adım 2")
        ctx.add_assistant_message(
            content=None,
            tool_calls=[
                {"id": "call_2a", "function": {"name": "tool_a", "arguments": "{}"}},
                {"id": "call_2b", "function": {"name": "tool_b", "arguments": "{}"}},
            ],
        )
        ctx.add_tool_result("call_2a", "tool_a", "Sonuç 2a")

        ctx.compact(aggressive=True)

        final_msgs = ctx.get_messages()
        self.assertEqual(final_msgs[0]["role"], "system")
        history = final_msgs[1:]

        if history:
            self.assertNotEqual(history[0]["role"], "tool")

        for idx, m in enumerate(history):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                expected_ids = {tc["id"] for tc in m["tool_calls"] if isinstance(tc, dict)}
                subsequent_tools = set()
                j = idx + 1
                while j < len(history) and history[j].get("role") == "tool":
                    subsequent_tools.add(history[j].get("tool_call_id"))
                    j += 1
                for eid in expected_ids:
                    self.assertIn(eid, subsequent_tools)

    def test_agent_cancel_calls_stop_all_on_bg_process_manager(self) -> None:
        """Verify agent cancel calls stop all on bg process manager."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = load_config()
            cfg.workspace_root = Path(tmp_dir)
            mock_llm = MagicMock()

            orch = AgentOrchestrator(cfg, llm_client=mock_llm)
            mock_bg = MagicMock()
            orch.bg_process_manager = mock_bg

            orch.cancel()

            self.assertTrue(orch._cancel_requested)
            mock_bg.stop_all.assert_called_once()
            orch.close()


if __name__ == "__main__":
    unittest.main()
