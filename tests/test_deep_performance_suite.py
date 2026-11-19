"""Unit tests for Phase 2 deep performance, vision caching,
sliding-window tool result compaction, and symbol memory extraction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import load_config
from evren_code.core.context import ToolCallItem
from evren_code.core.turn_pipeline.context import TurnContext
from evren_code.core.turn_pipeline.tool_runner import TurnToolRunner, extract_exported_symbols_from_disk
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.vision import register_vision_tools, _VISION_CACHE


class TestDeepPerformanceSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.config = load_config(self.workspace)
        self.config.workspace_root = self.workspace
        self.registry = ToolRegistry()
        _VISION_CACHE.clear()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        _VISION_CACHE.clear()

    def test_vision_cache_hit_returns_instant_response(self) -> None:
        """Verify vision cache hit returns instant response."""
        img_path = self.workspace / "test_logo.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake_image_bytes_for_testing_cache")

        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Stylized letter A in green and white."
        mock_llm_client.chat.return_value = mock_response

        register_vision_tools(self.registry, self.config, mock_llm_client)
        analyze_fn = self.registry.get("analyze_image").handler

        res1 = analyze_fn(image_path="test_logo.png", question="Describe logo")
        self.assertIn("Görsel Analiz Sonucu", res1)
        self.assertEqual(mock_llm_client.chat.call_count, 1)

        res2 = analyze_fn(image_path="test_logo.png", question="Describe logo")
        self.assertIn("Vision Cache / Önbellekten Getirildi", res2)
        self.assertIn("Stylized letter A", res2)
        self.assertEqual(mock_llm_client.chat.call_count, 1)

    def test_sliding_window_historical_tool_eviction(self) -> None:
        """Verify sliding window historical tool eviction."""
        from evren_code.core.context import ContextManager

        ctx = ContextManager(self.config, base_system_prompt="You are Evren-Code")

        def _tc(call_id: str) -> list:
            return [{"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]

        # Turn 1
        ctx.add_user_message("Create header")
        ctx.add_assistant_message("Creating header", tool_calls=_tc("tc_1"))
        ctx.add_tool_result("tc_1", "read_file", "X" * 3000)

        # Turn 2
        ctx.add_user_message("Create hero")
        ctx.add_assistant_message("Creating hero", tool_calls=_tc("tc_2"))
        ctx.add_tool_result("tc_2", "read_file", "Y" * 4000)

        # Turn 3
        ctx.add_user_message("Create products")
        ctx.add_assistant_message("Creating products", tool_calls=_tc("tc_3"))
        ctx.add_tool_result("tc_3", "read_file", "Z" * 2500)

        # Turn 4
        ctx.add_user_message("Create footer")
        ctx.add_assistant_message("Creating footer", tool_calls=_tc("tc_4"))
        ctx.add_tool_result("tc_4", "read_file", "A" * 2000)

        # Turn 5 (En son turn)
        ctx.add_user_message("Test build")
        ctx.add_assistant_message("Building", tool_calls=_tc("tc_5"))
        ctx.add_tool_result("tc_5", "read_file", "B" * 1500)

        evicted = ctx.compact_sliding_window_tools(keep_recent_turns=3, max_chars_per_result=300)

        self.assertGreaterEqual(evicted, 2)

        messages = ctx.get_messages()
        old_tool_msg = [m for m in messages if m.get("role") == "tool" and "tc_1" in str(m.get("tool_call_id", ""))][0]
        self.assertIn("Auto-Compacted", old_tool_msg["content"])
        self.assertLess(len(old_tool_msg["content"]), 500)

        recent_tool_msg = [m for m in messages if m.get("role") == "tool" and "tc_5" in str(m.get("tool_call_id", ""))][0]
        self.assertNotIn("Auto-Compacted", recent_tool_msg["content"])
        self.assertEqual(len(recent_tool_msg["content"]), 1500)

    def test_symbol_memory_card_extraction(self) -> None:
        """Verify symbol memory card extraction."""
        ts_file = self.workspace / "service.ts"
        ts_content = """
        export type UserId = string;
        export interface UserProfile {
            name: string;
            age: number;
        }
        export const DEFAULT_TIMEOUT = 30;
        export async function fetchUserData(id: UserId): Promise<UserProfile> {
            return { name: "ExampleUser", age: 5 };
        }
        export default function MainApp() {}
        """
        ts_file.write_text(ts_content, encoding="utf-8")

        symbols = extract_exported_symbols_from_disk(ts_file)
        self.assertIn("UserId", symbols)
        self.assertIn("UserProfile", symbols)
        self.assertIn("DEFAULT_TIMEOUT", symbols)
        self.assertIn("fetchUserData", symbols)
        self.assertIn("MainApp", symbols)

    def test_tool_runner_enriches_write_result_with_exported_symbols(self) -> None:
        """Verify tool runner enriches write result with exported symbols."""
        mock_agent = MagicMock()
        mock_agent.workspace = self.workspace
        mock_agent.current_task_contract = None
        mock_agent._active_speculative_futures = {}
        mock_agent.event_bus = MagicMock()
        mock_agent.context = MagicMock()

        mock_loop_guard = MagicMock()
        mock_loop_guard.check_muted_tool.return_value = (False, False, "")
        mock_loop_guard.check_consecutive_tool_call_loop.return_value = (False, "")
        mock_loop_guard.inspect_and_guard_tool_call.return_value = (False, None, None)

        runner = TurnToolRunner(mock_agent, mock_loop_guard)
        turn_ctx = TurnContext(user_input="create component")
        turn_ctx.target_spec = MagicMock()

        comp_file = self.workspace / "Button.tsx"
        comp_file.write_text("export function Button() { return null; }\nexport type ButtonVariant = 'primary' | 'secondary';", encoding="utf-8")

        tc = ToolCallItem(
            id="call_write",
            name="write_file",
            arguments={"path": "Button.tsx", "content": comp_file.read_text(encoding="utf-8")},
            from_xml=False,
        )

        mock_response = MagicMock()
        mock_response.finish_reason = "stop"
        mock_response.is_truncated = False
        mock_response.thinking = "Creating button component"
        mock_response.tool_calls = [tc]

        mock_agent.tool_executor.execute.return_value = "Success: 'Button.tsx' written successfully."

        should_continue, step_results, terminal_msg = runner.handle_tool_calls(turn_ctx=turn_ctx, response=mock_response)

        self.assertEqual(len(step_results), 1)
        res_text = step_results[0]["result"]
        self.assertIn("Tanımlanan Semboller / Exported Symbols", res_text)
        self.assertIn("Button", res_text)
        self.assertIn("ButtonVariant", res_text)
        self.assertIn("button.tsx", turn_ctx.exported_symbols_cache)


if __name__ == "__main__":
    unittest.main()
