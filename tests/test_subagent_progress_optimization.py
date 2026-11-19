"""Test subagent progress optimization module."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from evren_code.core.events import EventBus, EventType
from evren_code.core.parser import ParsedAssistantResponse
from evren_code.core.subagents import (
    SubagentNote,
    SubagentProgressTracker,
)


class TestSubagentProgressOptimization(unittest.TestCase):
    """Test subagent progress optimization implementation."""

    def test_20_normal_tool_calls_consume_zero_llm_tokens(self) -> None:
        """Verify 20 normal tool calls consume zero llm tokens."""
        mock_llm = MagicMock()
        tracker = SubagentProgressTracker(
            role="researcher",
            subagent_id="test_res_1",
            heartbeat_interval=30.0,
            cooldown_seconds=60.0,
            max_model_calls=3,
        )

        for step in range(1, 21):
            tracker.record_step(
                step=step,
                tool_name="read_file",
                arguments={"path": f"src/module_{step}.py"},
                result_str="def calculate():\n    return 42\n",
            )
            msg, usage = tracker.generate_status_update(
                llm_client=mock_llm,
                compactor_model_id="fast-model",
            )
            self.assertEqual(usage["total_tokens"], 0)
            self.assertEqual(usage["prompt_tokens"], 0)
            self.assertFalse(usage["is_estimated"])
            self.assertIn("read_file", msg)
            self.assertIn(f"src/module_{step}.py", msg)

        self.assertEqual(tracker.model_calls_count, 0)
        self.assertEqual(tracker.total_progress_tokens["total_tokens"], 0)
        mock_llm.chat.assert_not_called()

    def test_repeated_critical_error_deduplication_and_cooldown(self) -> None:
        """Verify repeated critical error deduplication and cooldown."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ParsedAssistantResponse(
            content="Derleme hatası: Bar modülü bulunamadı.",
            usage={"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95},
        )

        tracker = SubagentProgressTracker(
            role="tester",
            subagent_id="test_qa_1",
            cooldown_seconds=60.0,
            max_model_calls=3,
        )

        tracker.record_step(
            step=1,
            tool_name="run_command",
            arguments={"command": "mvn test"},
            result_str="Command failed with exit code 1: Compilation error in Foo.java",
        )
        msg1, usage1 = tracker.generate_status_update(
            llm_client=mock_llm,
            compactor_model_id="compactor",
        )
        self.assertEqual(tracker.model_calls_count, 1)
        self.assertEqual(usage1["total_tokens"], 95)
        self.assertIn("Derleme hatası", msg1)

        for step in range(2, 11):
            tracker.record_step(
                step=step,
                tool_name="run_command",
                arguments={"command": "mvn test"},
                result_str="Command failed with exit code 1: Compilation error in Foo.java",
            )
            msg, usage = tracker.generate_status_update(
                llm_client=mock_llm,
                compactor_model_id="compactor",
            )
            self.assertEqual(usage["total_tokens"], 0)
            self.assertIn("[Tekrar Eden Hata]", msg)

        self.assertEqual(tracker.model_calls_count, 1)
        self.assertEqual(mock_llm.chat.call_count, 1)

    def test_heartbeat_consumes_zero_tokens_and_emits_events(self) -> None:
        """Verify heartbeat consumes zero tokens and emits events."""
        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(lambda ev: events_received.append(ev))

        tracker = SubagentProgressTracker(
            role="researcher",
            subagent_id="sub_hb_1",
            heartbeat_interval=0.01,
            event_bus=event_bus,
        )

        tracker.record_step(
            step=1,
            tool_name="read_file",
            arguments={"path": "package.json"},
            result_str='{"name": "test-pkg"}\n',
        )

        time.sleep(0.02)
        self.assertTrue(tracker.should_emit_heartbeat())
        tracker.emit_heartbeat()

        hb_events = [e for e in events_received if e.type == EventType.SUBAGENT_HEARTBEAT]
        self.assertEqual(len(hb_events), 1)
        self.assertEqual(hb_events[0].data.get("role"), "researcher")
        self.assertEqual(hb_events[0].data.get("subagent_id"), "sub_hb_1")
        self.assertIn("Dosya incelendi", hb_events[0].data.get("latest_summary"))

        self.assertEqual(tracker.model_calls_count, 0)
        self.assertEqual(tracker.total_progress_tokens["total_tokens"], 0)

    def test_scratchpad_isolation_between_subagents(self) -> None:
        """Verify scratchpad isolation between subagents."""
        sub1 = SubagentProgressTracker(role="researcher", subagent_id="agent_1")
        sub2 = SubagentProgressTracker(role="tester", subagent_id="agent_2")

        sub1.record_step(
            step=1,
            tool_name="read_file",
            arguments={"path": "secret_doc.md"},
            result_str="# Gizli Mimari\n",
        )

        sub2.record_step(
            step=1,
            tool_name="run_command",
            arguments={"command": "npm test"},
            result_str="Tests passed\n",
        )

        self.assertEqual(len(sub1.notes), 1)
        self.assertEqual(sub1.notes[0].target, "secret_doc.md")

        self.assertEqual(len(sub2.notes), 1)
        self.assertEqual(sub2.notes[0].target, "npm test")
        self.assertNotIn("secret_doc.md", [n.target for n in sub2.notes])

    def test_sensitive_data_masked_in_scratchpad_and_status(self) -> None:
        """Verify sensitive data masked in scratchpad and status."""
        tracker = SubagentProgressTracker(role="tester", subagent_id="agent_sec")

        sensitive_token = "sk-live-99238491823912839128"
        sensitive_cmd = f"curl -H 'Authorization: Bearer {sensitive_token}' https://internal.corp"

        note = tracker.record_step(
            step=1,
            tool_name="run_command",
            arguments={"command": sensitive_cmd},
            result_str=f"Hata: 401 Unauthorized for {sensitive_token}\nTraceback: failed auth",
        )

        self.assertNotIn(sensitive_token, note.target)
        self.assertNotIn(sensitive_token, note.summary)

        msg, _ = tracker.generate_status_update()
        self.assertNotIn(sensitive_token, msg)

    def test_heartbeat_thread_clean_termination_on_stop(self) -> None:
        """Verify heartbeat thread clean termination on stop."""
        tracker = SubagentProgressTracker(
            role="architect",
            subagent_id="agent_term",
            heartbeat_interval=1.0,
        )

        tracker.start_heartbeat()
        self.assertIsNotNone(tracker._heartbeat_thread)
        self.assertTrue(tracker._heartbeat_thread.is_alive())

        # Durdur
        tracker.stop()
        self.assertTrue(tracker._heartbeat_stop_event.is_set())
        self.assertIsNone(tracker._heartbeat_thread)

        tracker.stop()
        self.assertIsNone(tracker._heartbeat_thread)

    def test_token_accounting_real_usage_vs_fallback_estimation(self) -> None:
        """Verify token accounting real usage vs fallback estimation."""
        mock_real = MagicMock()
        mock_real.chat.return_value = ParsedAssistantResponse(
            content="Bilinmeyen derleme hatası oluştu.",
            usage={"prompt_tokens": 110, "completion_tokens": 20, "total_tokens": 130},
        )

        t1 = SubagentProgressTracker(role="tester", subagent_id="t1")
        t1.record_step(
            step=1,
            tool_name="run_command",
            arguments={"command": "gradle test"},
            result_str="Command failed with exit code 1: fatal exception",
        )
        _, usage1 = t1.generate_status_update(llm_client=mock_real, compactor_model_id="fast")
        self.assertFalse(usage1["is_estimated"])
        self.assertEqual(usage1["total_tokens"], 130)
        self.assertEqual(usage1["prompt_tokens"], 110)
        self.assertEqual(usage1["completion_tokens"], 20)

        mock_no_usage = MagicMock()
        mock_no_usage.chat.return_value = ParsedAssistantResponse(
            content="Sözdizimi hatası bulundu.",
            usage=None,
        )

        t2 = SubagentProgressTracker(role="reviewer", subagent_id="t2")
        t2.record_step(
            step=1,
            tool_name="read_file",
            arguments={"path": "bad.py"},
            result_str="Hata: SyntaxError in line 10",
        )
        _, usage2 = t2.generate_status_update(llm_client=mock_no_usage, compactor_model_id="fast")
        self.assertTrue(usage2["is_estimated"])
        self.assertGreater(usage2["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
