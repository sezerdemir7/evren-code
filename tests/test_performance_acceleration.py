"""Test performance acceleration module."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from pathlib import Path

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.context import ContextManager
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.core.multi_agent.agent_factory import ROLE_ALLOWED_TOOLS, SpecialistAgent
from evren_code.core.multi_agent.coordinator import MultiAgentCoordinator
from evren_code.core.multi_agent.models import AgentRole
from evren_code.core.parser import (
    ParsedAssistantResponse,
    ToolCallItem,
    analyze_reasoning_overthinking,
    parse_assistant_message,
    sanitize_thinking_for_context,
)
from evren_code.core.security import PermissionManager, ToolExecutor
from evren_code.core.stream_filter import StreamTagFilter
from evren_code.tools.registry import ToolRegistry


class TestPerformanceAcceleration(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(__file__).parent.parent
        self.config = load_config(workspace_root=self.workspace)

    def test_extended_repetition_guard_catches_sentences_and_whitespace_flood(self) -> None:
        """Verify extended repetition guard catches sentences and whitespace flood."""
        phrase = "Bu adimda ogrenci ekranlari mekanik goc ile tema token'larina aktarilacak ve dogrulanacaktir."
        repeated_sentence_arg = f'{{"notes": "{phrase} {phrase} {phrase}"}}'
        self.assertTrue(EvrenLLMClient._has_tool_arg_repetition(repeated_sentence_arg))

        newline_flood_arg = '{"content": "' + ("\n   \n   \n   " * 5) + '"}'
        self.assertTrue(EvrenLLMClient._has_tool_arg_repetition(newline_flood_arg))

        # 3. Normal girintili temiz kodda sahte pozitif vermemeli
        normal_code = '{"content": "function process() {\\n    const a = 1;\\n    const b = 2;\\n    return a + b;\\n}"}'
        self.assertFalse(EvrenLLMClient._has_tool_arg_repetition(normal_code))

    def test_smart_bugfix_filter_prevents_unnecessary_multi_agent_decomposition(self) -> None:
        """Verify smart bugfix filter prevents unnecessary multi agent decomposition."""
        orch = AgentOrchestrator(config=self.config)

        # Simulated AxiosError log from real-world frontend client
        error_prompt = (
            "[browser] Login error: AxiosError: Network Error\n"
            "    at async login (src/features/auth/service.ts:59:22)\n"
            "bu hatayı çöz backend 8080 de frontend 5174 de çalışıyor bunu çöz"
        )
        should_decomp, reason = orch.should_autonomously_decompose(error_prompt)
        self.assertFalse(should_decomp)
        self.assertIn("tekil hata teşhisi", reason)

        fullstack_prompt = "sıfırdan fullstack bir sınav takip sistemi kur ve hem frontend hem backend modüllerini geliştir"
        should_decomp_fs, _ = orch.should_autonomously_decompose(fullstack_prompt)
        self.assertTrue(should_decomp_fs)

    def test_planner_agent_tools_and_budget_are_streamlined(self) -> None:
        """Verify planner agent tools and budget are streamlined."""
        planner_tools = ROLE_ALLOWED_TOOLS[AgentRole.PLANNER]
        self.assertIn("get_project_tree", planner_tools)
        self.assertIn("get_project_modules", planner_tools)
        self.assertNotIn("read_file", planner_tools)
        self.assertNotIn("search_code", planner_tools)

        llm = EvrenLLMClient(self.config)
        reg = ToolRegistry()
        pm = PermissionManager(self.workspace)
        te = ToolExecutor(reg, pm, MagicMock())
        from evren_code.core.multi_agent.file_ownership import FileOwnershipManager
        fo = FileOwnershipManager(self.workspace)
        agent = SpecialistAgent(
            agent_id="test_planner",
            role=AgentRole.PLANNER,
            config=self.config,
            llm_client=llm,
            tool_executor=te,
            registry=reg,
            file_ownership=fo,
        )
        self.assertEqual(agent.role, AgentRole.PLANNER)

    def test_coordinator_injects_project_structure_to_planner(self) -> None:
        """Verify coordinator injects project structure to planner."""
        llm = EvrenLLMClient(self.config)
        reg = ToolRegistry()
        pm = PermissionManager(self.workspace)
        te = ToolExecutor(reg, pm, MagicMock())
        coord = MultiAgentCoordinator(
            config=self.config,
            llm_client=llm,
            tool_executor=te,
            registry=reg,
        )
        # Mock planner
        planner = coord.agent_registry.get_or_create_agent(AgentRole.PLANNER)
        captured_task = []

        def _mock_exec(task, context_data=None):
            captured_task.append(task)
            from evren_code.core.multi_agent.models import AgentTaskResult
            return AgentTaskResult(
                task_id=task.task_id,
                agent_id="mock",
                role=AgentRole.PLANNER,
                success=True,
                output='[{"task_id": "task_1", "title": "Test", "role": "developer"}]',
            )

        planner.execute_task = _mock_exec  # type: ignore
        graph = coord.plan_dag_from_goal("Frontend ve backend sayfalarını düzenle")
        self.assertEqual(len(graph.tasks), 1)
        self.assertTrue(len(captured_task) > 0)
        self.assertIn("MEVCUT PROJE YAPISI", captured_task[0].description)

    def test_overthinking_detector_catches_code_blocks_and_rumination(self) -> None:
        """Verify overthinking detector catches code blocks and rumination."""
        step_13_thinking = (
            "I need to stop searching and act. Plan: add a runtime switch in Dragon.cs.\n"
            "Hmm. Actually there's another approach: Resources.Load won't work.\n"
            "Alternative approach: keep FBX in place.\n"
            "Wait — big issue: Dragon.ApplyAnimation drives procedural wing meshes.\n"
            "Let me reconsider whether the model origin is at ground level.\n"
            "```csharp\n"
            "var prefab = Resources.Load<GameObject>(\"DragonModel\");\n"
            "if (prefab == null) return;\n"
            "var go = Instantiate(prefab, _root);\n"
            "```\n"
            "I'll grep for pelvis in the py file."
        )

        # 1. Overthinking & Rumination tespiti
        is_over, reason = analyze_reasoning_overthinking(step_13_thinking)
        self.assertTrue(is_over)
        self.assertIn("code_block_in_thinking", reason)
        self.assertIn("rumination_loop", reason)

        sanitized = sanitize_thinking_for_context(step_13_thinking, max_chars=200)
        self.assertNotIn("```", sanitized)
        self.assertNotIn("Resources.Load<GameObject>", sanitized)
        self.assertLessEqual(len(sanitized), 205)
        self.assertIn("I need to stop searching and act", sanitized)

        # 3. parse_assistant_message entegrasyonu
        parsed = parse_assistant_message({
            "role": "assistant",
            "content": "",
            "reasoning_content": step_13_thinking,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search_code", "arguments": '{"pattern": "pelvis"}'},
                }
            ],
        })
        self.assertTrue(parsed.overthinking_detected)
        self.assertIn("code_block_in_thinking", parsed.overthinking_reason)

    def test_stream_tag_filter_catches_reasoning_repetition(self) -> None:
        """Verify stream tag filter catches reasoning repetition."""
        emitted_thk = []
        flt = StreamTagFilter(
            on_text=lambda t: None,
            on_thinking=lambda thk: emitted_thk.append(thk),
            repetition_guard=True,
            repetition_threshold=3,
        )
        repeated_unit = "Wait, let me reconsider the prefab path in Resources folder. "
        for _ in range(4):
            flt.feed_thinking(repeated_unit)
        self.assertTrue(flt.repetition_detected)

    def test_turn_pipeline_injects_anti_overthinking_directive_after_rumination_step(self) -> None:
        """Verify turn pipeline injects anti overthinking directive after rumination step."""
        orch = AgentOrchestrator(config=self.config)
        captured_messages_per_step = []

        step_count = {"n": 0}

        def _mock_chat(messages, **kwargs):
            step_count["n"] += 1
            captured_messages_per_step.append(messages)
            if step_count["n"] == 1:
                return ParsedAssistantResponse(
                    content="",
                    thinking="Plan:\n```csharp\npublic class Dragon {}\n```\nHmm, actually wait — let me reconsider.",
                    overthinking_detected=True,
                    overthinking_reason="code_block_in_thinking, rumination_loop(3)",
                    tool_calls=[
                        ToolCallItem(
                            id="call_tree",
                            name="get_project_tree",
                            arguments={"max_depth": 1},
                        )
                    ],
                )
            return ParsedAssistantResponse(
                content="İşlem hızlıca tamamlandı.",
                thinking="Doğrudan yanıt veriyorum.",
                tool_calls=[],
            )

        orch.llm_client.chat = _mock_chat  # type: ignore
        orch.run_turn("Proje yapısını kontrol et")

        self.assertEqual(step_count["n"], 2)
        second_step_msgs = captured_messages_per_step[1]
        last_msg_content = str(second_step_msgs[-1].get("content") or "")
        self.assertIn("Anti-Overthinking", last_msg_content)
        self.assertIn("code_block_in_thinking", last_msg_content)

        asst_msgs = [m for m in second_step_msgs if m.get("role") == "assistant"]
        self.assertTrue(len(asst_msgs) >= 1)
        self.assertNotIn("```csharp", str(asst_msgs[0].get("content") or ""))
        orch.close()


if __name__ == "__main__":
    unittest.main()
