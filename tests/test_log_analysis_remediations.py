"""Test log analysis remediations module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import httpx

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.context import ContextManager
from evren_code.core.llm_client import EvrenAPIError, EvrenLLMClient
from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem


class TestLogAnalysisRemediations(unittest.TestCase):

    def test_compact_context_always_guarantees_user_query(self) -> None:
        """Verify compact context always guarantees user query."""
        cfg = load_config()
        ctx = ContextManager(cfg, "Sen bir frontend uzmanısın.")
        ctx.start_new_task("frontend ekranlarını daha temiz hale ve kullanışlı ve modern hale getir")

        ctx.add_assistant_message(
            content=None,
            tool_calls=[ToolCallItem(id="call_read_1", name="read_multiple_files", arguments={"files": ["a.tsx"]})],
        )
        ctx.add_tool_result("call_read_1", "read_multiple_files", "A" * 5000)

        ctx.add_assistant_message(
            content=None,
            tool_calls=[ToolCallItem(id="call_read_2", name="read_multiple_files", arguments={"files": ["b.tsx"]})],
        )
        ctx.add_tool_result("call_read_2", "read_multiple_files", "B" * 5000)

        ctx.compact(aggressive=True)

        messages = ctx.get_messages()
        self.assertEqual(messages[0]["role"], "system")
        user_msgs = [m for m in messages if m.get("role") == "user"]
        self.assertGreater(len(user_msgs), 0, "HATA: Bağlam sıkıştırması sonrası hiç 'user' mesajı kalmadı!")
        self.assertIn("frontend ekranlarını", user_msgs[0]["content"])

    def test_llm_client_auto_recovers_on_no_user_query_error(self) -> None:
        """Verify llm client auto recovers on no user query error."""
        cfg = load_config()
        cfg.api.api_key = "evren_llm_test_key_12345"
        mock_http = MagicMock()

        resp_400 = MagicMock()
        resp_400.status_code = 400
        resp_400.text = '{"error":{"message":"No user query found in messages.","type":"invalid_request_error"}}'

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Kullanıcı hedefine devam ediyorum."}}],
            "usage": {"total_tokens": 100},
        }

        mock_http.post.side_effect = [resp_400, resp_200]

        client = EvrenLLMClient(cfg, http_client=mock_http)
        client._api_key = "evren_llm_test_key_12345"

        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Previous assistant"},
        ]

        res = client.chat(messages=msgs, stream=False)
        self.assertIn("Kullanıcı hedefine devam ediyorum.", res.content)
        self.assertEqual(mock_http.post.call_count, 2)

    def test_agent_orchestrator_error_awareness_prevents_hallucination(self) -> None:
        """Verify agent orchestrator error awareness prevents hallucination."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = load_config()
            cfg.workspace_root = Path(tmp_dir)
            mock_llm = MagicMock()
            coder_model = cfg.models.get_model("coder")
            mock_llm.active_model = coder_model

            orch = AgentOrchestrator(cfg, llm_client=mock_llm)

            orch.last_turn_error = "EVREN API Akış Hatası (HTTP 400): No user query found in messages."

            mock_llm.chat.return_value = ParsedAssistantResponse(
                content="Önceki adımda bağlam sıkıştırması nedeniyle kullanıcı sorgusu eksik kalmıştı, bu sebeple HTTP 400 hatası alındı."
            )

            orch.run_turn("neden hata aldın")

            all_user_contents = [
                str(m.get("content"))
                for m in orch.context._messages
                if m.get("role") == "user"
            ]
            has_error_notice = any("SİSTEM HATASI BİLDİRİMİ" in c for c in all_user_contents)
            self.assertTrue(
                has_error_notice,
                "HATA: last_turn_error bir sonraki turun kullanıcı bağlamına enjekte edilmedi!",
            )
            self.assertIsNone(orch.last_turn_error, "HATA: last_turn_error temizlenmedi!")
            orch.close()

    def test_consecutive_tasks_preserve_full_analysis_and_discovery(self) -> None:
        """Verify consecutive tasks preserve full analysis and discovery."""
        cfg = load_config()
        ctx = ContextManager(cfg, "Sen bir yazılım mimarısın.")

        ctx.start_new_task("bana bu projeyi analiz et")
        ctx.add_assistant_message(
            content=None,
            tool_calls=[
                ToolCallItem(id="c1", name="get_project_tree", arguments={"path": "."}),
                ToolCallItem(id="c2", name="skeletonize_code", arguments={"path": "lib/ui/features/gameplay/view_models/gameplay_view_model.dart"}),
            ],
        )
        ctx.add_tool_result("c1", "get_project_tree", "lib/\n  domain/\n  data/\n  ui/")
        ctx.add_tool_result("c2", "skeletonize_code", "class GameplayViewModel extends ChangeNotifier {}")

        verification_header = "### Doğrulama Özeti Raporu (Verification Summary Report)\n| Metrik | Değer |\n|---|---|\n| Çalıştırılan | 20 |"
        detailed_analysis = (
            "# Demo Puzzle App — Project Analysis Report\n"
            + ("This project is a modular Flutter puzzle application. " * 10)
            + "\nArchitecture: Clean Architecture (domain, data, ui). Uses SystemSound.click."
            + "\n[Düşünce: Gizli iç düşünce sızmamalı]"
        )
        ctx.add_assistant_message(content=f"{verification_header}\n\n---\n\n{detailed_analysis}")

        ctx.start_new_task("peki sen bir girişimci yazılımcı olsan bu uygulamayı daha oynanabilir hale getirmek için neler yapardın")

        self.assertIn("Demo Puzzle App — Project Analysis Report", ctx.summary_memory)
        self.assertIn("SystemSound.click", ctx.summary_memory)
        self.assertIn("gameplay_view_model.dart", ctx.summary_memory)
        self.assertIn("tree(.)", ctx.summary_memory)
        self.assertNotIn("Gizli iç düşünce sızmamalı", ctx.summary_memory)
        sys_msg = ctx.build_system_message()["content"]
        self.assertIn("ÖNEMLİ KURAL", sys_msg)
        self.assertIn("Demo Puzzle App", sys_msg)

    def test_today_log_derived_9_bugfixes_2026_11_17(self) -> None:
        """Verifies 9 telemetry-derived bugfixes across editing, planning, and JSON repair."""
        from evren_code.core.llm_client import EvrenLLMClient
        from evren_code.core.parser import StatefulJsonRepair, repair_and_parse_json

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = load_config()
            cfg.workspace_root = Path(tmp_dir)
            orch = AgentOrchestrator(cfg)
            orch.trust_mode = True

            mesh_py = Path(tmp_dir) / "mesh_build.py"
            mesh_py.write_text("import bpy\nfrom mathutils import Vector as V\n\nx = 1\n", encoding="utf-8")
            edit_res = orch.registry.execute(
                "edit_file",
                {
                    "path": "mesh_build.py",
                    "target": "from mathutils import V",
                    "replacement": "# Blender 5.x compatibility\nfrom mathutils import Vector as V",
                },
            )
            self.assertIn("Success", edit_res)
            updated_py = mesh_py.read_text(encoding="utf-8")
            self.assertNotIn("Vector as Vector as V", updated_py)
            self.assertIn("# Blender 5.x compatibility\nfrom mathutils import Vector as V", updated_py)

            w_res = orch.registry.execute(
                "write_file",
                {
                    "path": "src/app/globals.css\n",
                    "content": "body { color: white; }\n",
                },
            )
            self.assertIn("Success", w_res)
            self.assertTrue((Path(tmp_dir) / "src/app/globals.css").is_file())

            r1 = orch.registry.execute("read_file", {"path": "src/app/globals.css"})
            self.assertIn("Dosya Değişmedi", r1)
            r2 = orch.registry.execute("read_file", {"path": "src/app/globals.css"})
            self.assertIn("body { color: white; }", r2)

            plan1 = orch.plan_manager.create_plan(
                "3D Mesh Refactor",
                '[1. Compatibility update", "2. Mesh restructuring", "3. Surface detailing]',
            )
            self.assertEqual(plan1.total_steps, 3)

            plan2 = orch.plan_manager.create_plan(
                "Web Portal Plan",
                [
                    {"step": "Scaffold Next.js 14", "3": "Build EN/TR i18n layer", "4": "Add SEO metadata"},
                    {"2": "Implement design system"},
                ],
            )
            self.assertGreaterEqual(plan2.total_steps, 4)
            self.assertFalse(any("{" in s.description for s in plan2.steps))

            hdr_file = Path(tmp_dir) / "Header.tsx"
            hdr_file.write_text("export function Header() {\n  const a = 1;\n  const b = 2;\n  return <div />;\n}\n", encoding="utf-8")

            step_calls = [0]
            def _mock_seq_edits(**kwargs):
                step_calls[0] += 1
                if step_calls[0] == 1:
                    return ParsedAssistantResponse(
                        content="",
                        tool_calls=[
                            ToolCallItem(id="e1", name="edit_file", arguments={"path": "Header.tsx", "target": "const a = 1;", "replacement": "const a = 10;"}),
                            ToolCallItem(id="e2", name="edit_file", arguments={"path": "Header.tsx", "target": "const b = 2;", "replacement": "const b = 20;"}),
                        ],
                        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    )
                return ParsedAssistantResponse(
                    content="İki düzenleme de uygulandı.",
                    tool_calls=[],
                    usage={"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140},
                )

            orch.llm_client.chat = _mock_seq_edits  # type: ignore
            orch.run_turn("Header.tsx içindeki a ve b değişkenlerini güncelle", stream=False)
            hdr_text = hdr_file.read_text(encoding="utf-8")
            self.assertIn("const a = 10;", hdr_text)
            self.assertIn("const b = 20;", hdr_text)

            broken_files_str = '[{"content": "export const Icon = () => <path d="M4.5 3.2"/>;", "path": "icon.tsx"}]'
            parsed_batch = repair_and_parse_json(broken_files_str)
            self.assertIsInstance(parsed_batch.get("value"), list)
            self.assertEqual(parsed_batch["value"][0]["path"], "icon.tsx")

            valid_css_arg = (
                "  .logo-tile {\n"
                "    box-shadow:\n"
                "      inset 0 0 0 1px rgba(8, 11, 18, 0.06),\n"
                "      inset -1px -1px -6px rgba(8, 11, 18, 0.1),\n"
                "      inset -2px -2px 6px rgba(8, 11, 18, 0.1);\n"
                "  }\n"
            )
            self.assertFalse(EvrenLLMClient._has_tool_arg_repetition(valid_css_arg))

            telem_dict = orch.logger.tracker.to_dict()
            self.assertGreaterEqual(telem_dict["operations"]["model_call_count"], 2)
            self.assertGreater(telem_dict["tokens"]["actual"]["total_tokens"], 0)
            orch.close()


if __name__ == "__main__":
    unittest.main()
