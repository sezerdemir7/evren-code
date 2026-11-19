"""v0.2.0 module."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List

import httpx

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.events import ApprovalRequest, EventBus, EventType
from evren_code.core.evren_auth import EvrenAuthError, EvrenAuthManager
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.core.parser import parse_assistant_message, repair_and_parse_json
from evren_code.core.path_setup import get_scripts_directory, is_dir_in_path
from evren_code.core.session import SessionManager
from evren_code.core.stream_filter import StreamTagFilter


class RejectAllApprovalDelegate:
    """Reject all approval delegate implementation."""

    def request_approval(self, request: ApprovalRequest) -> bool:
        return False


class EvrenCodeTestSuite(unittest.TestCase):
    """EvrenCodeTestSuite: Config, Auth, Parser, StreamFilter, Tools, Undo, Session, Orchestrator."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.config = load_config(workspace_root=self.workspace)
        self.config.api.storage_dir = self.workspace / ".evren-storage"
        self.config.api.api_key = "evren_llm_test_key_12345"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_config_and_model_roles(self) -> None:
        """Verify config and model roles."""
        self.assertEqual(self.config.api.base_url, "https://evren-llmapi.ssyz.org.tr/v1")
        self.assertEqual(self.config.models.default, "deepseek-v4.1-flash")

        arch = self.config.models.get_model("architect")
        self.assertEqual(arch.id, "deepseek-v4.1-flash")
        self.assertTrue(arch.thinking)

        coder = self.config.models.get_model("coder")
        self.assertEqual(coder.id, "qwen3.8-flash-next")
        self.assertTrue(coder.tools)

        vision = self.config.models.get_model("vision")
        self.assertEqual(vision.id, "qwen3-vl-30b")
        self.assertTrue(vision.supports_vision)

    def test_auth_key_validation_and_terms_flow(self) -> None:
        """Verify auth key validation and terms flow."""
        terms_accepted_state = {"accepted": False}

        def mock_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/models"):
                if not terms_accepted_state["accepted"]:
                    return httpx.Response(
                        403,
                        json={"error": {"code": "terms_not_accepted"}},
                    )
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "glm-5.3",
                                "task": "chat",
                                "context_length": 128000,
                                "default_max_output_tokens": 8192,
                                "capabilities": {"tools": True, "thinking": True},
                                "modalities": ["text"],
                            },
                            {
                                "id": "qwen3-vl-30b",
                                "task": "vision_chat",
                                "context_length": 64000,
                                "default_max_output_tokens": 4096,
                                "capabilities": {"tools": False, "thinking": False},
                                "modalities": ["text", "image"],
                            },
                        ]
                    },
                )
            if path.endswith("/terms/text"):
                return httpx.Response(200, json={"content": "EVREN Kullanım Şartları Metni v1"})
            if path.endswith("/terms/accept"):
                body = json.loads(request.content.decode("utf-8"))
                self.assertEqual(body.get("version"), 1)
                terms_accepted_state["accepted"] = True
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(404)

        mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        auth_mgr = EvrenAuthManager(self.config, http_client=mock_client)

        with self.assertRaises(EvrenAuthError):
            auth_mgr.validate_key_format("yanlis_prefix_123")

        callback_called = {"count": 0}

        def accept_cb(text: str, path: Path) -> bool:
            callback_called["count"] += 1
            self.assertIn("EVREN Kullanım Şartları", text)
            self.assertTrue(path.is_file())
            return True

        discovered = auth_mgr.verify_and_discover_models(
            "  'evren_llm_valid_999' \n",
            terms_acceptance_callback=accept_cb,
        )
        self.assertEqual(callback_called["count"], 1)
        self.assertIn("glm-5.3", discovered)
        self.assertIn("qwen3-vl-30b", discovered)
        self.assertEqual(discovered["qwen3-vl-30b"].context_length, 64000)

    def test_hybrid_parser_and_json_repair(self) -> None:
        """Verify hybrid parser and json repair."""
        repaired = repair_and_parse_json('{"path": "src/app.py", "start_line": 10, }')
        self.assertEqual(repaired["path"], "src/app.py")
        self.assertEqual(repaired["start_line"], 10)

        raw_msg = {
            "content": (
                "<think>Önce dosyayı okumalıyım.</think>\n"
                "Dosyayı kontrol ediyorum.\n"
                '<tool_call>{"name": "read_file", "arguments": {"path": "README.md",}}</tool_call>'
            )
        }
        parsed = parse_assistant_message(raw_msg)
        self.assertEqual(parsed.thinking, "Önce dosyayı okumalıyım.")
        self.assertEqual(parsed.content, "Dosyayı kontrol ediyorum.")
        self.assertEqual(len(parsed.tool_calls), 1)
        self.assertEqual(parsed.tool_calls[0].name, "read_file")
        self.assertTrue(parsed.tool_calls[0].from_xml)

    def test_stream_tag_filter_hides_tool_calls_and_routes_thinking(self) -> None:
        """Verify stream tag filter hides tool calls and routes thinking."""
        text_out: List[str] = []
        think_out: List[str] = []

        flt = StreamTagFilter(
            on_text=text_out.append,
            on_thinking=think_out.append,
        )
        chunks = [
            "<thi",
            "nk>Plan yapıyorum</think>Merhaba, inceliyorum:\n<tool_",
            'call>{"name": "get_project_tree", "arguments": {}}</tool_',
            "call>Bitti.",
        ]
        for c in chunks:
            flt.feed(c)
        flt.flush()

        self.assertEqual("".join(think_out), "Plan yapıyorum")
        self.assertEqual("".join(text_out), "Merhaba, inceliyorum:\nBitti.")

    def test_auto_continue_guard_on_thinking_only_turn(self) -> None:
        """Verify auto continue guard on thinking only turn."""
        step = {"n": 0}

        def mock_llm_handler(request: httpx.Request) -> httpx.Response:
            step["n"] += 1
            if step["n"] == 1:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "reasoning_content": "Şimdi frontend dizinini incelemeliyim.",
                                }
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Frontend analizi tamamlandı: Next.js kullanılıyor.",
                            }
                        }
                    ]
                },
            )

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_llm_handler))
        llm_client = EvrenLLMClient(self.config, http_client=mock_http)
        orchestrator = AgentOrchestrator(config=self.config, llm_client=llm_client)

        final_answer = orchestrator.run_turn("frontend'i analiz et", stream=False)
        self.assertEqual(step["n"], 2)
        self.assertIn("Next.js", final_answer)

    def test_project_tree_batch_read_multi_edit_and_undo(self) -> None:
        """Verify project tree batch read multi edit and undo."""
        sub = self.workspace / "frontend" / "app"
        sub.mkdir(parents=True)
        f1 = sub / "page.tsx"
        f2 = self.workspace / "frontend" / "package.json"
        f1.write_text("const a = 1;\nconst b = 2;\n", encoding="utf-8")
        f2.write_text('{"name": "my-app"}', encoding="utf-8")

        orchestrator = AgentOrchestrator(config=self.config)

        # 1. get_project_tree
        tree_res = orchestrator.registry.execute(
            "get_project_tree", {"path": "frontend", "max_depth": 3}
        )
        self.assertIn("page.tsx", tree_res)
        self.assertIn("package.json", tree_res)

        # 2. read_multiple_files
        batch_res = orchestrator.registry.execute(
            "read_multiple_files",
            {"paths": ["frontend/app/page.tsx", "frontend/package.json"]},
        )
        self.assertIn("my-app", batch_res)
        self.assertIn("const a = 1;", batch_res)

        f1.write_text("const a = 1;   \nconst b = 2;\n", encoding="utf-8")
        edit_res = orchestrator.registry.execute(
            "multi_edit_file",
            {
                "path": "frontend/app/page.tsx",
                "edits": [
                    {"target": "const a = 1;", "replacement": "const a = 1;"},  # No-op blok
                    {"target": "const a = 1;\nconst b = 2;", "replacement": "const a = 10;\nconst b = 20;"},
                ],
            },
        )
        self.assertIn("Başarılı", edit_res)
        self.assertEqual(f1.read_text(encoding="utf-8"), "const a = 10;\nconst b = 20;\n")

        # 4. write_multiple_files
        multi_write_res = orchestrator.registry.execute(
            "write_multiple_files",
            {
                "files": [
                    {"path": "frontend/types.ts", "content": "export type X = string;"},
                    {"path": "frontend/schema.ts", "content": "export const y = 1;"},
                ]
            },
        )
        self.assertIn("2 dosya", multi_write_res)
        self.assertTrue((self.workspace / "frontend" / "types.ts").is_file())

        # 5. Undo (/undo)
        undo_msg = orchestrator.undo_manager.undo_last()
        self.assertIn("Geri alındı", undo_msg)

    def test_session_save_and_load(self) -> None:
        """Verify session save and load."""
        orchestrator = AgentOrchestrator(config=self.config)
        orchestrator.context.add_user_message("Merhaba")
        orchestrator.context.add_assistant_message("Selam, nasıl yardımcı olabilirim?")

        session_mgr = SessionManager(self.config)
        session_mgr.save_session("test_oturum", orchestrator.context, "glm-5.3")

        orchestrator.context.clear()
        self.assertEqual(orchestrator.context.history_length, 0)

        loaded = session_mgr.load_session("test_oturum", orchestrator.context)
        self.assertEqual(loaded["model"], "glm-5.3")
        self.assertEqual(orchestrator.context.history_length, 2)

    def test_auto_save_and_workspace_resume_with_plan(self) -> None:
        """Verify auto save and workspace resume with plan."""
        orchestrator = AgentOrchestrator(config=self.config)
        orchestrator.context.add_user_message("Frontend refactoring yap")
        orchestrator.plan_manager.create_plan(
            "Refactoring", ["Tasarım sistemi", "Bileşenler", "Test"]
        )
        orchestrator.plan_manager.update_step(1, "completed", "Tamamlandı")
        orchestrator.plan_manager.update_step(2, "in_progress", "İşleniyor")

        session_mgr = SessionManager(self.config)
        saved_path = session_mgr.auto_save_workspace_session(
            context=orchestrator.context,
            active_model_id="deepseek-v4.1-flash",
            plan_manager=orchestrator.plan_manager,
        )
        self.assertIsNotNone(saved_path)
        self.assertTrue(saved_path.is_file())

        new_orchestrator = AgentOrchestrator(config=self.config)
        self.assertEqual(new_orchestrator.context.history_length, 0)
        self.assertIsNone(new_orchestrator.plan_manager.current_plan)

        # Son oturumu tespit et
        last_session = session_mgr.get_last_workspace_session()
        self.assertIsNotNone(last_session)
        self.assertEqual(len(last_session["messages"]), 1)
        self.assertIn("Refactoring", last_session["plan"]["title"])
        self.assertTrue(last_session["relative_time"] in ("az önce", "just now"))

        session_mgr.restore_session_to_orchestrator(new_orchestrator, last_session)
        self.assertEqual(new_orchestrator.context.history_length, 1)
        self.assertIsNotNone(new_orchestrator.plan_manager.current_plan)
        self.assertEqual(new_orchestrator.plan_manager.current_plan.completed_steps, 1)
        self.assertEqual(
            new_orchestrator.plan_manager.current_plan.steps[1].status.value, "in_progress"
        )

        cleared = session_mgr.clear_workspace_session()
        self.assertTrue(cleared)
        self.assertIsNone(session_mgr.get_last_workspace_session())

    def test_completion_gatekeeper_blocks_premature_exit(self) -> None:
        """Verify completion gatekeeper blocks premature exit."""
        step = {"n": 0}

        def mock_handler(request: httpx.Request) -> httpx.Response:
            step["n"] += 1
            if step["n"] == 1:
                return httpx.Response(
                    200,
                    json={
                        "id": "resp-1",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Görevi erkenden bitirdim sanıyorum.",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            else:
                return httpx.Response(
                    200,
                    json={
                        "id": "resp-2",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Kalan adımları uyguladım ve bitirdim.",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        llm = EvrenLLMClient(config=self.config, http_client=client)
        llm.set_model("deepseek-v4.1-flash")
        orchestrator = AgentOrchestrator(config=self.config, llm_client=llm)

        orchestrator.plan_manager.create_plan(
            "Eksik Görev", ["Veritabanı", "Servis Katmanı", "Arayüz"]
        )
        orchestrator.plan_manager.update_step(1, "completed")
        self.assertFalse(orchestrator.plan_manager.is_plan_finished())

        out = orchestrator.run_turn("Eksikleri gider")
        self.assertGreaterEqual(step["n"], 2)
        msgs_text = " ".join(str(m.get("content")) for m in orchestrator.context.raw_messages)
        self.assertIn("Erken Bitiş Engellendi", msgs_text)

    def test_shell_command_duplicate_cd_removal(self) -> None:
        """Verify shell command duplicate cd removal."""
        sub_dir = self.workspace / "backend"
        sub_dir.mkdir(parents=True, exist_ok=True)

        orchestrator = AgentOrchestrator(config=self.config)
        res = orchestrator.registry.execute(
            "run_command",
            {"command": "cd backend && echo test_ok", "cwd": "backend"},
        )
        self.assertIn("test_ok", res)
        self.assertNotIn("Sistem belirtilen yolu bulamıyor", res)

    def test_consecutive_tool_call_loop_breaker(self) -> None:
        """Verify consecutive tool call loop breaker."""
        from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem

        test_file = self.workspace / "dongu_test.txt"
        test_file.write_text("Döngü testi içeriği", encoding="utf-8")

        call_idx = {"n": 0}

        def _fake_chat(**kwargs) -> ParsedAssistantResponse:
            call_idx["n"] += 1
            if call_idx["n"] <= 3:
                return ParsedAssistantResponse(
                    content="",
                    tool_calls=[
                        ToolCallItem(
                            id=f"c_{call_idx['n']}",
                            name="read_file",
                            arguments={"path": str(test_file)},
                        )
                    ],
                    thinking="",
                )
            return ParsedAssistantResponse(
                content="Döngüden çıktım, tamamlandı.",
                tool_calls=[],
                thinking="",
            )

        events_caught: List[str] = []
        bus = EventBus()
        bus.subscribe(lambda ev: events_caught.append(ev.data.get("message", "")), EventType.WARNING)

        orchestrator = AgentOrchestrator(config=self.config, event_bus=bus)
        orchestrator.llm_client.chat = _fake_chat  # type: ignore

        orchestrator.run_turn("Dosyayı oku")

        tool_results_texts = [
            str(m.get("content"))
            for m in orchestrator.context.get_messages()
            if m.get("role") == "tool"
        ]
        has_loop_block = any("DÖNGÜ KORUMASI DEVREDE" in txt for txt in tool_results_texts)
        self.assertTrue(has_loop_block, f"Döngü kırıcı mesajı bulunamadı: {tool_results_texts}")
        has_warning = any("mükerrer çağrı engellendi" in w for w in events_caught)
        self.assertTrue(has_warning)

    def test_consecutive_tool_call_hard_circuit_breaker(self) -> None:
        """Verify consecutive tool call hard circuit breaker."""
        from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem

        test_file = self.workspace / "zombie_test.txt"
        test_file.write_text("Zombie loop test", encoding="utf-8")

        call_idx = {"n": 0}

        def _zombie_chat(**kwargs) -> ParsedAssistantResponse:
            call_idx["n"] += 1
            return ParsedAssistantResponse(
                content="",
                tool_calls=[
                    ToolCallItem(
                        id=f"c_{call_idx['n']}",
                        name="read_file",
                        arguments={"path": str(test_file)},
                    )
                ],
                thinking="",
            )

        events_caught: List[str] = []
        bus = EventBus()
        bus.subscribe(lambda ev: events_caught.append(ev.data.get("message", "")), EventType.WARNING)

        orchestrator = AgentOrchestrator(config=self.config, event_bus=bus)
        orchestrator.llm_client.chat = _zombie_chat  # type: ignore

        result = orchestrator.run_turn("Dosyayı oku")
        self.assertLessEqual(call_idx["n"], 6)
        self.assertIn("Sert Devre Kesici", result)
        has_cb_warning = any("Devre Kesici" in w for w in events_caught)
        self.assertTrue(has_cb_warning)

    def test_approval_gate_rejection(self) -> None:
        """Verify approval gate rejection."""
        target_file = self.workspace / "korunmus.txt"
        target_file.write_text("orijinal", encoding="utf-8")

        step = {"n": 0}

        def mock_handler(request: httpx.Request) -> httpx.Response:
            step["n"] += 1
            if step["n"] == 1:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call_write",
                                            "type": "function",
                                            "function": {
                                                "name": "write_file",
                                                "arguments": json.dumps(
                                                    {"path": "korunmus.txt", "content": "degistirildi"}
                                                ),
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Kullanıcı işlemi onaylamadığı için dosya değiştirilmedi.",
                            }
                        }
                    ]
                },
            )

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        llm_client = EvrenLLMClient(self.config, http_client=mock_http)
        orchestrator = AgentOrchestrator(
            config=self.config,
            llm_client=llm_client,
            approval_delegate=RejectAllApprovalDelegate(),
        )
        orchestrator.trust_mode = False

        orchestrator.run_turn("korunmus.txt dosyasını değiştir", stream=False)
        self.assertEqual(target_file.read_text(encoding="utf-8"), "orijinal")

    def test_path_setup_detection(self) -> None:
        """Verify path setup detection."""
        scripts_dir = get_scripts_directory()
        self.assertTrue(scripts_dir.exists())
        fake_path = f"C:\\SomeDir;{scripts_dir};C:\\OtherDir"
        self.assertTrue(is_dir_in_path(scripts_dir, fake_path))
        self.assertFalse(is_dir_in_path(Path("C:/NonExistentScriptsDir123"), fake_path))


    def test_slash_command_completer_only_on_slash(self) -> None:
        """Verify slash command completer only on slash."""
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document
        from evren_code.cli import SLASH_COMMANDS, SlashCommandCompleter

        completer = SlashCommandCompleter(
            commands=SLASH_COMMANDS,
            models=["glm-5.3", "deepseek-v4.1-flash"],
        )
        dummy_event = CompleteEvent()

        doc_normal = Document("bu projenin frontend dosyaları ")
        res_normal = list(completer.get_completions(doc_normal, dummy_event))
        self.assertEqual(len(res_normal), 0)

        doc_slash = Document("/m")
        res_slash = [c.text for c in completer.get_completions(doc_slash, dummy_event)]
        self.assertIn("/model", res_slash)
        self.assertNotIn("/undo", res_slash)

        doc_model = Document("/model deep")
        res_model = [c.text for c in completer.get_completions(doc_model, dummy_event)]
        self.assertEqual(res_model, ["deepseek-v4.1-flash"])

    def test_trajectory_logger_and_gitignore_protection(self) -> None:
        """Verify trajectory logger and gitignore protection."""
        from evren_code.core.telemetry import TrajectoryLogger

        gi_file = self.workspace / ".gitignore"
        gi_file.write_text("node_modules/\n", encoding="utf-8")

        self.config.telemetry.enabled = True
        self.config.telemetry.mode = "project"
        self.config.telemetry.auto_gitignore = True

        logger = TrajectoryLogger(self.config)
        self.assertTrue(logger.enabled)
        self.assertIn(".evren/", gi_file.read_text(encoding="utf-8"))

        model_spec = self.config.models.get_model("glm-5.3")
        logger.start_turn("frontend analizi yap", model_spec)
        logger.record_step(
            step=1,
            thinking="Dosyaları kontrol ediyorum",
            content="list_directory çağrıldı",
            tool_calls=[],
            tool_results=[{"name": "list_directory", "approved": True, "result": "ok"}],
        )
        logger.complete_turn("Analiz bitti.", estimated_tokens=500)

        self.assertTrue(logger.jsonl_path.is_file())
        self.assertTrue(logger.markdown_path.is_file())

        jsonl_content = logger.jsonl_path.read_text(encoding="utf-8")
        self.assertIn("frontend analizi yap", jsonl_content)
        self.assertIn("glm-5.3", jsonl_content)

        md_content = logger.markdown_path.read_text(encoding="utf-8")
        self.assertIn("Kullanıcı İstemi", md_content)
        self.assertIn("Analiz bitti", md_content)

    def test_fuzzy_file_similarity_suggestion(self) -> None:
        """Verify fuzzy file similarity suggestion."""
        from evren_code.tools.filesystem import register_filesystem_tools
        from evren_code.tools.registry import ToolRegistry

        skill_file = self.workspace / "nextjs-frontend-skill.md"
        skill_file.write_text("# Next.js Skill Guide\n", encoding="utf-8")

        reg = ToolRegistry()
        register_filesystem_tools(reg, self.config)

        result = reg.execute("read_file", {"path": "docs/nextjs-frontend-skill.md"})
        self.assertTrue("not found" in result.lower() or "bulunamadı" in result.lower())
        self.assertIn("İpucu: Benzer ada sahip mevcut dosyalar:", result)
        self.assertIn("nextjs-frontend-skill.md", result)

    def test_same_step_duplicate_edit_guard(self) -> None:
        """Verify same step duplicate edit guard."""
        from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem

        test_file = self.workspace / "client.ts"
        test_file.write_text("import axios from 'axios';\nconst api = axios.create();\n", encoding="utf-8")

        orch = AgentOrchestrator(config=self.config)

        mock_response = ParsedAssistantResponse(
            content="",
            tool_calls=[
                ToolCallItem(
                    id="call_1",
                    name="edit_file",
                    arguments={
                        "path": "client.ts",
                        "target": "const api = axios.create();",
                        "replacement": "const api = axios.create({ timeout: 5000 });",
                    },
                ),
                ToolCallItem(
                    id="call_2",
                    name="multi_edit_file",
                    arguments={
                        "path": "client.ts",
                        "edits": [
                            {"target": "import axios from 'axios';", "replacement": "import axios from 'axios'; // modified"}
                        ],
                    },
                ),
            ],
            thinking="",
        )

        def _fake_complete(*args, **kwargs):
            if not hasattr(_fake_complete, "called"):
                _fake_complete.called = True
                return mock_response
            return ParsedAssistantResponse(content="Tamamlandı.", tool_calls=[], thinking="")

        orch.llm_client.chat = _fake_complete  # type: ignore
        orch.run_turn("client.ts dosyasını düzenle")

        content = test_file.read_text(encoding="utf-8")
        self.assertIn("timeout: 5000", content)

        tool_results_text = " ".join([m.get("content", "") for m in orch.context.get_messages() if m.get("role") == "tool"])
        self.assertIn("Çakışma Önleme", tool_results_text)

    def test_windows_cmd_chain_command_execution(self) -> None:
        """Verify windows cmd chain command execution."""
        from evren_code.tools.registry import ToolRegistry
        from evren_code.tools.shell import register_shell_tools

        reg = ToolRegistry()
        register_shell_tools(reg, self.config)

        res = reg.execute("run_command", {"command": "echo step1 && echo step2"})
        self.assertIn("(Exit Code): 0", res)
        self.assertIn("step1", res)
        self.assertIn("step2", res)

    def test_subfolder_fuzzy_directory_hint(self) -> None:
        """Verify subfolder fuzzy directory hint."""
        from evren_code.tools.registry import ToolRegistry
        from evren_code.tools.search import register_search_tools

        frontend_src = self.workspace / "web-frontend" / "src"
        frontend_src.mkdir(parents=True, exist_ok=True)
        (frontend_src / "test.ts").write_text("// test", encoding="utf-8")

        reg = ToolRegistry()
        register_search_tools(reg, self.config)

        res = reg.execute("list_directory", {"path": "src"})
        self.assertTrue("Error: Directory 'src' not found" in res or "Hata: 'src' dizini bulunamadı." in res)
        self.assertIn("İpucu: Alt dizinlerde şu mevcut klasörler bulundu:", res)
        self.assertIn("web-frontend/src", res)

    def test_repo_map_ast_extraction(self) -> None:
        """Verify repo map ast extraction."""
        from evren_code.tools.registry import ToolRegistry
        from evren_code.tools.repo_map import register_repo_map_tools

        py_file = self.workspace / "service.py"
        py_file.write_text(
            "class UserService:\n"
            "    def authenticate(self, user: str) -> bool:\n"
            "        return True\n",
            encoding="utf-8",
        )

        ts_file = self.workspace / "types.ts"
        ts_file.write_text(
            "export interface UserProfile {\n"
            "  id: string;\n"
            "  email: string;\n"
            "}\n"
            "export type Role = 'ADMIN' | 'USER';\n",
            encoding="utf-8",
        )

        reg = ToolRegistry()
        register_repo_map_tools(reg, self.config)

        map_output = reg.execute("get_repo_map", {"directory": "."})
        self.assertIn("service.py:", map_output)
        self.assertIn("class UserService", map_output)
        self.assertIn("def authenticate", map_output)
        self.assertIn("types.ts:", map_output)
        self.assertIn("interface UserProfile", map_output)
        self.assertIn("type Role", map_output)

    def test_self_healing_verification_detection_and_loop(self) -> None:
        """Verify self healing verification detection and loop."""
        from evren_code.core.self_healing import SelfHealingManager, VerificationTask

        pkg_file = self.workspace / "package.json"
        pkg_file.write_text(
            json.dumps({"name": "test-pkg", "scripts": {"test": "npm test"}}),
            encoding="utf-8",
        )

        sh_mgr = SelfHealingManager(workspace_root=self.workspace, max_attempts=2)
        task = sh_mgr.detect_verification_task()
        self.assertIsNotNone(task)
        self.assertIn("npm test", task.command)

        fake_shell = lambda cmd, cwd: "Exit Code: 1\n--- STDERR ---\nTypeError: undefined is not a function"
        ok, cmd_run, out = sh_mgr.execute_verification(fake_shell, task)
        self.assertFalse(ok)
        self.assertEqual(cmd_run, task.command)
        self.assertIn("TypeError", out)
        self.assertEqual(sh_mgr.current_attempt, 1)
        self.assertTrue(sh_mgr.has_attempts_left)

    def test_windows_command_normalization_and_dynamic_timeout(self) -> None:
        """Verify windows command normalization and dynamic timeout."""
        from evren_code.tools.shell import calculate_effective_timeout, normalize_windows_command

        ps_cmd = normalize_windows_command("rm -rf node_modules .next", use_cmd=False)
        self.assertIn("Remove-Item -Recurse -Force -ErrorAction SilentlyContinue", ps_cmd)
        self.assertIn('"node_modules"', ps_cmd)
        self.assertIn('".next"', ps_cmd)

        cmd_cmd = normalize_windows_command("rm -rf dist && npm install", use_cmd=True)
        self.assertIn("rd /s /q dist", cmd_cmd)
        self.assertIn("npm install", cmd_cmd)

        normal_to = calculate_effective_timeout("git status", self.config)
        self.assertEqual(normal_to, self.config.tools.command_timeout_seconds)

        build_to = calculate_effective_timeout("npm install", self.config)
        self.assertGreaterEqual(build_to, self.config.tools.build_command_timeout_seconds)

        build_to2 = calculate_effective_timeout("cargo build --release", self.config)
        self.assertGreaterEqual(build_to2, self.config.tools.build_command_timeout_seconds)

    def test_stream_filter_repetition_guard(self) -> None:
        """Verify stream filter repetition guard."""
        emitted_text: List[str] = []
        emitted_thinking: List[str] = []

        filt = StreamTagFilter(
            on_text=lambda t: emitted_text.append(t),
            on_thinking=lambda t: emitted_thinking.append(t),
            repetition_guard=True,
            repetition_threshold=3,
        )

        repeated_sentence = "Bu durumu düzeltmek için virgül eklenmeli. "
        for _ in range(5):
            filt.feed(repeated_sentence)
        filt.flush()

        self.assertTrue(filt.repetition_detected)
        self.assertIn("Bu durumu düzeltmek", filt.repeated_phrase)
        combined = "".join(emitted_text)
        self.assertLess(combined.count("Bu durumu düzeltmek"), 4)

    def test_agent_repetition_loop_breaker(self) -> None:
        """Verify agent repetition loop breaker."""
        from evren_code.core.parser import ParsedAssistantResponse

        class LoopingFakeLLM:
            def __init__(self, cfg):
                self.config = cfg
                self.active_model = cfg.models.get_model("glm-5.3")
                self.calls = 0

            def chat(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return ParsedAssistantResponse(
                        content="Bu durumu düzeltmek için virgül eklenmeli. " * 4,
                        repetition_detected=True,
                    )
                return ParsedAssistantResponse(
                    content="Hemen dosyayı inceliyorum.",
                    tool_calls=[],
                )

        orchestrator = AgentOrchestrator(
            config=self.config,
            llm_client=LoopingFakeLLM(self.config),
        )

        res = orchestrator.run_turn("frontend'i incele")
        messages = orchestrator.context.get_messages()
        has_warning = any(
            "Tekrar Döngüsü" in m.get("content", "")
            for m in messages
        )
        self.assertTrue(has_warning)

    def test_edit_file_diagnose_mismatch_hint(self) -> None:
        """Verify edit file diagnose mismatch hint."""
        from evren_code.tools.registry import ToolRegistry
        from evren_code.tools.filesystem import register_filesystem_tools

        test_file = self.workspace / "component.tsx"
        test_file.write_text(
            "const Comp = () => {\n"
            "  return (\n"
            "    <div>test</div>\n"
            "  );\n"
            "});\n",
            encoding="utf-8",
        )

        reg = ToolRegistry()
        register_filesystem_tools(reg, self.config)

        wrong_target = "  );\n};\n"
        res = reg.execute("edit_file", {"path": "component.tsx", "target": wrong_target, "replacement": "  );\n};\n// change"})
        self.assertTrue("Error:" in res or "Hata:" in res)
        self.assertIn("Satır 5'de uyumsuzluk tespit edildi", res)
        self.assertIn("Beklenen: `};`", res)
        self.assertIn("Dosyadaki: `});`", res)

    def test_read_file_uncreated_file_hint(self) -> None:
        """Verify read file uncreated file hint."""
        from evren_code.tools.registry import ToolRegistry
        from evren_code.tools.filesystem import register_filesystem_tools

        reg = ToolRegistry()
        register_filesystem_tools(reg, self.config)

        res = reg.execute("read_file", {"path": "completely_uncreated_feature/service.ts"})
        self.assertTrue("Error:" in res or "Hata:" in res)
        self.assertIn("Bu dosya henüz mevcut değil", res)
        self.assertIn("write_file", res)

    def test_auto_continue_seamless_batches(self) -> None:
        """Verify auto continue seamless batches."""
        from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem

        self.config.agent.max_iterations = 2
        self.config.agent.auto_continue = True
        self.config.agent.max_auto_continues = 1
        # Total allowed = 2 * (1 + 1) = 4

        call_count = 0

        class ActiveWorkingLLM:
            def __init__(self, cfg):
                self.config = cfg
                self.active_model = cfg.models.get_model("glm-5.3")

            def chat(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    return ParsedAssistantResponse(
                        content="",
                        tool_calls=[ToolCallItem(id=f"c_{call_count}", name="git_info", arguments={})],
                    )
                return ParsedAssistantResponse(
                    content="Tüm adımlar başarıyla tamamlandı.",
                    tool_calls=[],
                )

        events_caught = []
        bus = EventBus()
        bus.subscribe(lambda ev: events_caught.append(ev.data.get("message", "")), EventType.WARNING)

        orchestrator = AgentOrchestrator(
            config=self.config,
            llm_client=ActiveWorkingLLM(self.config),
            event_bus=bus,
        )
        if hasattr(self.config, "multi_agent"):
            self.config.multi_agent.auto_decompose = False

        res = orchestrator.run_turn("büyük projeyi refactor et")
        self.assertIn("başarıyla tamamlandı", res)
        self.assertEqual(call_count, 3)
        has_auto_cont = any("Otomatik Devam" in msg for msg in events_caught)
        self.assertTrue(has_auto_cont)

    def test_task_planner_lifecycle_and_markdown(self) -> None:
        """Verify task planner lifecycle and markdown."""
        from evren_code.core.planner import PlanManager, PlanStepStatus

        bus = EventBus()
        plan_events = []
        bus.subscribe(lambda ev: plan_events.append(ev.data), EventType.PLAN_UPDATED)

        pm = PlanManager(bus)
        plan = pm.create_plan(
            title="Frontend Yeniden Yapılandırma",
            steps=[
                "Tasarım tokenlarını incele",
                "Layout bileşenini güncelle",
                "Derlemeyi doğrula",
            ],
        )

        self.assertEqual(plan.total_steps, 3)
        self.assertEqual(plan.completed_steps, 0)
        self.assertEqual(len(plan_events), 1)

        ok1 = pm.update_step(1, "in_progress", notes="globals.css açıldı")
        self.assertTrue(ok1)
        self.assertEqual(plan.steps[0].status, PlanStepStatus.IN_PROGRESS)
        self.assertEqual(plan.steps[0].notes, "globals.css açıldı")

        pm.update_step(1, "completed")
        pm.update_step(2, "in_progress")
        self.assertEqual(plan.completed_steps, 1)

        md = plan.render_markdown()
        self.assertIn("### Aktif Görev Planı: Frontend Yeniden Yapılandırma (1/3 Tamamlandı)", md)
        self.assertIn("[x] 1. Tasarım tokenlarını incele *(Tamamlandı)*", md)
        self.assertIn("[/] 2. Layout bileşenini güncelle *(Şu Anda İşleniyor)*", md)
        self.assertIn("[ ] 3. Derlemeyi doğrula", md)

        table = pm.render_table()
        self.assertEqual(table.title, "📋 Görev Planı: Frontend Yeniden Yapılandırma")

    def test_planner_tools_in_orchestrator(self) -> None:
        """Verify planner tools in orchestrator."""
        orchestrator = AgentOrchestrator(config=self.config)

        res_create = orchestrator.registry.execute(
            "create_task_plan",
            {"title": "Auth Modülü", "steps": ["Şema yaz", "API bağla", "Form doğrula"]},
        )
        self.assertIn("Auth Modülü", res_create)
        self.assertIn("3 adımla oluşturuldu", res_create)

        res_update = orchestrator.registry.execute(
            "update_plan_step",
            {"step_id": 1, "status": "completed", "notes": "Zod şeması hazır"},
        )
        self.assertIn("durumu 'completed' olarak güncellendi", res_update)

        res_get = orchestrator.registry.execute("get_task_plan", {})
        self.assertIn("[x] 1. Şema yaz *(Tamamlandı)*", res_get)
        self.assertIn("Not: Zod şeması hazır", res_get)

    def test_subagent_manager_delegation(self) -> None:
        """Verify subagent manager delegation."""
        from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem
        from evren_code.core.subagents import SubagentManager

        test_doc = self.workspace / "API.md"
        test_doc.write_text("# API Belgeleri\nGET /users - Kullanıcı listesi", encoding="utf-8")

        class MockSubagentLLM:
            def __init__(self, cfg):
                self.config = cfg
                self.calls = 0

            def chat(self, messages, tools_schema=None, model_override=None, stream=False):
                self.calls += 1
                if self.calls == 1:
                    return ParsedAssistantResponse(
                        content="",
                        tool_calls=[
                            ToolCallItem(
                                id="call_res_1",
                                name="read_file",
                                arguments={"path": "API.md"},
                            )
                        ],
                    )
                return ParsedAssistantResponse(
                    content="API.md dosyasında GET /users uç noktası bulundu.",
                    tool_calls=[],
                )

        events_caught = []
        bus = EventBus()
        bus.subscribe(lambda ev: events_caught.append(ev.type))

        mock_llm = MockSubagentLLM(self.config)
        orchestrator = AgentOrchestrator(config=self.config, llm_client=mock_llm, event_bus=bus)

        sub_report = orchestrator.registry.execute(
            "delegate_subagent",
            {"role": "researcher", "prompt": "API.md içindeki uç noktaları bul."},
        )

        self.assertIn("[Araştırmacı Ajan Raporu]", sub_report)
        self.assertIn("GET /users uç noktası bulundu", sub_report)
        self.assertIn(EventType.SUBAGENT_STARTED, events_caught)
        self.assertIn(EventType.SUBAGENT_COMPLETED, events_caught)

    def test_semantic_search_indexing_and_query(self) -> None:
        """Verify semantic search indexing and query."""
        py_service = self.workspace / "auth_service.py"
        py_service.write_text(
            "class AuthenticationService:\n"
            "    def verify_password_hash(self, plain: str, hashed: str) -> bool:\n"
            "        '''Kullanıcı şifresini güvenli şekilde doğrular.'''\n"
            "        return True\n",
            encoding="utf-8",
        )

        ts_chart = self.workspace / "efficiency_chart.tsx"
        ts_chart.write_text(
            "export interface WeeklyEfficiencyProps {\n"
            "  studentId: string;\n"
            "}\n"
            "export function calculateWeeklyScore(hours: number): number {\n"
            "  return hours * 10;\n"
            "}\n",
            encoding="utf-8",
        )

        orchestrator = AgentOrchestrator(config=self.config)

        res1 = orchestrator.registry.execute("semantic_search_code", {"query": "şifre doğrulama password hash"})
        self.assertIn("auth_service.py", res1)
        self.assertIn("verify_password_hash", res1)

        res2 = orchestrator.registry.execute("semantic_search_code", {"query": "weekly efficiency score"})
        self.assertIn("efficiency_chart.tsx", res2)
        self.assertIn("calculateWeeklyScore", res2)

    def test_lsp_diagnostics_unclosed_tags_and_syntax(self) -> None:
        """Verify lsp diagnostics unclosed tags and syntax."""
        orchestrator = AgentOrchestrator(config=self.config)

        broken_file = self.workspace / "broken_page.tsx"
        broken_file.write_text(
            "export default function BrokenPage() {\n"
            "  return (\n"
            "    <div className='container'>\n"
            "      <p>Test</p>\n",
            encoding="utf-8",
        )

        diag_res = orchestrator.registry.execute("lsp_get_diagnostics", {"file_path": "broken_page.tsx"})
        self.assertIn("Kapatılmamış süslü parantez", diag_res)
        self.assertIn("Kapatılmamış JSX etiketi", diag_res)

        clean_file = self.workspace / "clean.py"
        clean_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        clean_diag = orchestrator.registry.execute("lsp_get_diagnostics", {"file_path": "clean.py"})
        self.assertIn("Temiz ✔", clean_diag)

    def test_lsp_find_definition_and_references(self) -> None:
        """Verify lsp find definition and references."""
        orchestrator = AgentOrchestrator(config=self.config)

        service_file = self.workspace / "user_manager.ts"
        service_file.write_text(
            "export class UserManager {\n"
            "  getUser() {}\n"
            "}\n"
            "const mgr = new UserManager();\n"
            "mgr.getUser();\n",
            encoding="utf-8",
        )

        def_res = orchestrator.registry.execute("lsp_find_definition", {"symbol": "UserManager"})
        self.assertIn("user_manager.ts:1", def_res)
        self.assertIn("export class UserManager", def_res)

        ref_res = orchestrator.registry.execute("lsp_find_references", {"symbol": "UserManager"})
        self.assertIn("user_manager.ts", ref_res)
        self.assertIn("new UserManager", ref_res)

    def test_file_state_tracker_dedup_and_invalidation(self) -> None:
        """Verify file state tracker dedup and invalidation."""
        orchestrator = AgentOrchestrator(config=self.config)

        sample = self.workspace / "service.py"
        sample.write_text("def process():\n    return 42\n", encoding="utf-8")

        res1 = orchestrator.registry.execute("read_file", {"path": "service.py"})
        self.assertIn("def process():", res1)
        self.assertIn("Satır 1-2", res1)

        res2 = orchestrator.registry.execute("read_file", {"path": "service.py"})
        self.assertIn("Dosya Değişmedi / File Unchanged", res2)
        self.assertIn("Adım 1 içerisinde zaten okundu", res2)

        edit_res = orchestrator.registry.execute(
            "edit_file",
            {"path": "service.py", "target": "return 42", "replacement": "return 100"}
        )
        self.assertIn("Başarılı", edit_res)

        res3 = orchestrator.registry.execute("read_file", {"path": "service.py"})
        self.assertIn("def process():", res3)
        self.assertIn("return 100", res3)
        self.assertNotIn("Dosya Değişmedi", res3)

    def test_search_stack_hint_and_matching_directories(self) -> None:
        """Verify search stack hint and matching directories."""
        orchestrator = AgentOrchestrator(config=self.config)

        (self.workspace / "backend" / "src" / "main" / "java").mkdir(parents=True, exist_ok=True)
        (self.workspace / "backend" / "src" / "main" / "java" / "App.java").write_text("class App {}", encoding="utf-8")
        (self.workspace / "frontend").mkdir(parents=True, exist_ok=True)
        (self.workspace / "frontend" / "page.tsx").write_text("export default function() {}", encoding="utf-8")

        res = orchestrator.registry.execute("find_files", {"pattern": "*.py"})
        self.assertIn(".java", res)
        self.assertIn(".tsx", res)
        self.assertIn("İpucu: Bu projede en yaygın dosya uzantıları", res)

        dir_res = orchestrator.registry.execute(
            "find_files",
            {"pattern": "*.java", "directory": "backend/src/main/java/extra_fake"}
        )
        self.assertIn("backend/src/main/java", dir_res)

    def test_autonomous_loop_pivot_and_course_correction(self) -> None:
        """Verify autonomous loop pivot and course correction."""
        from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem

        call_step = {"n": 0}

        def _smart_pivot_chat(**kwargs) -> ParsedAssistantResponse:
            call_step["n"] += 1
            n = call_step["n"]
            if n <= 4:
                return ParsedAssistantResponse(
                    content="",
                    tool_calls=[
                        ToolCallItem(
                            id=f"c_search_{n}",
                            name="find_files",
                            arguments={"pattern": "*.py", "directory": f"dir_{n}"},
                        )
                    ],
                    thinking="",
                )
            elif n == 5:
                return ParsedAssistantResponse(
                    content="",
                    tool_calls=[
                        ToolCallItem(
                            id="c_write",
                            name="write_file",
                            arguments={"path": "success.txt", "content": "Tamamlandı"},
                        )
                    ],
                    thinking="",
                )
            return ParsedAssistantResponse(
                content="Görev başarıyla tamamlandı.",
                tool_calls=[],
                thinking="",
            )

        events_caught: List[str] = []
        bus = EventBus()
        bus.subscribe(lambda ev: events_caught.append(ev.data.get("message", "")), EventType.WARNING)

        orchestrator = AgentOrchestrator(config=self.config, event_bus=bus)
        orchestrator.llm_client.chat = _smart_pivot_chat  # type: ignore

        res = orchestrator.run_turn("Görevi yap")

        self.assertIn("Görev başarıyla tamamlandı", res)
        self.assertTrue((self.workspace / "success.txt").is_file())
        self.assertEqual((self.workspace / "success.txt").read_text(encoding="utf-8"), "Tamamlandı")
        has_pivot_warning = any("Döngü Kırıcı" in w or "mükerrer çağrı engellendi" in w or "Rota Düzeltme" in w or "Devre Kesici" in w for w in events_caught)
        self.assertTrue(has_pivot_warning)

    def test_option_a_intelligence_and_completion_guards(self) -> None:
        """Verify option a intelligence and completion guards."""
        from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem
        from evren_code.core.self_healing import SelfHealingManager

        orchestrator = AgentOrchestrator(config=self.config)
        orchestrator.trust_mode = True

        long_file = self.workspace / "long_module.py"
        long_file.write_text("line1 = 1\nline2 = 2\nline3 = 3\nline4 = 4\n", encoding="utf-8")

        partial_res = orchestrator.registry.execute(
            "read_file", {"path": "long_module.py", "start_line": 1, "end_line": 2}
        )
        self.assertIn("line1 = 1", partial_res)
        self.assertNotIn("line4 = 4", partial_res)

        full_res = orchestrator.registry.execute("read_file", {"path": "long_module.py"})
        self.assertNotIn("Dosya Değişmedi", full_res)
        self.assertIn("line4 = 4", full_res)

        stub_res = orchestrator.registry.execute("read_file", {"path": "long_module.py"})
        self.assertIn("Dosya Değişmedi", stub_res)

        dup_file = self.workspace / "dup.py"
        dup_file.write_text("def a():\n    return None\n\ndef b():\n    return None\n", encoding="utf-8")
        ambig_res = orchestrator.registry.execute(
            "edit_file",
            {"path": "dup.py", "target": "    return None", "replacement": "    return 1"},
        )
        self.assertTrue("multiple matches" in ambig_res.lower() or "çoklu eşleşme" in ambig_res.lower())
        self.assertIn("Satır 2, Satır 5", ambig_res)

        call_n = {"n": 0}

        def _syntax_gate_chat(**kwargs) -> ParsedAssistantResponse:
            call_n["n"] += 1
            if call_n["n"] == 1:
                return ParsedAssistantResponse(
                    content="",
                    thinking="Önce hesaplama modülünü oluşturmalıyım.",
                    tool_calls=[
                        ToolCallItem(
                            id="tc_w1",
                            name="write_file",
                            arguments={"path": "calc.py", "content": "def broken(\n    return 1\n"},
                        )
                    ],
                )
            elif call_n["n"] == 2:
                return ParsedAssistantResponse(
                    content="Modülü yazdım ve bitirdim.",
                    thinking="",
                    tool_calls=[],
                )
            elif call_n["n"] == 3:
                return ParsedAssistantResponse(
                    content="",
                    thinking="Sözdizimi hatasını düzeltiyorum.",
                    tool_calls=[
                        ToolCallItem(
                            id="tc_w2",
                            name="write_file",
                            arguments={"path": "calc.py", "content": "def broken():\n    return 1\n"},
                        )
                    ],
                )
            return ParsedAssistantResponse(
                content="Sözdizimi hatası giderildi ve görev tamamlandı.",
                thinking="",
                tool_calls=[],
            )

        orchestrator.llm_client.chat = _syntax_gate_chat  # type: ignore
        final_out = orchestrator.run_turn("calc.py oluştur")
        self.assertEqual(call_n["n"], 4)
        self.assertIn("Sözdizimi hatası giderildi", final_out)

        all_msgs = orchestrator.context.raw_messages
        self.assertTrue(
            any("[Muhakeme]: Önce hesaplama modülünü oluşturmalıyım." in str(m.get("content") or "") for m in all_msgs)
        )
        self.assertTrue(
            any("Sözdizimi Denetçisi - Hatalı Kod Teslimi Engellendi" in str(m.get("content") or "") for m in all_msgs)
        )

        fe_dir = self.workspace / "frontend"
        be_dir = self.workspace / "backend"
        fe_dir.mkdir(exist_ok=True)
        be_dir.mkdir(exist_ok=True)
        (fe_dir / "package.json").write_text('{"scripts": {"typecheck": "tsc --noEmit"}}', encoding="utf-8")
        (be_dir / "pom.xml").write_text("<project></project>", encoding="utf-8")

        sh = SelfHealingManager(self.workspace)
        be_task = sh.detect_verification_task(modified_paths=["backend/src/Main.java"])
        self.assertIsNotNone(be_task)
        self.assertIn("mvn test-compile", be_task.command)

        fe_task = sh.detect_verification_task(modified_paths=["frontend/src/App.tsx"])
        self.assertIsNotNone(fe_task)
        self.assertIn("npm run typecheck", fe_task.command)

    def test_context_length_exceeded_auto_recovery_and_tool_arg_eviction(self) -> None:
        """Verify context length exceeded auto recovery and tool arg eviction."""
        from evren_code.core.parser import ToolCallItem

        auto_spec = self.config.models.get_model("auto")
        self.assertEqual(auto_spec.context_length, 262144)

        orchestrator = AgentOrchestrator(config=self.config)
        huge_tsx = "export default function Page() {\n" + ("  const x = 'veri';\n" * 400) + "}\n"
        for i in range(6):
            orchestrator.context.add_user_message(f"Sayfa {i} oluştur")
            orchestrator.context.add_assistant_message(
                content="",
                tool_calls=[
                    ToolCallItem(
                        id=f"call_w_{i}",
                        name="write_file",
                        arguments={"path": f"src/page_{i}.tsx", "content": huge_tsx},
                    )
                ],
                native_tools_supported=True,
            )
            orchestrator.context.add_tool_result(
                tool_call_id=f"call_w_{i}",
                tool_name="write_file",
                result_text=f"Başarılı: src/page_{i}.tsx yazıldı.",
                native_tools_supported=True,
            )

        before_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in orchestrator.context.raw_messages)
        evicted = orchestrator.context.evict_large_tool_results(max_chars_per_result=600, keep_recent=2)
        after_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in orchestrator.context.raw_messages)
        self.assertGreater(evicted, 0)
        self.assertLess(after_chars, before_chars // 2)
        first_asst = [m for m in orchestrator.context.raw_messages if m.get("role") == "assistant"][0]
        parsed_arg = json.loads(first_asst["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(parsed_arg["path"], "src/page_0.tsx")
        self.assertIn("kırpıldı", parsed_arg["content"])

        call_step = {"n": 0}

        def mock_overflow_handler(request: httpx.Request) -> httpx.Response:
            call_step["n"] += 1
            body = json.loads(request.content.decode("utf-8"))
            if call_step["n"] == 1:
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": "İstem (278123) + çıktı tavanı (16384) modelin bağlam penceresini (262144) aşıyor",
                            "type": "invalid_request_error",
                            "param": None,
                            "code": "context_length_exceeded",
                        }
                    },
                )
            self.assertIn("tools", body)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Bağlam sıkıştırıldı ve 8. adım başarıyla tamamlandı.",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 12000, "total_tokens": 12050},
                },
            )

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_overflow_handler))
        llm_client = EvrenLLMClient(self.config, http_client=mock_http)
        llm_client.set_model("auto")
        orchestrator.llm_client = llm_client

        result = orchestrator.run_turn("devam et", stream=False)
        self.assertIn("8. adım başarıyla tamamlandı", result)
        self.assertTrue(llm_client.active_model.tools)
        self.assertEqual(call_step["n"], 2)

    def test_speed_token_economy_anti_bloat_and_eval_runner(self) -> None:
        """Verify speed token economy anti bloat and eval runner."""
        from evals.runner import EvalRunner

        orchestrator = AgentOrchestrator(config=self.config)
        orchestrator.trust_mode = True

        sample_py = self.workspace / "discount_service.py"
        sample_py.write_text("def apply_discount(p: float) -> float:\n    return p * 0.9\n", encoding="utf-8")

        preloaded = orchestrator._preload_prompt_mentioned_files(
            "discount_service.py dosyasındaki oranı 0.85 yap."
        )
        self.assertIn("[Otomatik Ön-Yüklenen Dosya İçeriği", preloaded)
        self.assertIn("def apply_discount", preloaded)

        bloated_code = "\n".join(
            ["def render_cards():"]
            + ["    card_html = '<div class=\"metric-card-wrapper-item\">Sabit Kart</div>'" for _ in range(8)]
            + [f"    x_{i} = {i}" for i in range(305)]
        )
        write_res = orchestrator.registry.execute(
            "write_file",
            {"path": "bloated_view.py", "content": bloated_code},
        )
        self.assertIn("[Kod Ekonomisi / Yalınlık Notu]", write_res)
        self.assertIn("önerilen tavan: 300 satır", write_res)
        self.assertIn("tekrarlandığı tespit edildi", write_res)

        lean_code = "def summarize(items: list[int]) -> int:\n    return sum(items)\n"
        lean_res = orchestrator.registry.execute(
            "write_file",
            {"path": "lean_view.py", "content": lean_code},
        )
        self.assertNotIn("[Kod Ekonomisi / Yalınlık Notu]", lean_res)

        big_original = "\n".join(f"line_{i} = {i}" for i in range(40)) + "\n"
        big_replacement = "\n".join(f"line_{i} = {i * 10}" for i in range(40))
        (self.workspace / "big_edit.py").write_text(big_original, encoding="utf-8")
        edit_res = orchestrator.registry.execute(
            "edit_file",
            {"path": "big_edit.py", "target": big_original.strip(), "replacement": big_replacement},
        )
        self.assertIn("Diff bağlam tasarrufu için özetlendi", edit_res)

        tasks_file = Path(__file__).resolve().parent.parent / "evals" / "tasks.json"
        runner = EvalRunner(tasks_file, task_filter="lean_code")
        filtered = runner.load_tasks()
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["id"], "eval_11_lean_code_anti_bloat")


if __name__ == "__main__":
    unittest.main()
