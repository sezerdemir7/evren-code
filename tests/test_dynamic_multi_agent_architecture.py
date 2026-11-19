"""Dynamic Multi-Agent Architecture Tests.

Verifies:
1. Dynamic, domain-agnostic task decomposition (Game dev, Mobile dev, Web, CLI).
2. No rigid 2-4 task count limitation (flexible 2, 4, 6+ tasks based on complexity).
3. Contract-Driven Synchronization (provides_contracts and consumes_contracts auto-linking).
4. ContextSharingEngine injects contract specifications into downstream agents.
5. Natural language multi-agent invocation in AgentOrchestrator.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.multi_agent.coordinator import MultiAgentCoordinator
from evren_code.core.multi_agent.models import (
    AgentRole,
    EngineeringTask,
    MultiAgentRunConfig,
    TaskStatus,
)
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.multi_agent.context_sharing import ContextSharingEngine
from evren_code.core.parser import ParsedAssistantResponse


class TestDynamicMultiAgentArchitecture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_test_dyn_multi_"))
        self.config = load_config()
        self.config.workspace_root = self.tmp_dir

    def test_game_dev_domain_agnostic_decomposition_and_contracts(self) -> None:
        """Verify game dev domain agnostic decomposition and contracts."""
        mock_llm = MagicMock()

        tasks_json = json.dumps([
            {
                "task_id": "gameplay_core",
                "title": "Oyuncu Hareketi ve Kontroller",
                "description": "Karakter kontrolcüsünü kodla",
                "domain": "gameplay",
                "role": "developer",
                "dependencies": [],
                "provides_contracts": ["PlayerStatsModel"],
                "consumes_contracts": [],
                "assigned_files": ["src/player.py"],
            },
            {
                "task_id": "audio_engine",
                "title": "Ses Efektleri Yöneticisi",
                "description": "Arka plan ve efekt seslerini yönet",
                "domain": "audio",
                "role": "developer",
                "dependencies": [],
                "provides_contracts": ["AudioClipConfig"],
                "consumes_contracts": [],
                "assigned_files": ["src/audio.py"],
            },
            {
                "task_id": "hud_ui",
                "title": "Oyun HUD ve Skor Ekranı",
                "description": "Oyuncu can ve skor göstergesini kodla",
                "domain": "hud_ui",
                "role": "developer",
                "dependencies": [],
                "provides_contracts": [],
                "consumes_contracts": ["PlayerStatsModel"],
                "assigned_files": ["src/hud.py"],
            },
            {
                "task_id": "physics_sim",
                "title": "Çarpışma ve Yerçekimi Fiziği",
                "description": "Basit 2D yerçekimi ve zıplama mekaniği",
                "domain": "physics",
                "role": "developer",
                "dependencies": [],
                "provides_contracts": ["PhysicsEngine"],
                "consumes_contracts": [],
                "assigned_files": ["src/physics.py"],
            },
            {
                "task_id": "integration_qa",
                "title": "Oyun Döngüsü Entegrasyon Testi",
                "description": "Tüm modüllerin oyun döngüsünde (main loop) çalıştığını doğrula",
                "domain": "testing",
                "role": "tester",
                "dependencies": ["gameplay_core", "hud_ui", "audio_engine", "physics_sim"],
                "provides_contracts": [],
                "consumes_contracts": [],
                "assigned_files": [],
            },
        ])

        mock_llm.chat.return_value = ParsedAssistantResponse(content=tasks_json)

        coordinator = MultiAgentCoordinator(
            config=self.config,
            llm_client=mock_llm,
            tool_executor=MagicMock(),
            registry=MagicMock(),
        )

        graph = coordinator.plan_dag_from_goal("2D Platformer Oyun Projesi Geliştir")

        self.assertEqual(len(graph.tasks), 5)

        domains = {t.task_id: t.domain for t in graph.tasks.values()}
        self.assertEqual(domains["gameplay_core"], "gameplay")
        self.assertEqual(domains["audio_engine"], "audio")
        self.assertEqual(domains["hud_ui"], "hud_ui")
        self.assertEqual(domains["physics_sim"], "physics")

        hud_task = graph.get_task("hud_ui")
        self.assertIsNotNone(hud_task)
        self.assertIn("gameplay_core", hud_task.dependencies)

        ready_tasks = [t.task_id for t in graph.get_ready_tasks()]
        self.assertIn("gameplay_core", ready_tasks)
        self.assertIn("audio_engine", ready_tasks)
        self.assertIn("physics_sim", ready_tasks)
        self.assertNotIn("hud_ui", ready_tasks)

    def test_context_sharing_injects_contract_data(self) -> None:
        """Verify context sharing injects contract data."""
        graph = TaskGraph()
        t1 = EngineeringTask(
            task_id="t1_model",
            title="Veri Modeli",
            description="User şeması",
            role=AgentRole.DEVELOPER,
            domain="backend",
            provides_contracts=["UserModel"],
        )
        t1.status = TaskStatus.COMPLETED
        t1.result = MagicMock()
        t1.result.success = True
        t1.result.modified_files = ["src/models.py"]
        t1.result.artifacts = {"UserModel": "class User(BaseModel): id: int, name: str"}
        t1.result.output = "Model tanımlandı."

        t2 = EngineeringTask(
            task_id="t2_ui",
            title="Kullanıcı Profili Ekranı",
            description="User verisini göster",
            role=AgentRole.DEVELOPER,
            domain="frontend",
            dependencies=["t1_model"],
            consumes_contracts=["UserModel"],
        )

        graph.add_task(t1)
        graph.add_task(t2)

        sharing_engine = ContextSharingEngine()
        ctx = sharing_engine.prepare_task_context(t2, graph)

        self.assertIn("contracts", ctx)
        self.assertIn("UserModel", ctx["contracts"])
        contract_info = ctx["contracts"]["UserModel"]
        self.assertEqual(contract_info["provider_task_id"], "t1_model")
        self.assertEqual(contract_info["domain"], "backend")
        self.assertIn("class User(BaseModel)", contract_info["contract_data"])

    def test_natural_language_multi_agent_routing(self) -> None:
        """Verify natural language multi agent routing."""
        orc = AgentOrchestrator(self.config)
        orc.run_multi_agent_workflow = MagicMock(return_value="Çoklu ajan tamamlandı.")  # type: ignore

        prompt = "bu projeyi alt ajanlara bölüp aynı anda yap"
        res = orc.run_turn(prompt)

        orc.run_multi_agent_workflow.assert_called_once()
        self.assertEqual(res, "Çoklu ajan tamamlandı.")

    def test_autonomous_multi_agent_decomposition_without_explicit_keywords(self) -> None:
        """Verify autonomous multi agent decomposition without explicit keywords."""
        self.config.multi_agent.enabled = True
        self.config.multi_agent.auto_decompose = True
        orc = AgentOrchestrator(self.config)
        orc.run_multi_agent_workflow = MagicMock(return_value="Otonom çoklu ajan tamamlandı.")  # type: ignore

        user_prompt = "frontend ekranlarını daha temiz hale ve kullanışlı ve modern hale getir"
        should_dec, reason = orc.should_autonomously_decompose(user_prompt)
        self.assertTrue(should_dec, f"Otonom karar True olmalıydı: {reason}")
        self.assertIn("Otonom çoklu ajan kararı", reason)

        res = orc.run_turn(user_prompt)
        orc.run_multi_agent_workflow.assert_called_once_with(user_prompt)
        self.assertEqual(res, "Otonom çoklu ajan tamamlandı.")

        fullstack_prompt = "full stack bir e-ticaret uygulaması geliştir hem backend api hem frontend ui olsun"
        should_fs, reason_fs = orc.should_autonomously_decompose(fullstack_prompt)
        self.assertTrue(should_fs, f"Fullstack görev otonom olarak bölünmeli: {reason_fs}")

        game_prompt = "2d uzay oyunu geliştir, fizik motoru, düşman yapay zekası ve ses yöneticisini kur"
        should_game, reason_game = orc.should_autonomously_decompose(game_prompt)
        self.assertTrue(should_game, f"Oyun sistemleri otonom olarak bölünmeli: {reason_game}")

    def test_autonomous_decomposition_skips_simple_tasks_and_questions(self) -> None:
        """Verify autonomous decomposition skips simple tasks and questions."""
        self.config.multi_agent.enabled = True
        self.config.multi_agent.auto_decompose = True
        orc = AgentOrchestrator(self.config)

        should_q, reason_q = orc.should_autonomously_decompose("neden hata aldın?")
        self.assertFalse(should_q)
        self.assertIn("bilgi/açıklama", reason_q)

        should_typo, reason_typo = orc.should_autonomously_decompose("utils.py içindeki typo düzelt")
        self.assertFalse(should_typo)
        self.assertIn("Basit mikro-düzeltme", reason_typo)

        # 3. Devam komutu
        should_cont, reason_cont = orc.should_autonomously_decompose("devam et")
        self.assertFalse(should_cont)
        self.assertIn("Devam/onay", reason_cont)

    def test_autonomous_decomposition_disabled_when_config_is_off(self) -> None:
        """Verify autonomous decomposition disabled when config is off."""
        self.config.multi_agent.enabled = False
        orc = AgentOrchestrator(self.config)

        should_dec, reason = orc.should_autonomously_decompose("frontend ekranlarını modernize et")
        self.assertFalse(should_dec)
        self.assertIn("kapalı", reason)

        should_explicit, _ = orc.should_autonomously_decompose("bu işi alt ajanlara bölerek yap")
        self.assertTrue(should_explicit)

    def test_event_type_info_and_multi_agent_report_generation(self) -> None:
        """Verify event type info and multi agent report generation."""
        import io
        from rich.console import Console
        from evren_code.core.events import EventBus, EventType
        from evren_code.core.multi_agent.models import AgentTaskResult, OrchestrationReport
        from evren_code.ui.console import ConsoleUI

        bus = EventBus()
        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False)
        ui = ConsoleUI(event_bus=bus, console=test_console)

        bus.emit(EventType.INFO, message="Test bilgi mesajı")
        self.assertIn("Test bilgi mesajı", buf.getvalue())

        orc = AgentOrchestrator(self.config, event_bus=bus)
        mock_report = OrchestrationReport(
            run_id="run_123",
            goal="Responsive ekran düzeltmeleri",
            total_tasks=2,
            completed_tasks=2,
            failed_tasks=0,
            blocked_tasks=0,
            total_execution_time_seconds=1.5,
            modified_files=["src/Sidebar.tsx"],
            task_results={
                "task_1": AgentTaskResult(
                    task_id="task_1",
                    agent_id="agent_dev_1",
                    role=AgentRole.DEVELOPER,
                    success=True,
                    output="Responsive düzenlendi.",
                    modified_files=["src/Sidebar.tsx"],
                    execution_time_seconds=1.2,
                )
            },
            success=True,
        )

        orc.multi_agent_coordinator.plan_dag_from_goal = MagicMock()  # type: ignore
        orc.multi_agent_coordinator.run_plan = MagicMock(return_value=mock_report)  # type: ignore

        summary = orc.run_multi_agent_workflow("ekranları responsive yap")
        self.assertIn("Çoklu Ajan Görev Raporu", summary)
        self.assertIn("src/Sidebar.tsx", summary)
        self.assertIn("Responsive düzenlendi", summary)

    def test_all_agent_roles_resolve_valid_catalog_models(self) -> None:
        """Verify all agent roles resolve valid catalog models."""
        registry = self.config.models.catalog

        for role in AgentRole:
            spec = self.config.models.get_model(role.value)
            self.assertIn(
                spec.id,
                registry,
                f"Rol '{role.value}' için çözümlenen model ID '{spec.id}' katalogda bulunamadı!",
            )
            self.assertNotEqual(spec.id, role.value)

        dev_spec = self.config.models.get_model(AgentRole.DEVELOPER.value)
        self.assertEqual(dev_spec.id, "qwen3.8-flash-next")

        plan_spec = self.config.models.get_model(AgentRole.PLANNER.value)
        self.assertEqual(plan_spec.id, "deepseek-v4.1-flash")

        fallback_spec = self.config.models.get_model("unknown_role_xyz")
        self.assertEqual(fallback_spec.id, self.config.models.default)


if __name__ == "__main__":
    unittest.main()
