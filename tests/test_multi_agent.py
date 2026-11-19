"""Test multi agent module."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from evren_code.config import AppConfig, ModelSpec, MultiAgentConfig, load_config
from evren_code.core.events import EventBus
from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem
from evren_code.core.multi_agent.agent_factory import SpecialistAgent
from evren_code.core.multi_agent.agent_registry import AgentRegistry
from evren_code.core.multi_agent.communication import AgentMessageBus
from evren_code.core.multi_agent.conflict_resolver import ConflictResolver
from evren_code.core.multi_agent.context_sharing import ContextSharingEngine
from evren_code.core.multi_agent.coordinator import MultiAgentCoordinator
from evren_code.core.multi_agent.file_ownership import (
    FILE_OWNERSHIP_VIOLATION,
    FileOwnershipManager,
)
from evren_code.core.multi_agent.models import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    AgentTaskResult,
    EngineeringTask,
    MessageType,
    MultiAgentRunConfig,
    TaskStatus,
)
from evren_code.core.multi_agent.persistence import MultiAgentPersistence
from evren_code.core.multi_agent.result_aggregator import ResultAggregator
from evren_code.core.multi_agent.scheduler import MultiAgentScheduler
from evren_code.core.multi_agent.task_graph import TaskGraph, TaskGraphError
from evren_code.core.security import PermissionManager, ToolExecutor
from evren_code.tools.filesystem import register_filesystem_tools
from evren_code.tools.registry import ToolRegistry


class TestTaskGraphDAG(unittest.TestCase):
    """Test task graph dag implementation."""

    def test_topological_sort_linear(self):
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Görev 1", "Tasarım", AgentRole.ARCHITECT)
        t2 = EngineeringTask("t2", "Görev 2", "Geliştirme", AgentRole.DEVELOPER, dependencies=["t1"])
        t3 = EngineeringTask("t3", "Görev 3", "Test", AgentRole.TESTER, dependencies=["t2"])

        graph.add_task(t3)
        graph.add_task(t1)
        graph.add_task(t2)

        order = graph.topological_sort()
        self.assertEqual(order, ["t1", "t2", "t3"])

    def test_topological_sort_diamond(self):
        # t1 -> t2, t3 -> t4
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Kök", "Tasarım", AgentRole.ARCHITECT)
        t2 = EngineeringTask("t2", "Modül A", "Geliştirme", AgentRole.DEVELOPER, dependencies=["t1"])
        t3 = EngineeringTask("t3", "Modül B", "Geliştirme", AgentRole.DEVELOPER, dependencies=["t1"])
        t4 = EngineeringTask("t4", "Entegrasyon", "Test", AgentRole.TESTER, dependencies=["t2", "t3"])

        for t in [t4, t2, t1, t3]:
            graph.add_task(t)

        order = graph.topological_sort()
        self.assertEqual(order[0], "t1")
        self.assertEqual(order[-1], "t4")
        self.assertIn("t2", order[1:3])
        self.assertIn("t3", order[1:3])

    def test_cycle_detection_direct(self):
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Görev 1", "", AgentRole.DEVELOPER, dependencies=["t2"])
        t2 = EngineeringTask("t2", "Görev 2", "", AgentRole.DEVELOPER, dependencies=["t1"])
        graph.add_task(t1)
        graph.add_task(t2)

        valid, errors = graph.validate_dag()
        self.assertFalse(valid)
        self.assertTrue(any("çevrim" in e for e in errors))
        with self.assertRaises(TaskGraphError):
            graph.topological_sort()

    def test_cycle_detection_self_dependency(self):
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Kendine Bağımlı", "", AgentRole.DEVELOPER, dependencies=["t1"])
        graph.add_task(t1)

        valid, errors = graph.validate_dag()
        self.assertFalse(valid)
        self.assertTrue(any("Öz-bağımlılık" in e for e in errors))

    def test_missing_dependency_detection(self):
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Görev 1", "", AgentRole.DEVELOPER, dependencies=["t_non_existent"])
        graph.add_task(t1)

        valid, errors = graph.validate_dag()
        self.assertFalse(valid)
        self.assertTrue(any("Eksik bağımlılık" in e for e in errors))

    def test_get_ready_tasks(self):
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Görev 1", "", AgentRole.ARCHITECT)
        t2 = EngineeringTask("t2", "Görev 2", "", AgentRole.DEVELOPER, dependencies=["t1"])

        graph.add_task(t1)
        graph.add_task(t2)

        ready = graph.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "t1")

        graph.mark_task_status("t1", TaskStatus.COMPLETED)
        ready2 = graph.get_ready_tasks()
        self.assertEqual(len(ready2), 1)
        self.assertEqual(ready2[0].task_id, "t2")

    def test_cascade_failure_blocks_downstream_tasks(self):
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Görev 1", "", AgentRole.DEVELOPER)
        t2 = EngineeringTask("t2", "Görev 2", "", AgentRole.DEVELOPER, dependencies=["t1"])
        t3 = EngineeringTask("t3", "Görev 3", "", AgentRole.TESTER, dependencies=["t2"])

        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        graph.mark_task_status("t1", TaskStatus.FAILED)
        self.assertEqual(graph.get_task("t1").status, TaskStatus.FAILED)
        self.assertEqual(graph.get_task("t2").status, TaskStatus.BLOCKED)
        self.assertEqual(graph.get_task("t3").status, TaskStatus.BLOCKED)
        self.assertTrue(graph.is_finished())
        self.assertFalse(graph.all_completed())

    def test_serialization_and_deserialization(self):
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Başlık 1", "Açıklama 1", AgentRole.ARCHITECT, assigned_files=["a.py"])
        graph.add_task(t1)

        d = graph.to_dict()
        restored = TaskGraph.from_dict(d)
        self.assertEqual(len(restored.tasks), 1)
        self.assertEqual(restored.get_task("t1").title, "Başlık 1")
        self.assertEqual(restored.get_task("t1").assigned_files, ["a.py"])

    def test_to_ascii_tree(self):
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Plan", "", AgentRole.PLANNER)
        graph.add_task(t1)
        ascii_tree = graph.to_ascii_tree()
        self.assertIn("Görev Çizgesi", ascii_tree)
        self.assertIn("t1", ascii_tree)


class TestFileOwnershipManager(unittest.TestCase):
    """Test file ownership manager implementation."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_test_locks_"))
        self.fom = FileOwnershipManager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_acquire_and_release_exclusive_write_lock(self):
        ok, err = self.fom.acquire_locks("t1", "agent_1", write_files=["src/main.py"])
        self.assertTrue(ok)
        self.assertIsNone(err)

        ok2, err2 = self.fom.acquire_locks("t2", "agent_2", write_files=["src/main.py"])
        self.assertFalse(ok2)
        self.assertIn(FILE_OWNERSHIP_VIOLATION, err2)

        self.fom.release_locks("t1")
        ok3, err3 = self.fom.acquire_locks("t2", "agent_2", write_files=["src/main.py"])
        self.assertTrue(ok3)
        self.assertIsNone(err3)

    def test_shared_read_lock_multiple_tasks(self):
        ok1, _ = self.fom.acquire_locks("t1", "agent_1", write_files=[], read_files=["src/common.py"])
        ok2, _ = self.fom.acquire_locks("t2", "agent_2", write_files=[], read_files=["src/common.py"])
        self.assertTrue(ok1)
        self.assertTrue(ok2)

        ok_w, err_w = self.fom.acquire_locks("t3", "agent_3", write_files=["src/common.py"])
        self.assertFalse(ok_w)
        self.assertIn("okunuyor", err_w)

    def test_path_normalization_case_and_slashes(self):
        ok, _ = self.fom.acquire_locks("t1", "agent_1", write_files=["src/Module/App.py"])
        self.assertTrue(ok)

        ok2, _ = self.fom.acquire_locks("t2", "agent_2", write_files=["src\\module\\app.py"])
        self.assertFalse(ok2)

    def test_verify_write_permission(self):
        self.fom.acquire_locks("t1", "agent_1", write_files=["src/auth.py"])

        can_write1, _ = self.fom.verify_write_permission("t1", "src/auth.py")
        self.assertTrue(can_write1)

        can_write2, err = self.fom.verify_write_permission("t2", "src/auth.py")
        self.assertFalse(can_write2)
        self.assertIn(FILE_OWNERSHIP_VIOLATION, err)

        can_unassigned, _ = self.fom.verify_write_permission("t1", "src/other.py")
        self.assertFalse(can_unassigned)


class TestConflictResolver(unittest.TestCase):
    """Test conflict resolver implementation."""

    def test_detect_and_sequentialize_write_overlap(self):
        resolver = ConflictResolver()
        graph = TaskGraph()

        t1 = EngineeringTask("t1", "Özellik 1", "", AgentRole.DEVELOPER, assigned_files=["src/config.py"])
        t2 = EngineeringTask("t2", "Özellik 2", "", AgentRole.DEVELOPER, assigned_files=["src/config.py"])
        graph.add_task(t1)
        graph.add_task(t2)

        conflicts = resolver.detect_conflicts(graph)
        self.assertEqual(len(conflicts), 1)
        self.assertIn("src/config.py", conflicts[0].conflicting_files)

        resolved = resolver.resolve_conflicts(graph, strategy="sequentialize")
        self.assertEqual(len(resolved), 1)
        self.assertTrue(resolved[0].resolved)

        self.assertTrue("t1" in t2.dependencies or "t2" in t1.dependencies)
        valid, _ = graph.validate_dag()
        self.assertTrue(valid)

    def test_no_conflict_when_already_dependent(self):
        resolver = ConflictResolver()
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Adım 1", "", AgentRole.DEVELOPER, assigned_files=["a.py"])
        t2 = EngineeringTask("t2", "Adım 2", "", AgentRole.DEVELOPER, dependencies=["t1"], assigned_files=["a.py"])
        graph.add_task(t1)
        graph.add_task(t2)

        conflicts = resolver.detect_conflicts(graph)
        self.assertEqual(len(conflicts), 0)


class TestSpecialistAgentAndRegistry(unittest.TestCase):
    """Test specialist agent and registry implementation."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_test_agents_"))
        self.config = load_config(workspace_root=self.tmp_dir)
        self.event_bus = EventBus()
        self.registry = ToolRegistry()
        self.pm = PermissionManager(self.tmp_dir)
        self.fom = FileOwnershipManager(self.tmp_dir)

        class AutoApprove:
            def request_approval(self, r): return True
            def ask_user_question(self, q, opts=None): return ""

        self.tool_executor = ToolExecutor(
            registry=self.registry,
            permission_manager=self.pm,
            approval_delegate=AutoApprove(),
            event_bus=self.event_bus,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_architect_cannot_call_modifying_tools(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ParsedAssistantResponse(
            content="",
            tool_calls=[ToolCallItem(id="tc1", name="write_file", arguments={"path": "test.txt", "content": "hi"})],
        )

        agent = SpecialistAgent(
            agent_id="arch_1",
            role=AgentRole.ARCHITECT,
            config=self.config,
            llm_client=mock_llm,
            tool_executor=self.tool_executor,
            registry=self.registry,
            file_ownership=self.fom,
        )

        task = EngineeringTask("t1", "Tasarım", "Mimari analiz yap", AgentRole.ARCHITECT)
        result = agent.execute_task(task)
        self.assertNotIn("test.txt", result.modified_files)

    def test_developer_blocked_if_file_not_owned(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ParsedAssistantResponse(
            content="",
            tool_calls=[ToolCallItem(id="tc1", name="write_file", arguments={"path": "unauthorized.txt", "content": "hacked"})],
        )

        agent = SpecialistAgent(
            agent_id="dev_1",
            role=AgentRole.DEVELOPER,
            config=self.config,
            llm_client=mock_llm,
            tool_executor=self.tool_executor,
            registry=self.registry,
            file_ownership=self.fom,
        )

        task = EngineeringTask(
            "t1", "Geliştirme", "Kod yaz", AgentRole.DEVELOPER,
            assigned_files=["allowed.txt"],
        )
        self.fom.acquire_locks("t1", "dev_1", write_files=["allowed.txt"])

        result = agent.execute_task(task)
        self.assertNotIn("unauthorized.txt", result.modified_files)

    def test_agent_registry_pool(self):
        mock_llm = MagicMock()
        pool = AgentRegistry(
            config=self.config,
            llm_client=mock_llm,
            tool_executor=self.tool_executor,
            registry=self.registry,
            file_ownership=self.fom,
        )

        a1 = pool.get_or_create_agent(AgentRole.DEVELOPER)
        self.assertEqual(a1.role, AgentRole.DEVELOPER)
        agents = pool.list_agents()
        self.assertEqual(len(agents), 1)


class TestCommunicationAndContextSharing(unittest.TestCase):
    """Test communication and context sharing implementation."""

    def test_message_bus_routing(self):
        bus = AgentMessageBus()
        msg1 = AgentMessage(sender_id="coord", recipient_id="agent_1", message_type=MessageType.TASK_ASSIGNMENT)
        msg2 = AgentMessage(sender_id="coord", recipient_id="broadcast", message_type=MessageType.SYSTEM_ALERT)

        bus.send_message(msg1)
        bus.send_message(msg2)

        inbox_1 = bus.receive_messages("agent_1")
        self.assertEqual(len(inbox_1), 2)  # msg1 + broadcast msg2

    def test_context_sharing_collects_dependency_artifacts(self):
        cse = ContextSharingEngine()
        graph = TaskGraph()

        t1 = EngineeringTask("t1", "Mimari Görev", "Tasarım", AgentRole.ARCHITECT)
        t1.result = AgentTaskResult(
            task_id="t1",
            agent_id="arch_1",
            role=AgentRole.ARCHITECT,
            success=True,
            output="Veritabanı şeması: users ve orders tabloları.",
        )
        graph.add_task(t1)

        t2 = EngineeringTask("t2", "Kodlama", "Model yaz", AgentRole.DEVELOPER, dependencies=["t1"])
        graph.add_task(t2)

        ctx = cse.prepare_task_context(t2, graph)
        self.assertIn("t1", ctx["dependency_artifacts"])
        self.assertIn("users ve orders", ctx["dependency_artifacts"]["t1"])


class TestEndToEndMultiAgentCoordination(unittest.TestCase):
    """Test end to end multi agent coordination implementation."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_test_e2e_multi_"))
        self.config = load_config(workspace_root=self.tmp_dir)
        self.event_bus = EventBus()
        self.registry = ToolRegistry()
        self.pm = PermissionManager(self.tmp_dir)

        class AutoApprove:
            def request_approval(self, r): return True
            def ask_user_question(self, q, opts=None): return ""

        self.tool_executor = ToolExecutor(
            registry=self.registry,
            permission_manager=self.pm,
            approval_delegate=AutoApprove(),
            event_bus=self.event_bus,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_e2e_plan_and_execution_with_persistence(self):
        # Sahte LLM istemcisi
        mock_llm = MagicMock()

        def _mock_chat(messages, tools_schema=None, model_override=None, stream=False):
            last_msg = messages[-1]["content"] if messages else ""
            if "JSON listesi" in last_msg or "plan_init" in last_msg:
                tasks_json = json.dumps([
                    {
                        "task_id": "step_1",
                        "title": "Arayüz Tanımı",
                        "description": "API arayüzünü oluştur",
                        "role": "architect",
                        "dependencies": [],
                        "assigned_files": [],
                        "acceptance_criteria": ["Arayüz net olmalı"],
                    },
                    {
                        "task_id": "step_2",
                        "title": "Modül Kodlaması",
                        "description": "API modülünü kodla",
                        "role": "developer",
                        "dependencies": ["step_1"],
                        "assigned_files": ["api.py"],
                        "acceptance_criteria": ["Dosya yazılmalı"],
                    },
                ])
                return ParsedAssistantResponse(content=tasks_json)

            if "Kod İnceleme" in last_msg or "acceptance criteria" in last_msg:
                return ParsedAssistantResponse(content="Kod uygun ve kabul kriterleri sağlandı. [APPROVED]")

            return ParsedAssistantResponse(content="Görev başarıyla tamamlandı ve test edildi.")

        mock_llm.chat.side_effect = _mock_chat

        multi_cfg = MultiAgentRunConfig(
            enabled=True,
            max_parallel_agents=2,
            auto_review=True,
            verify_with_engine=False,
            persistence_dir=".evren/multi_agent",
        )

        coordinator = MultiAgentCoordinator(
            config=self.config,
            llm_client=mock_llm,
            tool_executor=self.tool_executor,
            registry=self.registry,
            multi_config=multi_cfg,
        )

        report = coordinator.run_plan(goal="Kullanıcı API Servisi Geliştir")

        self.assertTrue(report.success)
        self.assertEqual(report.completed_tasks, 2)
        self.assertEqual(report.failed_tasks, 0)
        self.assertIn("BAŞARILI", report.summary)

        runs = coordinator.persistence.list_runs()
        self.assertGreaterEqual(len(runs), 1)
        run_id = runs[0]["run_id"]

        graph_restored, rep_restored, meta = coordinator.persistence.load_checkpoint(run_id)
        self.assertEqual(len(graph_restored.tasks), 2)
        self.assertEqual(rep_restored.run_id, run_id)
        self.assertEqual(meta.get("goal"), "Kullanıcı API Servisi Geliştir")

    def test_cancellation_releases_locks_and_marks_cancelled(self):
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Uzun Görev", "", AgentRole.DEVELOPER, assigned_files=["busy.py"])
        graph.add_task(t1)

        fom = FileOwnershipManager(self.tmp_dir)
        reg = AgentRegistry(self.config, MagicMock(), self.tool_executor, self.registry, fom)
        cse = ContextSharingEngine()
        res_agg = ResultAggregator()

        scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=reg,
            file_ownership=fom,
            context_sharing=cse,
            result_aggregator=res_agg,
        )

        scheduler.cancel()
        self.assertTrue(scheduler.is_cancelled)
        self.assertEqual(graph.get_task("t1").status, TaskStatus.CANCELLED)
        self.assertEqual(len(fom.get_active_locks()), 0)

    def test_parse_json_tasks_resilience(self):
        coordinator = MultiAgentCoordinator(
            config=self.config,
            llm_client=MagicMock(),
            tool_executor=self.tool_executor,
            registry=self.registry,
        )

        # 1. Markdown codeblock
        md_text = """Analiz tamamlandı. İşte görevler:
```json
[
  {
    "task_id": "task_1",
    "title": "Responsive Düzenleme",
    "role": "developer",
    "assigned_files": ["src/components/Sidebar.tsx"]
  }
]
```
Umarım uygundur."""
        tasks = coordinator._parse_json_tasks(md_text)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "task_1")

        # 2. Object with "tasks" key
        obj_text = """```json
{
  "tasks": [
    {"task_id": "task_a", "title": "A"},
    {"task_id": "task_b", "title": "B"}
  ]
}
```"""
        tasks2 = coordinator._parse_json_tasks(obj_text)
        self.assertEqual(len(tasks2), 2)
        self.assertEqual(tasks2[1]["task_id"], "task_b")

    def test_wildcard_file_ownership_dynamic_locking(self):
        fom = FileOwnershipManager(self.tmp_dir)
        ok, err = fom.acquire_locks("task_fallback", "agent_1", write_files=["*"])
        self.assertTrue(ok)
        self.assertIsNone(err)

        can_write, _ = fom.verify_write_permission("task_fallback", "src/components/Sidebar.tsx")
        self.assertTrue(can_write)

        can_write_other, err_other = fom.verify_write_permission("task_other", "src/components/Sidebar.tsx")
        self.assertFalse(can_write_other)
        self.assertIn(FILE_OWNERSHIP_VIOLATION, err_other)

        fom.release_locks("task_fallback")
        self.assertEqual(len(fom.get_active_locks()), 0)

    def test_scheduler_retry_loop_on_reviewer_rejection(self):
        mock_llm = MagicMock()
        attempts = {"count": 0}

        def _mock_chat(messages, tools_schema=None, model_override=None, stream=False):
            last_msg = messages[-1]["content"] if messages else ""
            system_msg = messages[0]["content"] if messages else ""
            is_reviewer = "Reviewer" in system_msg or "Lütfen şu görevin çıktısını" in last_msg
            if is_reviewer:
                if attempts["count"] == 1:
                    return ParsedAssistantResponse(content="Dosya değişikliği yok. [REJECTED: Değişiklik yok]")
                else:
                    return ParsedAssistantResponse(content="Düzeltmeler yapıldı ve doğrulandı. [APPROVED]")
            attempts["count"] += 1
            return ParsedAssistantResponse(content=f"Çalışma tamamlandı deneme={attempts['count']}")

        mock_llm.chat.side_effect = _mock_chat

        graph = TaskGraph()
        task = EngineeringTask(
            task_id="retry_task",
            title="Düzeltme Görevi",
            description="İlk seferde eksik kalabilir",
            role=AgentRole.DEVELOPER,
            assigned_files=["*"],
            acceptance_criteria=["Kod yazılmalı"],
        )
        graph.add_task(task)

        fom = FileOwnershipManager(self.tmp_dir)
        reg = AgentRegistry(self.config, mock_llm, self.tool_executor, self.registry, fom)
        cse = ContextSharingEngine()
        res_agg = ResultAggregator(agent_registry=reg)

        scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=reg,
            file_ownership=fom,
            context_sharing=cse,
            result_aggregator=res_agg,
            config=MultiAgentRunConfig(max_retry_attempts=2, auto_review=True),
        )

        success = scheduler.execute_all()
        self.assertTrue(success)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.retry_count, 1)
        self.assertIn("DÜZELTME VE TEKRAR DENEME #1", task.description)


class TestMultiAgentAdvancedEnhancements(unittest.TestCase):
    """Test multi agent advanced enhancements implementation."""

    def test_conflict_resolver_glob_and_wildcard(self):
        resolver = ConflictResolver()

        g1 = TaskGraph()
        t1 = EngineeringTask("t1", "Global Task", "", AgentRole.DEVELOPER, assigned_files=["*"])
        t2 = EngineeringTask("t2", "Specific Task", "", AgentRole.DEVELOPER, assigned_files=["src/main.py"])
        g1.add_task(t1)
        g1.add_task(t2)
        conflicts1 = resolver.detect_conflicts(g1)
        self.assertEqual(len(conflicts1), 1)
        self.assertTrue(any("*" in c for c in conflicts1[0].conflicting_files))

        g2 = TaskGraph()
        t3 = EngineeringTask("t3", "Backend Task", "", AgentRole.DEVELOPER, assigned_files=["backend/*"])
        t4 = EngineeringTask("t4", "Auth Task", "", AgentRole.DEVELOPER, assigned_files=["backend/controllers/auth.py"])
        g2.add_task(t3)
        g2.add_task(t4)
        conflicts2 = resolver.detect_conflicts(g2)
        self.assertEqual(len(conflicts2), 1)

        g3 = TaskGraph()
        t5 = EngineeringTask("t5", "Frontend Task", "", AgentRole.DEVELOPER, assigned_files=["frontend/*"])
        t6 = EngineeringTask("t6", "Backend Task", "", AgentRole.DEVELOPER, assigned_files=["backend/*"])
        g3.add_task(t5)
        g3.add_task(t6)
        conflicts3 = resolver.detect_conflicts(g3)
        self.assertEqual(len(conflicts3), 0)

    def test_result_aggregator_rejection_false_positive_prevention(self):
        mock_registry = MagicMock()
        mock_reviewer = MagicMock()
        mock_registry.get_or_create_agent.return_value = mock_reviewer

        aggregator = ResultAggregator(agent_registry=mock_registry)
        task = EngineeringTask("t1", "Kod Görevi", "", AgentRole.DEVELOPER, acceptance_criteria=["Temiz kod"])

        mock_reviewer.execute_task.return_value = AgentTaskResult(
            task_id="rev_t1",
            agent_id="rev",
            role=AgentRole.REVIEWER,
            success=True,
            output="Geliştirici çıktısı incelendi. No changes were rejected. All criteria met. [APPROVED]",
        )
        res1 = AgentTaskResult("t1", "dev", AgentRole.DEVELOPER, success=True, output="Tamamlandı", modified_files=["a.py"])
        reviewed1 = aggregator.review_task_result(task, res1)
        self.assertTrue(reviewed1.review_approved)

        mock_reviewer.execute_task.return_value = AgentTaskResult(
            task_id="rev_t1",
            agent_id="rev",
            role=AgentRole.REVIEWER,
            success=True,
            output="Kod başarılı, hiçbir şey reddedilmedi. [ONAYLANDI]",
        )
        res2 = AgentTaskResult("t1", "dev", AgentRole.DEVELOPER, success=True, output="Tamamlandı", modified_files=["b.py"])
        reviewed2 = aggregator.review_task_result(task, res2)
        self.assertTrue(reviewed2.review_approved)

        mock_reviewer.execute_task.return_value = AgentTaskResult(
            task_id="rev_t1",
            agent_id="rev",
            role=AgentRole.REVIEWER,
            success=True,
            output="Kod kabul edilemez. [REJECTED: Eksik testler var]",
        )
        res3 = AgentTaskResult("t1", "dev", AgentRole.DEVELOPER, success=True, output="Tamamlandı", modified_files=["c.py"])
        reviewed3 = aggregator.review_task_result(task, res3)
        self.assertFalse(reviewed3.review_approved)

    def test_context_sharing_contract_normalization_and_code_block_extraction(self):
        cse = ContextSharingEngine()
        graph = TaskGraph()

        t1 = EngineeringTask(
            task_id="t1",
            title="Backend Auth",
            description="",
            role=AgentRole.DEVELOPER,
            provides_contracts=["AuthSchema"],
        )
        t1.result = AgentTaskResult(
            task_id="t1",
            agent_id="dev1",
            role=AgentRole.DEVELOPER,
            success=True,
            output=(
                "Kimlik doğrulama tamamlandı.\n"
                "### AuthSchema\n"
                "```ts\n"
                "export interface AuthSchema {\n"
                "  token: string;\n"
                "  userId: string;\n"
                "}\n"
                "```\n"
            ),
        )

        t2 = EngineeringTask(
            task_id="t2",
            title="Frontend Auth",
            description="",
            role=AgentRole.DEVELOPER,
            dependencies=["t1"],
            consumes_contracts=["auth_schema"],
        )

        graph.add_task(t1)
        graph.add_task(t2)

        ctx = cse.prepare_task_context(t2, graph)
        self.assertIn("auth_schema", ctx["contracts"])
        contract_info = ctx["contracts"]["auth_schema"]
        self.assertEqual(contract_info["provider_task_id"], "t1")
        self.assertIn("export interface AuthSchema", contract_info["contract_data"])

    def test_coordinator_plan_dag_contract_linking_and_aliases(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="evren_test_coord_enh_"))
        try:
            config = load_config(workspace_root=tmp_dir)
            mock_llm = MagicMock()
            json_plan = """
            [
              {
                "task_id": "backend_api",
                "title": "Backend API",
                "description": "API uçlarını yazın",
                "role": "developer",
                "provides": ["UserStore"],
                "assigned_files": ["backend/api.py"]
              },
              {
                "task_id": "frontend_ui",
                "title": "Frontend UI",
                "description": "Kullanıcı arayüzünü src/components/Profile.tsx içinde uygulayın",
                "role": "developer",
                "consumes": ["user_store"]
              }
            ]
            """
            mock_llm.chat.return_value = ParsedAssistantResponse(content=json_plan)

            pm = PermissionManager(tmp_dir)
            fom = FileOwnershipManager(tmp_dir)

            class AutoApprove:
                def request_approval(self, r): return True
                def ask_user_question(self, q, opts=None): return ""

            tool_executor = ToolExecutor(
                registry=ToolRegistry(),
                permission_manager=pm,
                approval_delegate=AutoApprove(),
            )

            coord = MultiAgentCoordinator(
                config=config,
                llm_client=mock_llm,
                tool_executor=tool_executor,
                registry=ToolRegistry(),
            )

            graph = coord.plan_dag_from_goal("Kullanıcı Profil Sistemi")
            self.assertEqual(len(graph.tasks), 2)
            t_front = graph.get_task("frontend_ui")
            self.assertIsNotNone(t_front)
            self.assertIn("backend_api", t_front.dependencies)
            self.assertIn("src/components/Profile.tsx", t_front.assigned_files)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
