"""Test audit phase 2 4 1 module."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from evren_code.config import AppConfig, MultiAgentConfig, load_config
from evren_code.core.events import EventBus
from evren_code.core.multi_agent.agent_factory import SpecialistAgent
from evren_code.core.multi_agent.agent_registry import AgentRegistry
from evren_code.core.multi_agent.conflict_resolver import ConflictResolver
from evren_code.core.multi_agent.context_sharing import ContextSharingEngine
from evren_code.core.multi_agent.coordinator import MultiAgentCoordinator
from evren_code.core.multi_agent.file_ownership import (
    FILE_OWNERSHIP_VIOLATION,
    FileOwnershipManager,
)
from evren_code.core.multi_agent.models import (
    AgentRole,
    AgentStatus,
    AgentTaskResult,
    EngineeringTask,
    MultiAgentRunConfig,
    TaskStatus,
)
from evren_code.core.multi_agent.persistence import (
    MultiAgentPersistence,
    PersistenceCorruptionError,
)
from evren_code.core.multi_agent.result_aggregator import ResultAggregator
from evren_code.core.multi_agent.scheduler import MultiAgentScheduler
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem
from evren_code.core.security import PermissionManager, ToolExecutor
from evren_code.tools.registry import ToolRegistry


class TestAudit1RealParallelExecution(unittest.TestCase):
    """Test audit1 real parallel execution implementation."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_audit_par_"))
        self.config = load_config(workspace_root=self.tmp_dir)
        self.registry = ToolRegistry()
        self.pm = PermissionManager(self.tmp_dir)
        class _Delegate:
            def request_approval(self, r): return True
            def ask_user_question(self, q, opts=None): return ""
        self.tool_executor = ToolExecutor(self.registry, self.pm, _Delegate())
        self.fom = FileOwnershipManager(self.tmp_dir)
        self.res_agg = ResultAggregator()
        self.cse = ContextSharingEngine()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_two_independent_tasks_execute_concurrently_with_barrier(self):
        """Verify two independent tasks execute concurrently with barrier."""
        barrier = threading.Barrier(2, timeout=3.0)
        task_1_entered = threading.Event()
        task_2_entered = threading.Event()

        class MockBarrierAgent(SpecialistAgent):
            def execute_task(self, task, context_data=None):
                if task.task_id == "t1":
                    task_1_entered.set()
                elif task.task_id == "t2":
                    task_2_entered.set()
                barrier.wait()
                return AgentTaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    role=self.role,
                    success=True,
                    output="Bariyer aşıldı",
                )

        reg = AgentRegistry(self.config, MagicMock(), self.tool_executor, self.registry, self.fom)
        reg._agents["dev_b1"] = MockBarrierAgent("dev_b1", AgentRole.DEVELOPER, self.config, MagicMock(), self.tool_executor, self.registry, self.fom)
        reg._agents["dev_b2"] = MockBarrierAgent("dev_b2", AgentRole.DEVELOPER, self.config, MagicMock(), self.tool_executor, self.registry, self.fom)

        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Görev 1", "", AgentRole.DEVELOPER)
        t2 = EngineeringTask("t2", "Görev 2", "", AgentRole.DEVELOPER)
        graph.add_task(t1)
        graph.add_task(t2)

        scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=reg,
            file_ownership=self.fom,
            context_sharing=self.cse,
            result_aggregator=self.res_agg,
            config=MultiAgentRunConfig(max_parallel_agents=2, auto_review=False),
        )

        ok = scheduler.execute_all()
        self.assertTrue(ok)
        self.assertTrue(task_1_entered.is_set())
        self.assertTrue(task_2_entered.is_set())
        self.assertEqual(graph.get_task("t1").status, TaskStatus.COMPLETED)
        self.assertEqual(graph.get_task("t2").status, TaskStatus.COMPLETED)

    def test_maximum_concurrency_never_exceeded(self):
        """Verify maximum concurrency never exceeded."""
        active_threads = 0
        max_seen = 0
        lock = threading.Lock()

        class MockCountingAgent(SpecialistAgent):
            def execute_task(self, task, context_data=None):
                nonlocal active_threads, max_seen
                with lock:
                    active_threads += 1
                    if active_threads > max_seen:
                        max_seen = active_threads
                time.sleep(0.05)
                with lock:
                    active_threads -= 1
                return AgentTaskResult(task_id=task.task_id, agent_id=self.agent_id, role=self.role, success=True, output="ok")

        reg = AgentRegistry(self.config, MagicMock(), self.tool_executor, self.registry, self.fom)
        for i in range(4):
            aid = f"dev_c_{i}"
            reg._agents[aid] = MockCountingAgent(aid, AgentRole.DEVELOPER, self.config, MagicMock(), self.tool_executor, self.registry, self.fom)

        graph = TaskGraph()
        for i in range(4):
            graph.add_task(EngineeringTask(f"t{i}", f"G_{i}", "", AgentRole.DEVELOPER))

        scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=reg,
            file_ownership=self.fom,
            context_sharing=self.cse,
            result_aggregator=self.res_agg,
            config=MultiAgentRunConfig(max_parallel_agents=2, auto_review=False),
        )

        ok = scheduler.execute_all()
        self.assertTrue(ok)
        self.assertLessEqual(max_seen, 2)
        self.assertEqual(max_seen, 2)

    def test_dependent_task_never_starts_prematurely(self):
        """Verify dependent task never starts prematurely."""
        t1_running = threading.Event()
        t1_can_finish = threading.Event()
        t2_started_while_t1_running = False

        class MockDepAgent(SpecialistAgent):
            def execute_task(self, task, context_data=None):
                nonlocal t2_started_while_t1_running
                if task.task_id == "t1":
                    t1_running.set()
                    t1_can_finish.wait(timeout=2.0)
                elif task.task_id == "t2":
                    if t1_running.is_set() and not t1_can_finish.is_set():
                        t2_started_while_t1_running = True
                return AgentTaskResult(task_id=task.task_id, agent_id=self.agent_id, role=self.role, success=True, output="ok")

        reg = AgentRegistry(self.config, MagicMock(), self.tool_executor, self.registry, self.fom)
        reg._agents["dev_d1"] = MockDepAgent("dev_d1", AgentRole.DEVELOPER, self.config, MagicMock(), self.tool_executor, self.registry, self.fom)
        reg._agents["dev_d2"] = MockDepAgent("dev_d2", AgentRole.DEVELOPER, self.config, MagicMock(), self.tool_executor, self.registry, self.fom)

        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Kök", "", AgentRole.DEVELOPER)
        t2 = EngineeringTask("t2", "Bağımlı", "", AgentRole.DEVELOPER, dependencies=["t1"])
        graph.add_task(t1)
        graph.add_task(t2)

        scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=reg,
            file_ownership=self.fom,
            context_sharing=self.cse,
            result_aggregator=self.res_agg,
            config=MultiAgentRunConfig(max_parallel_agents=2, auto_review=False),
        )

        def _releaser():
            t1_running.wait(timeout=2.0)
            time.sleep(0.02)
            t1_can_finish.set()

        th = threading.Thread(target=_releaser)
        th.start()
        ok = scheduler.execute_all()
        th.join()

        self.assertTrue(ok)
        self.assertFalse(t2_started_while_t1_running)

    def test_failed_task_does_not_corrupt_parallel_running_task(self):
        """Verify failed task does not corrupt parallel running task."""
        class MockMixedAgent(SpecialistAgent):
            def execute_task(self, task, context_data=None):
                if task.task_id == "fail_task":
                    raise RuntimeError("Bilinçli test çökmesi")
                return AgentTaskResult(task_id=task.task_id, agent_id=self.agent_id, role=self.role, success=True, output="başarılı")

        reg = AgentRegistry(self.config, MagicMock(), self.tool_executor, self.registry, self.fom)
        reg._agents["m1"] = MockMixedAgent("m1", AgentRole.DEVELOPER, self.config, MagicMock(), self.tool_executor, self.registry, self.fom)
        reg._agents["m2"] = MockMixedAgent("m2", AgentRole.DEVELOPER, self.config, MagicMock(), self.tool_executor, self.registry, self.fom)

        graph = TaskGraph()
        t1 = EngineeringTask("fail_task", "Çöken Görev", "", AgentRole.DEVELOPER)
        t2 = EngineeringTask("ok_task", "Sağlam Görev", "", AgentRole.DEVELOPER)
        graph.add_task(t1)
        graph.add_task(t2)

        scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=reg,
            file_ownership=self.fom,
            context_sharing=self.cse,
            result_aggregator=self.res_agg,
            config=MultiAgentRunConfig(max_parallel_agents=2, auto_review=False),
        )

        ok = scheduler.execute_all()
        self.assertFalse(ok)
        self.assertEqual(graph.get_task("fail_task").status, TaskStatus.FAILED)
        self.assertEqual(graph.get_task("ok_task").status, TaskStatus.COMPLETED)
        self.assertIn("Bilinçli test çökmesi", graph.get_task("fail_task").result.error)

    def test_task_timeout_handling(self):
        """Verify task timeout handling."""
        class MockHangingAgent(SpecialistAgent):
            def execute_task(self, task, context_data=None):
                time.sleep(1.0)
                return AgentTaskResult(task_id=task.task_id, agent_id=self.agent_id, role=self.role, success=True, output="geç")

        reg = AgentRegistry(self.config, MagicMock(), self.tool_executor, self.registry, self.fom)
        reg._agents["hang_1"] = MockHangingAgent("hang_1", AgentRole.DEVELOPER, self.config, MagicMock(), self.tool_executor, self.registry, self.fom)

        graph = TaskGraph()
        t1 = EngineeringTask("t_hang", "Askıda Görev", "", AgentRole.DEVELOPER, assigned_files=["file.txt"])
        graph.add_task(t1)

        scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=reg,
            file_ownership=self.fom,
            context_sharing=self.cse,
            result_aggregator=self.res_agg,
            config=MultiAgentRunConfig(task_timeout_seconds=0.1, auto_review=False),
        )

        ok = scheduler.execute_all(poll_interval=0.02)
        self.assertFalse(ok)
        self.assertEqual(graph.get_task("t_hang").status, TaskStatus.FAILED)
        self.assertIn("zaman aşımı", graph.get_task("t_hang").result.output.lower())
        for _ in range(50):
            if len(self.fom.get_active_locks()) == 0:
                break
            time.sleep(0.05)
        self.assertEqual(len(self.fom.get_active_locks()), 0)


class TestAudit2FileOwnershipAndRaceConditions(unittest.TestCase):
    """Test audit2 file ownership and race conditions implementation."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_audit_files_"))
        self.fom = FileOwnershipManager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_exclusive_write_lock_conflict(self):
        ok1, _ = self.fom.acquire_locks("t1", "a1", write_files=["app.py"])
        self.assertTrue(ok1)
        ok2, err2 = self.fom.acquire_locks("t2", "a2", write_files=["app.py"])
        self.assertFalse(ok2)
        self.assertIn(FILE_OWNERSHIP_VIOLATION, err2)

    def test_read_only_access_concurrent(self):
        ok1, _ = self.fom.acquire_locks("t1", "a1", write_files=[], read_files=["data.json"])
        ok2, _ = self.fom.acquire_locks("t2", "a2", write_files=[], read_files=["data.json"])
        self.assertTrue(ok1)
        self.assertTrue(ok2)

    def test_out_of_scope_modification_prevented(self):
        self.fom.acquire_locks("t1", "a1", write_files=["allowed.py"])
        can_write, err = self.fom.verify_write_permission("t1", "forbidden.py")
        self.assertFalse(can_write)
        self.assertIn("atanmamış", err)

    def test_path_normalization_windows_posix_bypasses(self):
        self.fom.acquire_locks("t1", "a1", write_files=["src/Module/App.py"])
        can_write, _ = self.fom.verify_write_permission("t1", "src\\module\\app.py")
        self.assertTrue(can_write)

        ok, err = self.fom.acquire_locks("t2", "a2", write_files=["SRC\\MODULE\\APP.PY"])
        self.assertFalse(ok)

    def test_symlink_and_path_traversal_containment(self):
        """Verify symlink and path traversal containment."""
        ok, err = self.fom.acquire_locks("t1", "a1", write_files=["../../outside.py"])
        self.assertFalse(ok)
        self.assertIn(FILE_OWNERSHIP_VIOLATION, err)

        can_write, err2 = self.fom.verify_write_permission("t1", "../../outside.py")
        self.assertFalse(can_write)
        self.assertIn(FILE_OWNERSHIP_VIOLATION, err2)


class TestAudit3ToolExecutorEnforcement(unittest.TestCase):
    """Test audit3 tool executor enforcement implementation."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_audit_tools_"))
        self.config = load_config(workspace_root=self.tmp_dir)
        self.registry = ToolRegistry()
        self.pm = PermissionManager(self.tmp_dir)
        class _Delegate:
            def request_approval(self, r): return True
            def ask_user_question(self, q, opts=None): return ""
        self.tool_executor = ToolExecutor(self.registry, self.pm, _Delegate())
        self.fom = FileOwnershipManager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_agent_cannot_forge_role_or_bypass_permissions(self):
        """Verify agent cannot forge role or bypass permissions."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ParsedAssistantResponse(
            content="",
            tool_calls=[ToolCallItem("tc1", "run_command", {"command": "dir"})],
        )
        agent = SpecialistAgent("arch_test", AgentRole.ARCHITECT, self.config, mock_llm, self.tool_executor, self.registry, self.fom)
        task = EngineeringTask("t1", "Mimari Görev", "", AgentRole.ARCHITECT)
        res = agent.execute_task(task)
        self.assertTrue(res.success)
        self.assertNotIn("tc1", res.artifacts)

    def test_write_multiple_files_strictly_enforces_file_ownership(self):
        """Verify write multiple files strictly enforces file ownership."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ParsedAssistantResponse(
            content="",
            tool_calls=[ToolCallItem("tc1", "write_multiple_files", {
                "files": [
                    {"path": "owned.py", "content": "valid"},
                    {"path": "unauthorized.py", "content": "bypass_attempt"},
                ]
            })],
        )
        agent = SpecialistAgent("dev_wm", AgentRole.DEVELOPER, self.config, mock_llm, self.tool_executor, self.registry, self.fom)
        task = EngineeringTask("t1", "Toplu Yazma", "", AgentRole.DEVELOPER, assigned_files=["owned.py"])
        self.fom.acquire_locks("t1", "dev_wm", write_files=["owned.py"])

        res = agent.execute_task(task)
        self.assertNotIn("unauthorized.py", res.modified_files)
        self.assertNotIn("owned.py", res.modified_files)


class TestAudit4ReviewerGate(unittest.TestCase):
    """Test audit4 reviewer gate implementation."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_audit_rev_"))
        self.config = load_config(workspace_root=self.tmp_dir)
        self.registry = ToolRegistry()
        self.pm = PermissionManager(self.tmp_dir)
        class _Delegate:
            def request_approval(self, r): return True
            def ask_user_question(self, q, opts=None): return ""
        self.tool_executor = ToolExecutor(self.registry, self.pm, _Delegate())
        self.fom = FileOwnershipManager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_reviewer_rejection_marks_task_not_approved(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ParsedAssistantResponse(
            content="Eksik güvenlik kontrolleri tespit edildi. [REJECTED: Güvenlik açığı]"
        )
        reg = AgentRegistry(self.config, mock_llm, self.tool_executor, self.registry, self.fom)
        agg = ResultAggregator(agent_registry=reg)

        task = EngineeringTask("t1", "Auth Ekle", "", AgentRole.DEVELOPER, acceptance_criteria=["Güvenli olmalı"])
        result = AgentTaskResult("t1", "dev_1", AgentRole.DEVELOPER, success=True, output="kod yazıldı", modified_files=["auth.py"])

        res = agg.review_task_result(task, result)
        self.assertFalse(res.review_approved)
        self.assertIn("reddedildi", res.review_notes)

    def test_missing_or_failed_review_cannot_silently_become_approval(self):
        """Verify missing or failed review cannot silently become approval."""
        mock_llm = MagicMock()
        # Model ne onay ne red verdi (ambiguous output)
        mock_llm.chat.return_value = ParsedAssistantResponse(content="Dosyayı gördüm, fena görünmüyor.")
        reg = AgentRegistry(self.config, mock_llm, self.tool_executor, self.registry, self.fom)
        agg = ResultAggregator(agent_registry=reg)

        task = EngineeringTask("t1", "Özellik", "", AgentRole.DEVELOPER, acceptance_criteria=["Testler geçmeli"])
        result = AgentTaskResult("t1", "dev_1", AgentRole.DEVELOPER, success=True, output="ok", modified_files=["test.py"])

        res = agg.review_task_result(task, result)
        self.assertFalse(res.review_approved)
        self.assertIn("onayı verilmedi", res.review_notes)


class TestAudit5CheckpointAndRecovery(unittest.TestCase):
    """Test audit5 checkpoint and recovery implementation."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_audit_rec_"))
        self.config = load_config(workspace_root=self.tmp_dir)
        self.registry = ToolRegistry()
        self.pm = PermissionManager(self.tmp_dir)
        class _Delegate:
            def request_approval(self, r): return True
            def ask_user_question(self, q, opts=None): return ""
        self.tool_executor = ToolExecutor(self.registry, self.pm, _Delegate())
        self.persistence = MultiAgentPersistence(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_corrupted_state_json_raises_persistence_corruption_error(self):
        run_id = "run_corrupted_test"
        rdir = self.persistence._run_dir(run_id)
        state_file = rdir / "state.json"
        state_file.write_text("{ this is corrupted json !!!", encoding="utf-8")

        with self.assertRaises(PersistenceCorruptionError):
            self.persistence.load_checkpoint(run_id)

    def test_recovery_reconciliation_prevents_blind_rerun_of_modified_files(self):
        """Verify recovery reconciliation prevents blind rerun of modified files."""
        run_id = "run_interrupted_test"
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Yarıda Kalan", "", AgentRole.DEVELOPER, assigned_files=["partially_written.py"])
        t1.status = TaskStatus.RUNNING
        t1.created_at = time.time() - 10
        graph.add_task(t1)

        mod_file = self.tmp_dir / "partially_written.py"
        mod_file.write_text("yarım kod", encoding="utf-8")

        self.persistence.save_checkpoint(run_id, graph, metadata={"goal": "Kurtarma testi"})

        coordinator = MultiAgentCoordinator(
            config=self.config,
            llm_client=MagicMock(),
            tool_executor=self.tool_executor,
            registry=self.registry,
        )

        report = coordinator.resume_run(run_id)
        resumed_task = coordinator.current_graph.get_task("t1")
        self.assertEqual(resumed_task.status, TaskStatus.FAILED)
        self.assertIn("çifte yazma engellendi", resumed_task.result.output)


class TestAudit6ConflictResolution(unittest.TestCase):
    """Test audit6 conflict resolution implementation."""

    def test_three_tasks_sequentialization_without_cycles(self):
        resolver = ConflictResolver()
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "Görev 1", "", AgentRole.DEVELOPER, assigned_files=["config.json"])
        t2 = EngineeringTask("t2", "Görev 2", "", AgentRole.DEVELOPER, assigned_files=["config.json"])
        t3 = EngineeringTask("t3", "Görev 3", "", AgentRole.DEVELOPER, assigned_files=["config.json"])
        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        conflicts = resolver.resolve_conflicts(graph, strategy="sequentialize")
        self.assertGreaterEqual(len(conflicts), 1)

        valid, errors = graph.validate_dag()
        self.assertTrue(valid, f"DAG hataları: {errors}")
        sorted_nodes = graph.topological_sort()
        self.assertEqual(len(sorted_nodes), 3)

    def test_failed_sequentialized_task_blocks_downstream(self):
        graph = TaskGraph()
        t1 = EngineeringTask("t1", "İlk Görev", "", AgentRole.DEVELOPER, assigned_files=["common.py"])
        t2 = EngineeringTask("t2", "İkinci Görev", "", AgentRole.DEVELOPER, dependencies=["t1"], assigned_files=["common.py"])
        graph.add_task(t1)
        graph.add_task(t2)

        graph.mark_task_status("t1", TaskStatus.FAILED)
        self.assertEqual(graph.get_task("t2").status, TaskStatus.BLOCKED)


class TestAudit7EndToEndScenario(unittest.TestCase):
    """Test audit7 end to end scenario implementation."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_audit_e2e_"))
        self.config = load_config(workspace_root=self.tmp_dir)
        self.registry = ToolRegistry()
        self.pm = PermissionManager(self.tmp_dir)
        class _Delegate:
            def request_approval(self, r): return True
            def ask_user_question(self, q, opts=None): return ""
        self.tool_executor = ToolExecutor(self.registry, self.pm, _Delegate())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_complete_13_step_orchestration_scenario(self):
        execution_events = []

        mock_llm = MagicMock()
        def _mock_chat(messages, tools_schema=None, model_override=None, stream=False):
            nonlocal execution_events
            last = messages[-1]["content"] if messages else ""
            ts = time.time()

            if "JSON listesi" in last or "plan_init" in last:
                execution_events.append((ts, "PLANNER_GENERATED_DAG"))
                return ParsedAssistantResponse(content=json.dumps([
                    {
                        "task_id": "step_arch",
                        "title": "Veri Modelleri Tasarımı",
                        "description": "Modelleri tasarla",
                        "role": "architect",
                        "dependencies": [],
                        "assigned_files": [],
                        "acceptance_criteria": ["Tasarım net olmalı"],
                    },
                    {
                        "task_id": "step_dev1",
                        "title": "Servis Katmanı",
                        "description": "Servisi kodla",
                        "role": "developer",
                        "dependencies": ["step_arch"],
                        "assigned_files": ["service.py"],
                        "acceptance_criteria": ["Servis yazılmalı"],
                    },
                    {
                        "task_id": "step_dev2",
                        "title": "Router Katmanı",
                        "description": "Router'ı kodla",
                        "role": "developer",
                        "dependencies": ["step_arch"],
                        "assigned_files": ["router.py"],
                        "acceptance_criteria": ["Router yazılmalı"],
                    },
                ]))

            if "Kod İnceleme" in last:
                execution_events.append((ts, "REVIEWER_APPROVED"))
                return ParsedAssistantResponse(content="Kod uygun ve kabul kriterleri eksiksiz karşılandı. [APPROVED]")

            # Developer / Architect
            execution_events.append((ts, "AGENT_TASK_EXECUTED"))
            return ParsedAssistantResponse(content="Uygulama başarıyla tamamlandı.")

        mock_llm.chat.side_effect = _mock_chat

        coordinator = MultiAgentCoordinator(
            config=self.config,
            llm_client=mock_llm,
            tool_executor=self.tool_executor,
            registry=self.registry,
            multi_config=MultiAgentRunConfig(
                enabled=True,
                max_parallel_agents=2,
                auto_review=True,
                verify_with_engine=False,
            ),
        )

        report = coordinator.run_plan(goal="Kullanıcı Servisi ve Yönlendirici Modülü")
        self.assertTrue(report.success)
        self.assertEqual(report.total_tasks, 3)
        self.assertEqual(report.completed_tasks, 3)

        runs = coordinator.persistence.list_runs()
        self.assertGreaterEqual(len(runs), 1)
        run_id = runs[0]["run_id"]

        graph_restored, rep_restored, _ = coordinator.persistence.load_checkpoint(run_id)
        self.assertEqual(len(graph_restored.tasks), 3)
        self.assertEqual(rep_restored.completed_tasks, 3)

        event_types = [e[1] for e in execution_events]
        self.assertIn("PLANNER_GENERATED_DAG", event_types)
        self.assertIn("AGENT_TASK_EXECUTED", event_types)


if __name__ == "__main__":
    unittest.main()
