"""Test sprint hardening module."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import AppConfig
from evren_code.core.multi_agent.agent_factory import SpecialistAgent
from evren_code.core.multi_agent.agent_registry import AgentRegistry
from evren_code.core.multi_agent.context_sharing import ContextSharingEngine
from evren_code.core.multi_agent.coordinator import MultiAgentCoordinator
from evren_code.core.multi_agent.file_ownership import FileOwnershipManager
from evren_code.core.multi_agent.models import (
    AgentRole,
    AgentTaskResult,
    EngineeringTask,
    MultiAgentRunConfig,
    TaskStatus,
)
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.multi_agent.result_aggregator import ResultAggregator
from evren_code.core.multi_agent.scheduler import MultiAgentScheduler
from evren_code.core.security import ToolExecutor
from evren_code.core.sprint.acceptance import AcceptanceCriteriaEngine
from evren_code.core.sprint.completion import SprintCompletionEngine
from evren_code.core.sprint.events import SprintEvent, SprintEventType
from evren_code.core.sprint.exceptions import SprintExecutionError
from evren_code.core.sprint.execution_engine import SprintExecutionEngine
from evren_code.core.sprint.iteration_controller import IterationController
from evren_code.core.sprint.manager import SprintManager
from evren_code.core.sprint.models import (
    AcceptanceCriterion,
    AcceptanceMethod,
    AcceptanceStatus,
    Sprint,
    SprintConfig,
    SprintMetadata,
    SprintStatus,
    SprintTask,
)
from evren_code.core.sprint.planner import SprintPlanner
from evren_code.core.sprint.recovery import SprintPersistence
from evren_code.tools.registry import ToolRegistry


class TestSprintHardeningWorkerTimeoutAndLock(unittest.TestCase):
    """TestSprintHardeningWorkerTimeoutAndLock: Senaryo 1, 2, 3."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="evren_test_hard_timeout_")
        self.workspace = Path(self.temp_dir)
        self.fom = FileOwnershipManager(self.workspace)
        self.cse = ContextSharingEngine(self.workspace)
        self.res_agg = ResultAggregator()
        self.registry = ToolRegistry()
        self.app_cfg = AppConfig(
            api=MagicMock(),
            models=MagicMock(),
            agent=MagicMock(),
            tools=MagicMock(),
            telemetry=MagicMock(),
            workspace_root=self.workspace,
        )
        self.tool_executor = ToolExecutor(
            registry=self.registry,
            permission_manager=MagicMock(),
            approval_delegate=MagicMock(),
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_timeout_retains_lock_until_worker_exits(self):
        """Verify timeout retains lock until worker exits."""
        worker_can_finish = threading.Event()

        class HangingAgent(SpecialistAgent):
            def execute_task(self, task, context_data=None):
                worker_can_finish.wait(timeout=2.0)
                return AgentTaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    role=self.role,
                    success=True,
                    output="Bitti",
                )

        reg = AgentRegistry(self.app_cfg, MagicMock(), self.tool_executor, self.registry, self.fom)
        reg._agents["hang_worker"] = HangingAgent(
            "hang_worker", AgentRole.DEVELOPER, self.app_cfg, MagicMock(), self.tool_executor, self.registry, self.fom
        )

        graph = TaskGraph()
        t1 = EngineeringTask("t_hang", "Askıda Görev", "", AgentRole.DEVELOPER, assigned_files=["src/app.py"])
        graph.add_task(t1)

        # 0.1 saniye timeout
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

        active_locks = self.fom.get_active_locks()
        self.assertGreaterEqual(len(active_locks), 1, "Zaman aşımı sonrası çalışan thread bitmeden kilit serbest bırakılmamalı!")

        worker_can_finish.set()
        time.sleep(0.1)

    def test_lock_released_after_worker_finishes(self):
        """Verify lock released after worker finishes."""
        worker_can_finish = threading.Event()

        class ControlledAgent(SpecialistAgent):
            def execute_task(self, task, context_data=None):
                worker_can_finish.wait(timeout=2.0)
                return AgentTaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    role=self.role,
                    success=True,
                    output="Sonlandı",
                )

        reg = AgentRegistry(self.app_cfg, MagicMock(), self.tool_executor, self.registry, self.fom)
        reg._agents["worker_2"] = ControlledAgent(
            "worker_2", AgentRole.DEVELOPER, self.app_cfg, MagicMock(), self.tool_executor, self.registry, self.fom
        )

        graph = TaskGraph()
        t1 = EngineeringTask("t_ctrl", "Kontrollü Görev", "", AgentRole.DEVELOPER, assigned_files=["src/db.py"])
        graph.add_task(t1)

        scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=reg,
            file_ownership=self.fom,
            context_sharing=self.cse,
            result_aggregator=self.res_agg,
            config=MultiAgentRunConfig(task_timeout_seconds=0.1, auto_review=False),
        )

        scheduler.execute_all(poll_interval=0.02)
        self.assertEqual(graph.get_task("t_ctrl").status, TaskStatus.FAILED)

        worker_can_finish.set()

        for _ in range(50):
            if len(self.fom.get_active_locks()) == 0:
                break
            time.sleep(0.02)

        self.assertEqual(len(self.fom.get_active_locks()), 0, "Worker sonlandıktan sonra kilit serbest bırakılmalı!")

    def test_second_worker_blocked_on_same_files_after_timeout(self):
        """Verify second worker blocked on same files after timeout."""
        worker_can_finish = threading.Event()

        class BlockTestAgent(SpecialistAgent):
            def execute_task(self, task, context_data=None):
                worker_can_finish.wait(timeout=2.0)
                return AgentTaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    role=self.role,
                    success=True,
                    output="İşlem tamam",
                )

        reg = AgentRegistry(self.app_cfg, MagicMock(), self.tool_executor, self.registry, self.fom)
        reg._agents["w_first"] = BlockTestAgent(
            "w_first", AgentRole.DEVELOPER, self.app_cfg, MagicMock(), self.tool_executor, self.registry, self.fom
        )

        graph = TaskGraph()
        t1 = EngineeringTask("t_timeout", "Zaman Aşımı Görevi", "", AgentRole.DEVELOPER, assigned_files=["src/shared.py"])
        graph.add_task(t1)

        scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=reg,
            file_ownership=self.fom,
            context_sharing=self.cse,
            result_aggregator=self.res_agg,
            config=MultiAgentRunConfig(task_timeout_seconds=0.1, auto_review=False),
        )

        scheduler.execute_all(poll_interval=0.02)
        self.assertEqual(graph.get_task("t_timeout").status, TaskStatus.FAILED)

        acquired, err = self.fom.acquire_locks(
            task_id="t_second",
            agent_id="agent_intruder",
            write_files=["src/shared.py"],
        )
        self.assertFalse(acquired, "Zaman aşımına uğramış ama arka planda devam eden worker'ın dosyası başkasına tahsis edilemez!")
        self.assertTrue(any(w in err.lower() for w in ("kilit", "violation")))

        # Worker bitir
        worker_can_finish.set()
        time.sleep(0.05)


class TestSprintHardeningLifecycleAndStateTransitions(unittest.TestCase):
    """TestSprintHardeningLifecycleAndStateTransitions: Senaryo 4, 5, 6, 7."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="evren_test_hard_lifecycle_")
        self.workspace = Path(self.temp_dir)
        self.app_cfg = AppConfig(
            api=MagicMock(),
            models=MagicMock(),
            agent=MagicMock(),
            tools=MagicMock(),
            telemetry=MagicMock(),
            workspace_root=self.workspace,
        )
        self.manager = SprintManager(
            config=self.app_cfg,
            llm_client=None,
            tool_executor=MagicMock(),
            registry=MagicMock(),
            sprint_config=SprintConfig(enabled=True, max_iterations=2),
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_start_prevented(self):
        """Verify concurrent start prevented."""
        sprint = self.manager.create_and_plan("Hedef", "Eşzamanlılık Testi")
        self.manager.approve_sprint(sprint.id)

        with self.manager._execution_lock:
            self.manager._running_sprints.add(sprint.id)

        with self.assertRaises(SprintExecutionError) as ctx:
            self.manager.start_sprint(sprint.id)
        self.assertIn("zaten aktif olarak", str(ctx.exception))

        # Temizlik
        self.manager._running_sprints.discard(sprint.id)

    def test_draft_or_unapproved_planned_start_prevented(self):
        """Verify draft or unapproved planned start prevented."""
        sprint = Sprint(
            metadata=SprintMetadata("s_draft", "Onaysız Sprint", "Hedef"),
            config=SprintConfig(),
            status=SprintStatus.DRAFT,
        )
        self.manager.persistence.save_sprint(sprint)

        with self.assertRaises(SprintExecutionError) as ctx:
            self.manager.start_sprint(sprint.id)
        self.assertIn("DRAFT durumundaki sprint", str(ctx.exception))

        # PLANNED yap ama onaylama
        sprint.status = SprintStatus.PLANNED
        self.manager.active_sprint = sprint
        self.manager.persistence.save_sprint(sprint)

        with self.assertRaises(SprintExecutionError) as ctx:
            self.manager.start_sprint(sprint.id)
        self.assertIn("PLANNED durumundaki sprint", str(ctx.exception))

    def test_pause_sprint_breaks_loop(self):
        """Verify pause sprint breaks loop."""
        sprint = self.manager.create_and_plan("Pause Hedefi", "Pause Testi")
        self.manager.approve_sprint(sprint.id)

        self.manager.execution_engine.request_pause(sprint.id)

        executed = self.manager.execution_engine.run_sprint(sprint)
        self.assertEqual(executed.status, SprintStatus.PAUSED)

    def test_cancel_sprint_aborts_loop(self):
        """Verify cancel sprint aborts loop."""
        sprint = self.manager.create_and_plan("Cancel Hedefi", "Cancel Testi")
        self.manager.approve_sprint(sprint.id)

        self.manager.execution_engine.request_cancel(sprint.id)

        executed = self.manager.execution_engine.run_sprint(sprint)
        self.assertEqual(executed.status, SprintStatus.CANCELLED)


class TestSprintHardeningCrashRecoveryAndReconciliation(unittest.TestCase):
    """TestSprintHardeningCrashRecoveryAndReconciliation: Senaryo 8, 9."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="evren_test_hard_recovery_")
        self.workspace = Path(self.temp_dir)
        self.persistence = SprintPersistence(self.workspace)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_recovery_detects_partial_modifications_and_fails_dirty_tasks(self):
        """Verify recovery detects partial modifications and fails dirty tasks."""
        src_file = self.workspace / "src" / "worker.py"
        src_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.write_text("def clean(): pass\n", encoding="utf-8")

        sprint = Sprint(
            metadata=SprintMetadata("sprint_rec_1", "Kurtarma Testi", "Hedef"),
            config=SprintConfig(),
            status=SprintStatus.RUNNING,
        )

        task = SprintTask(
            task_id="task_dirty",
            epic_id="epic_1",
            title="Kısmi Değişiklik Görevi",
            description="Dosya düzenleme",
            role=AgentRole.DEVELOPER,
            assigned_files=["src/worker.py"],
            status=TaskStatus.RUNNING,
            metadata={
                "file_snapshots": {
                    "src/worker.py": self.persistence.get_file_fingerprint("src/worker.py"),
                }
            },
        )
        sprint.tasks[task.task_id] = task

        src_file.write_text("def dirty_broken_syntax( \n", encoding="utf-8")

        reconciled_sprint = self.persistence.reconcile_on_crash_recovery(sprint)

        rec_task = reconciled_sprint.tasks["task_dirty"]
        self.assertEqual(rec_task.status, TaskStatus.FAILED)
        self.assertIsNotNone(rec_task.result)
        self.assertIn("partial/dirty disk modifications", rec_task.result.error)

    def test_recovery_preserves_user_pre_existing_files(self):
        """Verify recovery preserves user pre existing files."""
        user_file = self.workspace / "important_data.csv"
        initial_content = "id,name,role\n1,Alice,Admin\n2,Bob,User\n"
        user_file.write_text(initial_content, encoding="utf-8")

        sprint = Sprint(
            metadata=SprintMetadata("sprint_rec_2", "Veri Koruma Testi", "Hedef"),
            config=SprintConfig(),
            status=SprintStatus.RUNNING,
        )

        task = SprintTask(
            task_id="task_clean",
            epic_id="epic_1",
            title="Temiz Görev",
            description="Hiç dokunulmadı",
            role=AgentRole.DEVELOPER,
            assigned_files=["important_data.csv"],
            status=TaskStatus.RUNNING,
            metadata={
                "file_snapshots": {
                    "important_data.csv": self.persistence.get_file_fingerprint("important_data.csv"),
                }
            },
        )
        sprint.tasks[task.task_id] = task

        reconciled = self.persistence.reconcile_on_crash_recovery(sprint)

        self.assertTrue(user_file.exists())
        self.assertEqual(user_file.read_text(encoding="utf-8"), initial_content)

        self.assertEqual(reconciled.tasks["task_clean"].status, TaskStatus.PENDING)


class TestSprintHardeningAcceptanceAndCompletion(unittest.TestCase):
    """TestSprintHardeningAcceptanceAndCompletion: Senaryo 10, 11, 12, 13, 16."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="evren_test_hard_acc_")
        self.workspace = Path(self.temp_dir)
        self.registry = ToolRegistry()
        self.app_cfg = AppConfig(
            api=MagicMock(),
            models=MagicMock(),
            agent=MagicMock(),
            tools=MagicMock(),
            telemetry=MagicMock(),
            workspace_root=self.workspace,
        )
        self.tool_executor = ToolExecutor(
            registry=self.registry,
            permission_manager=MagicMock(),
            approval_delegate=MagicMock(),
        )
        self.acceptance_engine = AcceptanceCriteriaEngine(
            workspace_root=self.workspace,
            tool_executor=self.tool_executor,
        )
        self.completion_engine = SprintCompletionEngine(
            acceptance_engine=self.acceptance_engine,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_success_matches_real_command_outcome(self):
        """Verify build success matches real command outcome."""
        sprint = Sprint(metadata=SprintMetadata("s_build", "Build Test", "Hedef"), config=SprintConfig())

        crit_ok = AcceptanceCriterion(
            id="c_build_ok",
            description="Derleme başarılı",
            method=AcceptanceMethod.BUILD_SUCCESS,
            required_evidence='python -c "exit(0)"',
        )

        mock_exec = MagicMock()
        mock_exec.execute.return_value = "Çıkış Kodu (Exit Code): 0\n--- STDOUT ---\nBuild OK"
        engine = AcceptanceCriteriaEngine(self.workspace, tool_executor=mock_exec)

        res_ok = engine.evaluate_criterion(crit_ok, sprint)
        self.assertEqual(res_ok.status, AcceptanceStatus.PASSED)
        self.assertEqual(res_ok.evidence_data.get("exit_code"), 0)

        crit_fail = AcceptanceCriterion(
            id="c_build_fail",
            description="Derleme hatalı",
            method=AcceptanceMethod.BUILD_SUCCESS,
            required_evidence='python -c "exit(1)"',
        )
        mock_exec.execute.return_value = "Çıkış Kodu (Exit Code): 1\n--- STDERR ---\nSyntaxError"
        res_fail = engine.evaluate_criterion(crit_fail, sprint)
        self.assertEqual(res_fail.status, AcceptanceStatus.FAILED)
        self.assertEqual(res_fail.evidence_data.get("exit_code"), 1)

    def test_test_result_matches_real_command_outcome(self):
        """Verify test result matches real command outcome."""
        sprint = Sprint(metadata=SprintMetadata("s_test", "Test Test", "Hedef"), config=SprintConfig())
        crit = AcceptanceCriterion(
            id="c_test_fail",
            description="Testler başarısız",
            method=AcceptanceMethod.TEST_RESULT,
            required_evidence="pytest",
        )

        mock_exec = MagicMock()
        mock_exec.execute.return_value = "Çıkış Kodu (Exit Code): 2\n--- STDERR ---\n2 failed, 1 passed"
        engine = AcceptanceCriteriaEngine(self.workspace, tool_executor=mock_exec)

        res = engine.evaluate_criterion(crit, sprint)
        self.assertEqual(res.status, AcceptanceStatus.FAILED)
        self.assertEqual(res.evidence_data.get("exit_code"), 2)

    def test_evidence_less_criteria_fail_closed(self):
        """Verify evidence less criteria fail closed."""
        sprint = Sprint(metadata=SprintMetadata("s_ev_less", "Kanıtsız Kriter Testi", "Hedef"), config=SprintConfig())
        crit_generic = AcceptanceCriterion(
            id="crit_empty",
            description="Kanıtsız ve ilişkisiz kriter",
            method=AcceptanceMethod.BUILD_SUCCESS,
            required_evidence="",
            associated_task_ids=[],
        )

        engine = AcceptanceCriteriaEngine(self.workspace, tool_executor=None, verification_engine=None)
        res = engine.evaluate_criterion(crit_generic, sprint)
        self.assertEqual(res.status, AcceptanceStatus.FAILED, "Kanıt ve ilişkili görev yoksa kriter FAILED olmalı!")

    def test_missing_reviewer_task_blocks_sprint_completion(self):
        """Verify missing reviewer task blocks sprint completion."""
        sprint = Sprint(
            metadata=SprintMetadata("s_rev", "Reviewer Kapı Testi", "Hedef"),
            config=SprintConfig(require_reviewer_approval=True),
        )
        t_dev = SprintTask("t_dev", "epic_1", "Geliştirme", "", AgentRole.DEVELOPER, status=TaskStatus.COMPLETED)
        t_dev.result = AgentTaskResult("t_dev", "agent_1", AgentRole.DEVELOPER, success=True, output="Tamamlandı")
        sprint.tasks[t_dev.task_id] = t_dev

        is_completed, final_status, blockers = self.completion_engine.evaluate_sprint_completion(sprint)
        self.assertFalse(is_completed)
        self.assertEqual(final_status, SprintStatus.FAILED)
        self.assertTrue(any("reviewer" in b.lower() for b in blockers))

    def test_incomplete_criteria_block_sprint_completion(self):
        """Verify incomplete criteria block sprint completion."""
        sprint = Sprint(
            metadata=SprintMetadata("s_crit", "Kriter Blokaj Testi", "Hedef"),
            config=SprintConfig(require_reviewer_approval=False),
        )
        crit = AcceptanceCriterion(
            id="c_missing_file",
            description="Var olmayan dosya",
            method=AcceptanceMethod.FILE_EXISTS,
            required_evidence="src/non_existent.py",
        )
        sprint.acceptance_criteria[crit.id] = crit

        is_completed, final_status, blockers = self.completion_engine.evaluate_sprint_completion(sprint)
        self.assertFalse(is_completed)
        self.assertEqual(final_status, SprintStatus.FAILED)
        self.assertTrue(any("kriter" in b.lower() for b in blockers))


class TestSprintHardeningRepetitionBreakerAndTimeout(unittest.TestCase):
    """TestSprintHardeningRepetitionBreakerAndTimeout: origin_task_id."""

    def test_origin_task_id_repetition_breaker(self):
        """Verify origin task id repetition breaker."""
        controller = IterationController()
        sprint = Sprint(
            metadata=SprintMetadata("s_rep", "Döngü Kırıcı Testi", "Hedef"),
            config=SprintConfig(max_retries_per_task=2, max_iterations=10),
            current_iteration=1,
        )

        origin_id = "task_root_dev"
        t1 = SprintTask(
            task_id="fix_iter_1_abc",
            epic_id="epic_1",
            title="Düzeltme 1",
            description="",
            role=AgentRole.DEVELOPER,
            origin_task_id=origin_id,
            status=TaskStatus.FAILED,
        )
        t1.result = AgentTaskResult("fix_iter_1_abc", "ag_1", AgentRole.DEVELOPER, False, "Hata", error="ModuleNotFoundError: foo")

        can_continue_1, _ = controller.should_continue(sprint, [t1], [])
        self.assertTrue(can_continue_1)

        t2 = SprintTask(
            task_id="fix_iter_2_xyz",
            epic_id="epic_1",
            title="Düzeltme 2",
            description="",
            role=AgentRole.DEVELOPER,
            origin_task_id=origin_id,
            status=TaskStatus.FAILED,
        )
        t2.result = AgentTaskResult("fix_iter_2_xyz", "ag_1", AgentRole.DEVELOPER, False, "Hata", error="ModuleNotFoundError: foo")

        can_continue_2, _ = controller.should_continue(sprint, [t2], [])
        self.assertTrue(can_continue_2)

        t3 = SprintTask(
            task_id="fix_iter_3_ijk",
            epic_id="epic_1",
            title="Düzeltme 3",
            description="",
            role=AgentRole.DEVELOPER,
            origin_task_id=origin_id,
            status=TaskStatus.FAILED,
        )
        t3.result = AgentTaskResult("fix_iter_3_ijk", "ag_1", AgentRole.DEVELOPER, False, "Hata", error="ModuleNotFoundError: foo")

        can_continue_3, reason = controller.should_continue(sprint, [t3], [])
        self.assertFalse(can_continue_3, "Yeni task_id'ler origin_task_id üzerinden tekrarlayan hata denetimine takılmalı!")
        self.assertIn("Tekrarlayan hata tespit edildi", reason)
        self.assertIn(origin_id, reason)

    def test_wall_clock_sprint_timeout_enforced(self):
        """Verify wall clock sprint timeout enforced."""
        controller = IterationController()
        sprint = Sprint(
            metadata=SprintMetadata("s_clock", "Duvar Saati Testi", "Hedef", started_at=time.time() - 3700.0),
            config=SprintConfig(sprint_timeout_seconds=3600.0),
            current_iteration=1,
        )

        t_dummy = SprintTask("t_d", "e_1", "Görev", "", AgentRole.DEVELOPER, status=TaskStatus.FAILED)
        can_continue, reason = controller.should_continue(sprint, [t_dummy], [])
        self.assertFalse(can_continue)
        self.assertIn("duvar saati zaman aşımına", reason)


if __name__ == "__main__":
    unittest.main()
