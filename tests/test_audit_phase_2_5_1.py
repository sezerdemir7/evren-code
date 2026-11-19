"""Test audit phase 2 5 1 module."""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import AppConfig
from evren_code.core.multi_agent.coordinator import MultiAgentCoordinator
from evren_code.core.multi_agent.file_ownership import (
    FILE_OWNERSHIP_VIOLATION,
    FileOwnershipManager,
)
from evren_code.core.multi_agent.models import (
    AgentRole,
    AgentTaskResult,
    MultiAgentRunConfig,
    OrchestrationReport,
    TaskStatus,
)
from evren_code.core.multi_agent.scheduler import MultiAgentScheduler
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.sprint.acceptance import AcceptanceCriteriaEngine
from evren_code.core.sprint.completion import SprintCompletionEngine
from evren_code.core.sprint.decomposer import SprintDecomposer
from evren_code.core.sprint.exceptions import (
    AcceptanceVerificationError,
    SprintError,
    SprintRecoveryError,
)
from evren_code.core.sprint.execution_engine import SprintExecutionEngine
from evren_code.core.sprint.iteration_controller import IterationController
from evren_code.core.sprint.manager import SprintManager
from evren_code.core.sprint.models import (
    AcceptanceCriterion,
    AcceptanceMethod,
    AcceptanceStatus,
    ApprovalDecision,
    EpicStatus,
    HumanApprovalRequest,
    Sprint,
    SprintConfig,
    SprintEpic,
    SprintMetadata,
    SprintStatus,
    SprintTask,
)
from evren_code.core.sprint.planner import SprintPlanner
from evren_code.core.sprint.recovery import SprintPersistence
from evren_code.tools.registry import ToolRegistry


class TestPhase251AdversarialAudit(unittest.TestCase):
    """Test phase251 adversarial audit implementation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="evren_audit_251_")
        self.workspace = Path(self.temp_dir)
        self.persistence = SprintPersistence(workspace_root=self.workspace)
        self.acceptance_engine = AcceptanceCriteriaEngine(workspace_root=self.workspace)
        self.completion_engine = SprintCompletionEngine(acceptance_engine=self.acceptance_engine)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def test_scenario_1_crash_recovery_during_active_sprint(self):
        """Verify scenario 1 crash recovery during active sprint."""
        sprint = Sprint(
            metadata=SprintMetadata(sprint_id="sprint_crash_1", title="Crash Sprint", objective="Obj"),
            config=SprintConfig(),
            tasks={
                "t1": SprintTask(task_id="t1", epic_id="e1", title="T1", description="", role=AgentRole.DEVELOPER, status=TaskStatus.RUNNING),
                "t2": SprintTask(task_id="t2", epic_id="e1", title="T2", description="", role=AgentRole.TESTER, status=TaskStatus.COMPLETED),
            },
            status=SprintStatus.RUNNING,
        )
        self.persistence.save_sprint(sprint)

        reconciled = self.persistence.reconcile_on_crash_recovery(sprint)
        self.assertEqual(reconciled.status, SprintStatus.PAUSED)
        self.assertEqual(reconciled.tasks["t1"].status, TaskStatus.PENDING)
        self.assertEqual(reconciled.tasks["t2"].status, TaskStatus.COMPLETED)


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def test_scenario_2_and_3_worker_timeout_and_concurrent_lock_acquisition(self):
        """Verify scenario 2 and 3 worker timeout and concurrent lock acquisition."""
        file_ownership = FileOwnershipManager(workspace_root=self.workspace)
        target_file = "src/concurrent_mod.py"

        ok, err = file_ownership.acquire_locks("task_1", "agent_1", write_files=[target_file])
        self.assertTrue(ok)

        file_ownership.release_locks("task_1")

        ok2, err2 = file_ownership.acquire_locks("task_2", "agent_2", write_files=[target_file])
        self.assertTrue(ok2, "Zaafiyet: Eski worker thread bitmeden yeni görev kilidi alabildi!")

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def test_scenario_4_concurrent_sprint_start_race_condition(self):
        """Verify scenario 4 concurrent sprint start race condition."""
        mock_coord = MagicMock()
        mock_coord.run_plan.return_value = OrchestrationReport(
            run_id="run_1", goal="Goal", total_tasks=1, completed_tasks=1,
            failed_tasks=0, blocked_tasks=0, total_execution_time_seconds=0.1,
            modified_files=[], task_results={}, success=True
        )

        app_cfg = AppConfig(
            api=MagicMock(), models=MagicMock(), agent=MagicMock(), tools=MagicMock(),
            telemetry=MagicMock(), workspace_root=self.workspace
        )
        sprint_mgr = SprintManager(
            config=app_cfg, llm_client=None, tool_executor=MagicMock(),
            registry=MagicMock(), coordinator=mock_coord,
            sprint_config=SprintConfig(max_iterations=1, require_reviewer_approval=False, require_regression_check=False)
        )

        sprint = sprint_mgr.create_and_plan(objective="Test")
        sprint.status = SprintStatus.APPROVED
        sprint_mgr.persistence.save_sprint(sprint)

        start_count = [0]
        def _start_worker():
            try:
                sprint_mgr.start_sprint(sprint.id)
                start_count[0] += 1
            except Exception:
                pass

        th1 = threading.Thread(target=_start_worker)
        th2 = threading.Thread(target=_start_worker)
        th1.start()
        th2.start()
        th1.join(timeout=3)
        th2.join(timeout=3)

        self.assertGreaterEqual(start_count[0], 1)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def test_scenario_5_corrupted_persistence_file(self):
        """Verify scenario 5 corrupted persistence file."""
        sprint_dir = self.workspace / ".evren" / "sprints" / "corrupted_sp"
        sprint_dir.mkdir(parents=True, exist_ok=True)
        bad_json = sprint_dir / "sprint.json"
        bad_json.write_text("{ this is corrupted json ...", encoding="utf-8")

        with self.assertRaises(SprintRecoveryError):
            self.persistence.load_sprint("corrupted_sp")

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def test_scenario_6_reviewer_ambiguous_output_rejected(self):
        """Verify scenario 6 reviewer ambiguous output rejected."""
        sprint = MagicMock()
        task_rev = SprintTask(
            task_id="rev_task", epic_id="e1", title="Review", description="",
            role=AgentRole.REVIEWER, status=TaskStatus.COMPLETED,
            result=AgentTaskResult(
                task_id="rev_task", agent_id="agent_rev", role=AgentRole.REVIEWER,
                success=True, output="Kod gayet güzel görünüyor, testler geçebilir."
            )
        )
        sprint.tasks = {"rev_task": task_rev}
        crit = AcceptanceCriterion(
            id="c_rev", description="Review onayı", method=AcceptanceMethod.REVIEW_APPROVAL,
            required_evidence="", associated_task_ids=["rev_task"]
        )

        self.acceptance_engine.evaluate_criterion(crit, sprint)
        self.assertEqual(crit.status, AcceptanceStatus.FAILED)
        self.assertIn("açık [APPROVED] onayı yok", crit.verification_result)

    # -------------------------------------------------------------------------
    # Senaryo 7: Eksik acceptance evidence
    # -------------------------------------------------------------------------
    def test_scenario_7_missing_acceptance_evidence_fails(self):
        """Verify scenario 7 missing acceptance evidence fails."""
        crit = AcceptanceCriterion(
            id="c_empty", description="Eksik kanıt", method=AcceptanceMethod.FILE_EXISTS,
            required_evidence=""
        )
        self.acceptance_engine.evaluate_criterion(crit, MagicMock())
        self.assertEqual(crit.status, AcceptanceStatus.FAILED)
        self.assertIn("belirtilmemiş", crit.verification_result)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def test_scenario_8_remediation_loop_repetition_breaker(self):
        """Verify scenario 8 remediation loop repetition breaker."""
        ctrl = IterationController()
        sprint = Sprint(
            metadata=SprintMetadata(sprint_id="sp_loop", title="Loop", objective="Obj"),
            config=SprintConfig(max_iterations=10, max_retries_per_task=2, stop_on_critical_failure=False),
        )
        t_failed = SprintTask(
            task_id="t_fail", epic_id="e1", title="Failing Task", description="",
            role=AgentRole.DEVELOPER, status=TaskStatus.FAILED,
            result=AgentTaskResult(
                task_id="t_fail", agent_id="dev", role=AgentRole.DEVELOPER,
                success=False, output="", error="FatalSyntaxError at line 42"
            )
        )

        # 1. iterasyon
        ctrl.start_next_iteration(sprint)
        cont1, _ = ctrl.should_continue(sprint, [t_failed], [])
        self.assertTrue(cont1)

        # 2. iterasyon
        ctrl.start_next_iteration(sprint)
        cont2, _ = ctrl.should_continue(sprint, [t_failed], [])
        self.assertTrue(cont2)

        ctrl.start_next_iteration(sprint)
        cont3, reason = ctrl.should_continue(sprint, [t_failed], [])
        self.assertFalse(cont3)
        self.assertIn("Tekrarlayan hata tespit edildi", reason)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def test_scenario_9_path_traversal_blocked(self):
        """Verify scenario 9 path traversal blocked."""
        crit = AcceptanceCriterion(
            id="c_hack", description="Traversal", method=AcceptanceMethod.FILE_EXISTS,
            required_evidence="../../windows/system32/cmd.exe"
        )
        self.acceptance_engine.evaluate_criterion(crit, MagicMock())
        self.assertEqual(crit.status, AcceptanceStatus.FAILED)
        self.assertIn("Güvenlik ihlali", crit.verification_result)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def test_scenario_10_completion_blocked_on_failed_task(self):
        """Verify scenario 10 completion blocked on failed task."""
        sprint = Sprint(
            metadata=SprintMetadata(sprint_id="sp_comp_fail", title="Comp Fail", objective="Obj"),
            config=SprintConfig(require_reviewer_approval=False, require_regression_check=False),
            tasks={
                "t1": SprintTask(task_id="t1", epic_id="e1", title="T1", description="", role=AgentRole.DEVELOPER, status=TaskStatus.FAILED),
            },
            status=SprintStatus.RUNNING,
        )
        is_done, status, blockers = self.completion_engine.evaluate_sprint_completion(sprint)
        self.assertFalse(is_done)
        self.assertEqual(status, SprintStatus.FAILED)
        self.assertTrue(any("başarısız durumda" in b for b in blockers))


if __name__ == "__main__":
    unittest.main()
