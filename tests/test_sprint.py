"""Test sprint module."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import AppConfig, SprintAppConfig
from evren_code.core.events import EventBus
from evren_code.core.multi_agent.coordinator import MultiAgentCoordinator
from evren_code.core.multi_agent.models import (
    AgentRole,
    AgentTaskResult,
    EngineeringTask,
    OrchestrationReport,
    TaskStatus,
)
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.security import ToolExecutor
from evren_code.core.sprint.acceptance import AcceptanceCriteriaEngine
from evren_code.core.sprint.completion import SprintCompletionEngine
from evren_code.core.sprint.decomposer import SprintDecomposer
from evren_code.core.sprint.events import SprintEvent, SprintEventType
from evren_code.core.sprint.exceptions import (
    AcceptanceVerificationError,
    ApprovalRequiredError,
    SprintError,
    SprintPlanningError,
    SprintValidationError,
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
from evren_code.core.sprint.progress_tracker import (
    ProgressTracker,
    SprintProgressMetrics,
)
from evren_code.core.sprint.recovery import SprintPersistence
from evren_code.core.sprint.reporting import SprintReporter
from evren_code.tools.registry import ToolRegistry


class TestSprintModels(unittest.TestCase):
    """Test sprint models implementation."""

    def test_sprint_task_to_and_from_engineering_task(self):
        st = SprintTask(
            task_id="task_1",
            epic_id="epic_core",
            title="Kullanıcı Doğrulama",
            description="JWT doğrulaması ekle",
            role=AgentRole.DEVELOPER,
            dependencies=["task_0"],
            assigned_files=["src/auth.py"],
            read_only_files=["src/models.py"],
            acceptance_criteria=["crit_jwt"],
            estimated_complexity="high",
            priority=1,
            status=TaskStatus.PENDING,
        )

        eng_task = st.to_engineering_task()
        self.assertIsInstance(eng_task, EngineeringTask)
        self.assertEqual(eng_task.task_id, "task_1")
        self.assertEqual(eng_task.role, AgentRole.DEVELOPER)
        self.assertEqual(eng_task.metadata["epic_id"], "epic_core")
        self.assertEqual(eng_task.metadata["estimated_complexity"], "high")

        restored_st = SprintTask.from_engineering_task(eng_task)
        self.assertEqual(restored_st.task_id, st.task_id)
        self.assertEqual(restored_st.epic_id, "epic_core")
        self.assertEqual(restored_st.assigned_files, ["src/auth.py"])
        self.assertEqual(restored_st.estimated_complexity, "high")

    def test_sprint_full_serialization(self):
        meta = SprintMetadata(
            sprint_id="sp_test_1",
            title="Test Sprint",
            objective="Kimlik doğrulama modülünü tamamla",
            scope=["auth.py"],
            out_of_scope=["billing.py"],
        )
        cfg = SprintConfig(max_iterations=4, max_tasks_per_sprint=15)
        epic = SprintEpic(id="ep_1", title="Epik 1", description="Giriş", task_ids=["t1"])
        task = SprintTask(
            task_id="t1",
            epic_id="ep_1",
            title="Token üret",
            description="JWT üretimi",
            role=AgentRole.DEVELOPER,
        )
        crit = AcceptanceCriterion(
            id="c1",
            description="auth.py var olmalı",
            method=AcceptanceMethod.FILE_EXISTS,
            required_evidence="auth.py",
        )
        appr = HumanApprovalRequest(
            approval_id="appr_1",
            title="Migration onayı",
            description="Veritabanı migration",
            operation_type="db_migration",
        )

        sprint = Sprint(
            metadata=meta,
            config=cfg,
            epics={"ep_1": epic},
            tasks={"t1": task},
            acceptance_criteria={"c1": crit},
            approval_requests={"appr_1": appr},
            status=SprintStatus.PLANNED,
        )

        s_dict = sprint.to_dict()
        restored = Sprint.from_dict(s_dict)

        self.assertEqual(restored.id, "sp_test_1")
        self.assertEqual(restored.status, SprintStatus.PLANNED)
        self.assertEqual(len(restored.epics), 1)
        self.assertEqual(len(restored.tasks), 1)
        self.assertEqual(len(restored.acceptance_criteria), 1)
        self.assertEqual(len(restored.approval_requests), 1)


class TestSprintDecomposer(unittest.TestCase):
    """Test sprint decomposer implementation."""

    def setUp(self):
        self.decomposer = SprintDecomposer(max_tasks=10)

    def test_parse_valid_json_decomposition(self):
        raw = json.dumps({
            "epics": [
                {"id": "epic_api", "title": "API Geliştirme", "description": "REST uçları"}
            ],
            "tasks": [
                {
                    "task_id": "task_api_1",
                    "epic_id": "epic_api",
                    "title": "Login Ucu",
                    "description": "POST /login hazırla",
                    "role": "developer",
                    "dependencies": [],
                    "assigned_files": ["src/api/auth.py"],
                    "acceptance_criteria": ["crit_login_test"],
                }
            ],
            "acceptance_criteria": [
                {
                    "id": "crit_login_test",
                    "description": "Login testi geçmeli",
                    "method": "test_result",
                    "required_evidence": "pytest tests/test_auth.py",
                    "associated_task_ids": ["task_api_1"],
                }
            ],
        })

        epics, tasks, criteria = self.decomposer.parse_decomposition_response(raw)
        self.assertEqual(len(epics), 1)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(criteria), 1)
        self.assertIn("task_api_1", epics["epic_api"].task_ids)

        errors = self.decomposer.validate_decomposition(epics, tasks, criteria)
        self.assertEqual(errors, [])

    def test_validate_detects_dag_cycle(self):
        raw = json.dumps({
            "epics": [{"id": "ep1", "title": "Epik 1", "description": "Test"}],
            "tasks": [
                {"task_id": "t1", "epic_id": "ep1", "title": "T1", "dependencies": ["t2"], "role": "developer"},
                {"task_id": "t2", "epic_id": "ep1", "title": "T2", "dependencies": ["t1"], "role": "developer"},
            ],
            "acceptance_criteria": [],
        })
        epics, tasks, criteria = self.decomposer.parse_decomposition_response(raw)
        errors = self.decomposer.validate_decomposition(epics, tasks, criteria)
        self.assertTrue(any("Çevrim" in e or "DAG" in e for e in errors))

    def test_validate_detects_missing_epic_and_max_tasks(self):
        decomposer = SprintDecomposer(max_tasks=2)
        raw = json.dumps({
            "epics": [{"id": "ep1", "title": "Epik 1", "description": "Test"}],
            "tasks": [
                {"task_id": "t1", "epic_id": "ep_missing", "title": "T1", "role": "developer"},
                {"task_id": "t2", "epic_id": "ep1", "title": "T2", "role": "developer"},
                {"task_id": "t3", "epic_id": "ep1", "title": "T3", "role": "developer"},
            ],
            "acceptance_criteria": [],
        })
        epics, tasks, criteria = decomposer.parse_decomposition_response(raw)
        self.assertEqual(len(tasks), 3)
        errors = decomposer.validate_decomposition(epics, tasks, criteria)
        self.assertTrue(any("maksimum sınırı" in e for e in errors))


class TestSprintPlanner(unittest.TestCase):
    """Test sprint planner implementation."""

    def test_offline_fallback_plan(self):
        planner = SprintPlanner(llm_client=None)
        sprint = planner.plan_sprint(
            objective="CLI komut geçmişi ekle",
            title="Geçmiş Ekleme Sprinti",
            scope=["cli.py"],
        )

        self.assertEqual(sprint.status, SprintStatus.PLANNED)
        self.assertGreaterEqual(len(sprint.epics), 1)
        self.assertGreaterEqual(len(sprint.tasks), 3)
        self.assertIn("task_dev", sprint.tasks)
        self.assertIn("task_test", sprint.tasks)
        self.assertIn("task_review", sprint.tasks)
        self.assertGreaterEqual(len(sprint.acceptance_criteria), 2)

    def test_plan_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps({
            "epics": [{"id": "epic_db", "title": "DB Güncelleme", "description": "Şema"}],
            "tasks": [
                {
                    "task_id": "task_schema",
                    "epic_id": "epic_db",
                    "title": "Şema hazırla",
                    "description": "SQL tabloları",
                    "role": "developer",
                    "dependencies": [],
                    "assigned_files": ["db/schema.sql"],
                    "acceptance_criteria": ["c_file"],
                }
            ],
            "acceptance_criteria": [
                {
                    "id": "c_file",
                    "description": "Şema dosyası varlığı",
                    "method": "file_exists",
                    "required_evidence": "db/schema.sql",
                    "associated_task_ids": ["task_schema"],
                }
            ],
        })

        planner = SprintPlanner(llm_client=mock_llm)
        sprint = planner.plan_sprint(objective="Veritabanı oluştur")
        self.assertEqual(sprint.status, SprintStatus.PLANNED)
        self.assertIn("epic_db", sprint.epics)
        self.assertIn("task_schema", sprint.tasks)
        self.assertIn("c_file", sprint.acceptance_criteria)

    def test_remediation_planning(self):
        planner = SprintPlanner(llm_client=None)
        sprint = planner.plan_sprint(objective="Örnek Sprint")

        failed_task = SprintTask(
            task_id="t_failed",
            epic_id=next(iter(sprint.epics.keys())),
            title="Başarısız Görev",
            description="Hata verdi",
            role=AgentRole.DEVELOPER,
            status=TaskStatus.FAILED,
        )
        failed_crit = AcceptanceCriterion(
            id="c_failed",
            description="Test geçmedi",
            method=AcceptanceMethod.TEST_RESULT,
            required_evidence="pytest",
            status=AcceptanceStatus.FAILED,
            verification_result="AssertionError",
        )

        remediation_tasks = planner.plan_remediation(
            sprint=sprint,
            failed_tasks=[failed_task],
            failed_criteria=[failed_crit],
            iteration=1,
        )
        self.assertGreaterEqual(len(remediation_tasks), 1)
        self.assertIn("t_failed", remediation_tasks[0].dependencies)


class TestAcceptanceCriteriaEngine(unittest.TestCase):
    """Test acceptance criteria engine implementation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="evren_test_acc_")
        self.workspace = Path(self.temp_dir)
        self.engine = AcceptanceCriteriaEngine(workspace_root=self.workspace)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_file_exists_verification(self):
        crit = AcceptanceCriterion(
            id="c_file",
            description="config.json var olmalı",
            method=AcceptanceMethod.FILE_EXISTS,
            required_evidence="config.json",
        )
        dummy_sprint = MagicMock()
        self.engine.evaluate_criterion(crit, dummy_sprint)
        self.assertEqual(crit.status, AcceptanceStatus.FAILED)

        test_file = self.workspace / "config.json"
        test_file.write_text("{\"app\": \"evren\"}", encoding="utf-8")
        self.engine.evaluate_criterion(crit, dummy_sprint)
        self.assertEqual(crit.status, AcceptanceStatus.PASSED)
        self.assertTrue(crit.evidence_data.get("exists"))

    def test_file_exists_rejects_path_traversal(self):
        crit = AcceptanceCriterion(
            id="c_traversal",
            description="Gizli sistem dosyası",
            method=AcceptanceMethod.FILE_EXISTS,
            required_evidence="../../outside.txt",
        )
        dummy_sprint = MagicMock()
        self.engine.evaluate_criterion(crit, dummy_sprint)
        self.assertEqual(crit.status, AcceptanceStatus.FAILED)
        self.assertIn("Güvenlik ihlali", crit.verification_result)

    def test_symbol_exists_fallback_text_search(self):
        py_file = self.workspace / "auth.py"
        py_file.write_text("def authenticate_user(token):\n    return True\n", encoding="utf-8")

        crit = AcceptanceCriterion(
            id="c_sym",
            description="authenticate_user fonksiyonu",
            method=AcceptanceMethod.SYMBOL_EXISTS,
            required_evidence="auth.py:authenticate_user",
        )
        dummy_sprint = MagicMock()
        self.engine.evaluate_criterion(crit, dummy_sprint)
        self.assertEqual(crit.status, AcceptanceStatus.PASSED)
        self.assertEqual(crit.evidence_data.get("symbol"), "authenticate_user")

    def test_review_approval_verification(self):
        sprint = MagicMock()
        task_rev = SprintTask(
            task_id="rev_1",
            epic_id="ep1",
            title="Review",
            description="Rev",
            role=AgentRole.REVIEWER,
            status=TaskStatus.COMPLETED,
            result=AgentTaskResult(
                task_id="rev_1",
                agent_id="agent_rev",
                role=AgentRole.REVIEWER,
                success=True,
                output="Kod fena değil ama [REJECTED] eksik var.",
            ),
        )
        sprint.tasks = {"rev_1": task_rev}
        crit = AcceptanceCriterion(
            id="c_rev",
            description="Review onayı",
            method=AcceptanceMethod.REVIEW_APPROVAL,
            required_evidence="",
            associated_task_ids=["rev_1"],
        )

        self.engine.evaluate_criterion(crit, sprint)
        self.assertEqual(crit.status, AcceptanceStatus.FAILED)

        task_rev.result.output = "Değişiklikler incelendi. [APPROVED] Her şey standartlara uygun."
        self.engine.evaluate_criterion(crit, sprint)
        self.assertEqual(crit.status, AcceptanceStatus.PASSED)

    def test_manual_approval_verification(self):
        sprint = MagicMock()
        crit = AcceptanceCriterion(
            id="c_manual",
            description="Veritabanı silme onayı",
            method=AcceptanceMethod.MANUAL_APPROVAL,
            required_evidence="",
            associated_task_ids=["task_db"],
        )

        sprint.approval_requests = {}
        self.engine.evaluate_criterion(crit, sprint)
        self.assertEqual(crit.status, AcceptanceStatus.REQUIRES_USER_INPUT)

        appr = HumanApprovalRequest(
            approval_id="appr_db",
            title="DB onayı",
            description="DB",
            operation_type="db",
            task_id="task_db",
            decision=ApprovalDecision.APPROVED,
        )
        sprint.approval_requests = {"appr_db": appr}
        self.engine.evaluate_criterion(crit, sprint)
        self.assertEqual(crit.status, AcceptanceStatus.PASSED)


class TestProgressAndIteration(unittest.TestCase):
    """Test progress and iteration implementation."""

    def test_progress_metrics_and_ascii_dashboard(self):
        meta = SprintMetadata(sprint_id="sp_dash", title="Dashboard Test", objective="Test")
        cfg = SprintConfig(max_iterations=3)
        epic = SprintEpic(id="ep1", title="Epik 1", description="Test", task_ids=["t1", "t2"])
        t1 = SprintTask(task_id="t1", epic_id="ep1", title="T1", description="", role=AgentRole.DEVELOPER, status=TaskStatus.COMPLETED)
        t2 = SprintTask(task_id="t2", epic_id="ep1", title="T2", description="", role=AgentRole.TESTER, status=TaskStatus.PENDING)
        c1 = AcceptanceCriterion(id="c1", description="C1", method=AcceptanceMethod.FILE_EXISTS, required_evidence="", status=AcceptanceStatus.PASSED)

        sprint = Sprint(
            metadata=meta,
            config=cfg,
            epics={"ep1": epic},
            tasks={"t1": t1, "t2": t2},
            acceptance_criteria={"c1": c1},
            status=SprintStatus.RUNNING,
            current_iteration=1,
        )

        tracker = ProgressTracker()
        metrics = tracker.compute_metrics(sprint)
        self.assertEqual(metrics.total_tasks, 2)
        self.assertEqual(metrics.completed_tasks, 1)
        self.assertEqual(metrics.total_criteria, 1)
        self.assertEqual(metrics.passed_criteria, 1)
        self.assertGreater(metrics.completion_percentage, 0.0)

        dashboard = tracker.render_ascii_dashboard(sprint)
        self.assertIn("EVREN SPRINT", dashboard)
        self.assertIn("sp_dash", dashboard)
        self.assertIn("Epik 1", dashboard)

    def test_iteration_controller_repetition_breaker(self):
        ctrl = IterationController()
        sprint = Sprint(
            metadata=SprintMetadata(sprint_id="sp_rep", title="Repetition Test", objective="Test"),
            config=SprintConfig(max_iterations=5, max_retries_per_task=2),
            status=SprintStatus.RUNNING,
        )

        failed_task = SprintTask(
            task_id="t_bug",
            epic_id="ep1",
            title="Sürekli aynı hatayı veren görev",
            description="",
            role=AgentRole.DEVELOPER,
            status=TaskStatus.FAILED,
            result=AgentTaskResult(
                task_id="t_bug",
                agent_id="agent_dev",
                role=AgentRole.DEVELOPER,
                success=False,
                output="",
                error="IndexError: list index out of range",
            ),
        )

        # 1. Deneme
        ctrl.start_next_iteration(sprint)
        cont, _ = ctrl.should_continue(sprint, [failed_task], [])
        self.assertTrue(cont)

        ctrl.start_next_iteration(sprint)
        cont, _ = ctrl.should_continue(sprint, [failed_task], [])
        self.assertTrue(cont)

        ctrl.start_next_iteration(sprint)
        cont, reason = ctrl.should_continue(sprint, [failed_task], [])
        self.assertFalse(cont)
        self.assertIn("Tekrarlayan hata tespit edildi", reason)


class TestSprintPersistenceAndRecovery(unittest.TestCase):
    """Test sprint persistence and recovery implementation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="evren_test_persist_")
        self.workspace = Path(self.temp_dir)
        self.persistence = SprintPersistence(workspace_root=self.workspace)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_sprint(self):
        meta = SprintMetadata(sprint_id="sp_recov_1", title="Kalıcılık Testi", objective="Obj")
        sprint = Sprint(
            metadata=meta,
            config=SprintConfig(),
            epics={"ep1": SprintEpic(id="ep1", title="Ep1", description="", task_ids=["t1"])},
            tasks={"t1": SprintTask(task_id="t1", epic_id="ep1", title="T1", description="", role=AgentRole.DEVELOPER)},
            acceptance_criteria={"c1": AcceptanceCriterion(id="c1", description="C1", method=AcceptanceMethod.FILE_EXISTS, required_evidence="")},
            status=SprintStatus.PLANNED,
        )

        self.persistence.save_sprint(sprint)
        loaded = self.persistence.load_sprint("sp_recov_1")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.id, "sp_recov_1")
        self.assertEqual(loaded.metadata.title, "Kalıcılık Testi")
        self.assertIn("t1", loaded.tasks)

        sprints_list = self.persistence.list_sprints()
        self.assertEqual(len(sprints_list), 1)
        self.assertEqual(sprints_list[0]["sprint_id"], "sp_recov_1")

    def test_crash_recovery_reconciliation(self):
        meta = SprintMetadata(sprint_id="sp_crash", title="Crash Test", objective="Obj")
        t_running = SprintTask(
            task_id="t_run",
            epic_id="ep1",
            title="Çöken Görev",
            description="",
            role=AgentRole.DEVELOPER,
            status=TaskStatus.RUNNING,
        )
        sprint = Sprint(
            metadata=meta,
            config=SprintConfig(),
            tasks={"t_run": t_running},
            status=SprintStatus.RUNNING,
        )

        reconciled = self.persistence.reconcile_on_crash_recovery(sprint)
        self.assertEqual(reconciled.status, SprintStatus.PAUSED)
        self.assertEqual(reconciled.tasks["t_run"].status, TaskStatus.PENDING)


class TestSprintCompletionEngine(unittest.TestCase):
    """Test sprint completion engine implementation."""

    def test_completion_gates(self):
        mock_acc = MagicMock()
        completion_engine = SprintCompletionEngine(acceptance_engine=mock_acc)

        sprint = Sprint(
            metadata=SprintMetadata(sprint_id="sp_gate", title="Gate", objective="Obj"),
            config=SprintConfig(require_reviewer_approval=False),
            tasks={"t1": SprintTask(task_id="t1", epic_id="ep", title="T1", description="", role=AgentRole.DEVELOPER, status=TaskStatus.FAILED)},
            status=SprintStatus.RUNNING,
        )
        is_done, status, blockers = completion_engine.evaluate_sprint_completion(sprint)
        self.assertFalse(is_done)
        self.assertEqual(status, SprintStatus.FAILED)
        self.assertTrue(any("başarısız durumda" in b for b in blockers))

        sprint.tasks["t1"].status = TaskStatus.COMPLETED
        crit = AcceptanceCriterion(id="c1", description="C1", method=AcceptanceMethod.FILE_EXISTS, required_evidence="", status=AcceptanceStatus.FAILED)
        sprint.acceptance_criteria = {"c1": crit}
        is_done, status, blockers = completion_engine.evaluate_sprint_completion(sprint)
        self.assertFalse(is_done)
        self.assertEqual(status, SprintStatus.FAILED)

        crit.status = AcceptanceStatus.PASSED
        is_done, status, reasons = completion_engine.evaluate_sprint_completion(sprint)
        self.assertTrue(is_done)
        self.assertEqual(status, SprintStatus.COMPLETED)


class TestSprintEndToEndLifecycle(unittest.TestCase):
    """Test sprint end to end lifecycle implementation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="evren_test_e2e_sprint_")
        self.workspace = Path(self.temp_dir)

        # Mock coordinator
        self.mock_coord = MagicMock()
        def _mock_run_plan(goal, graph=None, on_update=None, run_id=None):
            if graph:
                for t in graph.tasks.values():
                    t.status = TaskStatus.COMPLETED
                    t.result = AgentTaskResult(
                        task_id=t.task_id,
                        agent_id="agent_dev",
                        role=t.role,
                        success=True,
                        output="Geliştirme tamamlandı. [APPROVED] Mimari doğrulandı.",
                        modified_files=["src/feature.py"],
                    )
                    if on_update:
                        on_update(t)
            rep = OrchestrationReport(
                run_id=run_id or "run_mock",
                goal=goal,
                total_tasks=len(graph.tasks) if graph else 0,
                completed_tasks=len(graph.tasks) if graph else 0,
                failed_tasks=0,
                blocked_tasks=0,
                total_execution_time_seconds=0.1,
                modified_files=["src/feature.py"],
                task_results={},
                success=True,
                summary="Mock run success",
            )
            return rep

        self.mock_coord.run_plan.side_effect = _mock_run_plan

        app_cfg = AppConfig(
            api=MagicMock(),
            models=MagicMock(),
            agent=MagicMock(),
            tools=MagicMock(),
            telemetry=MagicMock(),
            workspace_root=self.workspace,
        )

        self.sprint_manager = SprintManager(
            config=app_cfg,
            llm_client=None,
            tool_executor=MagicMock(),
            registry=MagicMock(),
            coordinator=self.mock_coord,
            sprint_config=SprintConfig(
                enabled=True,
                max_iterations=3,
                require_reviewer_approval=True,
                require_regression_check=False,
            ),
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_e2e_autonomous_sprint_execution(self):
        # 1. Planlama
        sprint = self.sprint_manager.create_and_plan(
            objective="Yeni loglama modülünü sisteme ekle",
            title="Loglama Modülü Sprinti",
            scope=["src/logger.py"],
        )
        self.assertEqual(sprint.status, SprintStatus.PLANNED)
        self.assertGreaterEqual(len(sprint.tasks), 1)

        approved = self.sprint_manager.approve_sprint(sprint.id)
        self.assertTrue(approved)
        self.assertEqual(sprint.status, SprintStatus.APPROVED)

        (self.workspace / "src").mkdir(parents=True, exist_ok=True)
        (self.workspace / "src" / "logger.py").write_text("class Logger:\n    pass\n", encoding="utf-8")

        executed_sprint = self.sprint_manager.start_sprint(sprint.id)

        self.assertIn(executed_sprint.status, (SprintStatus.COMPLETED, SprintStatus.COMPLETED_WITH_WARNINGS))
        self.assertGreaterEqual(executed_sprint.current_iteration, 1)

        self.assertIsNotNone(executed_sprint.report)
        md_rep = self.sprint_manager.get_markdown_report(executed_sprint.id)
        self.assertIn("Sprint Teslimat Raporu", md_rep)
        self.assertIn("Loglama Modülü Sprinti", md_rep)
        self.assertIn("src/feature.py", md_rep)


if __name__ == "__main__":
    unittest.main()
