"""Test tool execution reliability module."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import load_config
from evren_code.core.events import EventBus, EventType
from evren_code.core.multi_agent.agent_factory import SpecialistAgent
from evren_code.core.multi_agent.agent_registry import AgentRegistry
from evren_code.core.multi_agent.context_sharing import ContextSharingEngine
from evren_code.core.multi_agent.file_ownership import FileOwnershipManager
from evren_code.core.multi_agent.models import (
    AgentRole,
    AgentTaskResult,
    EngineeringTask,
    MultiAgentRunConfig,
    TaskStatus,
)
from evren_code.core.multi_agent.result_aggregator import ResultAggregator
from evren_code.core.multi_agent.scheduler import MultiAgentScheduler
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem
from evren_code.core.security import PermissionManager, PermissionMode, ToolExecutor
from evren_code.core.tool_reliability import (
    BatchExecutionManager,
    BoundedRetryManager,
    CancellationManager,
    CentralizedToolExecutionManager,
    ExecutionLifecycleStatus,
    ExecutionWatchdog,
    HeartbeatMonitor,
    OutputSizeLimiter,
    ToolCancelledError,
    ToolPolicyRegistry,
    ToolTimeoutPolicy,
    get_current_cancellation_token,
    report_tool_progress,
)
from evren_code.tools.filesystem import register_filesystem_tools
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.shell import execute_command_with_tree_kill


class _AutoApproveDelegate:
    def request_approval(self, request) -> bool:
        return True

    def ask_user_question(self, question: str, options=None) -> str:
        return options[0] if options else "evet"


class TestToolExecutionReliabilitySuite(unittest.TestCase):
    """Test tool execution reliability suite implementation."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_rel_test_"))
        self.config = load_config(workspace_root=self.tmp_dir)
        self.event_bus = EventBus()
        self.registry = ToolRegistry()
        self.pm = PermissionManager(self.tmp_dir, initial_mode=PermissionMode.BYPASS)
        self.executor = ToolExecutor(
            registry=self.registry,
            permission_manager=self.pm,
            approval_delegate=_AutoApproveDelegate(),
            event_bus=self.event_bus,
            max_output_chars=4000,
        )

    def tearDown(self) -> None:
        try:
            self.executor.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. Never-returning tool
    # ------------------------------------------------------------------
    def test_01_never_returning_tool_times_out_cleanly(self) -> None:
        """Verify 01 never returning tool times out cleanly."""
        stop_bg = threading.Event()

        def never_returning_tool() -> str:
            stop_bg.wait(timeout=10.0)
            return "asla_donmemeli"

        self.registry.register(
            name="infinite_wait_tool",
            description="Sonsuz bekleyen test aracı",
            parameters={"type": "object", "properties": {}},
            handler=never_returning_tool,
        )
        self.executor.execution_manager.policy_registry.register_policy(
            ToolTimeoutPolicy(
                tool_name="infinite_wait_tool",
                timeout_seconds=0.2,
                max_retries=0,
                idempotent=False,
            )
        )

        t0 = time.perf_counter()
        res = self.executor.execute("infinite_wait_tool", {})
        elapsed = time.perf_counter() - t0
        stop_bg.set()

        self.assertLess(elapsed, 1.5, "Never-returning tool zaman aşımı sınırında kesilmeliydi!")
        self.assertTrue(res.startswith("Hata:"))
        self.assertTrue("zaman aşımı" in res.lower() or "tamamlanamadı" in res.lower())

        records = self.executor.execution_manager.get_recent_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, ExecutionLifecycleStatus.TIMED_OUT)
        self.assertTrue(records[0].timed_out)

    # ------------------------------------------------------------------
    # 2. Slow tool
    # ------------------------------------------------------------------
    def test_02_slow_tool_respects_policy_and_heartbeat(self) -> None:
        """Verify 02 slow tool respects policy and heartbeat."""
        heartbeats_received = []
        self.event_bus.subscribe(
            lambda ev: heartbeats_received.append(ev.data),
            EventType.TOOL_HEARTBEAT,
        )

        def slow_working_tool(sleep_sec: float = 0.15) -> str:
            for step in range(3):
                report_tool_progress(f"step_{step + 1}", completed_units=step + 1, total_units=3)
                time.sleep(sleep_sec / 3.0)
            return "yavas_ama_basarili"

        self.registry.register(
            name="slow_tool",
            description="Yavaş test aracı",
            parameters={"type": "object", "properties": {"sleep_sec": {"type": "number"}}},
            handler=slow_working_tool,
        )
        self.executor.execution_manager.policy_registry.register_policy(
            ToolTimeoutPolicy(
                tool_name="slow_tool",
                timeout_seconds=0.8,
                max_retries=0,
                idempotent=True,
            )
        )

        ok_res = self.executor.execute("slow_tool", {"sleep_sec": 0.15})
        self.assertEqual(ok_res, "yavas_ama_basarili")
        self.assertGreaterEqual(len(heartbeats_received), 3)

        fail_res = self.executor.execute("slow_tool", {"sleep_sec": 1.0}, timeout=0.15)
        self.assertIn("Hata:", fail_res)

    # ------------------------------------------------------------------
    # 3. Tool exception
    # ------------------------------------------------------------------
    def test_03_tool_exception_isolated_and_recorded(self) -> None:
        """Verify 03 tool exception isolated and recorded."""
        def crashing_tool(x: int) -> str:
            raise ValueError(f"Geçersiz girdi değeri: {x}")

        self.registry.register(
            name="crashing_tool",
            description="Hata fırlatan araç",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            handler=crashing_tool,
        )

        res = self.executor.execute("crashing_tool", {"x": 42})
        self.assertIn("Hata: 'crashing_tool' aracı çalışırken beklenmeyen bir istisna oluştu", res)
        self.assertIn("Geçersiz girdi değeri: 42", res)

        rec = self.executor.execution_manager.get_recent_records()[-1]
        self.assertEqual(rec.status, ExecutionLifecycleStatus.FAILED)
        self.assertIn("42", rec.error or "")

    # ------------------------------------------------------------------
    # 4. Partial batch failure (read_multiple_files & BatchExecutionManager)
    # ------------------------------------------------------------------
    def test_04_partial_batch_failure_isolates_errors_per_file(self) -> None:
        """Verify 04 partial batch failure isolates errors per file."""
        register_filesystem_tools(self.registry, self.config)

        f1 = self.tmp_dir / "valid_one.py"
        f2 = self.tmp_dir / "valid_two.py"
        f1.write_text("print('one')\n", encoding="utf-8")
        f2.write_text("print('two')\n", encoding="utf-8")

        res = self.executor.execute(
            "read_multiple_files",
            {
                "paths": [
                    "valid_one.py",
                    "missing_nonexistent.py",
                    "../../outside_forbidden.py",
                    "valid_two.py",
                ]
            },
        )

        self.assertIn("Toplu Dosya Okuma (4 dosya):", res)
        self.assertIn("print('one')", res)
        self.assertIn("print('two')", res)
        self.assertIn("missing_nonexistent.py", res)
        self.assertIn("outside_forbidden.py", res)

    # ------------------------------------------------------------------
    # 5. Large file / tool output
    # ------------------------------------------------------------------
    def test_05_large_output_size_limiter_prevents_context_explosion(self) -> None:
        """Verify 05 large output size limiter prevents context explosion."""
        def huge_output_tool() -> str:
            head_marker = "HEAD_MARKER_START_" + ("A" * 2000)
            middle_noise = "M" * 50000
            tail_marker = ("Z" * 2000) + "_TAIL_MARKER_END"
            return head_marker + middle_noise + tail_marker

        self.registry.register(
            name="huge_output_tool",
            description="Devasa çıktı üreten araç",
            parameters={"type": "object", "properties": {}},
            handler=huge_output_tool,
        )

        res = self.executor.execute("huge_output_tool", {})
        self.assertIn("HEAD_MARKER_START_", res)
        self.assertIn("_TAIL_MARKER_END", res)
        self.assertIn("Araç çıktısı bağlam taşmasını önlemek için kısaltıldı", res)
        self.assertLess(len(res), 6000)

        rec = self.executor.execution_manager.get_recent_records()[-1]
        self.assertTrue(rec.output_truncated)

    # ------------------------------------------------------------------
    # 6. Subprocess hang
    # ------------------------------------------------------------------
    def test_06_subprocess_hang_killed_on_tool_timeout(self) -> None:
        """Verify 06 subprocess hang killed on tool timeout."""
        pid_file = self.tmp_dir / "child_pid.txt"

        def hanging_subprocess_tool() -> str:
            code = (
                "import os, time, pathlib; "
                f"pathlib.Path(r'{pid_file}').write_text(str(os.getpid()), encoding='utf-8'); "
                "time.sleep(30)"
            )
            rc, out, err = execute_command_with_tree_kill(
                args=[sys.executable, "-c", code],
                cwd=str(self.tmp_dir),
                timeout=30,
                shell=False,
            )
            return f"rc={rc} out={out} err={err}"

        self.registry.register(
            name="hanging_subprocess_tool",
            description="Alt süreç askıda bırakan araç",
            parameters={"type": "object", "properties": {}},
            handler=hanging_subprocess_tool,
        )

        t0 = time.perf_counter()
        res = self.executor.execute("hanging_subprocess_tool", {}, timeout=0.4)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 3.0)
        self.assertIn("Hata:", res)

        if pid_file.is_file():
            child_pid = int(pid_file.read_text(encoding="utf-8").strip())
            time.sleep(0.2)
            is_alive = False
            if os.name == "nt":
                chk = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {child_pid}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                is_alive = str(child_pid) in (chk.stdout or "")
            else:
                try:
                    os.kill(child_pid, 0)
                    is_alive = True
                except OSError:
                    is_alive = False
            self.assertFalse(is_alive, f"Alt süreç (PID={child_pid}) timeout sonrası öldürülmeliydi!")

    # ------------------------------------------------------------------
    # 7. Async cancellation
    # ------------------------------------------------------------------
    def test_07_async_tool_execution_and_cancellation(self) -> None:
        """Verify 07 async tool execution and cancellation."""
        cancelled_inside_coro = threading.Event()

        async def async_tool(mode: str = "fast") -> str:
            try:
                if mode == "fast":
                    await asyncio.sleep(0.02)
                    return "async_basarili"
                await asyncio.sleep(10.0)
                return "async_zaman_asimi_olmamali"
            except asyncio.CancelledError:
                cancelled_inside_coro.set()
                raise

        self.registry.register(
            name="async_tool",
            description="Asenkron araç",
            parameters={"type": "object", "properties": {"mode": {"type": "string"}}},
            handler=async_tool,  # type: ignore[arg-type]
        )

        ok_res = self.executor.execute("async_tool", {"mode": "fast"}, timeout=1.0)
        self.assertEqual(ok_res, "async_basarili")

        timeout_res = self.executor.execute("async_tool", {"mode": "hang"}, timeout=0.2)
        self.assertIn("Hata:", timeout_res)
        self.assertTrue(cancelled_inside_coro.wait(timeout=1.0))

    # ------------------------------------------------------------------
    # 8. Concurrent tool execution
    # ------------------------------------------------------------------
    def test_08_concurrent_tool_execution_one_hangs_others_succeed(self) -> None:
        """Verify 08 concurrent tool execution one hangs others succeed."""
        unblock = threading.Event()

        def mixed_parallel_tool(name: str) -> str:
            if name == "slow_hang":
                unblock.wait(timeout=5.0)
                return "gec_dondu"
            return f"hizli_sonuc_{name}"

        self.registry.register(
            name="read_file",
            description="Paralel okuma testi",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}},
            handler=mixed_parallel_tool,
        )
        self.executor.execution_manager.policy_registry.register_policy(
            ToolTimeoutPolicy(
                tool_name="read_file",
                timeout_seconds=0.25,
                max_retries=0,
                idempotent=False,
            )
        )

        results: dict[str, str] = {}

        def _worker(key: str) -> None:
            results[key] = self.executor.execute("read_file", {"name": key})

        threads = [
            threading.Thread(target=_worker, args=("fast_1",)),
            threading.Thread(target=_worker, args=("fast_2",)),
            threading.Thread(target=_worker, args=("slow_hang",)),
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        elapsed = time.perf_counter() - t0
        unblock.set()

        self.assertLess(elapsed, 1.5)
        self.assertEqual(results.get("fast_1"), "hizli_sonuc_fast_1")
        self.assertEqual(results.get("fast_2"), "hizli_sonuc_fast_2")
        self.assertTrue(results.get("slow_hang", "").startswith("Hata:"))

    # ------------------------------------------------------------------
    # 9. Watchdog intervention (Stall detection)
    # ------------------------------------------------------------------
    def test_09_watchdog_intervenes_on_stalled_tool(self) -> None:
        """Verify 09 watchdog intervenes on stalled tool."""
        def stalled_tool() -> str:
            tok = get_current_cancellation_token()
            for _ in range(50):
                if tok and tok.is_cancelled:
                    tok.check_cancelled()
                time.sleep(0.03)
            return "tamamlandi"

        self.registry.register(
            name="stalled_tool",
            description="Heartbeat üretmeden donan araç",
            parameters={"type": "object", "properties": {}},
            handler=stalled_tool,
        )
        # Wall-clock timeout 5.0sn ama stall_timeout 0.15sn!
        self.executor.execution_manager.policy_registry.register_policy(
            ToolTimeoutPolicy(
                tool_name="stalled_tool",
                timeout_seconds=5.0,
                stall_timeout_seconds=0.15,
                max_retries=0,
                idempotent=False,
            )
        )

        t0 = time.perf_counter()
        res = self.executor.execute("stalled_tool", {})
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 1.5, "Watchdog stall timeout 5sn beklemeden müdahale etmeliydi!")
        self.assertIn("Hata:", res)
        rec = self.executor.execution_manager.get_recent_records()[-1]
        self.assertTrue(rec.watchdog_intervened)
        self.assertEqual(rec.status, ExecutionLifecycleStatus.TIMED_OUT)
        self.assertGreaterEqual(self.executor.execution_manager.watchdog.interventions_count, 1)

    # ------------------------------------------------------------------
    # 10. Retry exhaustion & Idempotency enforcement
    # ------------------------------------------------------------------
    def test_10_bounded_retry_exhaustion_and_non_idempotent_guard(self) -> None:
        """Verify 10 bounded retry exhaustion and non idempotent guard."""
        idempotent_calls = 0
        non_idempotent_calls = 0

        def flaky_idempotent() -> str:
            nonlocal idempotent_calls
            idempotent_calls += 1
            raise ConnectionError("Temporary connection reset by peer")

        def flaky_non_idempotent() -> str:
            nonlocal non_idempotent_calls
            non_idempotent_calls += 1
            raise ConnectionError("Temporary connection reset by peer")

        self.registry.register(
            name="flaky_read",
            description="Geçici hata veren okuma aracı",
            parameters={"type": "object", "properties": {}},
            handler=flaky_idempotent,
        )
        self.registry.register(
            name="flaky_write",
            description="Geçici hata veren yazma aracı",
            parameters={"type": "object", "properties": {}},
            handler=flaky_non_idempotent,
        )

        self.executor.execution_manager.policy_registry.register_policy(
            ToolTimeoutPolicy(
                tool_name="flaky_read",
                timeout_seconds=1.0,
                max_retries=2,
                idempotent=True,
                retry_base_delay=0.02,
            )
        )
        self.executor.execution_manager.policy_registry.register_policy(
            ToolTimeoutPolicy(
                tool_name="flaky_write",
                timeout_seconds=1.0,
                max_retries=3,
                idempotent=False,
                retry_base_delay=0.02,
            )
        )

        res_read = self.executor.execute("flaky_read", {})
        self.assertIn("Hata:", res_read)
        self.assertEqual(idempotent_calls, 3, "1 ilk deneme + 2 retry = 3 çağrı yapılmalıydı.")

        res_write = self.executor.execute("flaky_write", {})
        self.assertIn("Hata:", res_write)
        self.assertEqual(non_idempotent_calls, 1, "İdempotent olmayan araç ASLA tekrarlanmamalı!")

    # ------------------------------------------------------------------
    # 11. DAG recovery & Orphan lock livelock prevention
    # ------------------------------------------------------------------
    def test_11_dag_recovery_on_timeout_and_orphan_lock_reclaim(self) -> None:
        """Verify 11 dag recovery on timeout and orphan lock reclaim."""
        fom = FileOwnershipManager(self.tmp_dir)
        cse = ContextSharingEngine()
        res_agg = ResultAggregator()
        attempts_by_task: dict[str, int] = {}

        class RecoveringAgent(SpecialistAgent):
            def execute_task(self, task, context_data=None):
                cnt = attempts_by_task.get(task.task_id, 0) + 1
                attempts_by_task[task.task_id] = cnt
                if task.task_id == "t1" and cnt == 1:
                    if task.cancel_event:
                        task.cancel_event.wait(timeout=1.0)
                    else:
                        time.sleep(0.5)
                    return AgentTaskResult(
                        task_id=task.task_id,
                        agent_id=self.agent_id,
                        role=self.role,
                        success=False,
                        output="timeout",
                    )
                return AgentTaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    role=self.role,
                    success=True,
                    output=f"başarılı_deneme_{cnt}",
                )

        reg = AgentRegistry(self.config, MagicMock(), self.executor, self.registry, fom)
        reg._agents["dev_rec_1"] = RecoveringAgent(
            "dev_rec_1", AgentRole.DEVELOPER, self.config, MagicMock(), self.executor, self.registry, fom
        )
        reg._agents["dev_rec_2"] = RecoveringAgent(
            "dev_rec_2", AgentRole.DEVELOPER, self.config, MagicMock(), self.executor, self.registry, fom
        )

        graph = TaskGraph()
        t1 = EngineeringTask(
            "t1",
            "Kurtarılacak Kök Görev",
            "İlk denemede yavaş, ikinci denemede hızlı",
            AgentRole.DEVELOPER,
            assigned_files=["shared.py"],
            metadata={"recover_on_timeout": True},
        )
        t2 = EngineeringTask(
            "t2",
            "Bağımlı Görev",
            "t1 tamamlanınca çalışır",
            AgentRole.DEVELOPER,
            dependencies=["t1"],
            assigned_files=["shared.py"],
        )
        graph.add_task(t1)
        graph.add_task(t2)

        run_cfg = MultiAgentRunConfig(
            max_parallel_agents=2,
            task_timeout_seconds=0.12,
            max_retry_attempts=2,
            auto_review=False,
        )
        scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=reg,
            file_ownership=fom,
            context_sharing=cse,
            result_aggregator=res_agg,
            config=run_cfg,
        )

        ok = scheduler.execute_all(poll_interval=0.02)
        self.assertTrue(ok, "t1 zaman aşımından kurtarılıp t2 ile birlikte tamamlanmalıydı!")
        self.assertEqual(graph.get_task("t1").status, TaskStatus.COMPLETED)
        self.assertEqual(graph.get_task("t2").status, TaskStatus.COMPLETED)
        self.assertGreaterEqual(scheduler.recovered_timeout_tasks, 1)

    # ------------------------------------------------------------------
    # 12. Late tool response after timeout / cancellation
    # ------------------------------------------------------------------
    def test_12_late_tool_response_after_timeout_cannot_mutate_state(self) -> None:
        """Verify 12 late tool response after timeout cannot mutate state."""
        finish_late_tool = threading.Event()

        def late_returning_tool() -> str:
            finish_late_tool.wait(timeout=2.0)
            return "GEC_GELEN_BASARILI_SONUC"

        self.registry.register(
            name="late_tool",
            description="Timeout sonrası geç dönen araç",
            parameters={"type": "object", "properties": {}},
            handler=late_returning_tool,
        )

        res, record = self.executor.execution_manager.execute_managed(
            tool_name="late_tool",
            handler=late_returning_tool,
            arguments={},
            timeout_override=0.1,
        )

        self.assertEqual(record.status, ExecutionLifecycleStatus.TIMED_OUT)
        self.assertIn("Hata:", res)

        finish_late_tool.set()
        time.sleep(0.1)

        self.assertEqual(record.status, ExecutionLifecycleStatus.TIMED_OUT)
        self.assertTrue(record.late_response_ignored)
        self.assertGreaterEqual(self.executor.execution_manager.late_ignored_count, 1)


if __name__ == "__main__":
    unittest.main()
