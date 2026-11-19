"""Test architecture remediation package a module."""

import concurrent.futures
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.events import EventBus, EventType
from evren_code.core.file_tracker import FileStateTracker


class TestArchitectureRemediationPackageA(unittest.TestCase):

    def test_config_default_yaml_exact_sync(self) -> None:
        """Verify config default yaml exact sync."""
        root_dir = Path(__file__).resolve().parent.parent
        root_cfg = root_dir / "config.default.yaml"
        pkg_cfg = root_dir / "src" / "evren_code" / "config.default.yaml"

        self.assertTrue(root_cfg.is_file(), f"Kök config.default.yaml bulunamadı: {root_cfg}")
        self.assertTrue(pkg_cfg.is_file(), f"Paket içi config.default.yaml bulunamadı: {pkg_cfg}")

        root_content = root_cfg.read_text(encoding="utf-8")
        pkg_content = pkg_cfg.read_text(encoding="utf-8")

        self.assertEqual(
            root_content,
            pkg_content,
            "HATA: Kök 'config.default.yaml' ile 'src/evren_code/config.default.yaml' senkronize değil! "
            "Her iki dosya da birebir aynı olmalıdır.",
        )

    def test_file_state_tracker_thread_safety(self) -> None:
        """Verify file state tracker thread safety."""
        tracker = FileStateTracker()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            files = [tmp_path / f"file_{i}.txt" for i in range(10)]
            for idx, f in enumerate(files):
                f.write_text(f"content {idx}", encoding="utf-8")

            def worker_task(thread_id: int) -> None:
                for _ in range(50):
                    f = files[thread_id % len(files)]
                    tracker.record_read(str(f), current_step=thread_id)
                    tracker.check_file(str(f), current_step=thread_id)
                    if thread_id % 3 == 0:
                        tracker.record_write(str(f), current_step=thread_id)
                    if thread_id % 7 == 0:
                        tracker.invalidate(str(f))

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(worker_task, i) for i in range(30)]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()

        self.assertTrue(hasattr(tracker, "_lock"))
        tracker.reset()
        self.assertEqual(len(tracker._states), 0)

    def test_event_bus_thread_safety(self) -> None:
        """Verify event bus thread safety."""
        bus = EventBus()
        counter = {"count": 0}

        def listener(evt) -> None:
            counter["count"] += 1

        def worker_task(tid: int) -> None:
            for _ in range(50):
                bus.subscribe(listener, EventType.STEP_STARTED)
                bus.emit(EventType.STEP_STARTED, tid=tid)
                bus.emit(EventType.WARNING, warning_msg="test")
                bus.unsubscribe(listener, EventType.STEP_STARTED)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(worker_task, i) for i in range(20)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

        self.assertGreater(counter["count"], 0)

    def test_orchestrator_graceful_close(self) -> None:
        """Verify orchestrator graceful close."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = load_config()
            cfg.workspace_root = Path(tmp_dir)
            mock_llm = MagicMock()

            orchestrator = AgentOrchestrator(config=cfg, llm_client=mock_llm)
            self.assertIsNotNone(orchestrator._speculative_pool)
            self.assertFalse(orchestrator._speculative_pool._shutdown)

            # Mock bg_process_manager
            mock_bg = MagicMock()
            orchestrator.bg_process_manager = mock_bg

            orchestrator.close()

            self.assertTrue(orchestrator._speculative_pool._shutdown)
            mock_bg.stop_all.assert_called_once()

    def test_orchestrator_context_manager(self) -> None:
        """Verify orchestrator context manager."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = load_config()
            cfg.workspace_root = Path(tmp_dir)
            mock_llm = MagicMock()

            with AgentOrchestrator(config=cfg, llm_client=mock_llm) as orch:
                self.assertFalse(orch._speculative_pool._shutdown)

            self.assertTrue(orch._speculative_pool._shutdown)


if __name__ == "__main__":
    unittest.main()
