"""Checkpoint module."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evren_code.core.multi_agent.models import OrchestrationReport
from evren_code.core.multi_agent.task_graph import TaskGraph


class PersistenceCorruptionError(Exception):
    """PersistenceCorruptionError: checkpoint."""
    pass


class MultiAgentPersistence:
    """Multi agent persistence implementation."""

    def __init__(self, workspace_root: Path, base_dirname: str = ".evren/multi_agent") -> None:
        self.workspace_root = workspace_root
        self.base_dir = self.workspace_root / base_dirname
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        p = self.runs_dir / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_checkpoint(
        self,
        run_id: str,
        task_graph: TaskGraph,
        report: Optional[OrchestrationReport] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint."""
        rdir = self._run_dir(run_id)
        state_file = rdir / "state.json"
        temp_file = rdir / "state.json.tmp"

        payload = {
            "run_id": run_id,
            "saved_at": time.time(),
            "metadata": metadata or {},
            "graph": task_graph.to_dict(),
            "report": report.to_dict() if report else None,
        }

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        os.replace(str(temp_file), str(state_file))

        return state_file

    def load_checkpoint(self, run_id: str) -> Tuple[TaskGraph, Optional[OrchestrationReport], Dict[str, Any]]:
        """Load checkpoint."""
        rdir = self.runs_dir / run_id
        state_file = rdir / "state.json"
        if not state_file.is_file():
            raise FileNotFoundError(f"'{run_id}' için durum dosyası bulunamadı: {state_file}")

        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            temp_file = rdir / "state.json.tmp"
            if temp_file.is_file():
                try:
                    with open(temp_file, "r", encoding="utf-8") as tf:
                        data = json.load(tf)
                except Exception:
                    raise PersistenceCorruptionError(f"Durum dosyası ('{state_file}') bozulmuş: {exc}")
            else:
                raise PersistenceCorruptionError(f"Durum dosyası ('{state_file}') bozulmuş: {exc}")
        except Exception as exc:
            raise PersistenceCorruptionError(f"Durum dosyası okunamadı: {exc}")

        graph = TaskGraph.from_dict(data.get("graph", {}))
        rep_data = data.get("report")
        report = OrchestrationReport.from_dict(rep_data) if rep_data else None
        meta = data.get("metadata", {})

        return graph, report, meta

    def log_event(self, run_id: str, message: str) -> None:
        """Log event (execution.log)."""
        rdir = self._run_dir(run_id)
        log_file = rdir / "execution.log"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")

    def list_runs(self) -> List[Dict[str, Any]]:
        """List runs."""
        if not self.runs_dir.is_dir():
            return []

        runs: List[Dict[str, Any]] = []
        for rdir in sorted(self.runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if rdir.is_dir():
                sfile = rdir / "state.json"
                if sfile.is_file():
                    try:
                        with open(sfile, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        runs.append({
                            "run_id": data.get("run_id", rdir.name),
                            "saved_at": data.get("saved_at", 0),
                            "total_tasks": len(data.get("graph", {}).get("tasks", {})),
                            "path": str(rdir),
                        })
                    except Exception:
                        runs.append({"run_id": rdir.name, "path": str(rdir)})
        return runs
