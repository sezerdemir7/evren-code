"""SprintPersistence & Recovery module."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evren_code.core.multi_agent.models import AgentTaskResult, TaskStatus
from evren_code.core.sprint.events import SprintEvent, SprintEventType
from evren_code.core.sprint.exceptions import SprintRecoveryError
from evren_code.core.sprint.models import Sprint, SprintStatus

logger = logging.getLogger(__name__)


class SprintPersistence:
    """Sprint persistence implementation."""

    def __init__(self, workspace_root: Path, base_dir: str = ".evren/sprints") -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.base_dir = (self.workspace_root / base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_sprint_dir(self, sprint_id: str) -> Path:
        sdir = (self.base_dir / sprint_id).resolve()
        sdir.mkdir(parents=True, exist_ok=True)
        return sdir

    def save_sprint(self, sprint: Sprint) -> None:
        """Save sprint."""
        sdir = self._get_sprint_dir(sprint.id)

        # 1. sprint.json (Ana durum)
        self._atomic_write_json(sdir / "sprint.json", sprint.to_dict())

        plan_data = {
            "metadata": sprint.metadata.to_dict(),
            "epics": {eid: e.to_dict() for eid, e in sprint.epics.items()},
            "tasks": {tid: t.to_dict() for tid, t in sprint.tasks.items()},
        }
        self._atomic_write_json(sdir / "plan.json", plan_data)

        # 3. acceptance.json (Kabul kriterleri)
        acc_data = {aid: a.to_dict() for aid, a in sprint.acceptance_criteria.items()}
        self._atomic_write_json(sdir / "acceptance.json", acc_data)

    def load_sprint(self, sprint_id: str) -> Optional[Sprint]:
        """Load sprint."""
        sdir = (self.base_dir / sprint_id).resolve()
        sfile = sdir / "sprint.json"
        if not sfile.exists():
            return None

        try:
            with open(sfile, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Sprint.from_dict(data)
        except Exception as e:
            logger.error(f"Sprint '{sprint_id}' yüklenirken hata: {e}")
            raise SprintRecoveryError(f"Sprint dosyası bozuk veya okunamıyor: {e}")

    def list_sprints(self) -> List[Dict[str, Any]]:
        """List sprints."""
        results = []
        if not self.base_dir.exists():
            return results

        for p in self.base_dir.iterdir():
            if p.is_dir():
                sfile = p / "sprint.json"
                if sfile.exists():
                    try:
                        with open(sfile, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        meta = data.get("metadata", {})
                        results.append({
                            "sprint_id": meta.get("sprint_id", p.name),
                            "title": meta.get("title", ""),
                            "status": data.get("status", "unknown"),
                            "iteration": data.get("current_iteration", 0),
                            "tasks_count": len(data.get("tasks", {})),
                            "created_at": meta.get("created_at"),
                        })
                    except Exception:
                        pass
        return sorted(results, key=lambda x: x.get("created_at") or 0, reverse=True)

    def log_event(self, sprint_id: str, event: SprintEvent) -> None:
        """Log event (audit log)."""
        sdir = self._get_sprint_dir(sprint_id)
        events_file = sdir / "events.jsonl"
        try:
            with open(events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.warning(f"Sprint olayı kaydedilemedi: {e}")

    def get_file_fingerprint(self, rel_path: str) -> Dict[str, Any]:
        """Get file fingerprint (mevcudiyet, boyut, sha256)."""
        p = (self.workspace_root / rel_path).resolve()
        if p.exists() and p.is_file():
            try:
                content = p.read_bytes()
                sha = hashlib.sha256(content).hexdigest()
                return {"exists": True, "size": len(content), "hash": sha}
            except Exception:
                return {"exists": True, "size": 0, "hash": ""}
        return {"exists": False}

    def reconcile_on_crash_recovery(self, sprint: Sprint) -> Sprint:
        """Reconcile on crash recovery."""
        reconciled = False
        for task in sprint.tasks.values():
            if task.status == TaskStatus.RUNNING:
                file_snapshots = task.metadata.get("file_snapshots", {})
                has_dirty_modifications = False
                dirty_details = []

                if file_snapshots:
                    for rel_path, snap in file_snapshots.items():
                        current_snap = self.get_file_fingerprint(rel_path)
                        if snap.get("exists") != current_snap.get("exists"):
                            has_dirty_modifications = True
                            dirty_details.append(
                                f"{rel_path}: varlık durumu değişti ({snap.get('exists')} -> {current_snap.get('exists')})"
                            )
                        elif snap.get("exists") and snap.get("hash") != current_snap.get("hash"):
                            has_dirty_modifications = True
                            dirty_details.append(f"{rel_path}: dosya içeriği/hash değişti")
                elif task.assigned_files:
                    pass

                if has_dirty_modifications:
                    logger.warning(
                        f"Kurtarma: Görev '{task.task_id}' yarıda kalmış ve disk üzerinde kısmi değişiklikler yapılmış: "
                        f"{', '.join(dirty_details)}. FAILED olarak işaretleniyor."
                    )
                    task.status = TaskStatus.FAILED
                    task.result = AgentTaskResult(
                        task_id=task.task_id,
                        agent_id=task.assigned_agent_id or "unknown",
                        role=task.role,
                        success=False,
                        output=f"Kurtarma başarısız: {', '.join(dirty_details)}",
                        error=f"Crash recovery detected partial/dirty disk modifications on assigned files: {', '.join(dirty_details)}",
                    )
                else:
                    logger.info(f"Kurtarma: Kesintiye uğramış temiz görev '{task.task_id}' PENDING durumuna çekiliyor.")
                    task.status = TaskStatus.PENDING

                reconciled = True

        if sprint.status == SprintStatus.RUNNING:
            sprint.status = SprintStatus.PAUSED
            reconciled = True

        if reconciled:
            self.save_sprint(sprint)
        return sprint

    def _atomic_write_json(self, target_path: Path, data: Dict[str, Any]) -> None:
        """Atomic write json."""
        tmp_path = target_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)


SprintRecovery = SprintPersistence

