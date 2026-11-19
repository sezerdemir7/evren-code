"""History module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evren_code.core.verification.models import VerificationReport


class VerificationHistoryStore:
    """Verification history store implementation."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()
        self.storage_dir = self.workspace_root / ".evren" / "verification"
        self.latest_file = self.storage_dir / "verification_latest.json"

    def ensure_dir(self) -> None:
        """Ensure dir."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, report: VerificationReport) -> Path:
        """Save report (latest)."""
        self.ensure_dir()
        ts_clean = report.timestamp.replace(":", "-").replace(".", "-")
        hist_file = self.storage_dir / f"verification_{ts_clean}.json"

        data = report.to_dict()
        json_content = json.dumps(data, indent=2, ensure_ascii=False)

        hist_file.write_text(json_content, encoding="utf-8")

        self.latest_file.write_text(json_content, encoding="utf-8")

        return hist_file

    def get_latest_report(self) -> Optional[VerificationReport]:
        """Get latest report."""
        if not self.latest_file.exists():
            return None
        try:
            data = json.loads(self.latest_file.read_text(encoding="utf-8"))
            return VerificationReport.from_dict(data)
        except Exception:
            return None

    def list_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List history."""
        if not self.storage_dir.exists():
            return []

        entries: List[Dict[str, Any]] = []
        files = sorted(
            [f for f in self.storage_dir.glob("verification_*.json") if f.name != "verification_latest.json"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for f in files[:limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                entries.append({
                    "file_name": f.name,
                    "timestamp": data.get("timestamp"),
                    "final_status": data.get("final_status"),
                    "task_description": data.get("task_description"),
                    "changed_files_count": len(data.get("changed_files", [])),
                    "steps_count": len(data.get("step_results", [])),
                    "correction_attempts_count": len(data.get("correction_attempts", [])),
                })
            except Exception:
                continue

        return entries

    def get_report_by_name(self, file_name: str) -> Optional[VerificationReport]:
        """Get report by name."""
        target = self.storage_dir / file_name
        if not target.exists():
            return None
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            return VerificationReport.from_dict(data)
        except Exception:
            return None
