"""Session Persistence, Checkpointing, and Resume Manager (`SessionManager`).

Enables:
1. Saving and loading sessions across projects via `/save <name>` and `/load <name>`.
2. Automatic session checkpointing to `.evren/session.json` upon turn completion,
   allowing seamless recovery with `evren --resume`.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evren_code.config import AppConfig
from evren_code.core.context import ContextManager


def format_relative_time(iso_str: str) -> str:
    """Converts an ISO 8601 timestamp into relative human-readable format ('5 minutes ago', etc.)."""
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        diff_sec = max(0, int((now - dt).total_seconds()))

        if diff_sec < 60:
            return "just now"
        if diff_sec < 3600:
            return f"{diff_sec // 60} minutes ago"
        if diff_sec < 86400:
            return f"{diff_sec // 3600} hours ago"
        days = diff_sec // 86400
        return f"{days} days ago"
    except Exception:
        return iso_str[:19]


class SessionManager:
    """Session manager implementation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.sessions_dir: Path = config.api.sessions_dir_path
        self.workspace_session_file: Path = (
            config.workspace_root / ".evren" / "session.json"
        )

    @staticmethod
    def _sanitize_name(name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
        return cleaned or "varsayilan"

    def save_session(
        self,
        name: str,
        context: ContextManager,
        active_model_id: str,
        plan_manager: Optional[Any] = None,
    ) -> Path:
        """Save session."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._sanitize_name(name)
        file_path = self.sessions_dir / f"{safe_name}.json"

        plan_data = plan_manager.to_dict() if plan_manager else None

        payload: Dict[str, Any] = {
            "name": safe_name,
            "workspace": str(self.config.workspace_root),
            "model": active_model_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "summary_memory": context.summary_memory,
            "messages": context.raw_messages,
            "plan": plan_data,
        }
        file_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return file_path

    def load_session(self, name: str, context: ContextManager) -> Dict[str, Any]:
        """Load session."""
        safe_name = self._sanitize_name(name)
        file_path = self.sessions_dir / f"{safe_name}.json"
        if not file_path.is_file():
            raise FileNotFoundError(f"'{safe_name}' adında kayıtlı bir oturum bulunamadı.")

        payload: Dict[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
        context.restore_state(
            messages=payload.get("messages", []),
            summary_memory=payload.get("summary_memory", ""),
        )
        return payload

    def auto_save_workspace_session(
        self,
        context: ContextManager,
        active_model_id: str,
        plan_manager: Optional[Any] = None,
    ) -> Optional[Path]:
        """Auto save workspace session."""
        raw_msgs = context.raw_messages
        user_msgs = [m for m in raw_msgs if m.get("role") == "user"]
        if not user_msgs:
            return None

        try:
            self.workspace_session_file.parent.mkdir(parents=True, exist_ok=True)
            plan_data = plan_manager.to_dict() if plan_manager else None

            last_prompt = user_msgs[-1].get("content", "")
            if isinstance(last_prompt, list):
                last_prompt = str(last_prompt)

            payload: Dict[str, Any] = {
                "workspace": str(self.config.workspace_root),
                "model": active_model_id,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "summary_memory": context.summary_memory,
                "messages": raw_msgs,
                "plan": plan_data,
                "last_prompt": str(last_prompt)[:200],
            }
            self.workspace_session_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return self.workspace_session_file
        except Exception:
            return None

    def get_last_workspace_session(self) -> Optional[Dict[str, Any]]:
        """Get last workspace session."""
        if not self.workspace_session_file.is_file():
            return None

        try:
            payload: Dict[str, Any] = json.loads(
                self.workspace_session_file.read_text(encoding="utf-8")
            )
            messages = payload.get("messages", [])
            if not messages:
                return None

            saved_at = payload.get("saved_at", "")
            payload["relative_time"] = format_relative_time(saved_at) if saved_at else "bilinmiyor"
            return payload
        except Exception:
            return None

    def restore_session_to_orchestrator(
        self,
        orchestrator: Any,
        payload: Dict[str, Any],
    ) -> None:
        """Restore session to orchestrator."""
        orchestrator.context.restore_state(
            messages=payload.get("messages", []),
            summary_memory=payload.get("summary_memory", ""),
        )

        plan_data = payload.get("plan")
        if plan_data and hasattr(orchestrator, "plan_manager"):
            orchestrator.plan_manager.restore_from_dict(plan_data)

        saved_model = payload.get("model")
        if saved_model and hasattr(orchestrator, "llm_client"):
            try:
                orchestrator.llm_client.set_model(saved_model)
            except Exception:
                pass

    def clear_workspace_session(self) -> bool:
        """Clear workspace session."""
        try:
            if self.workspace_session_file.is_file():
                self.workspace_session_file.unlink()
                return True
        except Exception:
            pass
        return False

    def list_sessions(self) -> List[Dict[str, str]]:
        """List sessions."""
        if not self.sessions_dir.is_dir():
            return []
        items: List[Dict[str, str]] = []
        for fpath in sorted(self.sessions_dir.glob("*.json")):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                items.append(
                    {
                        "name": data.get("name", fpath.stem),
                        "workspace": data.get("workspace", "-"),
                        "model": data.get("model", "-"),
                        "saved_at": data.get("saved_at", "-")[:19],
                        "messages": str(len(data.get("messages", []))),
                    }
                )
            except Exception:
                continue
        return items
