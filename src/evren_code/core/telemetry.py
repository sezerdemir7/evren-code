"""Trajectory and Agent Telemetry Logging Module (`telemetry.py`).

Records agent turn executions, tool calls, errors, and reasoning chains in both
machine-readable JSONL (`trajectory.jsonl`) and human-readable Markdown (`session_journal.md`).

Features:
- Project-local (`.evren/logs/`) or global user directory (`~/.evren-code/telemetry/`) modes.
- Automatically appends `.evren/` to `.gitignore` if present to keep repos clean.
- Fully toggled via configuration or `--no-log` CLI flag.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evren_code.config import AppConfig, ModelSpec
from evren_code.core.parser import ToolCallItem


@dataclass
class PhaseTiming:
    """Execution timing for an agent development phase (planning, coding, testing, recovery)."""
    start_time: float = 0.0
    accumulated_seconds: float = 0.0
    is_running: bool = False

    def start(self) -> None:
        if not self.is_running:
            self.start_time = time.perf_counter()
            self.is_running = True

    def stop(self) -> None:
        if self.is_running:
            self.accumulated_seconds += max(0.0, time.perf_counter() - self.start_time)
            self.is_running = False

    @property
    def total(self) -> float:
        if self.is_running:
            return round(self.accumulated_seconds + max(0.0, time.perf_counter() - self.start_time), 3)
        return round(self.accumulated_seconds, 3)


class ExecutionTelemetryTracker:
    """Execution telemetry tracker implementation."""

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root
        self.task_start_perf: float = 0.0
        self.task_stop_perf: float = 0.0
        self.is_running: bool = False

        self.phases: Dict[str, PhaseTiming] = {
            "planning": PhaseTiming(),
            "coding": PhaseTiming(),
            "testing": PhaseTiming(),
            "recovery": PhaseTiming(),
        }

        self.actual_input_tokens: int = 0
        self.actual_output_tokens: int = 0
        self.estimated_input_tokens: int = 0
        self.estimated_output_tokens: int = 0

        self.model_call_count: int = 0
        self.tool_call_count: int = 0
        self.file_read_count: int = 0
        self.file_write_count: int = 0
        self.retry_count: int = 0
        self.loop_breaker_count: int = 0
        self.agent_delegation_count: int = 0
        self.failed_operations: int = 0
        self.tool_timeout_count: int = 0
        self.tool_cancelled_count: int = 0
        self.watchdog_interventions: int = 0

    def start_task(self) -> None:
        self.task_start_perf = time.perf_counter()
        self.is_running = True

    def stop_task(self) -> None:
        if self.is_running:
            self.task_stop_perf = time.perf_counter()
            self.is_running = False
            for p in self.phases.values():
                p.stop()

    def start_phase(self, phase_name: str) -> None:
        if phase_name in self.phases:
            for name, p in self.phases.items():
                if name != phase_name and p.is_running:
                    p.stop()
            self.phases[phase_name].start()

    def stop_phase(self, phase_name: str) -> None:
        if phase_name in self.phases:
            self.phases[phase_name].stop()

    @property
    def total_wall_clock_seconds(self) -> float:
        if self.is_running:
            return round(max(0.0, time.perf_counter() - self.task_start_perf), 3)
        if self.task_start_perf > 0:
            return round(max(0.0, self.task_stop_perf - self.task_start_perf), 3)
        return 0.0

    def record_model_call(self, prompt_tokens: int, completion_tokens: int, is_estimated: bool = False) -> None:
        self.model_call_count += 1
        if is_estimated:
            self.estimated_input_tokens += prompt_tokens
            self.estimated_output_tokens += completion_tokens
        else:
            self.actual_input_tokens += prompt_tokens
            self.actual_output_tokens += completion_tokens

    def record_tool_call(self, tool_name: str, success: bool = True) -> None:
        self.tool_call_count += 1
        if not success:
            self.failed_operations += 1
        if tool_name in ("read_file", "read_multiple_files"):
            self.file_read_count += 1
        elif tool_name in ("write_file", "edit_file", "multi_edit_file", "write_multiple_files"):
            self.file_write_count += 1

    def record_retry(self) -> None:
        self.retry_count += 1

    def record_loop_breaker(self) -> None:
        self.loop_breaker_count += 1

    def record_delegation(self) -> None:
        self.agent_delegation_count += 1

    def record_tool_timeout(self, watchdog_intervened: bool = False) -> None:
        self.tool_timeout_count += 1
        self.failed_operations += 1
        if watchdog_intervened:
            self.watchdog_interventions += 1

    def record_tool_cancelled(self) -> None:
        self.tool_cancelled_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall_clock_duration_seconds": self.total_wall_clock_seconds,
            "phase_durations_seconds": {
                name: p.total for name, p in self.phases.items()
            },
            "tokens": {
                "actual": {
                    "input_tokens": self.actual_input_tokens,
                    "output_tokens": self.actual_output_tokens,
                    "total_tokens": self.actual_input_tokens + self.actual_output_tokens,
                },
                "estimated": {
                    "input_tokens": self.estimated_input_tokens,
                    "output_tokens": self.estimated_output_tokens,
                    "total_tokens": self.estimated_input_tokens + self.estimated_output_tokens,
                },
                "has_estimated_tokens": (self.estimated_input_tokens + self.estimated_output_tokens) > 0,
            },
            "operations": {
                "model_call_count": self.model_call_count,
                "tool_call_count": self.tool_call_count,
                "file_read_count": self.file_read_count,
                "file_write_count": self.file_write_count,
                "retry_count": self.retry_count,
                "loop_breaker_activations": self.loop_breaker_count,
                "agent_delegation_count": self.agent_delegation_count,
                "failed_operations": self.failed_operations,
                "tool_timeout_count": self.tool_timeout_count,
                "tool_cancelled_count": self.tool_cancelled_count,
                "watchdog_interventions": self.watchdog_interventions,
            },
        }

    def export_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


class TrajectoryLogger:
    """Trajectory logger implementation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.enabled = bool(config.telemetry.enabled and config.telemetry.mode != "off")
        self._log_dir: Optional[Path] = None

        if self.enabled:
            self._init_storage()

        self._current_turn: Optional[Dict[str, Any]] = None
        self._turn_start_perf: float = 0.0
        self.tracker: ExecutionTelemetryTracker = ExecutionTelemetryTracker(workspace_root=config.workspace_root)

    def _init_storage(self) -> None:
        mode = self.config.telemetry.mode.lower()
        if mode == "project":
            self._log_dir = self.config.workspace_root / ".evren" / "logs"
            if self.config.telemetry.auto_gitignore:
                self._ensure_gitignore()
        else:
            safe_ws_name = self.config.workspace_root.name or "workspace"
            self._log_dir = (
                self.config.api.storage_dir / "telemetry" / safe_ws_name
            )

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.enabled = False

    def _ensure_gitignore(self) -> None:
        """Ensure gitignore."""
        gi_file = self.config.workspace_root / ".gitignore"
        if not gi_file.is_file():
            return
        try:
            content = gi_file.read_text(encoding="utf-8", errors="ignore")
            lines = [line.strip() for line in content.splitlines()]
            if ".evren/" not in lines and ".evren" not in lines:
                append_entry = "\n# Evren-Code Yerel Günlük ve Kayıtlar\n.evren/\n"
                with open(gi_file, "a", encoding="utf-8") as f:
                    f.write(append_entry)
        except OSError:
            pass

    @property
    def log_directory(self) -> Optional[Path]:
        return self._log_dir

    def clear_logs(self) -> bool:
        """Clear logs."""
        if not self._log_dir or not self._log_dir.is_dir():
            return False
        try:
            for item in self._log_dir.glob("*"):
                if item.is_file():
                    item.unlink()
            return True
        except Exception:
            return False

    @property
    def jsonl_path(self) -> Optional[Path]:
        return self._log_dir / "trajectory.jsonl" if self._log_dir else None

    @property
    def markdown_path(self) -> Optional[Path]:
        return self._log_dir / "session_journal.md" if self._log_dir else None

    def start_turn(self, user_prompt: str, model_spec: ModelSpec) -> None:
        """Start turn."""
        if not self.enabled:
            return
        self._turn_start_perf = time.perf_counter()
        self._turn_start_time = time.time()
        self.tracker.start_task()
        self._current_turn = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_prompt": user_prompt,
            "model_id": model_spec.id,
            "model_name": model_spec.name,
            "status": "in_progress",
            "steps": [],
            "final_response": "",
            "duration_seconds": 0.0,
            "tokens_estimated": 0,
        }
        if self.config.telemetry.record_trajectory_jsonl and self.jsonl_path:
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"event": "turn_started", **self._current_turn}, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def record_step(
        self,
        step: int,
        thinking: str,
        content: str,
        tool_calls: List[ToolCallItem],
        tool_results: List[Dict[str, Any]],
    ) -> None:
        """Record step."""
        if not self.enabled or self._current_turn is None:
            return

        for idx, tc in enumerate(tool_calls):
            tr_item = tool_results[idx] if idx < len(tool_results) else {}
            res_txt = str(tr_item.get("result", ""))
            approved = bool(tr_item.get("approved", True))
            is_ok = approved and not (
                res_txt.startswith("Hata:")
                or res_txt.startswith("Error:")
                or res_txt.startswith("[GÜVENLİK ENGELİ")
                or res_txt.startswith("[KULLANICI REDDİ")
                or res_txt.startswith("🛑")
            )
            self.tracker.record_tool_call(tc.name, success=is_ok)

        max_chars = self.config.telemetry.max_log_chars_per_result
        serialized_calls = [
            {"name": tc.name, "arguments": tc.arguments, "from_xml": tc.from_xml}
            for tc in tool_calls
        ]
        serialized_results = []
        for tr in tool_results:
            raw_res = str(tr.get("result", ""))
            truncated = (
                raw_res[:max_chars] + f" ...[+{len(raw_res) - max_chars} karakter]"
                if len(raw_res) > max_chars
                else raw_res
            )
            serialized_results.append(
                {
                    "name": tr.get("name"),
                    "approved": tr.get("approved", True),
                    "result": truncated,
                }
            )

        step_data = {
            "step": step,
            "thinking": thinking,
            "content": content,
            "tool_calls": serialized_calls,
            "tool_results": serialized_results,
        }
        self._current_turn["steps"].append(step_data)

        if self.config.telemetry.record_trajectory_jsonl and self.jsonl_path:
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"event": "step_recorded", "step": step, "tool_calls": serialized_calls, "tool_results": serialized_results}, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def complete_turn(
        self,
        final_response: str,
        estimated_tokens: int = 0,
    ) -> None:
        """Complete turn."""
        if not self.enabled or self._current_turn is None or not self._log_dir:
            return

        self.tracker.stop_task()
        duration = round(time.perf_counter() - self._turn_start_perf, 2)
        self._current_turn["status"] = "completed"
        self._current_turn["duration_seconds"] = duration
        self._current_turn["final_response"] = final_response
        self._current_turn["tokens_estimated"] = estimated_tokens
        self._current_turn["telemetry_metrics"] = self.tracker.to_dict()

        turn_snapshot = self._current_turn
        self._current_turn = None

        if self.config.telemetry.record_trajectory_jsonl and self.jsonl_path:
            try:
                line = json.dumps({"event": "turn_completed", **turn_snapshot}, ensure_ascii=False)
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

        try:
            self.tracker.export_json(self._log_dir / "telemetry_report.json")
        except Exception:
            pass

        if self.config.telemetry.record_markdown_summary and self.markdown_path:
            try:
                md_block = self._format_markdown_entry(turn_snapshot)
                with open(self.markdown_path, "a", encoding="utf-8") as f:
                    f.write(md_block + "\n\n")
            except OSError:
                pass

    @staticmethod
    def _format_markdown_entry(turn: Dict[str, Any]) -> str:
        dt = turn.get("timestamp", "")[:19].replace("T", " ")
        model = f"{turn.get('model_name')} (`{turn.get('model_id')}`)"
        prompt = turn.get("user_prompt", "").strip()
        duration = turn.get("duration_seconds", 0)
        tokens = turn.get("tokens_estimated", 0)
        steps = turn.get("steps", [])

        lines = [
            f"### 🕒 [{dt}] — Model: {model}",
            f"**Kullanıcı İstemi:** `{prompt}`",
            f"* **Süre:** {duration}s | **Adım:** {len(steps)} | **Aktif Token:** ~{tokens}",
        ]

        tool_lines: List[str] = []
        for s in steps:
            for tc in s.get("tool_calls", []):
                tname = tc.get("name")
                args_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in tc.get("arguments", {}).items())
                tool_lines.append(f"- `{tname}({args_str})`")

        if tool_lines:
            lines.append("* **Kullanılan Araçlar:**\n  " + "\n  ".join(tool_lines))

        final_resp = turn.get("final_response", "").strip()
        if final_resp:
            preview = "\n> ".join(final_resp.splitlines()[:5])
            if len(final_resp.splitlines()) > 5:
                preview += "\n> ..."
            lines.append(f"* **Nihai Yanıt Özeti:**\n> {preview}")

        lines.append("---")
        return "\n".join(lines)
