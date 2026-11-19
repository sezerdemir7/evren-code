"""Evren-Code Event and Hook Manager (`hooks.py`).

Enables projects to configure custom lifecycle commands (linters, formatters, test runners)
via `.evren/hooks.yaml` on PreToolUse, PostToolUse, or OnTurnComplete events.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evren_code.core.events import AgentEvent, EventBus, EventType


@dataclass
class HookDefinition:
    """Individual hook definition rule."""
    event: str  # "pre_tool_use", "post_tool_use", "turn_completed"
    command: str
    match_tool: Optional[str] = None  # If configured only for a specific tool


class HookManager:
    """Loads and executes project hooks via the EventBus."""

    def __init__(self, workspace_root: Path, event_bus: EventBus) -> None:
        self.workspace_root = workspace_root.resolve()
        self.event_bus = event_bus
        self.hooks: List[HookDefinition] = []
        self._load_hooks()
        self._wire_events()

    def _hooks_config_path(self) -> Path:
        return self.workspace_root / ".evren" / "hooks.yaml"

    def _load_hooks(self) -> None:
        hpath = self._hooks_config_path()
        if not hpath.is_file():
            return
        try:
            import yaml
            data = yaml.safe_load(hpath.read_text(encoding="utf-8")) or {}
            raw_hooks = data.get("hooks", [])
            for item in raw_hooks:
                if isinstance(item, dict) and item.get("event") and item.get("command"):
                    self.hooks.append(
                        HookDefinition(
                            event=str(item["event"]).lower(),
                            command=str(item["command"]),
                            match_tool=item.get("tool"),
                        )
                    )
        except Exception:
            pass

    def _wire_events(self) -> None:
        """Wire events."""
        if not self.hooks:
            return

        self.event_bus.subscribe(self._on_tool_started, EventType.TOOL_EXECUTION_STARTED)
        self.event_bus.subscribe(self._on_tool_completed, EventType.TOOL_EXECUTION_COMPLETED)
        self.event_bus.subscribe(self._on_turn_completed, EventType.TURN_COMPLETED)

    def _on_tool_started(self, event: AgentEvent) -> None:
        tool_name = event.data.get("tool_name")
        for hook in self.hooks:
            if hook.event in ("pre_tool_use", "tool_execution_started"):
                if not hook.match_tool or hook.match_tool == tool_name:
                    self._run_hook_command(hook.command, {"TOOL_NAME": str(tool_name)})

    def _on_tool_completed(self, event: AgentEvent) -> None:
        tool_name = event.data.get("tool_name")
        for hook in self.hooks:
            if hook.event in ("post_tool_use", "tool_execution_completed"):
                if not hook.match_tool or hook.match_tool == tool_name:
                    self._run_hook_command(hook.command, {"TOOL_NAME": str(tool_name)})

    def _on_turn_completed(self, event: AgentEvent) -> None:
        for hook in self.hooks:
            if hook.event in ("turn_completed", "stop"):
                self._run_hook_command(hook.command, {})

    def _run_hook_command(self, cmd_template: str, extra_env: Dict[str, str]) -> None:
        from evren_code.core.security import CommandClassifier
        is_dangerous, reason = CommandClassifier.analyze(cmd_template)
        if is_dangerous:
            if self.event_bus:
                self.event_bus.emit(
                    EventType.WARNING,
                    message=f"[Hook Güvenlik Engeli]: Hook komutu yüksek riskli olduğu için engellendi: {reason} (`{cmd_template}`)",
                )
            return

        env = dict(os.environ)
        env.update(extra_env)
        try:
            subprocess.run(
                cmd_template,
                shell=True,
                cwd=str(self.workspace_root),
                env=env,
                capture_output=True,
                timeout=20,
            )
        except Exception:
            pass
