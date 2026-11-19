"""Git Integration Tool (`git_info`).

Safely queries read-only Git status (`status`), diffs (`diff`), staged changes (`staged_diff`),
recent commit history (`log`), and branch list (`branch`) without requiring confirmation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional

from evren_code.config import AppConfig
from evren_code.tools.registry import ToolRegistry


class ShadowGitManager:
    """Manages invisible shadow checkpoints and isolated worktrees for multi-agent workflows
    on the project git repository (Isolated Worktree Standard).
    """

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()
        self.has_git = (self.workspace_root / ".git").is_dir()
        self._checkpoints: List[str] = []

    def create_checkpoint(self, label: str = "") -> Optional[str]:
        """Creates an invisible git stash snapshot object prior to a turn or destructive operation;
        does not reset the working directory, only stores the current state.
        """
        if not self.has_git:
            return None
        try:
            proc = subprocess.run(
                ["git", "stash", "create", f"evren_checkpoint_{label}"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                check=False,
            )
            commit_hash = proc.stdout.strip()
            if commit_hash:
                self._checkpoints.append(commit_hash)
                return commit_hash
        except Exception:
            pass
        return None

    def revert_to_checkpoint(self, checkpoint_hash: Optional[str] = None) -> bool:
        """Atomically rolls back the workspace to the specified or most recent shadow checkpoint."""
        if not self.has_git:
            return False
        target = checkpoint_hash or (self._checkpoints[-1] if self._checkpoints else None)
        if not target:
            return False
        try:
            subprocess.run(
                ["git", "checkout", target, "--", "."],
                cwd=str(self.workspace_root),
                capture_output=True,
                check=False,
            )
            return True
        except Exception:
            return False

    def create_worktree(self, task_id: str) -> Optional[Path]:
        """Creates an independent, isolated git worktree for a sub-agent."""
        if not self.has_git:
            return None
        wt_dir = self.workspace_root / ".evren" / "worktrees" / task_id
        try:
            wt_dir.parent.mkdir(parents=True, exist_ok=True)
            if wt_dir.exists():
                return wt_dir
            branch_name = f"evren-code-{task_id}"
            subprocess.run(
                ["git", "worktree", "add", "-b", branch_name, str(wt_dir), "HEAD"],
                cwd=str(self.workspace_root),
                capture_output=True,
                check=False,
            )
            return wt_dir if wt_dir.is_dir() else None
        except Exception:
            return None

    def remove_worktree(self, task_id: str) -> None:
        """Safely removes the git worktree and branch of a finished sub-agent."""
        if not self.has_git:
            return
        wt_dir = self.workspace_root / ".evren" / "worktrees" / task_id
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt_dir)],
                cwd=str(self.workspace_root),
                capture_output=True,
                check=False,
            )
            branch_name = f"evren-code-{task_id}"
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                cwd=str(self.workspace_root),
                capture_output=True,
                check=False,
            )
        except Exception:
            pass


def register_git_tools(registry: ToolRegistry, config: AppConfig) -> None:
    """Registers the read-only Git query tool (`git_info`)."""

    def git_info(action: str = "status", target: Optional[str] = None) -> str:
        cmd_map = {
            "status": ["git", "status", "--short", "--branch"],
            "diff": ["git", "diff"] + ([target] if target else []),
            "staged_diff": ["git", "diff", "--staged"] + ([target] if target else []),
            "log": ["git", "log", "--oneline", "-n", "15"],
            "branch": ["git", "branch", "-a"],
        }
        args = cmd_map.get(action.lower())
        if not args:
            return (
                f"Error: Invalid git_info action '{action}'. "
                "Available actions: status, diff, staged_diff, log, branch. "
                f""
            )

        try:
            from evren_code.tools.shell import execute_command_with_tree_kill

            rc, stdout_str, stderr_str = execute_command_with_tree_kill(
                args=args,
                cwd=str(config.workspace_root),
                timeout=config.tools.command_timeout_seconds,
                shell=False,
            )
        except Exception as exc:
            return f"Error: Failed to execute git command: {exc}"

        if rc != 0:
            err = (stderr_str or "").strip()
            return f"Git Error (Exit code {rc}): {err}"

        out = (stdout_str or "").strip()
        if not out:
            return f"Git ({action}): Clean workspace, no modifications or output (Temiz)."

        max_chars = config.tools.max_output_chars
        if len(out) > max_chars:
            out = out[:max_chars] + f"\n... [Output truncated at {max_chars} characters / Çıktı kırpıldı]"
        return f"Git ({action}):\n{out}"

    registry.register(
        name="git_info",
        description=(
            "Inspects git repository status, diffs, staged changes, commit history (log), or branches safely and quickly."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "diff", "staged_diff", "log", "branch"],
                    "description": "Read-only git query action to run (default: 'status').",
                },
                "target": {
                    "type": "string",
                    "description": "Optional specific file or directory path for diff.",
                },
            },
            "required": [],
        },
        handler=git_info,
        requires_approval=False,
    )
