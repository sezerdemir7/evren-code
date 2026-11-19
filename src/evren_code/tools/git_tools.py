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

    @property
    def last_checkpoint_commit(self) -> Optional[str]:
        """Returns the most recent shadow checkpoint hash if available."""
        return self._checkpoints[-1] if self._checkpoints else None

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

    def git_commit_assistant(scope: Optional[str] = None, commit_type: Optional[str] = None) -> str:
        """Analyzes workspace diffs and generates a Conventional Commit message and PR summary draft."""
        proposal = build_conventional_commit_proposal(
            workspace_root=config.workspace_root,
            scope=scope,
            commit_type=commit_type,
        )
        if not proposal.get("changed_files"):
            return "Git Commit Assistant: Working tree is clean; no uncommitted changes found."
        return str(proposal["formatted_report"])

    registry.register(
        name="git_commit_assistant",
        description=(
            "Inspects current uncommitted or staged git changes, computes line additions/deletions (+X/-Y), "
            "and drafts a structured Conventional Commit message and PR summary without committing."
        ),
        parameters={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Optional Conventional Commit scope (e.g. 'auth', 'cli', 'core')."},
                "commit_type": {
                    "type": "string",
                    "description": "Optional commit type override ('feat', 'fix', 'refactor', 'docs', 'test', 'chore').",
                },
            },
            "required": [],
        },
        handler=git_commit_assistant,
        requires_approval=False,
    )


def build_conventional_commit_proposal(
    workspace_root: Path,
    scope: Optional[str] = None,
    commit_type: Optional[str] = None,
) -> dict:
    """Computes diff statistics and synthesizes a Conventional Commit message and PR summary."""
    changed_files: List[str] = []
    added_lines = 0
    deleted_lines = 0

    try:
        st_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in (st_proc.stdout or "").splitlines():
            if len(line) > 3:
                path_str = line[3:].strip().strip('"')
                if " -> " in path_str:
                    path_str = path_str.split(" -> ")[-1].strip()
                changed_files.append(path_str)

        for diff_args in (["git", "diff", "--numstat"], ["git", "diff", "--staged", "--numstat"]):
            d_proc = subprocess.run(
                diff_args,
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            for dline in (d_proc.stdout or "").splitlines():
                parts = dline.split("\t")
                if len(parts) >= 3:
                    if parts[0].isdigit():
                        added_lines += int(parts[0])
                    if parts[1].isdigit():
                        deleted_lines += int(parts[1])
    except Exception:
        pass

    if not changed_files:
        return {
            "changed_files": [],
            "added_lines": 0,
            "deleted_lines": 0,
            "commit_message": "",
            "formatted_report": "Working tree clean.",
        }

    inferred_type = (commit_type or "").strip().lower()
    if not inferred_type:
        if all(f.endswith(".md") for f in changed_files):
            inferred_type = "docs"
        elif all("test" in f.lower() for f in changed_files):
            inferred_type = "test"
        else:
            inferred_type = "feat"

    inferred_scope = (scope or "").strip().lower()
    if not inferred_scope:
        top_dirs = []
        for f in changed_files:
            norm = f.replace("\\", "/")
            segs = [s for s in norm.split("/") if s and s not in ("src", "evren_code")]
            if len(segs) > 1:
                top_dirs.append(segs[0])
            elif segs:
                top_dirs.append(Path(segs[0]).stem)
        inferred_scope = top_dirs[0] if top_dirs else "core"

    file_stems = ", ".join(Path(f).name for f in changed_files[:3])
    more_sfx = f" (+{len(changed_files) - 3} more)" if len(changed_files) > 3 else ""
    subject = f"{inferred_type}({inferred_scope}): update {file_stems}{more_sfx}"
    bullets = "\n".join(f"- Updated `{f}`" for f in changed_files[:10])
    commit_msg = f"{subject}\n\n{bullets}"

    report = (
        f"### 📝 Git Commit & PR Draft\n"
        f"- **Changed Files:** {len(changed_files)} (`+{added_lines}` / `-{deleted_lines}` lines)\n"
        f"- **Suggested Commit Message:**\n"
        f"```text\n{commit_msg}\n```"
    )
    return {
        "changed_files": changed_files,
        "added_lines": added_lines,
        "deleted_lines": deleted_lines,
        "commit_message": commit_msg,
        "formatted_report": report,
    }
