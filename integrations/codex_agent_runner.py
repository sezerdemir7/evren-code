"""Codex / OpenAI Agents / Python script client for Evren-Code Subagent."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_evren_subagent(
    task: str,
    workspace: str = ".",
    role: str = "orchestrator",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Evren-Code as an isolated autonomous subagent and return structured JSON.
    
    Args:
        task: Description of the engineering task.
        workspace: Path to the target workspace directory.
        role: Specialist role ('orchestrator', 'architect', 'coder', 'reviewer', 'tester').
        model: Specific model override (optional).
        
    Returns:
        Dict: Structured result object containing status, summary, modified_files, etc.
    """
    cmd = [
        sys.executable,
        "-m", "evren_code.cli",
        "--headless",
        "--json",
        "--role", role,
        "-C", str(Path(workspace).resolve()),
    ]
    if model:
        cmd.extend(["-m", model])
    cmd.append(task)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if proc.returncode != 0 and not proc.stdout.strip():
        return {
            "status": "error",
            "exit_code": proc.returncode,
            "error": proc.stderr.strip() or "Unknown subagent execution error.",
        }

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "raw_output": proc.stdout,
            "stderr": proc.stderr,
        }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        res = run_evren_subagent(task=prompt)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("Usage: python codex_agent_runner.py '<task>'")
