"""Agent-Facing Protocol and Usage Guide for Host Agents (`agent_spec.py`).

Provides machine-readable JSON and Markdown instructions for host AI systems
such as Claude Code, Codex, Antigravity, and Cursor to reliably orchestrate
`evren-code` as an autonomous subagent.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def get_agent_spec_data() -> Dict[str, Any]:
    """Returns structured JSON protocol specification for host agents."""
    return {
        "name": "evren-code",
        "description": "Autonomous software engineering subagent and LLM bridge powered by the EVREN AI platform.",
        "version": "0.2.0",
        "primary_capabilities": [
            "Autonomous Software Engineering (ReAct Loop, Surgical Diff, Automated Test and Repair)",
            "Direct Raw LLM Completions (DeepSeek-V4.1 Flash, Qwen-3.8, GLM-5.3, Gemma-4)",
            "Model Context Protocol (MCP) stdio Server",
        ],
        "available_models": {
            "deepseek-v4.1-flash": {
                "role": "architect / reviewer",
                "strengths": "Deep reasoning (<think>), complex systems architecture, root-cause bug diagnosis, and security code reviews.",
            },
            "qwen3.8-flash-next": {
                "role": "coder / tester",
                "strengths": "Fast surgical code generation, precise file editing, unit test synthesis, regex and algorithmic implementation.",
            },
            "glm-5.3": {
                "role": "orchestrator",
                "strengths": "General coordination, multi-step plan execution, and Turkish / multilingual fluency.",
            },
            "gemma-4-31b": {
                "role": "compactor",
                "strengths": "Long-context summarization and conversational token conservation.",
            },
            "qwen3-vl-30b": {
                "role": "vision",
                "strengths": "Headless browser screenshots, UI inspection, and visual defect analysis.",
            },
        },
        "execution_modes": [
            {
                "mode": "Direct LLM Completion (No Filesystem Changes)",
                "command": "evren ask --json -m <model_id> \"<prompt>\"",
                "example": "evren ask --json -m deepseek-v4.1-flash \"Explain the algorithmic complexity of...\"",
                "output_schema": {
                    "status": "success | error",
                    "model": "model_id",
                    "content": "Model output response",
                    "thinking": "Internal reasoning chain (<think>)",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
            {
                "mode": "Autonomous Subagent Task (Autonomous Coding & File Edits)",
                "command": "evren --headless --json --role <role> -C <workspace_dir> \"<task>\"",
                "example": "evren --headless --json --role coder \"Add refresh_token rotation to auth service\"",
                "flags_required": [
                    "--headless: Autonomous execution mode without TTY confirmation prompts.",
                    "--json: Outputs structured JSON results to stdout.",
                    "--role <architect|coder|reviewer|tester|orchestrator>: Automatically selects specialist model.",
                    "-C <dir>: Target project workspace root directory.",
                ],
                "output_schema": {
                    "status": "success | error",
                    "task": "Task description",
                    "model_used": "Model utilized",
                    "role": "Selected role",
                    "summary": "Summary of executed operations",
                    "modified_files": ["List of modified file paths"],
                },
            },
            {
                "mode": "Model Context Protocol (MCP)",
                "command": "evren mcp",
                "tools": [
                    "evren_llm_call(prompt, model, system_prompt, include_thinking): Direct raw LLM API call.",
                    "evren_task(task, workspace, role): Autonomous coding or refactoring task.",
                    "evren_architect(task, workspace): Architecture planning and risk assessment.",
                    "evren_code_review(path, workspace): Security and code quality review.",
                    "evren_models(): List active model catalog and token limits.",
                ],
            },
        ],
        "subagent_rules_for_host_agents": [
            "Always supply `--headless --json` together when calling via terminal; otherwise the agent prompts for interactive `[y/N/t]` confirmation and will block.",
            "Always specify `-C <workspace_dir>` if the target workspace differs from current working directory.",
            "Use `evren ask --json -m ...` when only reasoning or Q&A is required; avoids unnecessary filesystem overhead.",
            "For major architectural changes, first generate a plan with `--role architect`, then execute implementation with `--role coder`.",
        ],
    }


def get_agent_instructions(as_json: bool = False) -> str:
    """Generates human-readable Markdown or JSON instructions for host agents."""
    data = get_agent_spec_data()
    if as_json:
        return json.dumps(data, ensure_ascii=False, indent=2)

    md = []
    md.append(f"# 🤖 {data['name']} - Agent-to-Agent Protocol & Usage Guide")
    md.append("")
    md.append(f"> **Description:** {data['description']}")
    md.append("")
    md.append("## 📌 Guidelines for Host Agents")
    for rule in data["subagent_rules_for_host_agents"]:
        md.append(f"* {rule}")
    md.append("")
    md.append("## ⚡ Execution Modes")
    for em in data["execution_modes"]:
        md.append(f"### 🔹 {em['mode']}")
        md.append(f"**Command:** `{em['command']}`")
        if "example" in em:
            md.append(f"**Example:** `{em['example']}`")
        if "tools" in em:
            md.append("**Exposed Tools:**")
            for t in em["tools"]:
                md.append(f"  - `{t}`")
        md.append("")

    md.append("## 🧠 Available Models and Roles")
    for mid, spec in data["available_models"].items():
        md.append(f"* **`{mid}`** (Role: *{spec['role']}*): {spec['strengths']}")
    md.append("")

    return "\n".join(md)
