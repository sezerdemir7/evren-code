"""Tools module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from evren_code.core.verification.engine import VerificationEngine
from evren_code.tools.registry import ToolRegistry


def register_verification_tools(
    registry: ToolRegistry,
    verification_engine: VerificationEngine,
    llm_client: Optional[Any] = None,
) -> None:
    """Registers verification and remediation tools with ToolRegistry."""

    def run_verification(
        changed_files: str = "",
        task_description: str = "",
    ) -> str:
        """Run verification."""
        files = [f.strip() for f in changed_files.split(",") if f.strip()] if changed_files else []
        report = verification_engine.run_verification(
            changed_files=files,
            task_description=task_description,
        )
        return verification_engine.format_report_markdown(report)

    def get_verification_report(report_name: Optional[str] = None) -> str:
        """Get verification report."""
        if report_name:
            rep = verification_engine.history_store.get_report_by_name(report_name)
        else:
            rep = verification_engine.last_report or verification_engine.history_store.get_latest_report()

        if not rep:
            return "Kayıtlı bir doğrulama raporu bulunamadı."
        return verification_engine.format_report_markdown(rep)

    def get_verification_history(limit: int = 10) -> str:
        """Get verification history."""
        items = verification_engine.history_store.list_history(limit=limit)
        if not items:
            return "Henüz kaydedilmiş bir doğrulama geçmişi yok."

        lines = [f"### Doğrulama Geçmişi (Son {len(items)} Koşum):"]
        for it in items:
            lines.append(
                f"- `{it['file_name']}` | Durum: **{it['final_status']}** | "
                f"Değişen Dosyalar: {it['changed_files_count']} | Adımlar: {it['steps_count']} | "
                f"Tarih: {it.get('timestamp', '')[:19]}"
            )
        return "\n".join(lines)

    registry.register(
        name="run_verification",
        description="Executes targeted test suites and build verifications based on modified files and symbol impact analysis.",
        parameters={
            "type": "object",
            "properties": {
                "changed_files": {
                    "type": "string",
                    "description": "Comma-separated list of changed files to analyze and verify (e.g. 'src/auth.py,src/user.py').",
                },
                "task_description": {
                    "type": "string",
                    "description": "Optional brief summary of the task or modification being verified.",
                },
            },
        },
        handler=run_verification,
        requires_approval=False,
    )

    registry.register(
        name="get_verification_report",
        description="Retrieves the latest verification report or a specific named report.",
        parameters={
            "type": "object",
            "properties": {
                "report_name": {
                    "type": "string",
                    "description": "Optional report filename (shows latest report if omitted).",
                },
            },
        },
        handler=get_verification_report,
        requires_approval=False,
    )

    def synthesize_unit_test(
        target_file: str,
        test_framework: Optional[str] = None,
        execute_immediately: bool = True,
    ) -> str:
        """Synthesize unit test."""
        workspace = verification_engine.workspace_root
        target_path = (workspace / target_file.strip().replace("\\", "/")).resolve()
        if not target_path.is_file():
            found = list(workspace.rglob(Path(target_file).name))
            if found:
                target_path = found[0]
            else:
                return f"Error: Target file '{target_file}' for test synthesis not found."

        try:
            code_content = target_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Error: Cannot read file: {exc}"

        suffix = target_path.suffix.lower()
        try:
            rel_target = target_path.relative_to(workspace).as_posix()
        except ValueError:
            rel_target = target_path.name

        test_file_name = f"test_{target_path.stem}{suffix}" if suffix == ".py" else f"{target_path.stem}.test{suffix}"
        test_dir = workspace / "tests" if (workspace / "tests").is_dir() else (target_path.parent / "tests")
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file_path = test_dir / test_file_name
        try:
            rel_test_path = test_file_path.relative_to(workspace).as_posix()
        except ValueError:
            rel_test_path = test_file_path.name

        fw = test_framework or ("unittest" if suffix == ".py" else "jest")
        prompt = (
            f"You are an expert QA and Test Automation Engineer. Generate comprehensive unit and edge-case "
            f"tests with 100% test coverage target for the following source code.\n\n"
            f"Source File: `{rel_target}`\n"
            f"Test Framework: {fw}\n"
            f"Source Code:\n```\n{code_content[:4000]}\n```\n\n"
            "Rules:\n"
            "1. Return ONLY valid executable test code in a single markdown code block.\n"
            "2. Import the module under test correctly.\n"
            "3. Cover standard scenarios, erroneous inputs, null values, and boundary conditions."
        )

        test_code = ""
        if llm_client:
            try:
                roles = getattr(llm_client.config.models, "roles", {})
                coder_model = roles.get("coder") if isinstance(roles, dict) else "coder"
                resp = llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model_override=coder_model,
                    stream=False,
                    silent=True,
                )
                raw_code = resp.content or ""
                if "```" in raw_code:
                    parts = raw_code.split("```")
                    for p in parts[1::2]:
                        lines = p.splitlines()
                        if lines and lines[0].strip().lower() in ("python", "py", "javascript", "js", "typescript", "ts", "go", "java"):
                            test_code = "\n".join(lines[1:]).strip()
                            break
                        elif lines:
                            test_code = p.strip()
                            break
                else:
                    test_code = raw_code.strip()
            except Exception as exc:
                return f"Error: LLM test synthesis failed: {exc}"
        else:
            return "Error: No active LLM client found for test synthesis."

        if not test_code:
            return "Error: Failed to generate test code."

        try:
            test_file_path.write_text(test_code, encoding="utf-8")
        except OSError as exc:
            return f"Error: Failed to write test file: {exc}"

        exec_report = ""
        if execute_immediately and hasattr(verification_engine, "tool_executor"):
            cmd = f'py -3.11 -m unittest {rel_test_path}' if suffix == ".py" else f'npm test -- {rel_test_path}'
            run_out = verification_engine.tool_executor.execute(
                "run_command",
                {"command": cmd},
                caller_role="tester",
                caller_agent="test_synthesizer",
            )
            exec_report = f"\n\n**Run Report / Koşum Raporu:**\n```\n{run_out[:1200]}\n```"

        return (
            f"🧪 **[Autonomous Test Synthesis Completed / Otomatik Test Sentezi Tamamlandı]**\n"
            f"- **Target File / Hedef Dosya:** `{rel_target}`\n"
            f"- **Synthesized Test / Üretilen Test:** `{rel_test_path}` ({len(test_code.splitlines())} lines)\n"
            f"- **Framework / Çerçeve:** {fw}"
            f"{exec_report}"
        )

    registry.register(
        name="synthesize_unit_test",
        description=(
            "Autonomous Test Synthesizer: Analyzes source code files and generates/executes comprehensive unit test suites covering edge cases."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target_file": {
                    "type": "string",
                    "description": "Path to target source code file (e.g. 'src/utils.py').",
                },
                "test_framework": {
                    "type": "string",
                    "description": "Optional test framework (e.g. 'unittest', 'pytest', 'jest').",
                },
                "execute_immediately": {
                    "type": "boolean",
                    "description": "Whether to immediately run the generated test and report results (default: true).",
                },
            },
            "required": ["target_file"],
        },
        handler=synthesize_unit_test,
        requires_approval=False,
    )

    registry.register(
        name="get_verification_history",
        description="Retrieves a summary history of past verification and test runs.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of history items to return (default: 10).",
                },
            },
        },
        handler=get_verification_history,
        requires_approval=False,
    )
