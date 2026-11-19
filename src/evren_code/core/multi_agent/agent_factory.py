"""Specialist Agents module."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from evren_code.config import AppConfig, ModelSpec
from evren_code.core.context import ContextManager
from evren_code.core.events import EventBus, EventType
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.core.multi_agent.file_ownership import (
    FILE_OWNERSHIP_VIOLATION,
    FileOwnershipManager,
)
from evren_code.core.multi_agent.models import (
    AgentRole,
    AgentStatus,
    AgentTaskResult,
    EngineeringTask,
    TaskStatus,
)
from evren_code.core.security import SECURITY_ERR_ROLE_RESTRICTION, ToolExecutor
from evren_code.tools.registry import ToolRegistry


ARCHITECT_ROLE_PROMPT = """You are the EVREN Chief Software Architect Agent (Architect Agent).
Your role: Analyze codebase architecture, define module boundaries, and specify interface contracts.
You never modify code files directly (you do not have `write_file` / `edit_file` permissions).
ENGINEERING PRINCIPLES:
1. LEVERAGE EXISTING SOLUTIONS: Identify existing classes, functions, and utilities before designing new ones.
2. MINIMAL ABSTRACTION: Avoid premature layers, extra classes, or design pattern bloat; choose the simplest viable architecture.
3. OUTPUT: Provide clear architectural decisions, data structures, interface signatures, and concrete implementation strategy."""

DEVELOPER_ROLE_PROMPT = """You are the EVREN Senior Software Developer Agent (Developer Agent).
Your role: Implement engineering objectives thoroughly by executing concrete, surgical code modifications.
RULES AND ENGINEERING PRINCIPLES:
1. ACTION-ORIENTED EXECUTION: After at most 1–2 exploration/read steps (`read_file`, `search_code`), IMMEDIATELY begin applying code edits using `edit_file` or `write_file`. Do not waste step limits on repetitive reads.
2. REUSE EXISTING CODE: Extend existing components instead of creating parallel functions or duplicate files.
3. MINIMAL BLAST RADIUS: Modify only targeted files directly relevant to the objective. Avoid unrelated refactoring or redundant dependencies.
4. MAINTAIN QUALITY & SAFETY: Never strip error handling, security validations, or edge-case guards for the sake of brevity.
5. ASSIGNED SCOPE: Focus on your assigned files (`assigned_files`). If unspecified, accurately locate and modify relevant files.
6. SURGICAL MODIFICATIONS: Perform clean, targeted edits preserving surrounding working logic.
7. DATA-DRIVEN DESIGN: Avoid magic numbers and hardcoded values; write dynamic, configurable code.
8. SUMMARY: Summarize all modifications, touched files, and outcomes clearly upon completion."""

TESTER_ROLE_PROMPT = """You are the EVREN QA and Verification Engineer Agent (Tester Agent).
Your role: Execute build, test, and lint commands (`run_command`), verify test coverage, and diagnose failures.
RULES & PRINCIPLES:
1. Never delete or modify source code files.
2. Run only test, compilation, linting, and diagnostic commands.
3. Ensure test suites cover critical edge cases, boundary conditions, and regression risks.
4. Analyze compiler/test failure outputs, line numbers, and root causes accurately."""

REVIEWER_ROLE_PROMPT = """You are the EVREN Senior Code Reviewer and Acceptance Criteria Auditor (Reviewer Agent).
Your role: Inspect developer modifications, diffs, and adherence to acceptance criteria.
RULES & PRINCIPLES:
1. Never modify files or execute shell commands; read and analyze only.
2. CODE QUALITY AUDIT: Flag duplicate implementations, unnecessary abstractions, overengineering, and redundant dependencies.
3. SAFETY VERIFICATION: Ensure optimizations have not degraded error handling, logging, or security.
4. VERDICT: Explicitly approve (APPROVED) or reject with actionable feedback (REJECTED)."""

RESEARCHER_ROLE_PROMPT = """You are the EVREN Senior Codebase Researcher Agent (Researcher Agent).
Your role: Deeply inspect codebase symbols, documentation, and interface contracts to provide technical findings.
Operate strictly read-only; never modify files."""

PLANNER_ROLE_PROMPT = """You are the EVREN Task Decomposition and Planning Architect Agent (Planner Agent).
Your role: Decompose the objective into independent, parallelizable tasks with unambiguous acceptance criteria and file scopes (Directed Acyclic Graph - DAG).
RULES:
1. RAPID ARCHITECTURAL SCAN: Use at most 1–2 high-level overview tools (e.g., `get_project_tree` or `get_repo_map`) to understand directory hierarchy.
2. DIRECT OUTPUT: Do not spend turn steps reading individual code files line-by-line; file paths and directory layout are sufficient for decomposition.
3. OUTPUT FORMAT: Present your final decomposition strictly as a valid ```json [ ... ] ``` array of task objects."""


ROLE_ALLOWED_TOOLS: Dict[AgentRole, Set[str]] = {
    AgentRole.ARCHITECT: {
        "get_project_tree", "get_repo_map", "get_project_overview", "get_project_modules",
        "get_module_details", "inspect_project", "get_important_files", "read_file",
        "read_multiple_files", "find_files", "search_code", "semantic_search_code",
        "search_symbols", "find_symbol_definition", "find_symbol_references",
        "analyze_change_impact", "find_related_tests", "get_file_symbols",
    },
    AgentRole.DEVELOPER: {
        "read_file", "read_multiple_files", "write_file", "write_multiple_files", "edit_file", "multi_edit_file",
        "find_files", "search_code", "run_command", "lsp_get_diagnostics",
        "lsp_find_definition", "lsp_find_references", "search_symbols",
        "find_symbol_definition", "find_symbol_references", "analyze_change_impact",
        "get_file_symbols", "list_directory", "get_project_tree",
    },
    AgentRole.TESTER: {
        "run_command", "read_file", "read_multiple_files", "lsp_get_diagnostics",
        "git_info", "find_files", "search_code", "find_related_tests", "list_directory",
    },
    AgentRole.REVIEWER: {
        "git_info", "read_file", "read_multiple_files", "lsp_get_diagnostics",
        "search_code", "search_symbols", "find_symbol_definition", "find_symbol_references",
        "analyze_change_impact", "find_related_tests", "get_file_symbols", "get_project_tree",
    },
    AgentRole.RESEARCHER: {
        "get_project_tree", "get_repo_map", "get_project_overview", "get_project_modules",
        "get_module_details", "inspect_project", "get_important_files", "list_directory",
        "read_file", "read_multiple_files", "find_files", "search_code",
        "semantic_search_code", "search_symbols", "find_symbol_definition",
        "find_symbol_references", "analyze_change_impact", "find_related_tests",
        "get_file_symbols", "fetch_url",
    },
    AgentRole.PLANNER: {
        "get_project_tree", "get_repo_map", "get_project_overview", "get_project_modules",
        "inspect_project", "get_important_files", "find_files",
    },
}

ROLE_PROMPTS: Dict[AgentRole, str] = {
    AgentRole.ARCHITECT: ARCHITECT_ROLE_PROMPT,
    AgentRole.DEVELOPER: DEVELOPER_ROLE_PROMPT,
    AgentRole.TESTER: TESTER_ROLE_PROMPT,
    AgentRole.REVIEWER: REVIEWER_ROLE_PROMPT,
    AgentRole.RESEARCHER: RESEARCHER_ROLE_PROMPT,
    AgentRole.PLANNER: PLANNER_ROLE_PROMPT,
}


class SpecialistAgent:
    """Specialist agent implementation."""

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        config: AppConfig,
        llm_client: EvrenLLMClient,
        tool_executor: ToolExecutor,
        registry: ToolRegistry,
        file_ownership: FileOwnershipManager,
        event_bus: Optional[EventBus] = None,
        model_spec: Optional[ModelSpec] = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.config = config
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.registry = registry
        self.file_ownership = file_ownership
        self.event_bus = event_bus or EventBus()
        self.model_spec = model_spec or config.models.get_model(role.value)
        self.system_prompt = ROLE_PROMPTS.get(role, DEVELOPER_ROLE_PROMPT)
        self.allowed_tools = ROLE_ALLOWED_TOOLS.get(role, set())
        self.status = AgentStatus.IDLE

    def execute_task(
        self,
        task: EngineeringTask,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> AgentTaskResult:
        """Execute task."""
        self.status = AgentStatus.RUNNING
        start_time = time.time()
        tool_calls_count = 0
        modified_files: List[str] = []

        prompt_parts = [
            f"### GÖREV: {task.title} (ID: {task.task_id})",
            f"**Rol:** {self.role.value}",
            f"**Alan / Uzmanlık (Domain):** {getattr(task, 'domain', 'general')}",
            f"**Açıklama:** {task.description}",
        ]
        if task.assigned_files:
            prompt_parts.append(f"**Yazma İzni Verilen Dosyalar:** {', '.join(task.assigned_files)}")
        if task.read_only_files:
            prompt_parts.append(f"**Okuma Kapsamındaki Dosyalar:** {', '.join(task.read_only_files)}")
        if getattr(task, "provides_contracts", None):
            prompt_parts.append(f"**Üretilmesi Beklenen Sözleşmeler/Tipler (Provides):** {', '.join(task.provides_contracts)}")
        if getattr(task, "consumes_contracts", None):
            prompt_parts.append(f"**Tüketilecek Sözleşmeler/Tipler (Consumes):** {', '.join(task.consumes_contracts)}")
        if task.acceptance_criteria:
            criteria_str = "\n".join(f"- {c}" for c in task.acceptance_criteria)
            prompt_parts.append(f"**Kabul Kriterleri (Acceptance Criteria):**\n{criteria_str}")

        if context_data and context_data.get("contracts"):
            prompt_parts.append("\n**Sözleşme / Arayüz Şemaları (Consumed Contracts):**")
            for c_name, c_info in context_data["contracts"].items():
                prompt_parts.append(
                    f"--- [Sözleşme: {c_name} (Sağlayıcı: {c_info.get('provider_title')}, Alan: {c_info.get('domain')})] ---\n"
                    f"{c_info.get('contract_data')}"
                )

        if context_data and context_data.get("dependency_artifacts"):
            prompt_parts.append("\n**Önceki Bağımlı Görev Çıktıları:**")
            for dep_id, art in context_data["dependency_artifacts"].items():
                prompt_parts.append(f"--- [Görev: {dep_id}] ---\n{art}")

        preloaded_files: List[str] = []
        target_candidates = list(task.assigned_files) + list(task.read_only_files)
        ws_root = getattr(self.config, "workspace_root", Path.cwd())

        for rel_f in target_candidates[:3]:
            if not rel_f or "*" in rel_f or "?" in rel_f:
                continue
            try:
                p = (ws_root / rel_f).resolve()
                if p.is_file() and p.stat().st_size <= 65536:
                    txt = p.read_text(encoding="utf-8", errors="replace")
                    lines = txt.splitlines()
                    if lines and len(lines) <= 250:
                        numbered = "\n".join(f"{i:3d} | {ln}" for i, ln in enumerate(lines, start=1))
                        preloaded_files.append(f"### [Ön-Yüklenen Dosya]: `{rel_f}`\n```\n{numbered}\n```")
            except Exception:
                pass

        if preloaded_files:
            prompt_parts.append(
                "\n**Ön-Yüklenen Dosya İçerikleri (Bu dosyaları tekrar `read_file` ile okumanıza gerek yoktur; doğrudan `edit_file` / `write_file` uygulayabilirsiniz):**\n"
                + "\n\n".join(preloaded_files)
            )

        task_prompt = "\n\n".join(prompt_parts)

        context_mgr = ContextManager(self.config, self.system_prompt)
        context_mgr.add_user_message(task_prompt)

        effective_allowed_tools = set(self.allowed_tools)
        desc_lower = (task.description or "").lower()
        needs_write_or_exec = bool(task.assigned_files) or any(
            w in desc_lower for w in ("oluştur", "yaz", "çalıştır", "implement", "create", "write", "run", "execute")
        )
        if needs_write_or_exec and self.role in (AgentRole.ARCHITECT, AgentRole.RESEARCHER, AgentRole.DEVELOPER):
            effective_allowed_tools.update({
                "write_file", "write_multiple_files", "edit_file", "multi_edit_file",
                "run_command", "read_file", "read_multiple_files", "search_code", "find_files", "list_directory"
            })

        all_schemas = self.registry.get_openai_schemas()
        filtered_schemas = [
            s for s in all_schemas
            if s.get("function", {}).get("name") in effective_allowed_tools
        ]

        max_steps = (
            4 if self.role == AgentRole.PLANNER else self.config.agent.subagent_max_iterations
        )
        final_output = ""
        error_msg: Optional[str] = None
        success = True
        cancel_event: Optional[Any] = getattr(task, "cancel_event", None) or (
            task.metadata.get("cancel_event") if hasattr(task, "metadata") else None
        )

        try:
            for step in range(1, max_steps + 1):
                if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
                    final_output = "Görev iptal edildi veya zaman aşımına uğradı."
                    error_msg = "Task cancelled by scheduler"
                    success = False
                    break

                if self.event_bus:
                    self.event_bus.emit(
                        EventType.STEP_STARTED,
                        step=step,
                        max_steps=max_steps,
                        model_id=self.model_spec.id,
                        model_name=f"{self.role.value.capitalize()}",
                    )

                if context_mgr.needs_compaction(self.model_spec):
                    context_mgr.compact(aggressive=True)

                messages = context_mgr.get_messages()
                try:
                    response = self.llm_client.chat(
                        messages=messages,
                        tools_schema=filtered_schemas if self.model_spec.tools else None,
                        model_override=self.model_spec.id,
                        stream=True,
                        silent=True,
                        cancel_event=cancel_event,
                    )
                except TypeError:
                    try:
                        response = self.llm_client.chat(
                            messages=messages,
                            tools_schema=filtered_schemas if self.model_spec.tools else None,
                            model_override=self.model_spec.id,
                            stream=False,
                        )
                    except TypeError:
                        response = self.llm_client.chat(
                            messages=messages,
                            tools_schema=filtered_schemas if self.model_spec.tools else None,
                            model_override=self.model_spec.id,
                        )

                if not response.tool_calls:
                    final_output = response.content.strip()
                    break

                context_mgr.add_assistant_message(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    native_tools_supported=self.model_spec.tools,
                )

                def _exec_single_tc(tc: Any) -> tuple[Any, str, List[str]]:
                    tname = tc.name
                    mod_files_local: List[str] = []
                    if tname not in effective_allowed_tools:
                        return tc, (
                            f"[ROL KISITLAMASI / {SECURITY_ERR_ROLE_RESTRICTION}]: "
                            f"'{tname}' aracı '{self.role.value}' rolü için yetkilendirilmemiştir."
                        ), mod_files_local

                    if tname in ("write_file", "edit_file", "multi_edit_file", "write_multiple_files"):
                        target_files_to_check: List[str] = []
                        if tname == "write_multiple_files":
                            for item in tc.arguments.get("files", []):
                                if isinstance(item, dict) and "path" in item:
                                    target_files_to_check.append(str(item["path"]))
                        else:
                            p = tc.arguments.get("path") or tc.arguments.get("file_path", "")
                            if p:
                                target_files_to_check.append(str(p))

                        for tf in target_files_to_check:
                            can_write, reason = self.file_ownership.verify_write_permission(task.task_id, tf)
                            if not can_write:
                                return tc, reason or f"[{FILE_OWNERSHIP_VIOLATION}] '{tf}' için yazma yetkisi reddedildi.", mod_files_local

                        try:
                            res_str = self.tool_executor.execute(
                                tool_name=tname,
                                arguments=tc.arguments,
                                caller_role=self.role.value,
                                caller_agent=self.agent_id,
                                is_trust_mode=True,
                                cancel_event=cancel_event,
                            )
                        except TypeError:
                            res_str = self.tool_executor.execute(
                                tool_name=tname,
                                arguments=tc.arguments,
                                caller_role=self.role.value,
                                caller_agent=self.agent_id,
                                is_trust_mode=True,
                            )
                        for tf in target_files_to_check:
                            if tf:
                                mod_files_local.append(tf)
                        return tc, res_str, mod_files_local
                    else:
                        try:
                            res_str = self.tool_executor.execute(
                                tool_name=tname,
                                arguments=tc.arguments,
                                caller_role=self.role.value,
                                caller_agent=self.agent_id,
                                is_trust_mode=True,
                                cancel_event=cancel_event,
                            )
                        except TypeError:
                            res_str = self.tool_executor.execute(
                                tool_name=tname,
                                arguments=tc.arguments,
                                caller_role=self.role.value,
                                caller_agent=self.agent_id,
                                is_trust_mode=True,
                            )
                        return tc, res_str, mod_files_local

                from evren_code.core.turn_pipeline.tool_runner import is_parallel_safe_tool
                can_parallel_read = (
                    len(response.tool_calls) > 1
                    and all(is_parallel_safe_tool(tc.name) for tc in response.tool_calls)
                )

                if can_parallel_read:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(response.tool_calls), 4)) as pool:
                        futs = [pool.submit(_exec_single_tc, tc) for tc in response.tool_calls]
                        results = [fut.result() for fut in futs]
                else:
                    results = [_exec_single_tc(tc) for tc in response.tool_calls]

                for tc, res_str, mod_files_local in results:
                    if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
                        final_output = "Görev iptal edildi veya zaman aşımına uğradı."
                        error_msg = "Task cancelled by scheduler"
                        success = False
                        break

                    tool_calls_count += 1
                    for mf in mod_files_local:
                        if mf not in modified_files:
                            modified_files.append(mf)

                    if step >= max_steps - 2:
                        urgency_hint = (
                            f"\n[SİSTEM UYARISI]: Adım sınırına ulaşıyorsun ({step}/{max_steps}). "
                            "Araştırma araçlarını çağırmayı durdur ve hemen nihai çıktını üret!"
                        )
                        if self.role == AgentRole.PLANNER:
                            urgency_hint += " Görev listesini ```json [ ... ] ``` bloğunda gecikmeden döndür."
                        elif self.role == AgentRole.DEVELOPER and not modified_files:
                            urgency_hint += " Henüz hiçbir kod değişikliği yapmadın! Gerekli dosyaları `edit_file` veya `write_file` ile derhal düzenle."
                        res_str += urgency_hint

                    context_mgr.add_tool_result(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result_text=res_str,
                        native_tools_supported=self.model_spec.tools,
                        from_xml=tc.from_xml,
                    )

            if not final_output:
                assistant_texts = [
                    m.get("content", "") for m in context_mgr.get_messages()
                    if m.get("role") == "assistant" and m.get("content")
                ]
                for a_text in reversed(assistant_texts):
                    if a_text.strip():
                        final_output = a_text.strip()
                        break

            if not final_output:
                if modified_files:
                    final_output = f"Görev başarıyla tamamlandı. Düzenlenen dosyalar: {', '.join(modified_files)}."
                else:
                    final_output = f"Ajan ({self.role.value}) görevi {max_steps} adımda işledi ancak nihai metin üretmedi."

        except Exception as exc:
            success = False
            error_msg = str(exc)
            final_output = f"Görev yürütülürken hata oluştu: {exc}"

        exec_time = time.time() - start_time
        self.status = AgentStatus.COMPLETED if success else AgentStatus.FAILED

        return AgentTaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            role=self.role,
            success=success,
            output=final_output,
            modified_files=modified_files,
            artifacts={"output": final_output},
            error=error_msg,
            execution_time_seconds=exec_time,
            tool_calls_count=tool_calls_count,
        )
