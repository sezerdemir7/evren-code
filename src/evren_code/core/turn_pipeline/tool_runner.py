"""Turn Tool Runner: Manages permissions, approvals, parallel/sequential execution, and state tracking for tool calls."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from evren_code.core.events import EventType
from evren_code.core.parser import ToolCallItem, sanitize_thinking_for_context
from evren_code.core.planner import TaskContractEngine
from evren_code.core.turn_pipeline.context import TurnContext
from evren_code.core.turn_pipeline.loop_guard import TurnLoopGuard
from evren_code.core.verification.test_reporter import TestOutputParser


def extract_exported_symbols_from_disk(file_path: Path) -> List[str]:
    """Extract exported symbols from disk."""
    try:
        if not file_path.exists() or not file_path.is_file():
            return []
        suffix = file_path.suffix.lower()
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        symbols: List[str] = []
        if suffix in (".ts", ".tsx", ".js", ".jsx"):
            for m in re.finditer(
                r"\bexport\s+(?:default\s+)?(?:async\s+)?(?:function|const|let|class|type|interface)\s+([A-Za-z0-9_]+)",
                content,
            ):
                sym = m.group(1)
                if sym not in symbols and sym not in ("default", "const", "let", "function"):
                    symbols.append(sym)
        elif suffix == ".py":
            for m in re.finditer(r"^(?:def|class)\s+([A-Za-z0-9_]+)", content, re.MULTILINE):
                sym = m.group(1)
                if not sym.startswith("_") and sym not in symbols:
                    symbols.append(sym)
        return symbols[:8]
    except Exception:
        return []

PARALLEL_SAFE_TOOLS: Set[str] = {
    "read_file",
    "read_multiple_files",
    "list_directory",
    "get_project_tree",
    "find_files",
    "search_code",
    "semantic_search_code",
    "get_repo_map",
    "git_info",
    "fetch_url",
    "validate_web_contract",
    "analyze_image",
    "capture_web_screenshot",
    "inspect_web_page_visually",
    "lsp_get_diagnostics",
    "lsp_find_definition",
    "lsp_find_references",
    "get_task_plan",
    "get_project_memory",
    "query_project_memory",
    "synthesize_unit_test",
    "get_background_output",
    "get_project_overview",
    "get_project_modules",
    "get_module_details",
    "inspect_project",
    "get_important_files",
    "refresh_project_map",
    "search_symbols",
    "find_symbol_definition",
    "find_symbol_references",
    "analyze_change_impact",
    "find_related_tests",
    "get_file_symbols",
    "index_codebase",
    "get_verification_report",
    "get_verification_history",
    "skeletonize_code",
    "profile_code_performance",
    "analyze_implementation_strategy",
    "evaluate_code_reuse",
    "check_implementation_quality",
}


def is_parallel_safe_tool(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> bool:
    """Returns True if the tool is side-effect-free (read-only) and safe for parallel execution."""
    if tool_name in PARALLEL_SAFE_TOOLS:
        return True
    if tool_name == "delegate_subagent" and isinstance(arguments, dict):
        role = str(arguments.get("role", "")).lower()
        if role in ("researcher", "architect", "reviewer"):
            return True
    return False


class TurnToolRunner:
    """Manages speculative/parallel and sequential tool execution within an agent turn step."""

    def __init__(self, agent: Any, loop_guard: TurnLoopGuard) -> None:
        self.agent = agent
        self.loop_guard = loop_guard

    def handle_tool_calls(
        self,
        turn_ctx: TurnContext,
        response: Any,
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """Executes tool calls requested by the model in the current step.
        
        Returns:
            (should_continue_loop, step_tool_results, terminal_break_msg)
        """
        # Truncation Guard
        if response.finish_reason == "length" or getattr(response, "is_truncated", False):
            self.agent.event_bus.emit(
                EventType.WARNING,
                message=(
                    "[Truncation Guard / Kesik Çıktı Koruması]: Model output reached token limit "
                    f"(finish_reason='{response.finish_reason}') and was cut short. Tools aborted "
                    "to prevent file corruption."
                ),
            )
            self.agent.context.add_assistant_message(
                content=(response.content or "") + "\n... [Truncated due to token limit / Çıktı token limiti nedeniyle kesildi]",
                tool_calls=None,
                native_tools_supported=turn_ctx.target_spec.tools,
                thinking=response.thinking,
            )
            self.agent.context.add_user_message(
                "[System Warning - Truncation / Kesik Çıktı]: Your response or tool call was cut short "
                f"by the output token ceiling (finish_reason='{response.finish_reason}')! "
                "Tool calls were ABORTED to avoid writing corrupt or incomplete files.\n"
                "Please retry with smaller chunks:\n"
                "1. Rather than writing a massive file at once, create the skeleton with `write_file`.\n"
                "2. Add subsequent sections incrementally with `append_file` or `edit_file`.\n"
                "3. Avoid generating oversized payloads in a single turn."
            )
            return True, [], None

        asst_content = response.content or ""
        if response.thinking and "[Muhakeme]:" not in asst_content:
            compact_thk = sanitize_thinking_for_context(response.thinking, max_chars=200)
            if compact_thk:
                asst_content = f"[Muhakeme]: {compact_thk}" + (f"\n\n{asst_content}" if asst_content else "")

        self.agent.context.add_assistant_message(
            content=asst_content,
            tool_calls=response.tool_calls,
            native_tools_supported=turn_ctx.target_spec.tools,
            thinking=response.thinking,
        )

        step_tool_results: List[Dict[str, Any]] = []
        modified_paths_in_step: Dict[str, str] = {}
        prefetched_parallel_results: Dict[str, str] = {}

        # 1. Collect results from speculatively started tasks
        if getattr(self.agent, "_active_speculative_futures", None):
            for idx, tc in enumerate(response.tool_calls):
                fut = self.agent._active_speculative_futures.get(tc.id) or self.agent._active_speculative_futures.get(f"call_spec_{idx}")
                if fut:
                    try:
                        prefetched_parallel_results[tc.id] = fut.result(timeout=30.0)
                    except Exception as exc:
                        prefetched_parallel_results[tc.id] = f"Error: Speculative execution error: {exc}"
            self.agent._active_speculative_futures.clear()

        # 2. Plan parallel execution candidates
        parallel_candidates = []
        write_candidates = []
        seen_write_paths: Set[str] = set()
        write_path_conflict = False
        read_paths: Set[str] = set()

        for tc in response.tool_calls:
            if tc.id in prefetched_parallel_results or tc.name in turn_ctx.muted_tools or getattr(tc, "is_truncated", False):
                continue
            if tc.name in ("read_file", "search_code", "validate_web_contract") and isinstance(tc.arguments, dict):
                rp = str(tc.arguments.get("path") or tc.arguments.get("file_path") or "").strip().lower()
                if rp:
                    read_paths.add(rp)
            elif tc.name == "write_file" and isinstance(tc.arguments, dict):
                wp = str(tc.arguments.get("path", "")).strip().lower()
                if wp:
                    if wp in seen_write_paths:
                        write_path_conflict = True
                    seen_write_paths.add(wp)
                    write_candidates.append(tc)

        for tc in response.tool_calls:
            if tc.id in prefetched_parallel_results or tc.name in turn_ctx.muted_tools or getattr(tc, "is_truncated", False):
                continue
            if is_parallel_safe_tool(tc.name, tc.arguments if isinstance(tc.arguments, dict) else {}):
                tool_def = self.agent.registry.get(tc.name)
                configured_req = bool(
                    getattr(tool_def, "requires_approval", False)
                    or tc.name in self.agent.permission_manager.configured_requires_approval
                )
                dec = self.agent.permission_manager.check(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    is_trust_mode=self.agent.trust_mode,
                    configured_requires_approval=configured_req,
                    caller_role="main",
                    caller_id="orchestrator",
                )
                if dec.allowed and not dec.requires_approval:
                    parallel_candidates.append(tc)

        if len(write_candidates) > 1 and not write_path_conflict and not (seen_write_paths & read_paths):
            for tc in write_candidates:
                tool_def = self.agent.registry.get(tc.name)
                configured_req = bool(
                    getattr(tool_def, "requires_approval", False)
                    or tc.name in self.agent.permission_manager.configured_requires_approval
                )
                dec = self.agent.permission_manager.check(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    is_trust_mode=self.agent.trust_mode,
                    configured_requires_approval=configured_req,
                    caller_role="main",
                    caller_id="orchestrator",
                )
                if dec.allowed and not dec.requires_approval:
                    parallel_candidates.append(tc)

        if len(parallel_candidates) > 1:
            pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(8, len(parallel_candidates)),
                thread_name_prefix="evren_parallel",
            )
            try:
                future_map = {
                    tc.id: pool.submit(
                        self.agent.tool_executor.execute,
                        tc.name,
                        tc.arguments,
                        caller_role="main",
                        caller_agent="orchestrator_parallel",
                        is_trust_mode=self.agent.trust_mode,
                    )
                    for tc in parallel_candidates
                }
                for tc_id, fut in future_map.items():
                    try:
                        prefetched_parallel_results[tc_id] = fut.result(timeout=65.0)
                    except Exception as exc:
                        prefetched_parallel_results[tc_id] = f"Error: Parallel execution failure for `{tc_id}`: {exc}"
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

        for tc in response.tool_calls:
            self.agent.event_bus.emit(
                EventType.TOOL_CALL_DETECTED,
                tool_id=tc.id,
                tool_name=tc.name,
                arguments=tc.arguments,
            )

            # Truncated code protection
            if getattr(tc, "is_truncated", False) and tc.name in ("write_file", "edit_file", "multi_edit_file", "write_multiple_files", "apply_patch"):
                trunc_err_msg = (
                    f"Error [{tc.name}]: Output payload was truncated mid-string due to max_tokens limit (incomplete output)! "
                    "Call aborted to prevent file corruption. Please apply changes in smaller chunks using 'edit_file' or 'append_file'."
                )
                self.agent.event_bus.emit(
                    EventType.WARNING,
                    message=f"[Truncated Code Guard / Kesik Kod Koruması]: '{tc.name}' aborted due to incomplete output.",
                )
                self.agent.event_bus.emit(
                    EventType.TOOL_EXECUTION_COMPLETED,
                    tool_name=tc.name,
                    approved=False,
                    result=trunc_err_msg,
                )
                step_tool_results.append({
                    "name": tc.name,
                    "approved": False,
                    "result": trunc_err_msg,
                })
                self.agent.context.add_tool_result(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result_text=trunc_err_msg,
                    native_tools_supported=turn_ctx.target_spec.tools,
                    from_xml=tc.from_xml,
                )
                turn_ctx.last_step_had_error = True
                continue

            # Muted tool check
            is_blocked, is_terminal_break, mute_msg = self.loop_guard.check_muted_tool(turn_ctx, tc.name)
            if is_blocked:
                if is_terminal_break:
                    self.agent.last_circuit_breaker_info = tc.name
                    self.agent.context.add_assistant_message(
                        content=mute_msg or "",
                        tool_calls=None,
                        native_tools_supported=turn_ctx.target_spec.tools,
                    )
                    return False, step_tool_results, mute_msg
                else:
                    self.agent.event_bus.emit(
                        EventType.TOOL_EXECUTION_COMPLETED,
                        tool_name=tc.name,
                        approved=True,
                        result=mute_msg,
                    )
                    step_tool_results.append({
                        "name": tc.name,
                        "approved": True,
                        "result": mute_msg,
                    })
                    self.agent.context.add_tool_result(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result_text=mute_msg or "",
                        native_tools_supported=turn_ctx.target_spec.tools,
                        from_xml=tc.from_xml,
                    )
                    continue

            # Loop guard & search blindness check
            is_guarded, guard_result_msg, guard_user_msg = self.loop_guard.inspect_and_guard_tool_call(
                turn_ctx=turn_ctx,
                tool_name=tc.name,
                arguments=tc.arguments if isinstance(tc.arguments, dict) else {},
            )
            if is_guarded:
                self.agent.event_bus.emit(
                    EventType.TOOL_EXECUTION_COMPLETED,
                    tool_name=tc.name,
                    approved=True,
                    result=guard_result_msg,
                )
                step_tool_results.append({
                    "name": tc.name,
                    "approved": True,
                    "result": guard_result_msg,
                })
                self.agent.context.add_tool_result(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result_text=guard_result_msg or "",
                    native_tools_supported=turn_ctx.target_spec.tools,
                    from_xml=tc.from_xml,
                )
                if guard_user_msg:
                    self.agent.context.add_user_message(guard_user_msg)
                continue

            # Same-step file modification collision guard:
            target_path = None
            if isinstance(tc.arguments, dict):
                target_path = tc.arguments.get("path") or tc.arguments.get("file_path")
            if target_path and tc.name in ("write_file", "edit_file", "append_file", "multi_edit_file"):
                norm_p = str(Path(str(target_path).strip()).as_posix()).lower()
                if norm_p in modified_paths_in_step:
                    prev_tool = modified_paths_in_step[norm_p]
                    allow_seq_edit = (
                        prev_tool == "edit_file"
                        and tc.name == "edit_file"
                        and norm_p not in turn_ctx.files_with_syntax_errors
                    )
                    if not allow_seq_edit:
                        conflict_msg = (
                            f"Error [{tc.name}] [Collision Prevention / Çakışma Önleme]: File '{target_path}' was already modified in this step "
                            f"by '{prev_tool}'! Multiple conflicting modifications to the same file in a single step are prevented. "
                            f"Please proceed in the next step. (Hata [{tc.name}] [Çakışma Önleme]: '{target_path}' dosyası bu adım içinde daha önce "
                            f"'{prev_tool}' ile zaten değiştirildi! Aynı adımda aynı dosyaya mükerrer yazma yapılamaz.)"
                        )
                        self.agent.event_bus.emit(
                            EventType.WARNING,
                            message=f"[Collision Guard / Adım İçi Çakışma]: Sequential operation on '{target_path}' blocked.",
                        )
                        self.agent.event_bus.emit(
                            EventType.TOOL_EXECUTION_COMPLETED,
                            tool_name=tc.name,
                            approved=True,
                            result=conflict_msg,
                        )
                        step_tool_results.append({
                            "name": tc.name,
                            "approved": True,
                            "result": conflict_msg,
                        })
                        self.agent.context.add_tool_result(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            result_text=conflict_msg,
                            native_tools_supported=turn_ctx.target_spec.tools,
                            from_xml=tc.from_xml,
                        )
                        continue

            # Task Contract and Scope Validation
            if self.agent.current_task_contract:
                is_valid_act, act_reason = TaskContractEngine.validate_action(
                    self.agent.current_task_contract, tc.name, tc.arguments
                )
                if not is_valid_act:
                    contract_msg = f"[TASK CONTRACT & SCOPE BARRIER / GÖREV SÖZLEŞMESİ VE KAPSAM ENGELİ]: {act_reason}"
                    self.agent.event_bus.emit(EventType.WARNING, message=contract_msg)
                    self.agent.event_bus.emit(
                        EventType.TOOL_EXECUTION_COMPLETED,
                        tool_name=tc.name,
                        approved=False,
                        result=contract_msg,
                    )
                    step_tool_results.append({
                        "name": tc.name,
                        "approved": False,
                        "result": contract_msg,
                    })
                    self.agent.context.add_tool_result(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result_text=contract_msg,
                        native_tools_supported=turn_ctx.target_spec.tools,
                        from_xml=tc.from_xml,
                    )
                    continue

            result_str = self.agent.tool_executor.execute(
                tool_name=tc.name,
                arguments=tc.arguments,
                caller_role="main",
                caller_agent="orchestrator",
                is_trust_mode=self.agent.trust_mode,
                prefetched_result=prefetched_parallel_results.get(tc.id),
            )

            # Phase and test output tracking
            if tc.name in ("create_task_plan", "update_plan_step"):
                self.agent.logger.tracker.start_phase("planning")
            elif tc.name in ("write_file", "edit_file", "multi_edit_file", "write_multiple_files"):
                self.agent.logger.tracker.start_phase("coding")
            elif tc.name in ("lsp_get_diagnostics", "validate_web_contract"):
                turn_ctx.has_verification_executed_in_turn = True
                self.agent.logger.tracker.start_phase("testing")
            elif tc.name == "run_command":
                cmd_str = str(tc.arguments.get("command") or "").strip().lower()
                non_test_prefixes = (
                    "get-process", "get-content", "select-string", "get-childitem", "git status",
                    "git diff", "git log", "git show", "dir", "ls", "cat", "echo", "pwd", "which", "where"
                )
                is_pure_inspection = any(cmd_str.startswith(p) for p in non_test_prefixes)
                test_indicators = (
                    "pytest", "unittest", "test", "npm test", "yarn test", "pnpm test", "mvn test",
                    "gradle test", "cargo test", "go test", "dotnet test", "ctest", "make test", "build"
                )
                is_test_cmd = not is_pure_inspection and any(k in cmd_str for k in test_indicators)

                if is_test_cmd:
                    turn_ctx.has_verification_executed_in_turn = True
                    self.agent.logger.tracker.start_phase("testing")
                    pm = TestOutputParser.parse_output(
                        result_str,
                        workspace_root=self.agent.config.workspace_root,
                    )
                    if pm.framework != "unknown" and pm.tests_executed > 0:
                        turn_ctx.latest_test_metrics = pm

            if result_str.startswith("[GÜVENLİK ENGELİ") or result_str.startswith("[KULLANICI REDDİ") or result_str.startswith("[SECURITY POLICY BLOCK") or result_str.startswith("[USER REJECTION"):
                turn_ctx.last_step_had_error = True
                step_tool_results.append({
                    "name": tc.name,
                    "approved": False,
                    "result": result_str,
                })
                self.agent.context.add_tool_result(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result_text=result_str,
                    native_tools_supported=turn_ctx.target_spec.tools,
                    from_xml=tc.from_xml,
                )
                continue

            approved = True
            if tc.name == "remember_project_fact" and hasattr(self.agent.context, "reload_project_rules"):
                self.agent.context.reload_project_rules()
            if (
                result_str.startswith("Hata:")
                or result_str.startswith("Error:")
                or "[Canlı Sözdizimi/AST Uyarısı]" in result_str
                or "Syntax Warning" in result_str
                or "[Otonom Sözdizimi Kontrolü" in result_str
            ):
                turn_ctx.last_step_had_error = True

            if result_str.startswith("Hata:") or result_str.startswith("Error:"):
                err_cnt = turn_ctx.consecutive_tool_errors.get(tc.name, 0) + 1
                turn_ctx.consecutive_tool_errors[tc.name] = err_cnt
                if err_cnt >= 3:
                    turn_ctx.muted_tools[tc.name] = 3
                    turn_ctx.consecutive_tool_errors[tc.name] = 0
                    if tc.name == "read_file":
                        err_pivot_hint = (
                            "\n\n🛑 [Autonomous Error Breaker / Otonom Hata Kırıcı / Hallucination Guard]: 'read_file' failed 3 consecutive times "
                            "with nonexistent file error and has been temporarily muted! "
                            "Stop guessing file names immediately. "
                            "Use 'find_files' or 'get_project_tree' to locate existing files, or use 'write_file' to create a new one."
                        )
                    elif tc.name in ("edit_file", "multi_edit_file"):
                        err_pivot_hint = (
                            f"\n\n🛑 [Autonomous Error Breaker / Otonom Hata Kırıcı / Edit Guard]: '{tc.name}' failed 3 consecutive times "
                            "with target block mismatch and has been temporarily muted! "
                            "Stop blind edits: Read relevant line numbers with 'read_file' first or cleanly rewrite the file with 'write_file'."
                        )
                    else:
                        err_pivot_hint = (
                            f"\n\n🛑 [Autonomous Error Breaker / Otonom Hata Kırıcı]: Tool '{tc.name}' failed 3 consecutive times "
                            "and has been temporarily muted. Please pivot to a different tool or strategy."
                        )
                    result_str += err_pivot_hint
                    self.agent.event_bus.emit(
                        EventType.WARNING,
                        message=f"[Error Breaker / Hata Kırıcı / Otonom Hata Kırıcı]: '{tc.name}' failed 3 consecutive times and was muted.",
                    )
            else:
                turn_ctx.consecutive_tool_errors[tc.name] = 0
                if tc.name in ("read_file", "read_multiple_files"):
                    turn_ctx.muted_tools.pop("edit_file", None)
                    turn_ctx.muted_tools.pop("multi_edit_file", None)
                elif tc.name in ("find_files", "get_project_tree", "write_file", "write_multiple_files"):
                    turn_ctx.muted_tools.pop("read_file", None)

            # Add warning on 2nd consecutive invocation
            try:
                import json as _json
                tc_sig = f"{tc.name}:{_json.dumps(tc.arguments, sort_keys=True)}"
            except Exception:
                tc_sig = f"{tc.name}:{str(tc.arguments)}"
            if turn_ctx.consecutive_tool_counts.get(tc_sig, 0) == 2:
                result_str += (
                    "\n\n[Duplicate Call Notice / Mükerrer Çağrı Uyarısı]: You already executed this tool with identical parameters (az önce). "
                    "The content is present in your message history above. Instead of calling it again, "
                    "execute your next action (e.g. edit, test, or update plan)."
                )

            self.agent.event_bus.emit(
                EventType.TOOL_EXECUTION_COMPLETED,
                tool_name=tc.name,
                approved=True,
                result=result_str,
            )

            if approved and not result_str.startswith("Hata:") and not result_str.startswith("İşlem İptal"):
                has_syntax_warn = "[Canlı Sözdizimi/AST Uyarısı]" in result_str
                if tc.name in ("edit_file", "multi_edit_file", "write_file", "write_multiple_files", "append_file"):
                    turn_ctx.turn_has_file_modifications = True
                if target_path and tc.name in ("edit_file", "multi_edit_file", "write_file", "append_file"):
                    norm_path = str(Path(target_path).as_posix()).lower()
                    modified_paths_in_step[norm_path] = tc.name
                    if norm_path not in turn_ctx.modified_files_in_turn:
                        turn_ctx.modified_files_in_turn.append(norm_path)
                    if has_syntax_warn:
                        turn_ctx.files_with_syntax_errors[norm_path] = result_str.split("[Canlı Sözdizimi/AST Uyarısı]:")[-1]
                    else:
                        turn_ctx.files_with_syntax_errors.pop(norm_path, None)

                    try:
                        disk_p = Path(self.agent.workspace) / target_path if not Path(target_path).is_absolute() else Path(target_path)
                        syms = extract_exported_symbols_from_disk(disk_p)
                        if syms:
                            turn_ctx.exported_symbols_cache[norm_path] = syms
                            result_str += f"\n[Tanımlanan Semboller / Exported Symbols]: {', '.join(syms)}"
                    except Exception:
                        pass
                elif tc.name == "write_multiple_files" and isinstance(tc.arguments, dict):
                    batch_syms = []
                    for f in tc.arguments.get("files", []):
                        if isinstance(f, dict) and f.get("path"):
                            norm_p = str(Path(f["path"]).as_posix()).lower()
                            modified_paths_in_step[norm_p] = tc.name
                            if norm_p not in turn_ctx.modified_files_in_turn:
                                turn_ctx.modified_files_in_turn.append(norm_p)
                            if has_syntax_warn and f["path"] in result_str:
                                turn_ctx.files_with_syntax_errors[norm_p] = result_str
                            else:
                                turn_ctx.files_with_syntax_errors.pop(norm_p, None)

                            try:
                                disk_f = Path(self.agent.workspace) / f["path"] if not Path(f["path"]).is_absolute() else Path(f["path"])
                                syms = extract_exported_symbols_from_disk(disk_f)
                                if syms:
                                    turn_ctx.exported_symbols_cache[norm_p] = syms
                                    batch_syms.append(f"• `{f['path']}`: {', '.join(syms)}")
                            except Exception:
                                pass
                    if batch_syms:
                        result_str += "\n[Tanımlanan Semboller / Exported Symbols]:\n" + "\n".join(batch_syms)
                elif tc.name == "lsp_get_diagnostics" and isinstance(tc.arguments, dict):
                    diag_fp = str(tc.arguments.get("file_path") or "")
                    if diag_fp and ("Temiz ✔" in result_str or "No errors found" in result_str or "0 hata" in result_str.lower()):
                        norm_dp = str(Path(diag_fp).as_posix()).lower()
                        to_pop = [k for k in turn_ctx.files_with_syntax_errors if k == norm_dp or k.endswith(norm_dp) or norm_dp.endswith(k)]
                        for k in to_pop:
                            turn_ctx.files_with_syntax_errors.pop(k, None)
                elif tc.name == "run_command" and isinstance(tc.arguments, dict):
                    cmd_val = str(tc.arguments.get("command") or "").lower()
                    res_lower = result_str.lower()
                    has_zero_exit = (
                        "(exit code): 0" in res_lower
                        or "exit code: 0" in res_lower
                        or "exit=0" in res_lower
                        or "çıkış kodu): 0" in res_lower
                    )
                    if has_zero_exit and any(
                        chk in cmd_val for chk in ("tsc", "pytest", "python -m unittest", "cargo check", "go build", "npm test")
                    ):
                        turn_ctx.files_with_syntax_errors.clear()

            step_tool_results.append({
                "name": tc.name,
                "approved": approved,
                "result": result_str,
            })
            self.agent.context.add_tool_result(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result_text=result_str,
                native_tools_supported=turn_ctx.target_spec.tools,
                from_xml=tc.from_xml,
            )

        return True, step_tool_results, None
