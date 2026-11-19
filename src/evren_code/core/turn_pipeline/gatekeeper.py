"""Turn Gatekeeper: Guard managing quality gates, self-healing, DoD validation, and premature exit prevention."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evren_code.core.events import EventBus, EventType
from evren_code.core.planner import ComplexityLevel, GoalGate
from evren_code.core.self_healing import ErrorCategory, ErrorClassifier
from evren_code.core.turn_pipeline.context import TurnContext
from evren_code.core.verification.test_reporter import TestOutputParser


class TurnGatekeeper:
    """Manages completion, self-healing, pre-commit verification, DoD, syntax, and architectural review gates."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def check_all_completion_gates(
        self,
        turn_ctx: TurnContext,
        response: Any,
    ) -> bool:
        """Evaluates all completion gates sequentially.
        
        Returns:
            True: All gates passed; turn can conclude successfully.
            False: A gate intervened, feedback injected into context, step loop must continue.
        """
        # 0. Syntax Error Gatekeeper
        if not self._check_syntax_error_gate(turn_ctx, response):
            return False

        # 1. Self-Healing Verification Gate: Auto run tests after code changes, repair autonomously
        if not self._check_self_healing_gate(turn_ctx, response):
            return False

        # 2. Plan Completion Gatekeeper: Block premature completion if active plan steps remain
        if not self._check_plan_completion_gate(turn_ctx, response):
            return False

        # 3. Automated Pre-Commit Verification Gate:
        if not self._check_pre_commit_verification_gate(turn_ctx, response):
            return False

        # 4. Definition of Done & Verification Gate:
        if not self._check_dod_gate(turn_ctx, response):
            return False

        # 5. Dual-Model Pair Review / Architectural Review Gate:
        if not self._check_pair_review_gate(turn_ctx, response):
            return False

        return True

    def _check_syntax_error_gate(self, turn_ctx: TurnContext, response: Any) -> bool:
        if not turn_ctx.files_with_syntax_errors:
            return True

        try:
            from evren_code.tools.filesystem import _run_fast_inline_diagnostics
            to_remove = []
            for fp_str in list(turn_ctx.files_with_syntax_errors.keys()):
                target_path = Path(fp_str)
                if not target_path.is_absolute():
                    target_path = Path(self.agent.workspace) / fp_str
                if target_path.exists() and target_path.is_file():
                    content = target_path.read_text(encoding="utf-8", errors="ignore")
                    live_diag = _run_fast_inline_diagnostics(target_path, content)
                    if not live_diag:
                        to_remove.append(fp_str)
                elif not target_path.exists():
                    to_remove.append(fp_str)
            for fp_str in to_remove:
                turn_ctx.files_with_syntax_errors.pop(fp_str, None)
        except Exception:
            pass

        if not turn_ctx.files_with_syntax_errors:
            return True

        if turn_ctx.syntax_gate_retries < 2:
            turn_ctx.syntax_gate_retries += 1
            err_items = [f"• `{fp}`: {err.strip()}" for fp, err in turn_ctx.files_with_syntax_errors.items()]
            err_summary = "\n".join(err_items)
            self.agent.event_bus.emit(
                EventType.WARNING,
                message="[Syntax Gate / Sözdizimi Kapısı]: Unresolved syntax/AST error detected, premature exit prevented.",
            )
            native_tools = getattr(turn_ctx.target_spec, "tools", None) if turn_ctx.target_spec else None
            self.agent.context.add_assistant_message(
                content=response.content or "Proceeding to resolve syntax errors.",
                tool_calls=None,
                native_tools_supported=native_tools,
            )
            self.agent.context.add_user_message(
                f"[Syntax Auditor - Invalid Code Delivery Prevented / Sözdizimi Denetçisi - Hatalı Kod Teslimi Engellendi]: The recently edited files contain unresolved syntax/AST errors:\n{err_summary}\n\n"
                "Please fix these syntax issues using `edit_file` or `write_file` before concluding the task."
            )
            return False
        return True

    def _check_self_healing_gate(self, turn_ctx: TurnContext, response: Any) -> bool:
        if turn_ctx.has_verification_executed_in_turn and not turn_ctx.files_with_syntax_errors:
            if turn_ctx.latest_test_metrics and getattr(turn_ctx.latest_test_metrics, "tests_failed", 0) == 0:
                return True

        shm = self.agent.self_healing_manager
        if (
            turn_ctx.turn_has_file_modifications
            and shm.enabled
            and shm.has_attempts_left
        ):
            vtask = shm.detect_verification_task(
                modified_paths=turn_ctx.modified_files_in_turn
            )
            if vtask:
                self.agent.logger.tracker.start_phase("recovery")

                def _shell_exec(cmd: str, cwd: Optional[str] = None) -> str:
                    return self.agent.tool_executor.execute(
                        tool_name="run_command",
                        arguments={"command": cmd, "cwd": cwd or "."},
                        caller_role="tester",
                        caller_agent="self_healing",
                        is_trust_mode=self.agent.trust_mode,
                    )

                self.agent.event_bus.emit(
                    EventType.TOOL_EXECUTION_STARTED,
                    tool_name="self_healing_check",
                    arguments={"command": vtask.command, "target": vtask.description},
                )
                success, cmd_run, output = shm.execute_verification(
                    _shell_exec, vtask
                )
                turn_ctx.has_verification_executed_in_turn = True

                parsed_sh = TestOutputParser.parse_output(
                    output or "",
                    exit_code=0 if success else 1,
                    workspace_root=self.agent.config.workspace_root,
                )
                if parsed_sh.framework != "unknown" or parsed_sh.tests_executed > 0:
                    turn_ctx.latest_test_metrics = parsed_sh

                if not success:
                    is_sec_block = output and (
                        output.startswith("[GÜVENLİK ENGELİ")
                        or output.startswith("[KULLANICI REDDİ")
                        or output.startswith("İşlem Engellendi")
                        or output.startswith("İşlem İptal Edildi")
                    )
                    if is_sec_block:
                        self.agent.event_bus.emit(
                            EventType.WARNING,
                            message=f"[Self-Healing İptal]: Doğrulama komutu güvenlik politikası veya onay engeli nedeniyle çalıştırılmadı: {(output or '')[:120]}",
                        )
                    else:
                        err_cat, err_reason = ErrorClassifier.classify(output or "", command=cmd_run)
                        can_retry, retry_reason = shm.should_retry(err_cat, (output or "")[:100])

                        if not can_retry:
                            self.agent.event_bus.emit(
                                EventType.WARNING,
                                message=f"[Self-Healing Durduruldu / {err_cat.value}]: {retry_reason}",
                            )
                            self.agent.context.add_user_message(
                                f"[Self-Healing Terminated / {err_cat.value}]: {retry_reason}\n"
                                "Please explain the outcome to the user and summarize the current state."
                            )
                        else:
                            strategy_tip = shm.get_strategy_recommendation(
                                err_cat, cmd_run, (output or "")[:100]
                            )
                            attempt = shm.current_attempt
                            max_att = shm.max_attempts
                            self.agent.event_bus.emit(
                                EventType.WARNING,
                                message=(
                                    f"[Self-Healing {attempt}/{max_att} | {err_cat.value}]: '{cmd_run}' failed. "
                                    "Agent is adapting strategy to apply fix..."
                                ),
                            )
                            self.agent.context.add_assistant_message(
                                content=response.content or "Applied changes, executing verification now.",
                                tool_calls=None,
                                native_tools_supported=turn_ctx.target_spec.tools,
                            )
                            truncated_output = (output[:2500] + " ...[truncated]") if output and len(output) > 2500 else (output or "Exit code != 0")
                            self.agent.context.add_user_message(
                                f"[Autonomous Verification Failure (Self-Healing / {err_cat.value})]: Executed '{cmd_run}' following modifications and it failed:\n\n"
                                f"```\n{truncated_output}\n```\n\n"
                                f"{strategy_tip}\n"
                                "Please resolve this issue with a surgical code modification without repeating the failed approach."
                            )
                            return False
                else:
                    self.agent.event_bus.emit(
                        EventType.TOOL_EXECUTION_COMPLETED,
                        tool_name="self_healing_check",
                        approved=True,
                        result=f"Verification Succeeded / Doğrulama Başarılı: '{cmd_run}' tests and build passed successfully.",
                    )
                    if shm.current_attempt > 0 and hasattr(self.agent, "memory_manager"):
                        self.agent.memory_manager.remember_fact(
                            f"Project verification command `{cmd_run}` passes tests successfully.",
                            category="Build/Test",
                        )
        return True

    def _check_plan_completion_gate(self, turn_ctx: TurnContext, response: Any) -> bool:
        pm = self.agent.plan_manager
        if (
            pm.current_plan
            and pm.current_plan.active
            and not pm.current_plan.is_all_completed
            and turn_ctx.plan_gate_retries < 2
        ):
            rem_steps = pm.current_plan.remaining_steps
            rem_text = "\n".join(f"  • Step #{s.id}: {s.description}" for s in rem_steps)
            turn_ctx.plan_gate_retries += 1
            self.agent.event_bus.emit(
                EventType.WARNING,
                message="[Task Auditor / Görev Denetçisi]: Unfinished steps detected in active plan, continuing turn.",
            )
            self.agent.context.add_assistant_message(
                content=response.content or "Continuing to execute remaining plan steps.",
                tool_calls=None,
                native_tools_supported=turn_ctx.target_spec.tools,
            )
            self.agent.context.add_user_message(
                f"[Task Auditor - Early Conclusion Prevented / Görev Denetçisi - Erken Bitiş Engellendi]: You cannot conclude the task yet! "
                f"The active plan contains uncompleted steps:\n{rem_text}\n\n"
                "Please invoke the relevant tools to execute the remaining steps "
                "or update their status to 'completed' using 'update_plan_step'."
            )
            return False
        return True

    def _check_pre_commit_verification_gate(self, turn_ctx: TurnContext, response: Any) -> bool:
        if turn_ctx.turn_has_file_modifications and not turn_ctx.has_verification_executed_in_turn:
            has_web_files = any(
                fp.endswith((".html", ".htm", ".css", ".js"))
                for fp in turn_ctx.modified_files_in_turn
            ) or (self.agent.config.workspace_root / "index.html").is_file()

            if has_web_files and "validate_web_contract" in self.agent.registry._tools:
                web_contract_res = self.agent.registry.execute("validate_web_contract", {})
                if "[Web Arayüz Sözleşmesi Doğrulandı / PASS" in web_contract_res:
                    turn_ctx.has_verification_executed_in_turn = True
                    self.agent.event_bus.emit(
                        EventType.TOOL_EXECUTION_COMPLETED,
                        tool_name="validate_web_contract",
                        approved=True,
                        result=web_contract_res,
                    )
                elif "[Web Arayüz Sözleşmesi Uyuşmazlığı / FAILED]" in web_contract_res:
                    self.agent.event_bus.emit(
                        EventType.WARNING,
                        message="[Otonom Web Doğrulama]: Web arayüz sözleşmesinde uyuşmazlık tespit edildi, düzeltme talep ediliyor.",
                    )
                    self.agent.context.add_assistant_message(
                        content=response.content or "Değişiklikleri tamamladım, web arayüz sözleşmesini denetliyorum.",
                        tool_calls=None,
                        native_tools_supported=turn_ctx.target_spec.tools,
                    )
                    self.agent.context.add_user_message(
                        f"[Autonomous Web Verification Error - Contract Mismatch]:\n\n{web_contract_res}\n\n"
                        "Please fix missing DOM IDs, CSS classes, or asset mismatches using `edit_file`."
                    )
                    return False

            if (
                not turn_ctx.has_verification_executed_in_turn
                and "lsp_get_diagnostics" in self.agent.registry._tools
            ):
                code_files = [
                    fp for fp in turn_ctx.modified_files_in_turn
                    if fp.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".cs"))
                ]
                if code_files:
                    all_lsp_clean = True
                    lsp_issues = []
                    for cfp in code_files[:5]:
                        diag_out = self.agent.registry.execute("lsp_get_diagnostics", {"file_path": cfp, "path": cfp})
                        if "### LSP Teşhis Raporu" in diag_out or "### LSP Diagnostic Report" in diag_out or (diag_out.startswith("Hata:") and "bulunamadı" not in diag_out.lower()):
                            all_lsp_clean = False
                            lsp_issues.append(f"• `{cfp}`: {diag_out}")
                    if all_lsp_clean:
                        turn_ctx.has_verification_executed_in_turn = True
                        self.agent.event_bus.emit(
                            EventType.TOOL_EXECUTION_COMPLETED,
                            tool_name="lsp_get_diagnostics",
                            approved=True,
                            result="Verified 0 syntax/LSP issues in modified code files.",
                        )
                    elif lsp_issues:
                        self.agent.event_bus.emit(
                            EventType.WARNING,
                            message="[Otonom Statik Kod Doğrulama]: Değiştirilen dosyalarda LSP hataları tespit edildi.",
                        )
                        self.agent.context.add_assistant_message(
                            content=response.content or "Code written, checking static analysis.",
                            tool_calls=None,
                            native_tools_supported=turn_ctx.target_spec.tools,
                        )
                        self.agent.context.add_user_message(
                            "[Autonomous Static Code Verification Error]: Found syntax/type issues in edited files:\n"
                            + "\n".join(lsp_issues) + "\n\n"
                            "Please fix these issues using `edit_file`."
                        )
                        return False

        return True

    def _check_dod_gate(self, turn_ctx: TurnContext, response: Any) -> bool:
        is_dod_enabled = getattr(self.agent.config.agent, "dod_verification_gate_enabled", True)
        max_dod_retries = getattr(self.agent.config.agent, "max_dod_retries", 1)
        c_level = getattr(turn_ctx.complexity_assessment, "level", None)
        is_substantive_task = (
            c_level in (ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX)
            or (self.agent.current_task_contract and len(self.agent.current_task_contract.acceptance_criteria) > 0)
        ) and c_level != ComplexityLevel.SIMPLE

        needs_dod_verification = (
            is_dod_enabled
            and turn_ctx.dod_gate_retries < max_dod_retries
            and is_substantive_task
            and turn_ctx.turn_has_file_modifications
            and not turn_ctx.has_verification_executed_in_turn
        )
        if needs_dod_verification:
            turn_ctx.dod_gate_retries += 1
            self.agent.event_bus.emit(
                EventType.WARNING,
                message="[DoD Verification Gate / DoD Doğrulama Kapısı]: Task cannot conclude without verifying code changes or acceptance criteria.",
            )
            self.agent.context.add_assistant_message(
                content=response.content or "Completed modifications, proceeding to verification.",
                tool_calls=None,
                native_tools_supported=turn_ctx.target_spec.tools,
            )
            self.agent.context.add_user_message(
                "[DoD Verification Gate - Early Conclusion Prevented / DoD Doğrulama Kapısı - Erken Bitiş Engellendi]: You modified code in this turn "
                "or have mandatory acceptance criteria, but no verification tool "
                "(`run_command` with tests/build or `lsp_get_diagnostics`) HAS BEEN EXECUTED YET!\n\n"
                "Before concluding the task:\n"
                "1. Run a test or compilation command (`run_command`) proving the change works,\n"
                "2. Or verify with `lsp_get_diagnostics` that no errors remain.\n"
                "Conclude the task after verification is satisfied."
            )
            return False
        return True

    def _check_pair_review_gate(self, turn_ctx: TurnContext, response: Any) -> bool:
        if (
            getattr(self.agent.config.agent, "dual_review_enabled", False)
            and turn_ctx.turn_has_file_modifications
            and not turn_ctx.pair_review_passed
        ):
            arch_ok, critique = self.agent.review_code_with_architect(turn_ctx.modified_files_in_turn)
            if not arch_ok:
                self.agent.event_bus.emit(
                    EventType.WARNING,
                    message=f"[Architectural Review / Mimari Hakem / Pair Review]: Critical vulnerability/defect detected -> {critique[:120]}",
                )
                self.agent.context.add_assistant_message(
                    content=response.content or "Code implementation completed, submitting for architectural review.",
                    tool_calls=None,
                    native_tools_supported=turn_ctx.target_spec.tools,
                )
                self.agent.context.add_user_message(
                    f"[Architectural Review Warning - Changes Rejected]: Code review identified the following critical issue:\n\n"
                    f"{critique}\n\n"
                    "Please fix this issue using `edit_file` before concluding the task."
                )
                return False
            else:
                turn_ctx.pair_review_passed = True
                response.content = (response.content or "") + f"\n\n> 🛡️ **Mimari Hakem Onayı (Pair Review) / Architectural Approval:** {critique}"
        return True

    def evaluate_goal_gate(
        self,
        turn_ctx: TurnContext,
        step_tool_results: List[Dict[str, Any]],
    ) -> None:
        """Evaluates GoalGate completion based on executed commands and modified files."""
        recent_cmd_outputs = "\n".join(
            str(tr.get("result", ""))
            for tr in step_tool_results
            if tr.get("name") in ("run_command", "http_request")
        )
        if recent_cmd_outputs and turn_ctx.modified_files_in_turn:
            gg_res = GoalGate.evaluate(
                prompt=turn_ctx.user_input,
                modified_files=turn_ctx.modified_files_in_turn,
                recent_commands_output=recent_cmd_outputs,
                assessment=turn_ctx.complexity_assessment,
                contract=self.agent.current_task_contract,
            )
            if gg_res.is_satisfied:
                from evren_code.core.planner import ExecutionState
                self.agent.execution_state = ExecutionState.COMPLETED
                if self.agent.current_task_contract:
                    self.agent.current_task_contract.state = ExecutionState.COMPLETED
                self.agent.event_bus.emit(
                    EventType.TOOL_EXECUTION_COMPLETED,
                    tool_name="goal_gate",
                    approved=True,
                    result=f"✔ [Definition of Done - Goal Completed / Hedef Tamamlandı]: {gg_res.reason}",
                )
                self.agent.context.add_user_message(
                    f"✔ [Definition of Done (DoD) Satisfied]: {gg_res.reason}\n"
                    "Your primary objective has been successfully met. "
                    "Do NOT perform additional unsolicited actions or API data seeding. "
                    "Deliver your concise final report to the user in Turkish now."
                )
