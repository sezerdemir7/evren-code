"""Turn Pipeline: Orchestration engine coordinating phases of the ReAct agent turn loop."""

from __future__ import annotations

import difflib
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

from evren_code.core.events import EventType
from evren_code.core.planner import (
    ComplexityAssessment,
    ComplexityLevel,
    ExecutionBudget,
    ExecutionState,
    TaskComplexityClassifier,
    TaskContractEngine,
)
from evren_code.core.turn_pipeline.context import TurnContext
from evren_code.core.turn_pipeline.gatekeeper import TurnGatekeeper
from evren_code.core.turn_pipeline.loop_guard import TurnLoopGuard
from evren_code.core.turn_pipeline.tool_runner import TurnToolRunner
from evren_code.core.verification.test_reporter import ParsedTestMetrics


class TurnPipeline:
    """Modularly manages the lifecycle of a single agent turn (pre-turn, step loop, post-turn)."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.loop_guard = TurnLoopGuard(agent.event_bus)
        self.gatekeeper = TurnGatekeeper(agent)
        self.tool_runner = TurnToolRunner(agent, self.loop_guard)

    def run(
        self,
        user_input: str,
        model_override: Optional[str] = None,
        stream: bool = True,
    ) -> str:
        """Receives user input, performs preparation, executes step loop, and returns final answer."""
        turn_start_t = time.perf_counter()
        self.agent.last_turn_summary_badge = ""
        self.agent.last_turn_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # 1. Skills / slash commands expansion
        if user_input.strip().startswith("/"):
            parts = user_input.strip().split(None, 1)
            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            rendered = self.agent.skill_manager.render(cmd, args)
            if rendered:
                user_input = rendered

        # Clear completed plan from previous turn if all steps finished
        if (
            self.agent.plan_manager.current_plan
            and self.agent.plan_manager.current_plan.is_all_completed
        ):
            self.agent.plan_manager.clear()

        # Automatic User Preference Learning (Auto-Memory)
        learned_note = self.agent.context.auto_learn_user_preference(user_input)
        if learned_note:
            self.agent.event_bus.emit(EventType.INFO, message=f"🧠 {learned_note}")

        # Pre-load files mentioned in prompt and check continuation instruction
        enriched_input = self.agent._preload_prompt_mentioned_files(user_input)
        from evren_code.core.agent import is_continuation_prompt
        is_continuation_word = is_continuation_prompt(user_input)

        # Autonomous Multi-Agent Decomposition Check
        should_decompose, decompose_reason = self.agent.should_autonomously_decompose(user_input)
        if not is_continuation_word and should_decompose:
            clean_goal = user_input
            for trig in ("/multi plan", "alt ajanlara bölüp", "alt ajanlara bölerek", "alt ajanlara böl"):
                clean_goal = clean_goal.replace(trig, "").strip()
            self.agent.event_bus.emit(
                EventType.WARNING,
                message=f"⚡ [Multi-Agent Engaged / Çoklu Ajan Devrede]: {decompose_reason}",
            )
            multi_report_text = self.agent.run_multi_agent_workflow(clean_goal or user_input)
            last_rep = getattr(self.agent.multi_agent_coordinator, "last_report", None)
            
            # Graceful Fallback Safety Net:
            # If no tasks completed in multi-agent workflow (e.g. stalled or FAILED),
            # gracefully fall back to the singular ReAct loop rather than locking up.
            is_explicit_multi_cmd = user_input.strip().startswith("/multi") or any(
                tr in user_input.lower() for tr in ("alt ajanlara böl", "paralel ajan", "multi-agent")
            )
            has_no_completed = last_rep and last_rep.completed_tasks == 0
            if has_no_completed and not is_explicit_multi_cmd:
                fallback_reason = "Multi-agent workflow stalled on initial task (tamamlanamadı)"
                self.agent.event_bus.emit(
                    EventType.WARNING,
                    message=(
                        f"🛡 [Autonomous Fallback / Otomatik Devir]: {fallback_reason}. "
                        "Seamlessly handing task over to single-agent ReAct loop for step-by-step execution..."
                    ),
                )
                # Fall through to standard ReAct loop
            else:
                return multi_report_text

        # Task Boundary & Context Isolation
        if (
            self.agent.execution_state in (ExecutionState.COMPLETED, ExecutionState.CANCELLED, ExecutionState.IDLE)
            and not is_continuation_word
        ):
            self.agent.context.start_new_task(enriched_input)
            if hasattr(self.agent, "file_tracker") and self.agent.file_tracker is not None:
                self.agent.file_tracker.reset()
            self.agent.current_task_contract = TaskContractEngine.create_contract(user_input)
            self.agent._last_complexity_assessment = TaskComplexityClassifier.classify(user_input)
            self.agent._active_task_prompt = user_input
        else:
            if is_continuation_word:
                active_goal = self.agent._active_task_prompt or "Complete the previously started task thoroughly."
                continuation_text = (
                    f"[User Continuation Instruction / Kullanıcı Devam Talimatı]: {user_input}\n"
                    f"[Active Task Goal / Aktif Görev Hedefi]: {active_goal}"
                )
                self.agent.context.add_user_message(continuation_text)
                if self.agent._last_complexity_assessment and self.agent._last_complexity_assessment.level == ComplexityLevel.SIMPLE:
                    self.agent._last_complexity_assessment = TaskComplexityClassifier.classify(active_goal)
                    if self.agent._last_complexity_assessment.level == ComplexityLevel.SIMPLE:
                        self.agent._last_complexity_assessment = ComplexityAssessment(
                            level=ComplexityLevel.MEDIUM,
                            max_turns=20,
                            allow_delegation=True,
                            requires_formal_plan=True,
                            reason="Step budget expanded via continuation instruction.",
                        )
            else:
                self.agent.context.add_user_message(enriched_input)
            if not is_continuation_word or self.agent.current_task_contract is None:
                self.agent.current_task_contract = TaskContractEngine.create_contract(user_input)
                self.agent._last_complexity_assessment = TaskComplexityClassifier.classify(user_input)
                self.agent._active_task_prompt = user_input

        # Anti-loop and circuit breaker reminders
        if is_continuation_word and self.agent.last_circuit_breaker_info:
            anti_loop_reminder = (
                f"[System Reminder / Anti-Loop / Sistem Hatırlatması / Anti-Döngü]: In the previous step, tool '{self.agent.last_circuit_breaker_info}' was circuit-broken due to excessive repetition. "
                "Do not repeat failed calls with identical parameters! Inspect target file with `read_file` or cleanly rewrite with `write_file`."
            )
            self.agent.context.add_user_message(anti_loop_reminder)
            self.agent.last_circuit_breaker_info = None
        elif not is_continuation_word:
            self.agent.last_circuit_breaker_info = None

        if self.agent.last_turn_error:
            error_note = (
                f"[SYSTEM ERROR NOTIFICATION / SİSTEM HATASI BİLDİRİMİ]: In the previous turn, the following system/API error occurred: "
                f"'{self.agent.last_turn_error}'. The user may be asking about or expecting a fix. "
                "Explain it honestly and technically; never claim 'no error occurred'!"
            )
            self.agent.context.add_user_message(error_note)
            self.agent.last_turn_error = None

        self.agent.execution_state = ExecutionState.PLANNING
        self.agent._cancel_requested = False
        self.agent.last_turn_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }

        target_spec = (
            self.agent.config.models.get_model(model_override)
            if model_override
            else self.agent.llm_client.active_model
        )
        self.agent.logger.start_turn(user_input, target_spec)
        if hasattr(self.agent, "shadow_git") and self.agent.shadow_git:
            self.agent.shadow_git.create_checkpoint(label="turn_start")

        effective_task_prompt = self.agent._active_task_prompt or user_input
        complexity_assessment = (
            self.agent._last_complexity_assessment
            if (is_continuation_word and self.agent._last_complexity_assessment)
            else TaskComplexityClassifier.classify(effective_task_prompt)
        )
        self.agent._last_complexity_assessment = complexity_assessment

        base_max_steps = self.agent.config.agent.max_iterations
        if complexity_assessment.level == ComplexityLevel.SIMPLE and complexity_assessment.is_urgent:
            base_max_steps = min(base_max_steps, complexity_assessment.max_turns)
        auto_continue_enabled = getattr(self.agent.config.agent, "auto_continue", True)
        max_continues = (
            getattr(self.agent.config.agent, "max_auto_continues", 2)
            if auto_continue_enabled and not complexity_assessment.is_urgent
            else 0
        )
        total_allowed_steps = base_max_steps * (1 + max_continues)

        initial_budget_turns = max(complexity_assessment.max_turns, base_max_steps)
        execution_budget = ExecutionBudget(
            max_turns=initial_budget_turns,
            max_model_calls=initial_budget_turns * 2,
        )
        self.agent.logger.tracker.start_phase("planning")

        delegation_decision = self.agent.adaptive_delegation.evaluate_delegation(
            assessment=complexity_assessment,
            prompt=effective_task_prompt,
            token_budget_left=50000,
        )
        if delegation_decision.should_delegate:
            self.agent.logger.tracker.record_delegation()

        if self.agent.context.needs_compaction(target_spec):
            self.agent.compact_context()

        self.agent.permission_manager.current_prompt = effective_task_prompt
        self.agent.self_healing_manager.reset_turn()

        turn_ctx = TurnContext(
            user_input=user_input,
            model_override=model_override,
            stream=stream,
            effective_task_prompt=effective_task_prompt,
            enriched_input=enriched_input,
            is_continuation_word=is_continuation_word,
            target_spec=target_spec,
            complexity_assessment=complexity_assessment,
            base_max_steps=base_max_steps,
            max_continues=max_continues,
            total_allowed_steps=total_allowed_steps,
            execution_budget=execution_budget,
            max_empty_retries=self.agent.config.agent.max_empty_thinking_retries,
            compaction_cooldown_steps=max(3, getattr(self.agent.config.agent, "compaction_cooldown_steps", 6)),
        )

        step = 0
        while step < turn_ctx.total_allowed_steps:
            if self.agent._cancel_requested or self.agent.execution_state == ExecutionState.CANCELLED:
                self.agent.execution_state = ExecutionState.CANCELLED
                cancel_msg = "Operation cancelled by user request. (İşlem kullanıcı isteği ile iptal edildi.)"
                self.agent.logger.complete_turn(cancel_msg, self.agent.context.estimate_tokens())
                return cancel_msg

            if turn_ctx.execution_budget.is_exceeded():
                can_expand = (
                    step < turn_ctx.total_allowed_steps
                    and not (
                        any(cnt >= 3 for cnt in turn_ctx.consecutive_tool_counts.values())
                        or any(cnt >= 3 for cnt in turn_ctx.consecutive_tool_errors.values())
                    )
                    and (turn_ctx.turn_has_file_modifications or self.agent.plan_manager.has_active_plan or step < turn_ctx.base_max_steps)
                )
                if can_expand:
                    turn_ctx.execution_budget.max_turns = min(turn_ctx.total_allowed_steps, turn_ctx.execution_budget.max_turns + 20)
                    turn_ctx.execution_budget.max_model_calls = turn_ctx.execution_budget.max_turns * 2
                    self.agent.event_bus.emit(
                        EventType.WARNING,
                        message=f"[Dynamic Budget / Dinamik Bütçe]: Expanded step budget as task is making active progress ({step}/{turn_ctx.total_allowed_steps}).",
                    )
                else:
                    self.agent.execution_state = ExecutionState.BUDGET_EXCEEDED
                    self.agent.event_bus.emit(
                        EventType.WARNING,
                        message="[Budget Ceiling / Bütçe Sınırı]: Step budget allocated for task complexity reached.",
                    )
                    break

            step += 1
            turn_ctx.step = step
            turn_ctx.execution_budget.record_turn()
            self.agent.file_tracker.current_step = step
            self.agent.execution_state = ExecutionState.EXECUTING

            if turn_ctx.last_step_had_error:
                turn_ctx.turn_has_escalated_error = True
            turn_ctx.last_step_had_error = False

            # Update cooldowns for muted tools
            self.loop_guard.update_cooldowns(turn_ctx)

            if step > turn_ctx.base_max_steps and (step - 1) % turn_ctx.base_max_steps == 0:
                if (
                    any(cnt >= 3 for cnt in turn_ctx.consecutive_tool_counts.values())
                    or any(cnt >= 3 for cnt in turn_ctx.consecutive_tool_errors.values())
                ):
                    self.agent.event_bus.emit(
                        EventType.WARNING,
                        message="[Loop Prevention / Döngü Önleme]: Auto-continue blocked due to repetitive call loop.",
                    )
                    break

                turn_ctx.current_continue_count += 1
                self.agent.event_bus.emit(
                    EventType.WARNING,
                    message=(
                        f"[Auto-Continue / Otomatik Devam]: Reached {step - 1} step limit, "
                        f"but task is actively ongoing. Seamlessly continuing (Stage {turn_ctx.current_continue_count}/{turn_ctx.max_continues})..."
                    ),
                )

            self.agent.event_bus.emit(
                EventType.STEP_STARTED,
                step=step,
                max_steps=turn_ctx.total_allowed_steps,
                model_id=turn_ctx.target_spec.id,
                model_name=turn_ctx.target_spec.name,
            )

            # Sliding window micro eviction & historical tool pruning
            evict_interval = max(1, getattr(self.agent.config.agent, "sliding_eviction_step_interval", 5))
            evict_min_tokens = getattr(self.agent.config.agent, "sliding_eviction_min_tokens", 20000)
            evict_max_chars = getattr(self.agent.config.agent, "sliding_eviction_max_chars", 1800)
            if step > 1 and step % evict_interval == 0 and self.agent.context.estimate_tokens() >= evict_min_tokens:
                self.agent.context.evict_large_tool_results(max_chars_per_result=evict_max_chars, file_tracker=self.agent.file_tracker)

            if step > 3:
                self.agent.context.compact_sliding_window_tools(keep_recent_turns=3, max_chars_per_result=400)

            # Proactive mid-turn compaction
            if (
                step - turn_ctx.last_compaction_step >= turn_ctx.compaction_cooldown_steps
                and self.agent.context.needs_compaction(turn_ctx.target_spec)
            ):
                if self.agent.compact_context() > 0:
                    turn_ctx.last_compaction_step = step

            # Supervise background processes and reactively report crashes
            has_active_bg = False
            bg_manager = getattr(self.agent, "bg_process_manager", None)
            if bg_manager:
                for entry in bg_manager.list_processes():
                    rc = entry.process.poll()
                    if rc is None:
                        has_active_bg = True
                    elif rc != 0 and not getattr(entry, "_crash_reported", False):
                        entry._crash_reported = True
                        with entry.lock:
                            tail = list(entry.output_lines)[-10:]
                        tail_msg = "\n".join(tail) if tail else "(no output / çıktı yok)"
                        self.agent.context.add_user_message(
                            f"[System Notice - Background Process Crashed / Arka Plan Süreci Çöktü]: '{entry.process_id}' (`{entry.command}`) "
                            f"terminated unexpectedly with exit code {rc}.\nRecent outputs:\n{tail_msg}"
                        )
                        self.agent.event_bus.emit(
                            EventType.WARNING,
                            message=f"[Background Process Crashed / Arka Plan Süreci Çöktü]: {entry.process_id} (code: {rc})",
                        )

            ctx_text = ""
            if self.agent.context._messages:
                recent_msgs = self.agent.context._messages[-3:]
                ctx_text = " ".join(str(m.get("content", "")) for m in recent_msgs)

            comp_level = (
                turn_ctx.complexity_assessment.level.value
                if (turn_ctx.complexity_assessment and hasattr(turn_ctx.complexity_assessment.level, "value"))
                else None
            )
            if getattr(self.agent.config.agent, "lean_tools_mode", True):
                active_tools = self.agent.registry.select_active_tools(
                    context_text=ctx_text,
                    has_active_background=has_active_bg,
                    sticky_set=turn_ctx.active_tools_sticky,
                    complexity_level=comp_level,
                )
            else:
                active_tools = {t.name for t in self.agent.registry.list_tools()}
            turn_ctx.active_tools_sticky.update(active_tools)

            extra_sys = ""
            tools_schema = self.agent.registry.get_openai_schemas(tool_names=active_tools)
            if not turn_ctx.target_spec.tools:
                extra_sys = self.agent.registry.build_fallback_prompt_instructions(tool_names=active_tools)
                tools_schema = None

            if self.agent.sprint_manager.active_sprint:
                sp_guidance = self.agent.sprint_manager.render_active_sprint_prompt()
                if sp_guidance:
                    extra_sys = f"{extra_sys}\n\n{sp_guidance}".strip()

            active_plan_md = (
                self.agent.plan_manager.current_plan.render_markdown()
                if (self.agent.plan_manager.current_plan and hasattr(self.agent.plan_manager.current_plan, "render_markdown"))
                else ""
            )

            ephemeral_directives: List[str] = []
            if active_plan_md:
                ephemeral_directives.append(active_plan_md)
            if turn_ctx.last_step_overthinking:
                ephemeral_directives.append(
                    f"⚡ [Thinking Budget Notice / Anti-Overthinking ({turn_ctx.last_overthinking_reason})]: "
                    "In the previous step, code drafting or indecision was detected inside `<think>`. "
                    "In this step, limit thinking strictly to AT MOST 2 SENTENCES. "
                    "NEVER draft code inside thoughts; invoke concrete tools (`edit_file`, `write_file`, `read_file`, `run_command`) immediately!"
                )

            combined_ephemeral = "\n\n".join(ephemeral_directives)
            messages = self.agent.context.get_messages(
                extra_system_instructions=extra_sys,
                active_plan_markdown=combined_ephemeral,
            )

            try:
                speculate_cb = self.agent._speculate_tool if hasattr(self.agent, "_speculate_tool") else None
                response = self.agent.llm_client.chat(
                    messages=messages,
                    tools_schema=tools_schema,
                    model_override=turn_ctx.target_spec.id,
                    stream=turn_ctx.stream,
                    temperature=turn_ctx.temperature_override,
                    on_tool_speculate=speculate_cb,
                )
            except Exception as exc:
                err_str = str(exc)
                lower_err = err_str.lower()
                if "context_length_exceeded" in lower_err or "bağlam penceresini" in lower_err:
                    self.agent.context.evict_large_tool_results(max_chars_per_result=600, keep_recent=1)
                    continue

                self.agent.last_turn_error = err_str
                self.agent.event_bus.emit(
                    EventType.WARNING,
                    message=f"[LLM API Error / LLM API Hatası]: {err_str}",
                )
                self.agent.logger.complete_turn(f"Terminated with error: {err_str}", self.agent.context.estimate_tokens())
                self.agent._auto_save_session()
                return f"Error: Failed to communicate with LLM service: {err_str}"

            turn_ctx.execution_budget.record_model_call()
            if response.usage:
                self.agent.context.record_usage(response.usage)
                p_tok = int(response.usage.get("prompt_tokens", 0) or 0)
                c_tok = int(response.usage.get("completion_tokens", 0) or 0)
                self.agent.last_turn_usage["prompt_tokens"] += p_tok
                self.agent.last_turn_usage["completion_tokens"] += c_tok
                self.agent.last_turn_usage["total_tokens"] += int(response.usage.get("total_tokens", p_tok + c_tok) or 0)
                self.agent.logger.tracker.record_model_call(p_tok, c_tok, is_estimated=False)
                pt_details = response.usage.get("prompt_tokens_details") or {}
                cached_c = pt_details.get("cached_tokens", 0) if isinstance(pt_details, dict) else 0
                if not cached_c:
                    cached_c = response.usage.get("cache_read_input_tokens", 0) or response.usage.get("cachedContentTokenCount", 0) or 0
                self.agent.last_turn_usage["cached_tokens"] = self.agent.last_turn_usage.get("cached_tokens", 0) + int(cached_c or 0)
            else:
                est_in = self.agent.context.estimate_tokens()
                est_out = len(response.content or "") // 3
                self.agent.logger.tracker.record_model_call(est_in, est_out, is_estimated=True)

            # Overthinking / Rumination Detection and Tracking
            if getattr(response, "overthinking_detected", False):
                turn_ctx.last_step_overthinking = True
                turn_ctx.last_overthinking_reason = getattr(response, "overthinking_reason", "overthinking")
                turn_ctx.overthinking_incident_count += 1
            else:
                turn_ctx.last_step_overthinking = False
                turn_ctx.last_overthinking_reason = ""

            # When the model did not invoke any tools
            if not response.tool_calls:
                # 1. Repetition Loop Guard
                if getattr(response, "repetition_detected", False):
                    self.agent.event_bus.emit(
                        EventType.WARNING,
                        message=(
                            "[Loop Breaker / Döngü Kırıcı]: Repetitive text loop detected in model output, "
                            "stream terminated early. Directing model to invoke tools."
                        ),
                    )
                    partial_note = (
                        f"[System Intervention / Repetition Loop Breaker / Tekrar Döngüsü Kırıldı]: A repetitive loop of identical phrases "
                        "was detected in your output and cut short by the system.\n"
                        "Please cease repetitive phrasing and directly invoke concrete tools "
                        "(`edit_file`, `write_file`, `run_command`) to complete the task."
                    )
                    self.agent.context.add_assistant_message(
                        content=(response.content or "") + "\n... [Terminated due to repetitive text loop]",
                        tool_calls=None,
                        native_tools_supported=turn_ctx.target_spec.tools,
                        thinking=response.thinking,
                    )
                    self.agent.context.add_user_message(partial_note)
                    continue

                # 2. Auto-Continue Guard on Incomplete Turn
                if (
                    self.agent._is_incomplete_turn(response.content or "", response.thinking or "")
                    and turn_ctx.empty_retry_count < turn_ctx.max_empty_retries
                ):
                    turn_ctx.empty_retry_count += 1
                    partial_note = response.content or f"(Thinking: {(response.thinking or '')[:300]})"
                    self.agent.context.add_assistant_message(
                        content=partial_note,
                        tool_calls=None,
                        native_tools_supported=turn_ctx.target_spec.tools,
                    )
                    if "<tool_call>" in (response.content or ""):
                        self.agent.event_bus.emit(
                            EventType.WARNING,
                            message="[Syntax Recovery / Sözdizimi Kurtarma]: Raw <tool_call> detected, retrying invocation.",
                        )
                        self.agent.context.add_user_message(
                            "[Syntax Error / Sözdizimi Hatası]: You attempted to call a tool (<tool_call>) but the JSON syntax was invalid. "
                            "Please re-invoke the intended action (e.g. 'write_file' or 'edit_file') matching the standard tool schema."
                        )
                    else:
                        self.agent.context.add_user_message(
                            "[System Reminder]: You paused without invoking a tool or completing your final response. "
                            "Please invoke your planned tool now (e.g. `get_project_tree`, `read_multiple_files`, `read_file`) "
                            "or state your final answer directly."
                        )
                    continue

                # 3. Quality and Verification Gatekeepers
                gates_passed = self.gatekeeper.check_all_completion_gates(turn_ctx, response)
                if not gates_passed:
                    continue

                raw_assistant_answer = (response.content or "").strip()
                turn_ctx.final_content = raw_assistant_answer

                # Auto-record exploration summary into permanent memory if memory is empty
                if (
                    step >= 2
                    and raw_assistant_answer
                    and hasattr(self.agent, "memory_manager")
                    and self.agent.memory_manager
                ):
                    try:
                        existing_mem = self.agent.memory_manager.load_memory()
                        discovery_summary = self.agent.context._extract_turn_discovery_notes()
                        if not existing_mem and discovery_summary:
                            compact_answer = self.agent.context._clean_assistant_summary_text(raw_assistant_answer, max_chars=800)
                            auto_fact = f"Discovery Summary:\n{discovery_summary}\nArchitecture Note: {compact_answer}"
                            self.agent.memory_manager.remember_fact(auto_fact, category="Architecture Discovery")
                            self.agent.context.reload_project_rules()
                    except Exception:
                        pass

                self.agent.execution_state = ExecutionState.COMPLETED
                self.agent.context.add_assistant_message(
                    content=turn_ctx.final_content,
                    tool_calls=None,
                    native_tools_supported=turn_ctx.target_spec.tools,
                    thinking=response.thinking,
                )
                elapsed_sec = max(0.01, time.perf_counter() - turn_start_t)
                summary_badge = self.build_compact_turn_summary(turn_ctx, elapsed_sec)
                self.agent.last_turn_summary_badge = summary_badge
                if turn_ctx.modified_files_in_turn or turn_ctx.latest_test_metrics:
                    self.agent.event_bus.emit(EventType.TURN_COMPLETED, summary_badge=summary_badge)
                self.agent.logger.complete_turn(turn_ctx.final_content, self.agent.context.estimate_tokens())
                self.agent._auto_save_session()
                return turn_ctx.final_content

            # When tools were called -> execute with ToolRunner
            turn_ctx.empty_retry_count = 0
            cont, step_tool_results, term_msg = self.tool_runner.handle_tool_calls(turn_ctx, response)
            if term_msg:
                self.agent.logger.complete_turn(term_msg, self.agent.context.estimate_tokens())
                self.agent._auto_save_session()
                return term_msg
            if not cont:
                continue

            self.agent.logger.record_step(
                step=step,
                thinking=response.thinking,
                content=response.content,
                tool_calls=response.tool_calls,
                tool_results=step_tool_results,
            )

            # GoalGate evaluation
            self.gatekeeper.evaluate_goal_gate(turn_ctx, step_tool_results)
            self.agent._auto_save_session()

        # Step ceiling reached
        if not turn_ctx.final_content:
            active_plan_info = ""
            if self.agent.plan_manager.has_active_plan and self.agent.plan_manager.current_plan:
                plan = self.agent.plan_manager.current_plan
                completed = [s.description for s in plan.steps if s.status.value == "completed"]
                remaining = [s.description for s in plan.steps if s.status.value in ("pending", "in_progress")]
                if completed:
                    active_plan_info += f"\n\n**Completed Steps ({len(completed)}/{plan.total_steps}):**\n" + "\n".join(f"- ✔ {c}" for c in completed)
                if remaining:
                    active_plan_info += f"\n\n**Remaining Steps ({len(remaining)}):**\n" + "\n".join(f"- ⏳ {r}" for r in remaining)

            warning_msg = (
                f"Maximum step budget reached ({step}/{turn_ctx.total_allowed_steps}). "
                f"All changes and session state have been preserved. (Maksimum adım bütçesine ({step}/{turn_ctx.total_allowed_steps}) ulaşıldı. Tüm değişiklikler ve çalışma durumu kaydedildi.)"
                f"{active_plan_info}\n\n"
                "To continue seamlessly, you may reply with **'devam et'** / **'continue'**."
            )
        else:
            warning_msg = turn_ctx.final_content

        self.agent.event_bus.emit(EventType.WARNING, message=warning_msg)
        self.agent.logger.complete_turn(warning_msg, self.agent.context.estimate_tokens())
        self.agent._auto_save_session()
        return warning_msg

    def build_compact_turn_summary(self, turn_ctx: TurnContext, elapsed_sec: float) -> str:
        """Synthesizes a single-line engineering summary badge for the completed turn."""
        mod_files = list(dict.fromkeys(turn_ctx.modified_files_in_turn or []))
        added = 0
        removed = 0

        if mod_files and hasattr(self.agent, "undo_manager"):
            first_snaps: Dict[str, Any] = {}
            for snap in getattr(self.agent.undo_manager, "_stack", []):
                rel_key = getattr(snap, "relative_display", "").replace("\\", "/")
                if rel_key and rel_key not in first_snaps:
                    first_snaps[rel_key] = snap

            ws_root = self.agent.config.workspace_root
            for rel_fp in mod_files:
                norm_rel = rel_fp.replace("\\", "/")
                curr_p = ws_root / norm_rel
                new_lines: List[str] = []
                if curr_p.is_file():
                    try:
                        new_lines = curr_p.read_text(encoding="utf-8", errors="replace").splitlines()
                    except Exception:
                        new_lines = []
                snap_obj = first_snaps.get(norm_rel)
                if snap_obj is not None:
                    old_text = getattr(snap_obj, "previous_content", "") or ""
                    old_lines = old_text.splitlines() if getattr(snap_obj, "existed_before", False) else []
                else:
                    old_lines = []
                for dline in difflib.ndiff(old_lines, new_lines):
                    if dline.startswith("+ "):
                        added += 1
                    elif dline.startswith("- "):
                        removed += 1

        parts: List[str] = []
        if mod_files:
            parts.append(f"✔ {len(mod_files)} dosya düzenlendi (+{added} / -{removed} satır)")
        else:
            parts.append("✔ Tur tamamlandı")

        if turn_ctx.latest_test_metrics is not None:
            m = turn_ctx.latest_test_metrics
            total_passed = getattr(m, "passed", 0)
            total_failed = getattr(m, "failed", 0) + getattr(m, "errors", 0)
            if total_passed > 0 and total_failed == 0:
                parts.append(f"{total_passed} test geçti")
            elif total_failed > 0:
                parts.append(f"{total_passed} geçti / {total_failed} başarısız")

        parts.append(f"{elapsed_sec:.1f}s")

        tokens = (
            self.agent.last_turn_usage.get("total_tokens", 0)
            or self.agent.context.estimate_tokens()
        )
        if tokens >= 1000:
            parts.append(f"~{tokens / 1000.0:.1f}K token")
        else:
            parts.append(f"~{tokens} token")

        return " · ".join(parts)
