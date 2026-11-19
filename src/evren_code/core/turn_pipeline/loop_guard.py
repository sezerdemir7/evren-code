"""Turn Loop Guard: Guard breaking redundant tool call loops, search blindness, and infinite recursions."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from evren_code.core.events import EventBus, EventType
from evren_code.core.turn_pipeline.context import TurnContext


class TurnLoopGuard:
    """Manages redundant and repetitive tool invocations, cooldowns (mute), and circuit breakers."""

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus

    def update_cooldowns(self, turn_ctx: TurnContext) -> None:
        """Decrements cooldown step counts for muted tools and unmutes those that expired."""
        to_unmute = []
        for t_name in list(turn_ctx.muted_tools.keys()):
            turn_ctx.muted_tools[t_name] -= 1
            if turn_ctx.muted_tools[t_name] <= 0:
                to_unmute.append(t_name)
        for t_name in to_unmute:
            turn_ctx.muted_tools.pop(t_name, None)

    def check_muted_tool(
        self,
        turn_ctx: TurnContext,
        tool_name: str,
    ) -> Tuple[bool, bool, Optional[str]]:
        """Checks if a currently muted tool is being invoked.
        
        Returns:
            (is_blocked, is_terminal_break, message)
        """
        if tool_name not in turn_ctx.muted_tools:
            return False, False, None

        viol = turn_ctx.mute_violations.get(tool_name, 0) + 1
        turn_ctx.mute_violations[tool_name] = viol

        if viol >= 2:
            final_break = (
                f"🛑 [Hard Circuit Breaker / Sert Devre Kesici]: Tool '{tool_name}' was repeatedly called "
                "despite being muted and instructed to pivot. "
                "Terminating turn to prevent infinite recursion."
            )
            self.event_bus.emit(
                EventType.WARNING,
                message=f"[Circuit Breaker / Devre Kesici]: '{tool_name}' infinite mute violation, turn ended.",
            )
            return True, True, final_break

        mute_msg = (
            f"🛑 [Tool Muted / Araç Kısıtlandı]: Tool '{tool_name}' was temporarily muted by the system "
            f"due to repetitive call loop ({turn_ctx.muted_tools[tool_name]} steps remaining).\n\n"
            f"Recognize Error and Pivot (Rota Düzeltme):\n"
            f"- Stop calling this tool repeatedly.\n"
            f"- The workspace context is already available; directly edit target file ('edit_file'), "
            f"create missing files ('write_file'), or run tests ('run_command')."
        )
        self.event_bus.emit(
            EventType.WARNING,
            message=f"[Course Correction / Rota Düzeltme]: '{tool_name}' call blocked, directing agent to pivot.",
        )
        return True, False, mute_msg

    def inspect_and_guard_tool_call(
        self,
        turn_ctx: TurnContext,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Guards against duplicate tool calls and search blindness loops.
        
        Returns:
            (should_block, tool_result_msg, user_injection_msg)
        """
        # 1. Track identical call signatures:
        try:
            tool_sig = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        except Exception:
            tool_sig = f"{tool_name}:{str(arguments)}"

        if tool_sig == turn_ctx.last_executed_tool_sig:
            consecutive_count = turn_ctx.consecutive_tool_counts.get(tool_sig, 1) + 1
        else:
            consecutive_count = 1
            turn_ctx.consecutive_tool_counts.clear()
        turn_ctx.consecutive_tool_counts[tool_sig] = consecutive_count
        turn_ctx.last_executed_tool_sig = tool_sig

        # 2. Track consecutive calls of the same tool even if args change:
        if tool_name == turn_ctx.last_tool_name:
            name_consecutive_count = turn_ctx.consecutive_tool_name_counts.get(tool_name, 1) + 1
        else:
            name_consecutive_count = 1
            turn_ctx.consecutive_tool_name_counts.clear()
        turn_ctx.consecutive_tool_name_counts[tool_name] = name_consecutive_count
        turn_ctx.last_tool_name = tool_name

        # Search Blindness and Loop Breaker - Autonomous Pivot:
        is_search_tool = tool_name in ("find_files", "search_code", "list_directory")
        if is_search_tool and name_consecutive_count >= 4:
            turn_ctx.muted_tools[tool_name] = 3
            turn_ctx.consecutive_tool_name_counts.clear()
            turn_ctx.consecutive_tool_counts.clear()
            pivot_msg = (
                f"🛑 [Autonomous Pivot / Otonom Rota Değiştirme]: Search tool '{tool_name}' was called "
                f"{name_consecutive_count} consecutive times with different parameters and reached a dead end.\n\n"
                f"RECOGNIZE ERROR AND PIVOT (Rota Düzeltme):\n"
                f"- The pattern or directory does not exist. Tool '{tool_name}' is muted for 3 steps.\n"
                f"- Project structure is already in context. Exit this search loop and focus directly on the task: "
                f"Read or edit files with `read_file` / `edit_file`, or create files with `write_file`."
            )
            self.event_bus.emit(
                EventType.WARNING,
                message=f"[Loop Breaker / Döngü Kırıcı / Rota Düzeltme]: '{tool_name}' search loop broken, agent pivoting.",
            )
            user_msg = (
                f"[System Intervention]: Repeated unsuccessful searches detected. Recognize error and pivot (Rota Düzeltme). "
                f"Stop using '{tool_name}' and proceed directly with modifying or creating code files."
            )
            return True, pivot_msg, user_msg

        # Hard Circuit Breaker at 4 consecutive identical calls
        if consecutive_count >= 4:
            turn_ctx.muted_tools[tool_name] = 3
            turn_ctx.consecutive_tool_counts.clear()
            turn_ctx.consecutive_tool_name_counts.clear()
            pivot_msg = (
                f"🛑 [Hard Circuit Breaker / Otonom Rota Değiştirme]: Tool '{tool_name}' was repeated with identical arguments "
                f"4 consecutive times and deadlocked.\n\n"
                f"RECOGNIZE ERROR AND PIVOT:\n"
                f"- This action is not producing the expected result; stop forcing the same call.\n"
                f"- Tool '{tool_name}' is muted for 3 steps.\n"
                f"- Change strategy immediately: use a different concrete tool (`edit_file`, `write_file`, `run_command`)."
            )
            self.event_bus.emit(
                EventType.WARNING,
                message=f"[Circuit Breaker / Devre Kesici]: '{tool_name}' loop terminated at step 4.",
            )
            user_msg = (
                f"[System Intervention / Circuit Breaker]: Deadlocked on '{tool_name}'. Stop repeating identical calls. "
                "Change strategy and proceed by editing or writing code."
            )
            return True, pivot_msg, user_msg

        # Soft Loop Breaker at 3 consecutive identical calls
        if consecutive_count >= 3:
            loop_abort_msg = (
                f"[LOOP PROTECTION ENGAGED / DÖNGÜ KORUMASI DEVREDE]: Tool '{tool_name}' was called {consecutive_count} consecutive times with identical parameters. "
                f"This search or content is already present in your context. "
                f"Redundant call loop prevented by system (mükerrer çağrı engellendi).\n\n"
                f"NOTICE: Stop repetitive searching immediately and execute the next concrete action to accomplish the goal "
                f"(e.g., read file with `read_file`, edit with `edit_file`, or execute commands with `run_command`)."
            )
            self.event_bus.emit(
                EventType.WARNING,
                message=f"[Loop Breaker / Döngü Kırıcı]: Blocked 3rd consecutive identical call for '{tool_name}' (mükerrer çağrı engellendi).",
            )
            return True, loop_abort_msg, None

        return False, None, None
