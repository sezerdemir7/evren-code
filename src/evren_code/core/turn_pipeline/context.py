"""Turn Context: Data structure maintaining execution state, counters, and parameters for a single agent turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class TurnContext:
    """Execution state, step iterations, and gate counters for a single agent turn."""

    user_input: str
    model_override: Optional[str] = None
    stream: bool = True

    # Task and context fields
    effective_task_prompt: str = ""
    enriched_input: str = ""
    is_continuation_word: bool = False
    target_spec: Any = None
    complexity_assessment: Any = None

    # Step budgets and ceilings
    base_max_steps: int = 15
    max_continues: int = 2
    total_allowed_steps: int = 45
    execution_budget: Any = None
    step: int = 0
    current_continue_count: int = 0

    # Retry and gate counters
    empty_retry_count: int = 0
    max_empty_retries: int = 2
    plan_gate_retries: int = 0
    syntax_gate_retries: int = 0
    dod_gate_retries: int = 0
    repetition_incident_count: int = 0
    overthinking_incident_count: int = 0
    last_step_overthinking: bool = False
    last_overthinking_reason: str = ""

    # Model and parameter adaptations
    temperature_override: Optional[float] = None
    active_model_escalation: Optional[str] = None

    # Verification, file, and state tracking
    has_verification_executed_in_turn: bool = False
    turn_has_file_modifications: bool = False
    active_tools_sticky: Set[str] = field(default_factory=set)
    pair_review_passed: bool = False
    modified_files_in_turn: List[str] = field(default_factory=list)
    files_with_syntax_errors: Dict[str, str] = field(default_factory=dict)
    exported_symbols_cache: Dict[str, List[str]] = field(default_factory=dict)
    last_step_had_error: bool = False
    turn_has_escalated_error: bool = False

    # Loop prevention, mute, and circuit breaker counters
    consecutive_tool_counts: Dict[str, int] = field(default_factory=dict)
    last_executed_tool_sig: Optional[str] = None
    consecutive_tool_name_counts: Dict[str, int] = field(default_factory=dict)
    last_tool_name: Optional[str] = None
    consecutive_tool_errors: Dict[str, int] = field(default_factory=dict)
    muted_tools: Dict[str, int] = field(default_factory=dict)
    mute_violations: Dict[str, int] = field(default_factory=dict)

    # Micro-compaction tracking
    last_compaction_step: int = -999
    compaction_cooldown_steps: int = 6

    # Test reporting and result fields
    latest_test_metrics: Optional[Any] = None
    final_content: str = ""
