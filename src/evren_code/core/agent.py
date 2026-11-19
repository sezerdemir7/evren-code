"""ReAct Agent Orchestrator (`AgentOrchestrator`).

Manages the Think -> Act -> Observe -> Self-Heal turn loop:
- Auto-Continue Guard: Automatically resumes if model stops prematurely with thoughts only.
- UndoManager: Preserves undo history (`/undo`) across file modifications.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

from evren_code.config import AppConfig
from evren_code.core.context import ContextManager
from evren_code.core.events import (
    ApprovalDelegate,
    ApprovalRequest,
    EventBus,
    EventType,
)
from evren_code.core.file_tracker import FileStateTracker
from evren_code.core.llm_client import EvrenAPIError, EvrenLLMClient
from evren_code.core.memory import ProjectMemoryManager, register_memory_tools
from evren_code.core.repo_intelligence import (
    RepoIntelligenceEngine,
    register_repo_intelligence_tools,
)
from evren_code.core.symbol_graph import (
    CodebaseSymbolEngine,
    register_symbol_graph_tools,
)
from evren_code.core.verification import (
    VerificationEngine,
    register_verification_tools,
)
from evren_code.core.planner import (
    ComplexityAssessment,
    ComplexityLevel,
    ExecutionBudget,
    ExecutionState,
    GoalGate,
    PlanManager,
    TaskComplexityClassifier,
    TaskContract,
    TaskContractEngine,
    TaskPlan,
)
from evren_code.core.self_healing import (
    ErrorCategory,
    ErrorClassifier,
    SelfHealingManager,
)
from evren_code.core.verification.test_reporter import (
    ParsedTestMetrics,
    TestOutputParser,
    TestReportingEngine,
    TestVerificationStatus,
    VerificationSummaryReport,
)
from evren_code.core.multi_agent.adaptive_delegation import (
    AdaptiveDelegationEngine,
    DelegationDecision,
)
from evren_code.core.session import SessionManager
from evren_code.core.subagents import SubagentManager
from evren_code.core.hooks import HookManager
from evren_code.core.security import PermissionManager, ToolExecutor, PermissionMode
from evren_code.core.skills import SkillManager
from evren_code.core.telemetry import TrajectoryLogger
from evren_code.core.turn_pipeline import TurnPipeline
from evren_code.core.undo import UndoManager
from evren_code.tools.filesystem import register_filesystem_tools
from evren_code.tools.git_tools import register_git_tools
from evren_code.tools.lsp import register_lsp_tools
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.profiler import register_profiler_tools
from evren_code.tools.repo_map import register_repo_map_tools
from evren_code.tools.search import register_search_tools
from evren_code.tools.semantic_search import register_semantic_search_tools
from evren_code.tools.shell import register_shell_tools
from evren_code.tools.skeletonizer import register_skeletonizer_tools
from evren_code.tools.vision import register_vision_tools
from evren_code.tools.web import register_web_tools


def _load_default_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "system.md"
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8")
    return "You are Evren-Code, an autonomous AI software engineer."


def is_continuation_prompt(text: str) -> bool:
    """Determines whether user input is a continuation or confirmation prompt."""
    cleaned = text.strip().lower()
    exact_words = {
        "devam", "devam et", "devam ettir", "sürdür", "continue", "evet", "hayır", "yes", "no", "tamam", "ok", "proceed"
    }
    if cleaned in exact_words:
        return True
    prefixes = (
        "devam", "kaldığın yerden", "yarım kalan", "son işlemi", "kaldığı yerden", "devam ettir", "işleme devam"
    )
    if any(cleaned.startswith(p) for p in prefixes):
        return True
    if any(w in cleaned for w in ("yarım kalan", "kaldığın yerden", "devam ettir", "kaldığı yerden devam")):
        return True
    return False


class DefaultAutoApproveDelegate:
    """Default fail-safe approval delegate when no interactive delegate is provided."""

    def request_approval(self, request: ApprovalRequest) -> bool:
        # 1. Reject critical or dangerous operations
        if getattr(request, "risk_level", "standard").lower() in ("critical", "dangerous", "high"):
            return False

        # 2. Inspect destructive shell commands
        if request.tool_name == "run_command":
            cmd = str(request.arguments.get("command", ""))
            from evren_code.core.security import CommandClassifier
            is_danger, _ = CommandClassifier.analyze(cmd)
            if is_danger:
                return False

        # 3. Reject destructive git or delete operations
        if request.tool_name in ("git_reset_hard", "delete_file"):
            return False

        return True

    def ask_user_question(self, question: str, options: Optional[List[str]] = None) -> str:
        if options:
            return options[0]
        return "(Auto mode: Default option accepted / Otomatik mod: Varsayılan seçenek kabul edildi)"


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
}


class AgentOrchestrator:
    """Manages the full agent turn loop, tool execution, and multi-model orchestration."""

    def __init__(
        self,
        config: AppConfig,
        llm_client: Optional[EvrenLLMClient] = None,
        event_bus: Optional[EventBus] = None,
        approval_delegate: Optional[ApprovalDelegate] = None,
        system_prompt: Optional[str] = None,
        undo_manager: Optional[UndoManager] = None,
        permission_manager: Optional[PermissionManager] = None,
        hook_manager: Optional[HookManager] = None,
        skill_manager: Optional[SkillManager] = None,
        memory_manager: Optional[ProjectMemoryManager] = None,
        repo_intelligence: Optional[RepoIntelligenceEngine] = None,
        symbol_engine: Optional[CodebaseSymbolEngine] = None,
        file_tracker: Optional[FileStateTracker] = None,
        verification_engine: Optional[VerificationEngine] = None,
        session_manager: Optional[SessionManager] = None,
        plan_manager: Optional[PlanManager] = None,
        subagent_manager: Optional[SubagentManager] = None,
        turn_pipeline: Optional[TurnPipeline] = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus or EventBus()
        self.llm_client = llm_client or EvrenLLMClient(config, self.event_bus)
        self.approval_delegate: ApprovalDelegate = (
            approval_delegate or DefaultAutoApproveDelegate()
        )
        self._trust_mode: bool = False
        self.undo_manager = undo_manager or UndoManager(
            max_history=config.tools.max_undo_history,
        )
        self.permission_manager = permission_manager or PermissionManager(
            workspace_root=config.workspace_root,
            require_approval_for=config.tools.require_approval_for,
        )
        self.hook_manager = hook_manager or HookManager(config.workspace_root, self.event_bus)
        self.skill_manager = skill_manager or SkillManager(config.workspace_root)
        self.memory_manager = memory_manager or ProjectMemoryManager(config.workspace_root)
        self.repo_intelligence = repo_intelligence or RepoIntelligenceEngine(config.workspace_root, config)
        self.symbol_engine = symbol_engine or CodebaseSymbolEngine(config.workspace_root, config)
        self.file_tracker = file_tracker or FileStateTracker()
        from evren_code.tools.git_tools import ShadowGitManager
        self.shadow_git = ShadowGitManager(config.workspace_root)
        self.logger = TrajectoryLogger(config)
        self.event_bus.subscribe(
            lambda ev: self.logger.tracker.record_tool_timeout(
                watchdog_intervened=bool(ev.data.get("watchdog_intervened", False))
            ),
            EventType.TOOL_EXECUTION_TIMEOUT,
        )
        self.event_bus.subscribe(
            lambda _ev: self.logger.tracker.record_tool_cancelled(),
            EventType.TOOL_EXECUTION_CANCELLED,
        )
        self.last_circuit_breaker_info: Optional[str] = None
        self.last_turn_error: Optional[str] = None
        self.last_turn_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.self_healing_manager = SelfHealingManager(
            workspace_root=config.workspace_root,
            max_attempts=config.agent.self_healing.max_attempts,
            enabled=config.agent.self_healing.enabled,
            auto_detect=config.agent.self_healing.auto_detect,
        )

        base_prompt = system_prompt or _load_default_system_prompt()
        self.context = ContextManager(config, base_prompt)

        self.registry = ToolRegistry(
            require_approval_for=config.tools.require_approval_for
        )

        self.tool_executor = ToolExecutor(
            registry=self.registry,
            permission_manager=self.permission_manager,
            approval_delegate=self.approval_delegate,
            event_bus=self.event_bus,
            max_output_chars=getattr(config.tools, "max_output_chars", 16000),
        )

        self.verification_engine = verification_engine or VerificationEngine(
            workspace_root=config.workspace_root,
            tool_executor=self.tool_executor,
            repo_intelligence=self.repo_intelligence,
            symbol_engine=self.symbol_engine,
        )

        self.session_manager = session_manager or SessionManager(self.config)
        self.plan_manager = plan_manager or PlanManager(self.event_bus)
        self.subagent_manager = subagent_manager or SubagentManager(
            config=self.config,
            registry=self.registry,
            llm_client=self.llm_client,
            event_bus=self.event_bus,
            tool_executor=self.tool_executor,
            permission_manager=self.permission_manager,
            approval_delegate=self.approval_delegate,
        )

        from evren_code.core.multi_agent.coordinator import MultiAgentCoordinator
        from evren_code.core.multi_agent.models import MultiAgentRunConfig

        multi_run_cfg = MultiAgentRunConfig(
            enabled=getattr(self.config.multi_agent, "enabled", False),
            auto_decompose=getattr(self.config.multi_agent, "auto_decompose", True),
            max_parallel_agents=getattr(self.config.multi_agent, "max_parallel_agents", 4),
            task_timeout_seconds=getattr(self.config.multi_agent, "task_timeout_seconds", 300.0),
            max_retry_attempts=getattr(self.config.multi_agent, "max_retry_attempts", 2),
            auto_review=getattr(self.config.multi_agent, "auto_review", True),
            verify_with_engine=getattr(self.config.multi_agent, "verify_with_engine", True),
            heartbeat_interval_seconds=getattr(self.config.multi_agent, "heartbeat_interval_seconds", 10.0),
            recover_on_timeout=getattr(self.config.multi_agent, "recover_on_timeout", True),
            persistence_dir=getattr(self.config.multi_agent, "persistence_dir", ".evren/multi_agent"),
        )
        self.multi_agent_coordinator = MultiAgentCoordinator(
            config=self.config,
            llm_client=self.llm_client,
            tool_executor=self.tool_executor,
            registry=self.registry,
            repo_intelligence=self.repo_intelligence,
            symbol_engine=self.symbol_engine,
            verification_engine=self.verification_engine,
            event_bus=self.event_bus,
            multi_config=multi_run_cfg,
            shadow_git=self.shadow_git,
        )

        self.adaptive_delegation = AdaptiveDelegationEngine(
            workspace_root=self.config.workspace_root,
            file_ownership=self.multi_agent_coordinator.file_ownership,
        )

        from evren_code.core.sprint.manager import SprintManager
        from evren_code.core.sprint.models import SprintConfig

        sp_cfg = getattr(self.config, "sprint", None)
        sprint_run_cfg = SprintConfig(
            enabled=getattr(sp_cfg, "enabled", False) if sp_cfg else False,
            max_iterations=getattr(sp_cfg, "max_iterations", 5) if sp_cfg else 5,
            max_tasks_per_sprint=getattr(sp_cfg, "max_tasks_per_sprint", 30) if sp_cfg else 30,
            max_parallel_agents=getattr(sp_cfg, "max_parallel_agents", 3) if sp_cfg else 3,
            max_retries_per_task=getattr(sp_cfg, "max_retries_per_task", 2) if sp_cfg else 2,
            stop_on_critical_failure=getattr(sp_cfg, "stop_on_critical_failure", True) if sp_cfg else True,
            require_reviewer_approval=getattr(sp_cfg, "require_reviewer_approval", True) if sp_cfg else True,
            require_regression_check=getattr(sp_cfg, "require_regression_check", True) if sp_cfg else True,
            require_acceptance_criteria=getattr(sp_cfg, "require_acceptance_criteria", True) if sp_cfg else True,
            checkpoint_interval=getattr(sp_cfg, "checkpoint_interval", 1) if sp_cfg else 1,
            resume_interrupted_sprints=getattr(sp_cfg, "resume_interrupted_sprints", True) if sp_cfg else True,
            storage_dir=getattr(sp_cfg, "storage_dir", ".evren/sprints") if sp_cfg else ".evren/sprints",
        )
        self.sprint_manager = SprintManager(
            config=self.config,
            llm_client=self.llm_client,
            tool_executor=self.tool_executor,
            registry=self.registry,
            coordinator=self.multi_agent_coordinator,
            repo_intelligence=self.repo_intelligence,
            symbol_engine=self.symbol_engine,
            verification_engine=self.verification_engine,
            event_bus=self.event_bus,
            sprint_config=sprint_run_cfg,
        )

        self.execution_state: ExecutionState = ExecutionState.IDLE
        self.current_task_contract: Optional[TaskContract] = None
        self._last_complexity_assessment: Optional[Any] = None
        self._active_task_prompt: str = ""
        self._cancel_requested: bool = False

        self._speculative_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="evren_speculative"
        )
        self._active_speculative_futures: Dict[str, concurrent.futures.Future] = {}

        self._register_builtin_tools()
        try:
            setattr(
                self.llm_client,
                "_fallback_tool_instructions",
                self.registry.build_fallback_prompt_instructions(),
            )
        except Exception as exc:
            logger.debug("LLM istemcisine yedek araç yönergeleri atanırken hata: %s", exc)

        self.turn_pipeline = turn_pipeline or TurnPipeline(self)

    @property
    def trust_mode(self) -> bool:
        """Trust mode."""
        return getattr(self, "_trust_mode", False)

    @trust_mode.setter
    def trust_mode(self, val: bool) -> None:
        """Trust mode."""
        self._trust_mode = bool(val)
        new_mode = PermissionMode.BYPASS if val else PermissionMode.DEFAULT
        if hasattr(self, "permission_manager") and self.permission_manager:
            self.permission_manager.mode = new_mode
        if hasattr(self, "tool_executor") and self.tool_executor and hasattr(self.tool_executor, "permission_manager"):
            self.tool_executor.permission_manager.mode = new_mode

    def _register_builtin_tools(self) -> None:
        from evren_code.tools.lsp import LSPClient
        self.lsp_client = LSPClient(self.config.workspace_root, symbol_engine=self.symbol_engine)

        register_filesystem_tools(
            self.registry,
            self.config,
            self.undo_manager,
            self.file_tracker,
            lsp_manager=self.lsp_client,
        )
        register_search_tools(self.registry, self.config)
        register_repo_map_tools(self.registry, self.config)
        register_semantic_search_tools(self.registry, self.config)
        register_lsp_tools(self.registry, self.config, symbol_engine=self.symbol_engine)
        register_git_tools(self.registry, self.config)
        register_web_tools(self.registry, self.config)
        self.bg_process_manager = register_shell_tools(self.registry, self.config)
        register_vision_tools(self.registry, self.config, self.llm_client)
        register_memory_tools(self.registry, self.memory_manager)
        register_repo_intelligence_tools(self.registry, self.repo_intelligence)
        register_symbol_graph_tools(self.registry, self.symbol_engine)
        register_verification_tools(self.registry, self.verification_engine, llm_client=self.llm_client)
        register_skeletonizer_tools(self.registry, self.config)
        register_profiler_tools(self.registry, self.config)
        self._register_planner_tools()
        self._register_subagent_tools()
        self._register_interaction_tools()
        self._register_external_mcp_tools()

    def _register_external_mcp_tools(self) -> None:
        """Discovers and registers tools from external MCP servers configured in `.evren/mcp.json`."""
        try:
            from evren_code.core.mcp_client import MCPClientManager

            self.mcp_client_manager = MCPClientManager(
                workspace_root=self.config.workspace_root,
                global_dir=self.config.api.storage_dir,
            )
            self.mcp_client_manager.connect_and_register_tools(self.registry)
        except Exception as exc:
            logger.debug("External MCP client registration skipped: %s", exc)

    def _register_interaction_tools(self) -> None:
        """Register interaction tools."""

        def ask_user(question: str, options: Optional[List[str]] = None) -> str:
            if hasattr(self.approval_delegate, "ask_user_question"):
                ans = self.approval_delegate.ask_user_question(question, options)
            else:
                ans = options[0] if options else "(Yanıt alınamadı)"
            return f"Kullanıcı Yanıtı: {ans}"

        self.registry.register(
            name="ask_user",
            description=(
                "Asks the user a clarifying question when facing critical ambiguities, missing domain requirements, "
                "or competing architectural trade-offs, offering multiple-choice options when applicable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Clear question directed to the user."},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of recommended choices (2-4 options recommended).",
                    },
                },
                "required": ["question"],
            },
            handler=ask_user,
            requires_approval=False,
        )

    def _register_planner_tools(self) -> None:
        """Registers task planning and tracking tools ('create_task_plan', 'update_plan_step', 'get_task_plan')."""

        def create_task_plan(title: str, steps: Any) -> str:
            plan = self.plan_manager.create_plan(title, steps)
            return (
                f"Success: Task plan '{plan.title}' created with {plan.total_steps} steps ({plan.total_steps} adımla oluşturuldu).\n"
                f"{plan.render_markdown()}"
            )

        def update_plan_step(step_id: int, status: str, notes: Optional[str] = None) -> str:
            ok = self.plan_manager.update_step(step_id, status, notes or "")
            if not ok:
                return f"Error: Step #{step_id} not found in the active task plan."
            plan = self.plan_manager.current_plan
            progress = f" ({plan.completed_steps}/{plan.total_steps} completed)" if plan else ""
            return f"Success: Step #{step_id} updated to '{status}' (durumu '{status}' olarak güncellendi).{progress}"

        def get_task_plan() -> str:
            plan = self.plan_manager.current_plan
            if not plan:
                return "No active task plan found. Call 'create_task_plan' to initialize a new plan."
            return plan.render_markdown()

        self.registry.register(
            name="create_task_plan",
            description=(
                "Initializes a structured task plan for multi-step engineering tasks to maintain focus and prevent deviation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Concise plan title."},
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sequential list of 3-7 concrete execution steps.",
                    },
                },
                "required": ["title", "steps"],
            },
            handler=create_task_plan,
            requires_approval=False,
        )

        self.registry.register(
            name="update_plan_step",
            description=(
                "Updates step status in the active plan ('in_progress', 'completed', 'failed', 'skipped'). "
                "Always invoke alongside the concrete tool call in the same turn step."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "step_id": {"type": "integer", "description": "Step index (1, 2, ...)."},
                    "status": {
                        "type": "string",
                        "enum": ["in_progress", "completed", "failed", "skipped"],
                        "description": "Target step status.",
                    },
                    "notes": {"type": "string", "description": "Optional concise findings or notes."},
                },
                "required": ["step_id", "status"],
            },
            handler=update_plan_step,
            requires_approval=False,
        )

        self.registry.register(
            name="get_task_plan",
            description="Retrieves the current task plan and completion status.",
            parameters={"type": "object", "properties": {}},
            handler=get_task_plan,
            requires_approval=False,
        )

    def _register_subagent_tools(self) -> None:
        """Registers the subagent delegation tool ('delegate_subagent')."""

        def delegate_subagent(role: str, prompt: str) -> str:
            return self.subagent_manager.delegate(role=role, task_prompt=prompt)

        self.registry.register(
            name="delegate_subagent",
            description=(
                "Delegates a self-contained subtask to an isolated specialist subagent without cluttering main context. "
                "Roles: 'researcher' (read-only codebase/docs exploration), "
                "'architect' (high-level design and algorithmic evaluation), "
                "'tester' (runs test suites and diagnoses failures), "
                "'reviewer' (evaluates git diffs and acceptance criteria)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["researcher", "architect", "tester", "reviewer"],
                        "description": "Specialist subagent role.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Clear and detailed task description for the subagent.",
                    },
                },
                "required": ["role", "prompt"],
            },
            handler=delegate_subagent,
            requires_approval=False,
        )

    def _summarize_with_compactor_model(self, messages: List[Dict[str, Any]]) -> str:
        """Summarize with compactor model (micro-notes)."""
        compactor_spec = self.config.models.get_model("compactor")
        notes: List[str] = []
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                u_text = str(msg.get("content") or "")[:250].strip()
                if u_text:
                    notes.append(f"Kullanıcı İstemi: {u_text}")
            elif role == "assistant":
                tcs = msg.get("tool_calls")
                if tcs:
                    t_names = [
                        tc.get("function", {}).get("name")
                        for tc in tcs
                        if isinstance(tc, dict) and tc.get("function", {}).get("name")
                    ]
                    if t_names:
                        notes.append(f"Asistan Çağrısı: {', '.join(t_names)}")
                elif msg.get("content"):
                    c_text = str(msg["content"])[:200].strip()
                    if c_text and not c_text.startswith("{"):
                        notes.append(f"Asistan Notu: {c_text}")
            elif role == "tool":
                t_name = msg.get("name", "araç")
                raw_c = str(msg.get("content") or "")
                first_line = raw_c.splitlines()[0] if raw_c.splitlines() else ""
                notes.append(f"Sonuç ({t_name}): {first_line[:120]}")

        distilled_notes = "\n".join(notes[:40]) if notes else json.dumps(messages, ensure_ascii=False)[:4000]

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "Aşağıdaki yazılım geliştirme adım notlarını özetle. "
                    "Değiştirilen dosyaları, alınan mimari kararları ve açık kalan hedefleri maddeler halinde koru.\n\n"
                    "ÖNEMLİ KURALLAR:\n"
                    "- Asistanın başarısız olmuş araç denemelerini ('hedef blok bulunamadı', 'hatalı satırları düzeltiyorum' gibi niyet cümlelerini) aynen kopyalama.\n"
                    "- Yalnızca tamamlanmış somut dosya durumlarını ve açık kalan ana hedefleri teknik olarak özetle.\n"
                    "- Özet metninde asistan taklidi yapma, doğrudan üçüncü şahıs teknik durum özeti sun."
                ),
            },
            {"role": "user", "content": distilled_notes},
        ]
        resp = self.llm_client.chat(
            messages=prompt_messages,
            tools_schema=None,
            model_override=compactor_spec.id,
            stream=False,
            silent=True,
        )
        return resp.content

    def cancel(self) -> None:
        """Cancel."""
        self._cancel_requested = True
        self.execution_state = ExecutionState.CANCELLED
        if hasattr(self.llm_client, "cancel"):
            self.llm_client.cancel()
        if hasattr(self, "tool_executor") and self.tool_executor and hasattr(self.tool_executor, "cancel_all"):
            try:
                self.tool_executor.cancel_all(reason="User requested orchestrator cancellation")
            except Exception as exc:
                logger.warning("İptal anında aktif araç çağrıları durdurulurken hata: %s", exc)
        if hasattr(self, "bg_process_manager") and self.bg_process_manager:
            try:
                self.bg_process_manager.stop_all()
            except Exception as exc:
                logger.warning("İptal anında arka plan süreçleri durdurulurken hata: %s", exc)
        self.event_bus.emit(
            EventType.WARNING,
            message="[Görev İptal Edildi]: Çalışan tüm işlemler ve süreçler kullanıcı isteğiyle durduruldu.",
        )

    def compact_context(self, aggressive: bool = False) -> int:
        """Compact context."""
        compacted_count = self.context.compact(
            summarizer_fn=None if aggressive else self._summarize_with_compactor_model,
            aggressive=aggressive,
            file_tracker=self.file_tracker,
        )
        if compacted_count > 0:
            self.file_tracker.reset()
            self.event_bus.emit(
                EventType.CONTEXT_COMPACTED,
                compacted_messages=compacted_count,
                estimated_tokens=self.context.estimate_tokens(),
            )
            self._auto_save_session()
        return compacted_count

    def _speculate_tool(self, tool_name: str, arguments: Dict[str, Any], tc_id: str) -> None:
        """Speculate tool."""
        if tool_name not in PARALLEL_SAFE_TOOLS:
            return
        tool_def = self.registry.get(tool_name)
        if not tool_def:
            return
        configured_req = bool(
            getattr(tool_def, "requires_approval", False)
            or tool_name in self.permission_manager.configured_requires_approval
        )
        decision = self.permission_manager.check(
            tool_name=tool_name,
            arguments=arguments,
            is_trust_mode=self.trust_mode,
            configured_requires_approval=configured_req,
            caller_role="main",
            caller_id="orchestrator_speculative",
        )
        if not decision.allowed or decision.requires_approval:
            return
        if tc_id in self._active_speculative_futures:
            return

        fut = self._speculative_pool.submit(
            self.tool_executor.execute,
            tool_name,
            arguments,
            caller_role="main",
            caller_agent="orchestrator_speculative",
            is_trust_mode=self.trust_mode,
        )
        self._active_speculative_futures[tc_id] = fut

    def review_code_with_architect(
        self,
        modified_paths: List[str],
        step_diffs: str = "",
    ) -> Tuple[bool, str]:
        """Review code with architect (Dual-Model Pair Review / Consensus Engine)."""
        arch_model_id = self.config.models.roles.get("architect") or "deepseek-v4.1-flash"
        review_prompt = (
            "Sen kıdemli bir Sistem Mimarı ve Güvenlik Denetçisisin (Code Reviewer).\n"
            "Fast Coder tarafından az önce yapılan şu kod değişikliklerini denetle:\n\n"
            f"Değiştirilen Dosyalar: {', '.join(modified_paths)}\n"
            f"Diff Özeti:\n```diff\n{step_diffs[:3500]}\n```\n\n"
            "Denetim Kriterleri:\n"
            "1. Kritik Güvenlik Açığı (SQLi, XSS, Command Injection, hardcoded secrets).\n"
            "2. Bariz Çökme / Mantık Hatası (Null pointer, unhandled exception, bozuk imza).\n\n"
            "Karar Biçimi (Zorunlu):\n"
            "- Kod güvenli ise sadece 'APPROVED' yaz.\n"
            "- Eğer kritik hata veya açık varsa 'REJECTED: <hata açıklaması ve çözüm>' yaz."
        )

        try:
            resp = self.llm_client.chat(
                messages=[{"role": "user", "content": review_prompt}],
                model_override=arch_model_id,
                stream=False,
                silent=True,
            )
            content = (resp.content or "").strip()
            if content.startswith("APPROVED") or "APPROVED" in content[:30]:
                return True, "Kod mimari hakem tarafından onaylandı (APPROVED)."
            elif content.startswith("REJECTED") or "REJECTED" in content[:30]:
                critique = content.replace("REJECTED:", "").strip()
                return False, critique
            return True, "Onaylandı"
        except Exception as exc:
            return True, f"Hakem denetimi atlandı: {exc}"

    @staticmethod
    def _is_incomplete_turn(content: str, thinking: str) -> bool:
        """Is incomplete turn."""
        stripped = content.strip()
        if not stripped and thinking.strip():
            return True
        if stripped.endswith(":") and len(stripped) < 180:
            return True
        if "<tool_call>" in stripped:
            return True
        return False

    def run_verification(
        self,
        changed_files: Optional[List[str]] = None,
        task_description: str = "",
        auto_correct: bool = True,
    ) -> Any:
        """Run verification."""
        target_files = changed_files or []

        def _orchestrator_correction_handler(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
            if not auto_correct:
                return False, []
            prompt = plan.get("correction_prompt", "")
            before_files: Dict[str, str] = {}
            for f in plan.get("target_files", target_files):
                p = (self.config.workspace_root / f).resolve()
                if p.is_file():
                    before_files[f] = p.read_text(encoding="utf-8", errors="replace")

            self.run_turn(prompt)

            modified: List[str] = []
            for f, old_content in before_files.items():
                p = (self.config.workspace_root / f).resolve()
                if p.is_file() and p.read_text(encoding="utf-8", errors="replace") != old_content:
                    modified.append(f)
            return len(modified) > 0, modified

        report = self.verification_engine.run_verification(
            changed_files=target_files,
            task_description=task_description,
            orchestrator_correction_handler=_orchestrator_correction_handler if auto_correct else None,
            is_trust_mode=self.trust_mode,
        )
        return report

    def _preload_prompt_mentioned_files(self, user_input: str) -> str:
        """Preload prompt mentioned files."""
        import re
        from evren_code.tools.filesystem import resolve_safe_path

        if "[@" in user_input or "[Otomatik Ön-Yüklenen" in user_input:
            return user_input

        pattern = r"(?<![@\w/\\])([a-zA-Z0-9_\-\./\\]+\.(?:py|ts|tsx|js|jsx|json|yaml|yml|java|go|rs|css|html|cs))\b"
        raw_matches = re.findall(pattern, user_input)

        max_files = getattr(self.config.tools, "auto_preload_max_files", 3)
        max_lines = getattr(self.config.tools, "auto_preload_max_lines", 250)

        symbol_candidates = re.findall(r"\b([A-Z][a-zA-Z0-9_]{3,})\b", user_input)
        for sym in symbol_candidates:
            if len(raw_matches) >= max_files * 2:
                break
            for ext in (".cs", ".py", ".ts", ".js"):
                cand_name = f"{sym}{ext}"
                try:
                    resolved = resolve_safe_path(self.config.workspace_root, cand_name)
                    if resolved.is_file():
                        raw_matches.append(cand_name)
                        break
                except Exception:
                    pass

        if not raw_matches:
            return user_input

        seen: Set[str] = set()
        preloaded_blocks: List[str] = []

        for rel_cand in raw_matches:
            norm_cand = rel_cand.replace("\\", "/").strip("./")
            if not norm_cand or norm_cand in seen:
                continue
            seen.add(norm_cand)
            try:
                target = resolve_safe_path(self.config.workspace_root, norm_cand)
                if not target.is_file() or target.stat().st_size > 65536:
                    continue
                content = target.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                if not lines:
                    continue
                if len(lines) > max_lines:
                    try:
                        from evren_code.tools.skeletonizer import skeletonize_file_content
                        ok, skel_text, stats = skeletonize_file_content(
                            self.config.workspace_root,
                            norm_cand,
                            include_docstrings=True,
                        )
                        if ok and skel_text:
                            skel_lines = skel_text.splitlines()
                            numbered = "\n".join(f"{i:4d} | {ln}" for i, ln in enumerate(skel_lines, start=1))
                            preloaded_blocks.append(
                                f"Dosya Skeletoni (AST Skeleton - Büyük Dosya): {norm_cand} "
                                f"({stats['skeleton_lines']} satır / Orijinal {stats['original_lines']} satır, "
                                f"%{stats['savings_pct']} Tasarruf)\n{numbered}"
                            )
                            if len(preloaded_blocks) >= max_files:
                                break
                            continue
                    except Exception as skel_err:
                        logger.debug("Skeleton çıkarma hatası: %s", skel_err)
                    continue

                numbered = "\n".join(f"{i:4d} | {ln}" for i, ln in enumerate(lines, start=1))
                preloaded_blocks.append(
                    f"Dosya: {norm_cand} (Satır 1-{len(lines)} / Toplam {len(lines)})\n{numbered}"
                )
                if len(preloaded_blocks) >= max_files:
                    break
            except (OSError, UnicodeDecodeError) as exc:
                logger.debug("İstemde geçen '%s' dosyası ön yüklenirken hata: %s", rel_cand, exc)
                continue

        if not preloaded_blocks:
            return user_input

        joined = "\n\n---\n\n".join(preloaded_blocks)
        return (
            f"{user_input}\n\n"
            f"[Otomatik Ön-Yüklenen Dosya İçeriği / Auto Pre-loaded File Content] (No need to call `read_file` again; apply `edit_file` directly):\n"
            f"{joined}"
        )

    def run_turn(
        self,
        user_input: str,
        model_override: Optional[str] = None,
        stream: bool = True,
    ) -> str:
        """Run turn."""
        return self.turn_pipeline.run(
            user_input=user_input,
            model_override=model_override,
            stream=stream,
        )

    def _auto_save_session(self) -> None:
        """Auto save session (.evren/session.json)."""
        try:
            self.session_manager.auto_save_workspace_session(
                context=self.context,
                active_model_id=self.llm_client.active_model.id,
                plan_manager=self.plan_manager,
            )
        except Exception as exc:
            logger.debug("Oturum durumu kaydedilirken hata: %s", exc)

    def run_architect_workflow(self, task_description: str, stream: bool = True) -> str:
        """Run architect workflow (/architect)."""
        architect_spec = self.config.models.get_model("architect")
        coder_spec = self.config.models.get_model("coder")

        plan_prompt = (
            f"[MİMARİ PLANLAMA AŞAMASI - Model: {architect_spec.name}]\n"
            f"Görev: {task_description}\n"
            "Lütfen henüz hiçbir dosyayı değiştirme (`write_file` veya `edit_file` kullanma). "
            "Önce `get_project_tree` ve `read_multiple_files` araçlarıyla "
            "ilgili dosyaları incele ve hangi dosyada hangi değişikliklerin yapılacağını içeren "
            "net bir teknik uygulama planı çıkar."
        )
        self.run_turn(
            plan_prompt,
            model_override=architect_spec.id,
            stream=stream,
        )

        exec_prompt = (
            f"[KODLAMA VE DOĞRULAMA AŞAMASI - Model: {coder_spec.name}]\n"
            "Yukarıda hazırlanan mimari planı şimdi adım adım uygula. "
            "`edit_file` / `multi_edit_file` / `write_file` ile kod değişikliklerini yap ve `run_command` ile test/doğrulama çalıştır."
        )
        return self.run_turn(
            exec_prompt,
            model_override=coder_spec.id,
            stream=stream,
        )

    def should_autonomously_decompose(self, prompt: str) -> Tuple[bool, str]:
        """Autonomously determines whether the task should be decomposed into parallel subagents.

        Criteria:
        1. User did not provide continuation/confirmation prompt ("continue", "yes", etc.).
        2. Not a pure question/explanation query ("what is", "why", "explain") without coding action.
        3. Not a simple micro-edit (typo, single-line change, file inspection).
        4. Involves multi-domain components (frontend + backend, multi-system game) OR 
           comprehensive system overhaul (Complex tier).
        """
        p_lower = prompt.strip().lower()

        # 1. Explicit multi-agent triggers
        multi_triggers = (
            "alt ajanlara böl", "alt ajanlarla", "paralel ajan", "multi-agent", "multi agent",
            "aynı anda yap", "aynı anda çalıştır", "işi böl", "görevleri böl", "takım olarak",
            "paralel alt ajan", "paralel olarak yürüt"
        )
        if any(trig in p_lower for trig in multi_triggers) or prompt.strip().startswith("/multi plan"):
            return True, "Explicit multi-agent instruction trigger (Açık çoklu ajan tetikleyici talimatı)."

        # Multi-agent disabled check
        multi_cfg = getattr(self.config, "multi_agent", None)
        is_enabled = getattr(multi_cfg, "enabled", False)
        auto_decompose = getattr(multi_cfg, "auto_decompose", True)
        if not (is_enabled and auto_decompose):
            return False, "Multi-agent autonomous delegation is disabled in configuration (kapalı)."

        # Continuation prompts should stay in singular agent loop
        if is_continuation_prompt(prompt):
            return False, "Continuation/confirmation prompt (Devam/onay komutu)."

        # Sequential / step-by-step instructions: user wants step-by-step execution
        sequential_indicators = (
            "adım adım", "adim adim", "sırayla", "sirayla", "tek tek", "aşama aşama", "asama asama",
            "faz faz", "faz 0 dan", "faz 0'dan", "faz 1 den", "faz 1'den", "önce ... sonra",
            "sırasıyla", "sirasiyla", "sırayla yap", "adım adım geliştir", "adım adım kodla"
        )
        if any(seq_kw in p_lower for seq_kw in sequential_indicators):
            return False, "User requested sequential / step-by-step execution in single ReAct loop."

        # Pure question/information queries without coding action
        pure_question_keywords = ("nedir", "nerede", "açıkla", "neden", "hata neden", "hata aldın", "nasıl çalışır", "kimdir", "özetle")
        has_action = any(act in p_lower for act in ("yap", "geliştir", "oluştur", "kur", "yaz", "modernize", "refactor", "ekle", "entegre", "yenile", "temizle"))
        if any(qk in p_lower for qk in pure_question_keywords) and not has_action:
            return False, "Pure information/explanation query (bilgi/açıklama)."

        # Simple micro-edits
        simple_indicators = ("typo", "yazım hatası", "tek satır", "dosyayı oku", "satırını değiştir", "quick fix")
        if any(si in p_lower for si in simple_indicators):
            return False, "Simple micro-edit (Basit mikro-düzeltme)."

        # Error diagnostics and stacktrace fixes are faster in single agent
        is_error_investigation = (
            any(err_kw in p_lower for err_kw in (
                "bu hatayı çöz", "bu hatayı düzelt", "hatayı gider", "hata aldım",
                "hata var", "neden takılıyor", "axioserror", "network error",
                "traceback (most recent call last)", "syntaxerror", "typeerror",
                "referenceerror", "uncaught exception", "nullpointerexception",
                "failed to compile", "build failed", "cannot find module",
                "unhandledrejection", "status 500", "status 404", "status 403",
            ))
            and not any(exp in p_lower for exp in ("baştan sona", "tüm sistemi", "uçtan uca", "refactor", "yeniden yapılandır"))
        )
        if is_error_investigation:
            return False, "Error investigation (tekil hata teşhisi tek ajan ile daha hızlı)."

        # 2. Autonomous Multi-Domain & Component Analysis
        # A. Multi-domain concurrency (Fullstack, Frontend + Backend):
        has_frontend = any(w in p_lower for w in ("frontend", "arayüz", "ui", "ux", "ekran", "sayfa", "bileşen"))
        has_backend = any(w in p_lower for w in ("backend", "api", "servis", "service", "endpoint", "veritabanı", "database", "sql"))
        has_fullstack = ("full stack" in p_lower or "fullstack" in p_lower or (has_frontend and has_backend))

        # B. Multi-system game development:
        game_systems = [w for w in ("fizik", "physics", "ai", "yapay zeka", "düşman", "enemy", "ses", "audio", "hud", "karakter", "seviye", "level") if w in p_lower]
        is_multi_game = len(game_systems) >= 2 or ("oyun" in p_lower and len(game_systems) >= 1)

        # C. Comprehensive system modernization:
        is_broad_scope = any(w in p_lower for w in (
            "uçtan uca", "baştan sona", "sıfırdan", "tüm sistemi", "bütün ekranlar",
            "tüm ekranlar", "ekranları modernize", "modern hale", "tasarım sistemi", "yeniden yapılandır"
        ))

        # D. Complexity Assessment:
        assessment = TaskComplexityClassifier.classify(prompt)
        is_complex = assessment.level == ComplexityLevel.COMPLEX

        has_multi_system = has_fullstack or is_multi_game or is_broad_scope
        if has_multi_system or (is_complex and any(act in p_lower for act in ("geliştir", "inşa", "kur", "oluştur", "modernize", "refactor"))):
            return True, f"Otonom çoklu ajan kararı: Görev çoklu bileşen/alan ve geniş kapsam içeriyor (Karmaşıklık: {assessment.level.value})."

        return False, "Standard single-agent execution is sufficient."

    def run_multi_agent_workflow(self, goal: str) -> str:
        """Domain-Agnostic Multi-Agent DAG Workflow:
        1. Analyzes goal and dynamically decomposes into subtasks.
        2. Executes independent tasks concurrently in a thread pool.
        3. Synchronizes inter-dependent tasks via contract-first execution.
        4. Aggregates results into a unified summary report.
        """
        self.event_bus.emit(
            EventType.WARNING,
            message=f"🚀 [Multi-Agent System / Çoklu Ajan Sistemi]: Analyzing goal and planning dynamic task DAG: '{goal[:80]}'...",
        )
        graph = self.multi_agent_coordinator.plan_dag_from_goal(goal)

        tree_repr = graph.to_ascii_tree()
        self.event_bus.emit(
            EventType.INFO,
            message=f"[Task Graph Created / Görev Çizgesi Oluşturuldu]:\n{tree_repr}",
        )

        def _on_task_update(task: Any) -> None:
            if not hasattr(task, "status"):
                return
            st_val = task.status.value if hasattr(task.status, "value") else str(task.status)
            if st_val == "running":
                self.event_bus.emit(
                    EventType.STEP_STARTED,
                    step=len([t for t in graph.tasks.values() if (t.status.value if hasattr(t.status, "value") else str(t.status)) in ("running", "completed")]),
                    max_steps=len(graph.tasks),
                    model_id=f"agent-{task.role.value if hasattr(task.role, 'value') else str(task.role)}",
                    model_name=f"{(task.role.value if hasattr(task.role, 'value') else str(task.role)).capitalize()} [{getattr(task, 'domain', 'general')}]",
                )
            elif st_val == "completed":
                self.event_bus.emit(
                    EventType.TOOL_EXECUTION_COMPLETED,
                    tool_name=f"task_{task.task_id}",
                    approved=True,
                    result=f"✔ [Task Completed / Görev Tamamlandı]: {task.title}",
                )
            elif st_val == "failed":
                err_detail = getattr(task.result, "error", None) or "Unknown error"
                self.event_bus.emit(
                    EventType.ERROR,
                    message=f"[Task Failed / Görev Başarısız]: {task.title} (ID: {task.task_id}) -> {err_detail}",
                )
            elif st_val == "blocked":
                self.event_bus.emit(
                    EventType.WARNING,
                    message=f"[Task Blocked / Görev Engellendi]: {task.title} (ID: {task.task_id}) upstream dependency failed.",
                )
            elif st_val == "cancelled":
                self.event_bus.emit(
                    EventType.TOOL_EXECUTION_CANCELLED,
                    tool_name=f"task_{task.task_id}",
                )

        report = self.multi_agent_coordinator.run_plan(goal=goal, graph=graph, on_update=_on_task_update)

        exec_duration = getattr(report, "total_execution_time_seconds", getattr(report, "total_duration_seconds", 0.0))
        summary_lines = [
            f"## 🏆 Multi-Agent Task Report / Çoklu Ajan Görev Raporu ({report.run_id})",
            f"- **Goal / Ana Hedef:** {report.goal}",
            f"- **Total Tasks / Toplam Görev:** {report.total_tasks} | **Passed / Başarılı:** {report.completed_tasks} | **Failed / Başarısız:** {report.failed_tasks}",
            f"- **Duration / Yürütme Süresi:** {exec_duration:.2f}s",
            f"- **Concurrent Agents / Eşzamanlı Ajan:** {self.multi_agent_coordinator.multi_config.max_parallel_agents}",
            "",
            "### 📋 Completed Subtasks & Domains / Tamamlanan Görevler:",
        ]
        task_results_list = list(report.task_results.values()) if isinstance(report.task_results, dict) else report.task_results
        for tr in task_results_list:
            status_emoji = "✔" if tr.success else "❌"
            summary_lines.append(f"- {status_emoji} **[{tr.role.value.upper()}] {tr.task_id}**: {tr.output[:120]} ({tr.execution_time_seconds:.1f}s)")
            if tr.modified_files:
                summary_lines.append(f"  • *Modified Files:* {', '.join(tr.modified_files)}")

        if report.conflicts_detected:
            summary_lines.append(f"\n⚠ **File Conflicts Resolved ({len(report.conflicts_detected)}):** Resolved.")

        final_text = "\n".join(summary_lines)
        self.event_bus.emit(EventType.TEXT_DELTA, delta=f"\n\n{final_text}\n")
        self.event_bus.emit(EventType.TURN_COMPLETED)
        self.logger.complete_turn(final_text, self.context.estimate_tokens())
        self._auto_save_session()
        self.multi_agent_coordinator.last_report = report
        return final_text

    def close(self) -> None:
        """Close."""
        if hasattr(self, "_speculative_pool") and self._speculative_pool:
            try:
                self._speculative_pool.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                logger.debug("Spekülatif havuz kapatılırken hata: %s", exc)

        if hasattr(self, "tool_executor") and self.tool_executor and hasattr(self.tool_executor, "close"):
            try:
                self.tool_executor.close()
            except Exception as exc:
                logger.debug("ToolExecutor kapatılırken hata: %s", exc)

        if hasattr(self, "bg_process_manager") and self.bg_process_manager:
            try:
                self.bg_process_manager.stop_all()
            except Exception as exc:
                logger.debug("Arka plan süreçleri kapatılırken hata: %s", exc)

        if hasattr(self, "mcp_client_manager") and self.mcp_client_manager:
            try:
                self.mcp_client_manager.close_all()
            except Exception as exc:
                logger.debug("External MCP connections close error: %s", exc)

    def __enter__(self) -> "AgentOrchestrator":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
