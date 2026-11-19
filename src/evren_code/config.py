"""Data-Driven module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


@dataclass
class ModelSpec:
    """Model spec implementation."""

    id: str
    name: str
    task: str = "chat"
    context_length: int = 128000
    max_output_tokens: int = 8192
    tools: bool = True
    thinking: bool = False
    modalities: List[str] = field(default_factory=lambda: ["text"])

    @property
    def supports_vision(self) -> bool:
        return "image" in self.modalities or self.task == "vision_chat"


@dataclass
class ApiConfig:
    """Api config implementation."""

    base_url: str
    key_prefix: str
    timeout_seconds: float
    verify_timeout_seconds: float
    max_retries: int
    terms_version: int
    storage_dir: Path
    key_filename: str
    terms_filename: str
    models_cache_filename: str
    sessions_dirname: str = "sessions"
    api_key: Optional[str] = None

    @property
    def key_file_path(self) -> Path:
        return self.storage_dir / self.key_filename

    @property
    def terms_file_path(self) -> Path:
        return self.storage_dir / self.terms_filename

    @property
    def models_cache_path(self) -> Path:
        return self.storage_dir / self.models_cache_filename

    @property
    def sessions_dir_path(self) -> Path:
        return self.storage_dir / self.sessions_dirname


@dataclass
class ModelsConfig:
    """Models config implementation."""

    default: str
    roles: Dict[str, str]
    display_names: Dict[str, str]
    default_context_length: int
    default_max_output_tokens: int
    catalog: Dict[str, ModelSpec]
    role_aliases: Dict[str, str] = field(
        default_factory=lambda: {
            "auto": "auto",
            "evren-auto": "auto",
            "deepseek": "deepseek-v4.1-flash",
            "developer": "coder",
            "dev": "coder",
            "code": "coder",
            "coder": "coder",
            "planner": "architect",
            "plan": "architect",
            "architect": "architect",
            "tester": "coder",
            "test": "coder",
            "qa": "coder",
            "reviewer": "reviewer",
            "review": "reviewer",
            "researcher": "architect",
            "research": "architect",
            "orchestrator": "orchestrator",
        }
    )

    def get_model(self, model_or_role: Optional[str] = None) -> ModelSpec:
        """Get model."""
        raw_target = (model_or_role or self.default).strip()
        target = raw_target.lower()

        if target in self.roles:
            target = self.roles[target]
        elif target in self.role_aliases:
            alias_role = self.role_aliases[target]
            if alias_role in self.roles:
                target = self.roles[alias_role]
            elif alias_role in self.catalog:
                target = alias_role

        if target in self.catalog:
            return self.catalog[target]

        if self.default in self.catalog:
            return self.catalog[self.default]

        return ModelSpec(
            id=raw_target,
            name=self.display_names.get(raw_target, raw_target),
            context_length=self.default_context_length,
            max_output_tokens=self.default_max_output_tokens,
            tools=True,
            thinking=True,
        )


@dataclass
class SelfHealingConfig:
    """Self healing config implementation."""

    enabled: bool = True
    max_attempts: int = 3
    auto_detect: bool = True


@dataclass
class RepoMapConfig:
    """RepoMapConfig: Repo Map."""

    max_tokens: int = 1500
    max_files: int = 120


@dataclass
class AgentConfig:
    """Agent config implementation."""

    max_iterations: int
    subagent_max_iterations: int
    max_empty_thinking_retries: int
    temperature: float
    compaction_threshold_ratio: float
    keep_recent_messages: int
    chars_per_token_estimate: float
    project_rules_filenames: List[str]
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1
    repetition_guard_enabled: bool = True
    repetition_threshold_count: int = 3
    auto_continue: bool = True
    max_auto_continues: int = 2
    sliding_eviction_step_interval: int = 5
    sliding_eviction_min_tokens: int = 48000
    sliding_eviction_max_chars: int = 1800
    max_working_context_tokens: int = 96000
    subagent_heartbeat_interval_seconds: float = 30.0
    subagent_progress_cooldown_seconds: float = 60.0
    subagent_max_progress_model_calls: int = 3
    subagent_scratchpad_max_entries: int = 5
    dod_verification_gate_enabled: bool = True
    max_dod_retries: int = 1
    lean_tools_mode: bool = True
    dual_review_enabled: bool = False
    self_healing: SelfHealingConfig = field(default_factory=SelfHealingConfig)


@dataclass
class ToolsConfig:
    """ToolsConfig: shell."""

    max_file_read_lines: int
    max_file_size_bytes: int
    max_batch_read_files: int
    max_batch_lines_per_file: int
    default_tree_depth: int
    max_tree_depth: int
    max_tree_entries: int
    max_search_results: int
    max_directory_entries: int
    command_timeout_seconds: int
    max_output_chars: int
    max_undo_history: int
    url_fetch_timeout_seconds: float
    max_url_chars: int
    require_approval_for: List[str]
    ignored_dirs: List[str]
    build_command_timeout_seconds: int = 90
    max_recommended_file_lines: int = 300
    auto_preload_max_files: int = 3
    auto_preload_max_lines: int = 250
    repo_map: RepoMapConfig = field(default_factory=RepoMapConfig)


@dataclass
class TelemetryConfig:
    """TelemetryConfig: trajectory / run log."""

    enabled: bool = True
    mode: str = "project"  # "project", "user", "off"
    record_trajectory_jsonl: bool = True
    record_markdown_summary: bool = True
    auto_gitignore: bool = True
    max_log_chars_per_result: int = 3000


@dataclass
class MultiAgentConfig:
    """Multi agent config implementation."""

    enabled: bool = False
    auto_decompose: bool = True
    max_parallel_agents: int = 4
    task_timeout_seconds: float = 300.0
    max_retry_attempts: int = 2
    auto_review: bool = True
    verify_with_engine: bool = True
    heartbeat_interval_seconds: float = 10.0
    recover_on_timeout: bool = True
    persistence_dir: str = ".evren/multi_agent"


@dataclass
class SprintAppConfig:
    """Sprint app config implementation."""

    enabled: bool = False
    max_iterations: int = 5
    max_tasks_per_sprint: int = 30
    max_parallel_agents: int = 3
    max_retries_per_task: int = 2
    stop_on_critical_failure: bool = True
    require_reviewer_approval: bool = True
    require_regression_check: bool = True
    require_acceptance_criteria: bool = True
    checkpoint_interval: int = 1
    resume_interrupted_sprints: bool = True
    storage_dir: str = ".evren/sprints"
    sprint_timeout_seconds: float = 3600.0


@dataclass
class AppConfig:
    """App config implementation."""

    api: ApiConfig
    models: ModelsConfig
    agent: AgentConfig
    tools: ToolsConfig
    telemetry: TelemetryConfig
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    sprint: SprintAppConfig = field(default_factory=SprintAppConfig)
    workspace_root: Path = field(default_factory=Path.cwd)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge (recursive)."""
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _find_default_config_path() -> Path:
    """Find default config path."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "config.default.yaml",
        Path(__file__).resolve().parent / "config.default.yaml",
        Path.cwd() / "config.default.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("config.default.yaml bulunamadı.")


def load_config(
    custom_config_path: Optional[Path] = None,
    workspace_root: Optional[Path] = None,
) -> AppConfig:
    """Load config."""
    load_dotenv()

    default_path = _find_default_config_path()
    with open(default_path, "r", encoding="utf-8") as f:
        raw_data: Dict[str, Any] = yaml.safe_load(f) or {}

    storage_dir = Path(os.path.expanduser(raw_data["api"]["storage_dir"]))
    user_config_path = custom_config_path or (storage_dir / "config.yaml")
    if user_config_path.is_file():
        with open(user_config_path, "r", encoding="utf-8") as f:
            user_data = yaml.safe_load(f) or {}
            raw_data = _deep_merge(raw_data, user_data)

    api_raw = raw_data["api"]
    models_raw = raw_data["models"]
    agent_raw = raw_data["agent"]
    tools_raw = raw_data["tools"]
    telemetry_raw = raw_data.get("telemetry", {})

    base_url = os.environ.get("EVREN_BASE_URL", api_raw["base_url"]).rstrip("/")
    env_key = os.environ.get("EVREN_API_KEY")
    default_model = os.environ.get("EVREN_DEFAULT_MODEL", models_raw["default"])

    storage_dir = Path(os.path.expanduser(api_raw["storage_dir"]))

    api_cfg = ApiConfig(
        base_url=base_url,
        key_prefix=api_raw["key_prefix"],
        timeout_seconds=float(api_raw["timeout_seconds"]),
        verify_timeout_seconds=float(api_raw["verify_timeout_seconds"]),
        max_retries=int(api_raw["max_retries"]),
        terms_version=int(api_raw["terms_version"]),
        storage_dir=storage_dir,
        key_filename=api_raw["key_filename"],
        terms_filename=api_raw["terms_filename"],
        models_cache_filename=api_raw["models_cache_filename"],
        sessions_dirname=api_raw.get("sessions_dirname", "sessions"),
        api_key=env_key.strip() if env_key else None,
    )

    catalog: Dict[str, ModelSpec] = {}
    for mid, mdata in models_raw.get("catalog", {}).items():
        catalog[mid] = ModelSpec(
            id=mid,
            name=mdata.get("name", mid),
            task=mdata.get("task", "chat"),
            context_length=int(mdata.get("context_length", models_raw["defaults"]["context_length"])),
            max_output_tokens=int(mdata.get("max_output_tokens", models_raw["defaults"]["max_output_tokens"])),
            tools=bool(mdata.get("tools", True)),
            thinking=bool(mdata.get("thinking", False)),
            modalities=list(mdata.get("modalities", ["text"])),
        )

    models_cfg = ModelsConfig(
        default=default_model,
        roles=dict(models_raw.get("roles", {})),
        display_names=dict(models_raw.get("display_names", {})),
        default_context_length=int(models_raw["defaults"]["context_length"]),
        default_max_output_tokens=int(models_raw["defaults"]["max_output_tokens"]),
        catalog=catalog,
    )

    sh_raw = agent_raw.get("self_healing", {})
    self_healing_cfg = SelfHealingConfig(
        enabled=bool(sh_raw.get("enabled", True)),
        max_attempts=int(sh_raw.get("max_attempts", 3)),
        auto_detect=bool(sh_raw.get("auto_detect", True)),
    )

    agent_cfg = AgentConfig(
        max_iterations=int(agent_raw["max_iterations"]),
        subagent_max_iterations=int(agent_raw["subagent_max_iterations"]),
        max_empty_thinking_retries=int(agent_raw.get("max_empty_thinking_retries", 2)),
        temperature=float(agent_raw["temperature"]),
        compaction_threshold_ratio=float(agent_raw["compaction_threshold_ratio"]),
        keep_recent_messages=int(agent_raw["keep_recent_messages"]),
        chars_per_token_estimate=float(agent_raw["chars_per_token_estimate"]),
        project_rules_filenames=list(agent_raw["project_rules_filenames"]),
        frequency_penalty=float(agent_raw.get("frequency_penalty", 0.1)),
        presence_penalty=float(agent_raw.get("presence_penalty", 0.1)),
        repetition_guard_enabled=bool(agent_raw.get("repetition_guard_enabled", True)),
        repetition_threshold_count=int(agent_raw.get("repetition_threshold_count", 3)),
        auto_continue=bool(agent_raw.get("auto_continue", True)),
        max_auto_continues=int(agent_raw.get("max_auto_continues", 2)),
        sliding_eviction_step_interval=int(agent_raw.get("sliding_eviction_step_interval", 5)),
        sliding_eviction_min_tokens=int(agent_raw.get("sliding_eviction_min_tokens", 48000)),
        sliding_eviction_max_chars=int(agent_raw.get("sliding_eviction_max_chars", 1800)),
        max_working_context_tokens=int(agent_raw.get("max_working_context_tokens", 96000)),
        subagent_heartbeat_interval_seconds=float(agent_raw.get("subagent_heartbeat_interval_seconds", 30.0)),
        subagent_progress_cooldown_seconds=float(agent_raw.get("subagent_progress_cooldown_seconds", 60.0)),
        subagent_max_progress_model_calls=int(agent_raw.get("subagent_max_progress_model_calls", 3)),
        subagent_scratchpad_max_entries=int(agent_raw.get("subagent_scratchpad_max_entries", 5)),
        dod_verification_gate_enabled=bool(agent_raw.get("dod_verification_gate_enabled", True)),
        max_dod_retries=int(agent_raw.get("max_dod_retries", 1)),
        lean_tools_mode=bool(agent_raw.get("lean_tools_mode", True)),
        self_healing=self_healing_cfg,
    )

    rm_raw = tools_raw.get("repo_map", {})
    repo_map_cfg = RepoMapConfig(
        max_tokens=int(rm_raw.get("max_tokens", 1500)),
        max_files=int(rm_raw.get("max_files", 120)),
    )

    tools_cfg = ToolsConfig(
        max_file_read_lines=int(tools_raw["max_file_read_lines"]),
        max_file_size_bytes=int(tools_raw["max_file_size_bytes"]),
        max_batch_read_files=int(tools_raw.get("max_batch_read_files", 10)),
        max_batch_lines_per_file=int(tools_raw.get("max_batch_lines_per_file", 200)),
        default_tree_depth=int(tools_raw.get("default_tree_depth", 3)),
        max_tree_depth=int(tools_raw.get("max_tree_depth", 6)),
        max_tree_entries=int(tools_raw.get("max_tree_entries", 300)),
        max_search_results=int(tools_raw["max_search_results"]),
        max_directory_entries=int(tools_raw["max_directory_entries"]),
        command_timeout_seconds=int(tools_raw["command_timeout_seconds"]),
        max_output_chars=int(tools_raw["max_output_chars"]),
        max_undo_history=int(tools_raw.get("max_undo_history", 30)),
        url_fetch_timeout_seconds=float(tools_raw.get("url_fetch_timeout_seconds", 20.0)),
        max_url_chars=int(tools_raw.get("max_url_chars", 15000)),
        require_approval_for=list(tools_raw["require_approval_for"]),
        ignored_dirs=list(tools_raw["ignored_dirs"]),
        build_command_timeout_seconds=int(tools_raw.get("build_command_timeout_seconds", 90)),
        max_recommended_file_lines=int(tools_raw.get("max_recommended_file_lines", 300)),
        auto_preload_max_files=int(tools_raw.get("auto_preload_max_files", 3)),
        auto_preload_max_lines=int(tools_raw.get("auto_preload_max_lines", 250)),
        repo_map=repo_map_cfg,
    )

    telemetry_cfg = TelemetryConfig(
        enabled=bool(telemetry_raw.get("enabled", True)),
        mode=str(telemetry_raw.get("mode", "project")),
        record_trajectory_jsonl=bool(telemetry_raw.get("record_trajectory_jsonl", True)),
        record_markdown_summary=bool(telemetry_raw.get("record_markdown_summary", True)),
        auto_gitignore=bool(telemetry_raw.get("auto_gitignore", True)),
        max_log_chars_per_result=int(telemetry_raw.get("max_log_chars_per_result", 3000)),
    )

    ma_raw = raw_data.get("multi_agent", {})
    multi_agent_cfg = MultiAgentConfig(
        enabled=bool(ma_raw.get("enabled", False)),
        auto_decompose=bool(ma_raw.get("auto_decompose", True)),
        max_parallel_agents=int(ma_raw.get("max_parallel_agents", 4)),
        task_timeout_seconds=float(ma_raw.get("task_timeout_seconds", 300.0)),
        max_retry_attempts=int(ma_raw.get("max_retry_attempts", 2)),
        auto_review=bool(ma_raw.get("auto_review", True)),
        verify_with_engine=bool(ma_raw.get("verify_with_engine", True)),
        heartbeat_interval_seconds=float(ma_raw.get("heartbeat_interval_seconds", 10.0)),
        persistence_dir=str(ma_raw.get("persistence_dir", ".evren/multi_agent")),
    )

    sp_raw = raw_data.get("sprint", {})
    sprint_cfg = SprintAppConfig(
        enabled=bool(sp_raw.get("enabled", False)),
        max_iterations=int(sp_raw.get("max_iterations", 5)),
        max_tasks_per_sprint=int(sp_raw.get("max_tasks_per_sprint", 30)),
        max_parallel_agents=int(sp_raw.get("max_parallel_agents", 3)),
        max_retries_per_task=int(sp_raw.get("max_retries_per_task", 2)),
        stop_on_critical_failure=bool(sp_raw.get("stop_on_critical_failure", True)),
        require_reviewer_approval=bool(sp_raw.get("require_reviewer_approval", True)),
        require_regression_check=bool(sp_raw.get("require_regression_check", True)),
        require_acceptance_criteria=bool(sp_raw.get("require_acceptance_criteria", True)),
        checkpoint_interval=int(sp_raw.get("checkpoint_interval", 1)),
        resume_interrupted_sprints=bool(sp_raw.get("resume_interrupted_sprints", True)),
        storage_dir=str(sp_raw.get("storage_dir", ".evren/sprints")),
        sprint_timeout_seconds=float(sp_raw.get("sprint_timeout_seconds", 3600.0)),
    )

    return AppConfig(
        api=api_cfg,
        models=models_cfg,
        agent=agent_cfg,
        tools=tools_cfg,
        telemetry=telemetry_cfg,
        multi_agent=multi_agent_cfg,
        sprint=sprint_cfg,
        workspace_root=workspace_root or Path.cwd(),
    )
