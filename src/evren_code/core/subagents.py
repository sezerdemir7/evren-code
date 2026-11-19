"""Specialist Subagent Delegation Engine (`subagents.py`).

Prevents polluting the primary orchestrator context with exploratory reads or verbose test traces:
1. `researcher`: Read-only codebase and documentation investigation specialist.
2. `tester`: QA specialist running build, test, and lint steps to isolate failure root causes.
3. `reviewer`: Security and architecture audit specialist inspecting git diffs and changes.
"""

from __future__ import annotations

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from evren_code.config import AppConfig, ModelSpec
from evren_code.core.context import ContextManager, estimate_tokens
from evren_code.core.events import ApprovalDelegate, EventBus, EventType
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.core.security import (
    PermissionManager,
    SECURITY_ERR_ROLE_RESTRICTION,
    ToolExecutor,
)
from evren_code.core.turn_pipeline.tool_runner import is_parallel_safe_tool
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.shell import mask_sensitive_data


@dataclass
class SubagentNote:
    """Concise, structured scratchpad note for a subagent execution step."""

    step: int
    tool_name: str
    target: str
    status: str  # "success" | "error" | "warning"
    summary: str
    is_critical: bool
    error_signature: Optional[str]
    timestamp: float


class SubagentProgressTracker:
    """Subagent progress tracker implementation."""

    def __init__(
        self,
        role: str,
        subagent_id: str,
        heartbeat_interval: float = 30.0,
        cooldown_seconds: float = 60.0,
        max_model_calls: int = 3,
        max_entries: int = 5,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.role = role
        self.subagent_id = subagent_id
        self.heartbeat_interval = heartbeat_interval
        self.cooldown_seconds = cooldown_seconds
        self.max_model_calls = max_model_calls
        self.notes: deque[SubagentNote] = deque(maxlen=max_entries)
        self.event_bus = event_bus

        self.start_time: float = time.time()
        self.last_progress_event_time: float = self.start_time
        self.last_heartbeat_time: float = self.start_time
        self.last_model_summary_time: float = 0.0

        self.model_calls_count: int = 0
        self.last_error_signature: Optional[str] = None
        self.total_progress_tokens: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "is_estimated": False,
        }

        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def record_step(
        self,
        step: int,
        tool_name: str,
        arguments: Dict[str, Any],
        result_str: str,
    ) -> SubagentNote:
        """Record step."""
        now = time.time()
        self.last_progress_event_time = now

        raw_target = ""
        for key in ("path", "file_path", "command", "query", "url", "symbol"):
            if key in arguments and arguments[key]:
                raw_target = str(arguments[key])
                break
        if not raw_target and arguments:
            for v in arguments.values():
                if isinstance(v, str) and v:
                    raw_target = v
                    break

        target = mask_sensitive_data(raw_target).strip()
        if len(target) > 80:
            target = target[:77] + "..."

        status = "success"
        is_critical = False
        error_signature: Optional[str] = None

        if result_str.startswith("[ROL KISITLAMASI]"):
            status = "error"
            is_critical = True
            summary = "Yetkisiz araç çağrısı engellendi"
            error_signature = f"{tool_name}:role_restriction"
        elif result_str.startswith("Hata:") or result_str.startswith("Error:"):
            status = "error"
            first_line = result_str.splitlines()[0] if result_str else "Bilinmeyen hata"
            summary = mask_sensitive_data(first_line)[:100]
            is_critical = any(
                kw in result_str.lower()
                for kw in [
                    "failed",
                    "syntaxerror",
                    "compilation",
                    "build",
                    "permission",
                    "cannot",
                    "exception",
                    "traceback",
                ]
            )
            norm_err = re.sub(r"\d+", "", summary).strip().lower()
            norm_err = re.sub(r"\s+", " ", norm_err)
            error_signature = f"{tool_name}:{norm_err[:50]}"
        elif "Command failed with exit code" in result_str:
            status = "error"
            is_critical = True
            match = re.search(r"exit code (\d+)", result_str)
            code = match.group(1) if match else "?"
            summary = f"Komut başarısız oldu (exit code {code})"
            error_signature = f"{tool_name}:exit_code_{code}"
        else:
            if tool_name in ("read_file", "read_multiple_files"):
                lines = result_str.count("\n") + 1 if result_str else 0
                summary = f"Dosya incelendi ({lines} satır)"
            elif tool_name in ("search_code", "find_files", "semantic_search_code", "search_symbols"):
                lines = len(result_str.strip().splitlines()) if result_str.strip() else 0
                summary = f"{lines} eşleşme bulundu"
            elif tool_name == "run_command":
                summary = "Komut başarıyla çalıştırıldı (exit code 0)"
            elif tool_name == "git_info":
                summary = "Git durumu incelendi"
            else:
                summary = "İşlem başarılı"

        note = SubagentNote(
            step=step,
            tool_name=tool_name,
            target=target,
            status=status,
            summary=summary,
            is_critical=is_critical,
            error_signature=error_signature,
            timestamp=now,
        )
        self.notes.append(note)
        return note

    def should_emit_heartbeat(self) -> bool:
        """Should emit heartbeat."""
        return (time.time() - self.last_heartbeat_time) >= self.heartbeat_interval

    def emit_heartbeat(self) -> None:
        """Emit heartbeat."""
        self.last_heartbeat_time = time.time()
        latest = self.notes[-1] if self.notes else None
        latest_summary = latest.summary if latest else "Beklemede"
        latest_target = latest.target if latest else ""
        step = latest.step if latest else 0
        elapsed = round(self.last_heartbeat_time - self.start_time, 1)

        if self.event_bus:
            self.event_bus.emit(
                EventType.SUBAGENT_HEARTBEAT,
                role=self.role,
                subagent_id=self.subagent_id,
                step=step,
                elapsed_seconds=elapsed,
                target=latest_target,
                latest_summary=latest_summary,
            )

    def generate_status_update(
        self,
        llm_client: Optional[EvrenLLMClient] = None,
        compactor_model_id: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Generate status update."""
        if not self.notes:
            return (
                f"[{self.role}] Başlatıldı.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "is_estimated": False},
            )

        latest_note = self.notes[-1]
        now = time.time()

        zero_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "is_estimated": False,
        }
        deterministic_msg = (
            f"[{self.role}] Adım {latest_note.step}: {latest_note.tool_name} "
            f"({latest_note.target}) - {latest_note.summary}"
        )

        if not latest_note.is_critical:
            return deterministic_msg, zero_token_usage

        if latest_note.error_signature and latest_note.error_signature == self.last_error_signature:
            return f"{deterministic_msg} [Tekrar Eden Hata]", zero_token_usage

        if self.model_calls_count >= self.max_model_calls:
            return f"{deterministic_msg} [Model Sınırı]", zero_token_usage

        if self.last_model_summary_time > 0 and (now - self.last_model_summary_time) < self.cooldown_seconds:
            return f"{deterministic_msg} [Cooldown]", zero_token_usage

        # 5. LLM Client yoksa -> Deterministik (0 token)
        if llm_client is None:
            return deterministic_msg, zero_token_usage

        try:
            delta_notes = list(self.notes)[-2:]
            notes_ctx = "\n".join(
                f"- Adım {n.step}: {n.tool_name}({n.target}) -> {n.status}: {n.summary}"
                for n in delta_notes
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Sen bir alt-ajan durum özetleyicisisin. "
                        "Aşağıdaki son 2 işlem notunu incele ve 1 cümlelik çok kısa Türkçe teknik durum özeti üret. "
                        "Gereksiz açıklama yapma, sadece durumu bildir."
                    ),
                },
                {"role": "user", "content": f"Olaylar:\n{notes_ctx}"},
            ]

            response = llm_client.chat(
                messages=messages,
                model_override=compactor_model_id,
                stream=False,
            )

            self.model_calls_count += 1
            self.last_model_summary_time = now
            self.last_error_signature = latest_note.error_signature

            # Token muhasebesi
            if response.usage and isinstance(response.usage, dict):
                p_tok = int(response.usage.get("prompt_tokens", 0))
                c_tok = int(response.usage.get("completion_tokens", 0))
                t_tok = int(response.usage.get("total_tokens", p_tok + c_tok))
                is_est = False
            else:
                p_tok = estimate_tokens(str(messages))
                c_tok = estimate_tokens(response.content)
                t_tok = p_tok + c_tok
                is_est = True

            step_usage = {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": t_tok,
                "is_estimated": is_est,
            }

            self.total_progress_tokens["prompt_tokens"] += p_tok
            self.total_progress_tokens["completion_tokens"] += c_tok
            self.total_progress_tokens["total_tokens"] += t_tok
            if is_est:
                self.total_progress_tokens["is_estimated"] = True

            clean_summary = (
                response.content.strip().splitlines()[0] if response.content else latest_note.summary
            )
            clean_summary = mask_sensitive_data(clean_summary)[:150]
            return f"[{self.role}] Adım {latest_note.step}: {clean_summary}", step_usage

        except Exception:
            return deterministic_msg, zero_token_usage

    def start_heartbeat(self) -> None:
        """Start heartbeat."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_worker,
            daemon=True,
            name=f"SubagentHeartbeat_{self.subagent_id}",
        )
        self._heartbeat_thread.start()

    def _heartbeat_worker(self) -> None:
        """Heartbeat worker."""
        while not self._heartbeat_stop_event.wait(timeout=0.5):
            if self.should_emit_heartbeat():
                self.emit_heartbeat()

    def stop(self) -> None:
        """Stop."""
        self._heartbeat_stop_event.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)
        self._heartbeat_thread = None


@dataclass
class SubagentRoleSpec:
    """Subagent role spec implementation."""

    role: str
    display_name: str
    system_prompt: str
    allowed_tools: Set[str]
    model_role: str = "coder"


RESEARCHER_PROMPT = """Sen EVREN Kıdemli Kod Tabanı ve Dokümantasyon Araştırmacısısın (Researcher Subagent).
Görevin: Ana orkestratörün sorusuna yanıt bulmak için kod tabanını, fonksiyon imzalarını,
harici URL'leri veya dokümantasyonları derinlemesine incelemek ve ana ajana 2-4 paragraflık,
yoğun ve somut bir teknik rapor sunmaktır.
Asla kod dosyalarını değiştirme (`write_file` / `edit_file` yetkin yoktur); sadece oku ve sentezle.
Cevabında ilgili dosya yollarını ve önemli fonksiyon/tip isimlerini açıkça belirt.
"""

TESTER_PROMPT = """Sen EVREN QA ve Test/Derleme Mühendisisin (Tester Subagent).
Görevin: Projedeki derleme (`npm run build`, `tsc`), birim testleri (`npm test`, `pytest`) veya linter
komutlarını `run_command` ile çalıştırmak; derleyici hatalarını, stack trace'leri ve kırılan testleri analiz etmektir.
Çıktında hatanın kaynaklandığı dosya yolunu, satır numarasını ve doğrudan çözüm önerisini raporla.
"""

REVIEWER_PROMPT = """Sen EVREN Kıdemli Kod İnceleme ve Güvenlik Uzmanısın (Reviewer Subagent).
Görevin: Son yapılan değişiklikleri (`git_info`, `read_file`) incelemek; mimari hataları,
sözdizimi tutarsızlıklarını, hardcoded hassas bilgileri (secrets/tokens) ve eksik tip tanımlarını tespit etmektir.
Doğrudan ve maddeler halinde bir inceleme raporu sun.
"""

ARCHITECT_PROMPT = """Sen EVREN Baş Yazılım Mimarı Ajanısın (Architect Subagent).
Görevin: Karmaşık mühendislik ve tasarım problemlerini analiz etmek, ilgili kod yapısını inceleyip
somut, adım adım uygulanabilir mimari çözüm ve teknik tasarım sunmaktır.
Asla dosya değiştirme; sadece analiz et ve uygulama planı / teknik tasarım üret.
"""

DEFAULT_SUBAGENT_ROLES: Dict[str, SubagentRoleSpec] = {
    "researcher": SubagentRoleSpec(
        role="researcher",
        display_name="Araştırmacı Ajan",
        system_prompt=RESEARCHER_PROMPT,
        allowed_tools={
            "get_project_tree",
            "get_repo_map",
            "get_project_overview",
            "get_project_modules",
            "get_module_details",
            "inspect_project",
            "get_important_files",
            "list_directory",
            "read_file",
            "read_multiple_files",
            "find_files",
            "search_code",
            "semantic_search_code",
            "lsp_find_definition",
            "lsp_find_references",
            "search_symbols",
            "find_symbol_definition",
            "find_symbol_references",
            "analyze_change_impact",
            "find_related_tests",
            "get_file_symbols",
            "fetch_url",
        },
        model_role="coder",
    ),
    "architect": SubagentRoleSpec(
        role="architect",
        display_name="Mimari Tasarım Ajanı",
        system_prompt=ARCHITECT_PROMPT,
        allowed_tools={
            "get_project_tree",
            "get_repo_map",
            "get_project_overview",
            "get_project_modules",
            "get_module_details",
            "inspect_project",
            "get_important_files",
            "read_file",
            "read_multiple_files",
            "find_files",
            "search_code",
            "semantic_search_code",
            "search_symbols",
            "find_symbol_definition",
            "find_symbol_references",
            "analyze_change_impact",
            "find_related_tests",
            "get_file_symbols",
        },
        model_role="architect",
    ),
    "tester": SubagentRoleSpec(
        role="tester",
        display_name="Test & Doğrulama Ajanı",
        system_prompt=TESTER_PROMPT,
        allowed_tools={"run_command", "read_file", "lsp_get_diagnostics", "git_info"},
        model_role="coder",
    ),
    "reviewer": SubagentRoleSpec(
        role="reviewer",
        display_name="Kod İnceleme & Güvenlik Ajanı",
        system_prompt=REVIEWER_PROMPT,
        allowed_tools={
            "git_info",
            "read_file",
            "read_multiple_files",
            "lsp_get_diagnostics",
            "search_code",
            "search_symbols",
            "find_symbol_definition",
            "find_symbol_references",
            "analyze_change_impact",
            "find_related_tests",
        },
        model_role="reviewer",
    ),
}


class SubagentManager:
    """Subagent manager implementation."""

    def __init__(
        self,
        config: AppConfig,
        registry: ToolRegistry,
        llm_client: EvrenLLMClient,
        event_bus: Optional[EventBus] = None,
        tool_executor: Optional[ToolExecutor] = None,
        permission_manager: Optional[PermissionManager] = None,
        approval_delegate: Optional[ApprovalDelegate] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.llm_client = llm_client
        self.event_bus = event_bus or EventBus()
        if tool_executor is not None:
            self.tool_executor = tool_executor
        else:
            pm = permission_manager or PermissionManager(
                workspace_root=config.workspace_root,
                require_approval_for=config.tools.require_approval_for,
            )
            # Default auto approve if none provided (for tests / standalone usage)
            delegate = approval_delegate
            if delegate is None:
                class _AutoApprove:
                    def request_approval(self, req: Any) -> bool:
                        return True
                    def ask_user_question(self, q: str, opts: Any = None) -> str:
                        return ""
                delegate = _AutoApprove()
            self.tool_executor = ToolExecutor(
                registry=self.registry,
                permission_manager=pm,
                approval_delegate=delegate,
                event_bus=self.event_bus,
            )
        self.roles = dict(DEFAULT_SUBAGENT_ROLES)
        self._load_custom_roles()

    def _load_custom_roles(self) -> None:
        """Load custom roles."""
        agents_dir = self.config.workspace_root / ".evren" / "agents"
        if not agents_dir.is_dir():
            return
        import yaml
        for afile in agents_dir.glob("*.y*ml"):
            try:
                data = yaml.safe_load(afile.read_text(encoding="utf-8")) or {}
                role = str(data.get("role", afile.stem)).lower()
                self.roles[role] = SubagentRoleSpec(
                    role=role,
                    display_name=str(data.get("display_name", role.title())),
                    system_prompt=str(data.get("system_prompt", "")),
                    allowed_tools=set(data.get("allowed_tools", ["read_file"])),
                    model_role=str(data.get("model_role", "coder")),
                )
            except Exception:
                pass

    def delegate(
        self,
        role: str,
        task_prompt: str,
    ) -> str:
        """Delegate."""
        role_key = role.strip().lower()
        spec = self.roles.get(role_key)
        if not spec:
            available = ", ".join(f"'{r}'" for r in self.roles.keys())
            return f"Hata: Geçersiz alt-ajan rolü '{role}'. Mevcut roller: {available}."

        target_model = self.config.models.get_model(spec.model_role)
        compactor_model = None
        try:
            compactor_model = self.config.models.get_model("compactor")
        except Exception:
            pass
        compactor_model_id = compactor_model.id if compactor_model else target_model.id

        subagent_id = f"{spec.role}_{uuid.uuid4().hex[:8]}"

        tracker = SubagentProgressTracker(
            role=spec.role,
            subagent_id=subagent_id,
            heartbeat_interval=getattr(self.config.agent, "subagent_heartbeat_interval_seconds", 30.0),
            cooldown_seconds=getattr(self.config.agent, "subagent_progress_cooldown_seconds", 60.0),
            max_model_calls=getattr(self.config.agent, "subagent_max_progress_model_calls", 3),
            max_entries=getattr(self.config.agent, "subagent_scratchpad_max_entries", 5),
            event_bus=self.event_bus,
        )

        self.event_bus.emit(
            EventType.SUBAGENT_STARTED,
            role=spec.role,
            subagent_id=subagent_id,
            model_id=target_model.id,
            model_name=target_model.name,
            task=task_prompt[:120],
        )

        all_schemas = self.registry.get_openai_schemas()
        filtered_schemas = [
            s for s in all_schemas
            if s.get("function", {}).get("name") in spec.allowed_tools
        ]

        sub_context = ContextManager(self.config, spec.system_prompt)
        sub_context.add_user_message(task_prompt)

        max_steps = self.config.agent.subagent_max_iterations
        final_answer = ""

        tracker.start_heartbeat()

        try:
            for step in range(1, max_steps + 1):
                if sub_context.needs_compaction(target_model):
                    sub_context.compact(aggressive=True)
                messages = sub_context.get_messages()
                response = self.llm_client.chat(
                    messages=messages,
                    tools_schema=filtered_schemas if target_model.tools else None,
                    model_override=target_model.id,
                    stream=False,
                )

                if not response.tool_calls:
                    final_answer = response.content.strip()
                    break

                sub_context.add_assistant_message(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    native_tools_supported=target_model.tools,
                )

                def _exec_subagent_tool(item: ToolCallItem) -> tuple[ToolCallItem, str]:
                    if item.name not in spec.allowed_tools:
                        return item, (
                            f"[ROL KISITLAMASI / {SECURITY_ERR_ROLE_RESTRICTION}]: "
                            f"'{item.name}' aracı '{spec.role}' rolü için yetkilendirilmemiştir."
                        )
                    res = self.tool_executor.execute(
                        tool_name=item.name,
                        arguments=item.arguments,
                        caller_role=spec.role,
                        caller_agent=f"subagent_{spec.role}",
                        is_trust_mode=False,
                    )
                    return item, res

                can_parallel = (
                    len(response.tool_calls) > 1
                    and all(is_parallel_safe_tool(tc.name) for tc in response.tool_calls)
                )

                if can_parallel:
                    pool = ThreadPoolExecutor(max_workers=min(len(response.tool_calls), 8), thread_name_prefix="evren_subagent_par")
                    try:
                        futs = [(tc, pool.submit(_exec_subagent_tool, tc)) for tc in response.tool_calls]
                        executed_results = []
                        for tc_item, fut in futs:
                            try:
                                executed_results.append(fut.result(timeout=65.0))
                            except Exception as exc:
                                executed_results.append((tc_item, f"Hata: '{tc_item.name}' alt-ajan paralel yürütmesinde zaman aşımı veya hata: {exc}"))
                    finally:
                        pool.shutdown(wait=False, cancel_futures=True)
                else:
                    executed_results = [_exec_subagent_tool(tc) for tc in response.tool_calls]

                for tc, result_str in executed_results:
                    note = tracker.record_step(
                        step=step,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        result_str=result_str,
                    )
                    status_text, token_stats = tracker.generate_status_update(
                        llm_client=self.llm_client,
                        compactor_model_id=compactor_model_id,
                    )
                    self.event_bus.emit(
                        EventType.SUBAGENT_PROGRESS,
                        role=spec.role,
                        subagent_id=subagent_id,
                        step=step,
                        tool_name=tc.name,
                        status_text=status_text,
                        token_usage=token_stats,
                        is_llm_generated=(token_stats.get("total_tokens", 0) > 0),
                    )

                    sub_context.add_tool_result(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        result_text=result_str,
                        native_tools_supported=target_model.tools,
                        from_xml=tc.from_xml,
                    )

            if not final_answer:
                final_answer = f"Alt-ajan ({spec.display_name}) {max_steps} adımda araştırmasını tamamladı ancak nihai metin üretmedi."

        except Exception as exc:
            final_answer = f"Alt-ajan yürütmesi sırasında hata oluştu: {exc}"
        finally:
            tracker.stop()

        self.event_bus.emit(
            EventType.SUBAGENT_COMPLETED,
            role=spec.role,
            subagent_id=subagent_id,
            output=final_answer[:200],
            progress_token_usage=tracker.total_progress_tokens,
        )
        return (
            f"### [{spec.display_name} Raporu]\n{final_answer}"
        )
