"""Tool reliability module."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ExecutionLifecycleStatus(str, Enum):
    """Execution lifecycle status implementation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    FAILED = "failed"
    LATE_IGNORED = "late_ignored"


@dataclass
class ToolTimeoutPolicy:
    """ToolTimeoutPolicy: tekrarlanabilirlik."""

    tool_name: str
    timeout_seconds: float = 30.0
    stall_timeout_seconds: Optional[float] = None
    idempotent: bool = True
    max_retries: int = 0
    retry_base_delay: float = 0.25
    max_output_chars: Optional[int] = None
    is_async: bool = False
    allow_subprocess: bool = False


DEFAULT_TOOL_POLICIES: Dict[str, ToolTimeoutPolicy] = {
    "read_file": ToolTimeoutPolicy(
        tool_name="read_file",
        timeout_seconds=10.0,
        stall_timeout_seconds=5.0,
        idempotent=True,
        max_retries=1,
    ),
    "read_multiple_files": ToolTimeoutPolicy(
        tool_name="read_multiple_files",
        timeout_seconds=25.0,
        stall_timeout_seconds=10.0,
        idempotent=True,
        max_retries=1,
    ),
    "write_file": ToolTimeoutPolicy(
        tool_name="write_file",
        timeout_seconds=15.0,
        idempotent=False,
        max_retries=0,
    ),
    "append_file": ToolTimeoutPolicy(
        tool_name="append_file",
        timeout_seconds=15.0,
        idempotent=False,
        max_retries=0,
    ),
    "edit_file": ToolTimeoutPolicy(
        tool_name="edit_file",
        timeout_seconds=15.0,
        idempotent=False,
        max_retries=0,
    ),
    "multi_edit_file": ToolTimeoutPolicy(
        tool_name="multi_edit_file",
        timeout_seconds=20.0,
        idempotent=False,
        max_retries=0,
    ),
    "write_multiple_files": ToolTimeoutPolicy(
        tool_name="write_multiple_files",
        timeout_seconds=30.0,
        idempotent=False,
        max_retries=0,
    ),
    "get_project_tree": ToolTimeoutPolicy(
        tool_name="get_project_tree",
        timeout_seconds=15.0,
        idempotent=True,
        max_retries=1,
    ),
    "list_directory": ToolTimeoutPolicy(
        tool_name="list_directory",
        timeout_seconds=10.0,
        idempotent=True,
        max_retries=1,
    ),
    "find_files": ToolTimeoutPolicy(
        tool_name="find_files",
        timeout_seconds=15.0,
        idempotent=True,
        max_retries=1,
    ),
    "search_code": ToolTimeoutPolicy(
        tool_name="search_code",
        timeout_seconds=20.0,
        idempotent=True,
        max_retries=1,
        allow_subprocess=True,
    ),
    "semantic_search_code": ToolTimeoutPolicy(
        tool_name="semantic_search_code",
        timeout_seconds=25.0,
        idempotent=True,
        max_retries=1,
    ),
    "lsp_get_diagnostics": ToolTimeoutPolicy(
        tool_name="lsp_get_diagnostics",
        timeout_seconds=18.0,
        idempotent=True,
        max_retries=1,
        allow_subprocess=True,
    ),
    "lsp_find_definition": ToolTimeoutPolicy(
        tool_name="lsp_find_definition",
        timeout_seconds=15.0,
        idempotent=True,
        max_retries=1,
    ),
    "lsp_find_references": ToolTimeoutPolicy(
        tool_name="lsp_find_references",
        timeout_seconds=15.0,
        idempotent=True,
        max_retries=1,
    ),
    "git_info": ToolTimeoutPolicy(
        tool_name="git_info",
        timeout_seconds=15.0,
        idempotent=True,
        max_retries=1,
        allow_subprocess=True,
    ),
    "run_command": ToolTimeoutPolicy(
        tool_name="run_command",
        timeout_seconds=310.0,
        idempotent=False,
        max_retries=0,
        allow_subprocess=True,
    ),
    "get_background_output": ToolTimeoutPolicy(
        tool_name="get_background_output",
        timeout_seconds=10.0,
        idempotent=True,
        max_retries=1,
    ),
    "stop_background_process": ToolTimeoutPolicy(
        tool_name="stop_background_process",
        timeout_seconds=10.0,
        idempotent=True,
        max_retries=0,
    ),
    "fetch_url": ToolTimeoutPolicy(
        tool_name="fetch_url",
        timeout_seconds=25.0,
        idempotent=True,
        max_retries=1,
    ),
    "http_request": ToolTimeoutPolicy(
        tool_name="http_request",
        timeout_seconds=25.0,
        idempotent=False,
        max_retries=0,
    ),
    "validate_web_contract": ToolTimeoutPolicy(
        tool_name="validate_web_contract",
        timeout_seconds=15.0,
        idempotent=True,
        max_retries=1,
    ),
    "capture_web_screenshot": ToolTimeoutPolicy(
        tool_name="capture_web_screenshot",
        timeout_seconds=30.0,
        idempotent=True,
        max_retries=1,
        allow_subprocess=True,
    ),
    "analyze_image": ToolTimeoutPolicy(
        tool_name="analyze_image",
        timeout_seconds=90.0,
        idempotent=True,
        max_retries=1,
    ),
    "inspect_web_page_visually": ToolTimeoutPolicy(
        tool_name="inspect_web_page_visually",
        timeout_seconds=90.0,
        idempotent=True,
        max_retries=1,
        allow_subprocess=True,
    ),
    "delegate_subagent": ToolTimeoutPolicy(
        tool_name="delegate_subagent",
        timeout_seconds=240.0,
        idempotent=False,
        max_retries=0,
    ),
    "run_verification": ToolTimeoutPolicy(
        tool_name="run_verification",
        timeout_seconds=240.0,
        idempotent=False,
        max_retries=0,
    ),
    "ask_user": ToolTimeoutPolicy(
        tool_name="ask_user",
        timeout_seconds=3600.0,
        idempotent=False,
        max_retries=0,
    ),
}


class ToolPolicyRegistry:
    """Tool policy registry implementation."""

    def __init__(self, default_timeout: float = 30.0) -> None:
        self.default_timeout = default_timeout
        self._policies: Dict[str, ToolTimeoutPolicy] = dict(DEFAULT_TOOL_POLICIES)
        self._lock = threading.RLock()

    def register_policy(self, policy: ToolTimeoutPolicy) -> None:
        with self._lock:
            self._policies[policy.tool_name] = policy

    def get_policy(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolTimeoutPolicy:
        with self._lock:
            base = self._policies.get(tool_name)
        if base is None:
            return ToolTimeoutPolicy(
                tool_name=tool_name,
                timeout_seconds=self.default_timeout,
                idempotent=False,
                max_retries=0,
            )
        if arguments and isinstance(arguments, dict):
            explicit_to = arguments.get("timeout")
            if isinstance(explicit_to, (int, float)) and explicit_to > 0:
                return ToolTimeoutPolicy(
                    tool_name=base.tool_name,
                    timeout_seconds=float(explicit_to) + 5.0 if tool_name == "run_command" else float(explicit_to),
                    stall_timeout_seconds=base.stall_timeout_seconds,
                    idempotent=base.idempotent,
                    max_retries=base.max_retries,
                    retry_base_delay=base.retry_base_delay,
                    max_output_chars=base.max_output_chars,
                    is_async=base.is_async,
                    allow_subprocess=base.allow_subprocess,
                )
            if tool_name == "http_request":
                method = str(arguments.get("method", "GET")).upper()
                is_safe_get = method in ("GET", "HEAD", "OPTIONS")
                return ToolTimeoutPolicy(
                    tool_name=base.tool_name,
                    timeout_seconds=base.timeout_seconds,
                    stall_timeout_seconds=base.stall_timeout_seconds,
                    idempotent=is_safe_get,
                    max_retries=1 if is_safe_get else 0,
                    retry_base_delay=base.retry_base_delay,
                    max_output_chars=base.max_output_chars,
                    is_async=base.is_async,
                    allow_subprocess=base.allow_subprocess,
                )
        return base


def terminate_process_tree_safe(pid: int) -> None:
    """Terminate process tree safe (child processes)."""
    if not pid or pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=4,
                check=False,
            )
        except Exception:
            pass
    else:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


_ACTIVE_EXECUTION_TLS = threading.local()


def get_current_cancellation_token() -> Optional["CancellationToken"]:
    return getattr(_ACTIVE_EXECUTION_TLS, "token", None)


def get_current_heartbeat_monitor() -> Optional["HeartbeatMonitor"]:
    return getattr(_ACTIVE_EXECUTION_TLS, "heartbeat", None)


def report_tool_progress(message: str = "", completed_units: int = 0, total_units: int = 0) -> None:
    """Report tool progress."""
    hb = get_current_heartbeat_monitor()
    if hb is not None:
        hb.beat(message=message, completed_units=completed_units, total_units=total_units)


def register_active_subprocess_pid(pid: int) -> None:
    """Register active subprocess pid."""
    tok = get_current_cancellation_token()
    if tok is not None and pid and pid > 0:
        tok.register_pid(pid)


def unregister_active_subprocess_pid(pid: int) -> None:
    """Unregister active subprocess pid."""
    tok = get_current_cancellation_token()
    if tok is not None and pid and pid > 0:
        tok.unregister_pid(pid)


class ToolCancelledError(Exception):
    """Tool cancelled error implementation."""
    pass


class CancellationToken:
    """Cancellation token implementation."""

    def __init__(self, execution_id: str, tool_name: str = "") -> None:
        self.execution_id = execution_id
        self.tool_name = tool_name
        self._event = threading.Event()
        self._reason: str = ""
        self._pids: Set[int] = set()
        self._async_tasks: List[asyncio.Task] = []
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._lock = threading.RLock()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def check_cancelled(self) -> None:
        """Check cancelled."""
        if self._event.is_set():
            raise ToolCancelledError(self._reason or f"Execution '{self.execution_id}' cancelled.")

    def register_pid(self, pid: int) -> None:
        if not pid or pid <= 0:
            return
        with self._lock:
            is_already_cancelled = self._event.is_set()
            if not is_already_cancelled:
                self._pids.add(pid)
        if is_already_cancelled:
            terminate_process_tree_safe(pid)

    def unregister_pid(self, pid: int) -> None:
        with self._lock:
            self._pids.discard(pid)

    def register_async_task(self, task: asyncio.Task) -> None:
        with self._lock:
            if self._event.is_set():
                try:
                    task.cancel()
                except Exception:
                    pass
                return
            self._async_tasks.append(task)

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._event.is_set():
                try:
                    callback()
                except Exception:
                    pass
                return
            self._cleanup_callbacks.append(callback)

    def cancel(self, reason: str = "Cancelled") -> None:
        """Cancel (PID tree)."""
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
            pids = list(self._pids)
            self._pids.clear()
            tasks = list(self._async_tasks)
            self._async_tasks.clear()
            callbacks = list(self._cleanup_callbacks)
            self._cleanup_callbacks.clear()

        for pid in pids:
            terminate_process_tree_safe(pid)

        for t in tasks:
            try:
                loop = t.get_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(t.cancel)
                else:
                    t.cancel()
            except Exception:
                pass

        for cb in callbacks:
            try:
                cb()
            except Exception as exc:
                logger.debug("Cleanup callback hatası (%s): %s", self.execution_id, exc)


class CancellationManager:
    """Cancellation manager implementation."""

    def __init__(self) -> None:
        self._tokens: Dict[str, CancellationToken] = {}
        self._lock = threading.RLock()

    def create_token(self, execution_id: Optional[str] = None, tool_name: str = "") -> CancellationToken:
        exec_id = execution_id or f"exec_{uuid.uuid4().hex[:10]}"
        token = CancellationToken(execution_id=exec_id, tool_name=tool_name)
        with self._lock:
            self._tokens[exec_id] = token
        return token

    def get_token(self, execution_id: str) -> Optional[CancellationToken]:
        with self._lock:
            return self._tokens.get(execution_id)

    def cancel_execution(self, execution_id: str, reason: str = "Explicit cancellation") -> bool:
        with self._lock:
            token = self._tokens.get(execution_id)
        if token:
            token.cancel(reason=reason)
            return True
        return False

    def cancel_all(self, reason: str = "Global orchestrator cancellation") -> int:
        with self._lock:
            tokens = list(self._tokens.values())
        for tok in tokens:
            tok.cancel(reason=reason)
        return len(tokens)

    def release_token(self, execution_id: str) -> None:
        with self._lock:
            self._tokens.pop(execution_id, None)


class HeartbeatMonitor:
    """HeartbeatMonitor: liveness."""

    def __init__(
        self,
        execution_id: str,
        tool_name: str,
        event_bus: Optional[Any] = None,
    ) -> None:
        self.execution_id = execution_id
        self.tool_name = tool_name
        self.event_bus = event_bus
        self.started_at: float = time.perf_counter()
        self.last_heartbeat: float = self.started_at
        self.heartbeat_count: int = 0
        self.last_message: str = "started"
        self.completed_units: int = 0
        self.total_units: int = 0
        self._lock = threading.Lock()

    def beat(self, message: str = "", completed_units: int = 0, total_units: int = 0) -> None:
        now = time.perf_counter()
        with self._lock:
            self.last_heartbeat = now
            self.heartbeat_count += 1
            if message:
                self.last_message = message
            if completed_units > 0:
                self.completed_units = completed_units
            if total_units > 0:
                self.total_units = total_units

        if self.event_bus is not None:
            try:
                from evren_code.core.events import EventType
                if hasattr(EventType, "TOOL_HEARTBEAT"):
                    self.event_bus.emit(
                        EventType.TOOL_HEARTBEAT,
                        execution_id=self.execution_id,
                        tool_name=self.tool_name,
                        message=message or self.last_message,
                        completed_units=self.completed_units,
                        total_units=self.total_units,
                        elapsed_seconds=round(now - self.started_at, 3),
                    )
            except Exception:
                pass

    @property
    def idle_seconds(self) -> float:
        with self._lock:
            return max(0.0, time.perf_counter() - self.last_heartbeat)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.started_at)


@dataclass
class WatchdogEntry:
    execution_id: str
    tool_name: str
    token: CancellationToken
    heartbeat: HeartbeatMonitor
    max_duration_seconds: float
    stall_timeout_seconds: Optional[float] = None
    intervened: bool = False
    intervention_reason: str = ""


class ExecutionWatchdog:
    """Execution watchdog implementation."""

    def __init__(self, check_interval_seconds: float = 0.05) -> None:
        self.check_interval_seconds = check_interval_seconds
        self._entries: Dict[str, WatchdogEntry] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.interventions_count: int = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._watch_loop,
                daemon=True,
                name="EvrenExecutionWatchdog",
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=0.5)
        self._thread = None

    def watch(
        self,
        execution_id: str,
        tool_name: str,
        token: CancellationToken,
        heartbeat: HeartbeatMonitor,
        max_duration_seconds: float,
        stall_timeout_seconds: Optional[float] = None,
    ) -> WatchdogEntry:
        self.start()
        entry = WatchdogEntry(
            execution_id=execution_id,
            tool_name=tool_name,
            token=token,
            heartbeat=heartbeat,
            max_duration_seconds=max_duration_seconds,
            stall_timeout_seconds=stall_timeout_seconds,
        )
        with self._lock:
            self._entries[execution_id] = entry
        return entry

    def unwatch(self, execution_id: str) -> Optional[WatchdogEntry]:
        with self._lock:
            return self._entries.pop(execution_id, None)

    def _watch_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.check_interval_seconds):
            with self._lock:
                entries = list(self._entries.values())
            for entry in entries:
                if entry.intervened or entry.token.is_cancelled:
                    continue
                elapsed = entry.heartbeat.elapsed_seconds
                idle = entry.heartbeat.idle_seconds
                if elapsed > entry.max_duration_seconds:
                    entry.intervened = True
                    entry.intervention_reason = (
                        f"Watchdog wall-clock timeout ({elapsed:.2f}s > {entry.max_duration_seconds:.2f}s)"
                    )
                    self.interventions_count += 1
                    entry.token.cancel(reason=entry.intervention_reason)
                elif (
                    entry.stall_timeout_seconds is not None
                    and entry.stall_timeout_seconds > 0
                    and idle > entry.stall_timeout_seconds
                ):
                    entry.intervened = True
                    entry.intervention_reason = (
                        f"Watchdog stall timeout (no heartbeat for {idle:.2f}s > {entry.stall_timeout_seconds:.2f}s)"
                    )
                    self.interventions_count += 1
                    entry.token.cancel(reason=entry.intervention_reason)


class OutputSizeLimiter:
    """Output size limiter implementation."""

    def __init__(self, default_max_chars: int = 16000) -> None:
        self.default_max_chars = default_max_chars

    def limit(self, text: str, max_chars: Optional[int] = None, tool_name: str = "") -> Tuple[str, bool]:
        """Limit (limited_text, was_truncated)."""
        if not isinstance(text, str):
            text = str(text or "")
        effective_max = max_chars if (max_chars is not None and max_chars > 0) else self.default_max_chars
        if len(text) <= effective_max:
            return text, False

        notice_reserve = 220
        if effective_max <= notice_reserve + 40:
            return text[:effective_max], True

        usable = effective_max - notice_reserve
        head_size = max(100, usable // 2)
        tail_size = max(100, usable - head_size)
        head = text[:head_size]
        tail = text[-tail_size:]
        label = f" ({tool_name})" if tool_name else ""
        truncated_notice = (
            f"\n\n... [Araç çıktısı bağlam taşmasını önlemek için kısaltıldı{label}: "
            f"Toplam {len(text)} karakterden {effective_max} karaktere indirildi. "
            f"Başlangıç ({head_size} krk) ve Bitiş ({tail_size} krk) korundu] ...\n\n"
        )
        result = f"{head}{truncated_notice}{tail}"
        if len(result) > effective_max:
            result = result[:effective_max]
        return result, True


class BoundedRetryManager:
    """BoundedRetryManager: transient."""

    TRANSIENT_KEYWORDS = (
        "timed out",
        "zaman aşımı",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection error",
        "resource busy",
        "locked",
        "try again",
    )

    def should_retry(
        self,
        policy: ToolTimeoutPolicy,
        attempt: int,
        error_or_result: str,
        is_exception: bool = False,
    ) -> bool:
        """Should retry."""
        if not policy.idempotent:
            return False
        if attempt >= policy.max_retries:
            return False
        if is_exception:
            return True
        lower = (error_or_result or "").lower()
        if lower.startswith("hata:") or lower.startswith("error:"):
            if any(kw in lower for kw in self.TRANSIENT_KEYWORDS):
                return True
        return False

    def compute_delay(self, policy: ToolTimeoutPolicy, attempt: int) -> float:
        return min(2.0, policy.retry_base_delay * (2 ** max(0, attempt)))


@dataclass
class BatchItemResult:
    """BatchItemResult: batch."""

    item_key: str
    success: bool
    output: str
    duration_seconds: float = 0.0
    timed_out: bool = False
    error: Optional[str] = None


class BatchExecutionManager:
    """Batch execution manager implementation."""

    def __init__(
        self,
        per_item_timeout_seconds: float = 5.0,
        batch_timeout_seconds: float = 25.0,
        max_Output_chars_per_item: int = 8000,
    ) -> None:
        self.per_item_timeout_seconds = per_item_timeout_seconds
        self.batch_timeout_seconds = batch_timeout_seconds
        self.max_output_chars_per_item = max_Output_chars_per_item

    def execute_batch(
        self,
        items: List[Any],
        item_handler: Callable[[Any], str],
        key_fn: Callable[[Any], str] = str,
        cancellation_token: Optional[CancellationToken] = None,
        heartbeat: Optional[HeartbeatMonitor] = None,
    ) -> List[BatchItemResult]:
        results: List[BatchItemResult] = []
        batch_start = time.perf_counter()
        total = len(items)

        for idx, item in enumerate(items, start=1):
            key = key_fn(item)
            if cancellation_token and cancellation_token.is_cancelled:
                results.append(
                    BatchItemResult(
                        item_key=key,
                        success=False,
                        output=f"Hata: '{key}' işlemi iptal edildi ({cancellation_token.reason}).",
                        error="Cancelled",
                    )
                )
                continue

            elapsed_batch = time.perf_counter() - batch_start
            remaining_batch = self.batch_timeout_seconds - elapsed_batch
            if remaining_batch <= 0:
                results.append(
                    BatchItemResult(
                        item_key=key,
                        success=False,
                        output=f"Hata: '{key}' toplu işlem zaman aşımı sınırı ({self.batch_timeout_seconds}s) dolduğu için atlandı.",
                        timed_out=True,
                        error="Batch timeout exceeded",
                    )
                )
                continue

            eff_item_timeout = min(self.per_item_timeout_seconds, remaining_batch)
            item_start = time.perf_counter()

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="evren_batch_item")
            fut = executor.submit(item_handler, item)
            try:
                raw_out = fut.result(timeout=eff_item_timeout)
                dur = time.perf_counter() - item_start
                is_err = isinstance(raw_out, str) and (raw_out.startswith("Hata:") or raw_out.startswith("Error:"))
                results.append(
                    BatchItemResult(
                        item_key=key,
                        success=not is_err,
                        output=raw_out,
                        duration_seconds=dur,
                        error=raw_out if is_err else None,
                    )
                )
            except concurrent.futures.TimeoutError:
                dur = time.perf_counter() - item_start
                results.append(
                    BatchItemResult(
                        item_key=key,
                        success=False,
                        output=f"Hata: '{key}' okunurken/işlenirken zaman aşımına uğradı ({eff_item_timeout:.1f}s).",
                        duration_seconds=dur,
                        timed_out=True,
                        error=f"Item timeout after {eff_item_timeout:.1f}s",
                    )
                )
            except Exception as exc:
                dur = time.perf_counter() - item_start
                results.append(
                    BatchItemResult(
                        item_key=key,
                        success=False,
                        output=f"Hata: '{key}' işlenirken beklenmeyen istisna oluştu: {exc}",
                        duration_seconds=dur,
                        error=str(exc),
                    )
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            if heartbeat:
                heartbeat.beat(
                    message=f"Batch item {idx}/{total} ({key})",
                    completed_units=idx,
                    total_units=total,
                )

        return results


@dataclass
class ToolExecutionRecord:
    """ToolExecutionRecord: traceability."""

    execution_id: str
    tool_name: str
    caller_role: str
    caller_agent: str
    status: ExecutionLifecycleStatus
    started_at: float
    completed_at: float = 0.0
    duration_seconds: float = 0.0
    attempts: int = 1
    timed_out: bool = False
    cancelled: bool = False
    watchdog_intervened: bool = False
    output_truncated: bool = False
    late_response_ignored: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "tool_name": self.tool_name,
            "caller_role": self.caller_role,
            "caller_agent": self.caller_agent,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": round(self.duration_seconds, 4),
            "attempts": self.attempts,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "watchdog_intervened": self.watchdog_intervened,
            "output_truncated": self.output_truncated,
            "late_response_ignored": self.late_response_ignored,
            "error": self.error,
        }


class CentralizedToolExecutionManager:
    """Centralized tool execution manager implementation."""

    def __init__(
        self,
        policy_registry: Optional[ToolPolicyRegistry] = None,
        cancellation_manager: Optional[CancellationManager] = None,
        watchdog: Optional[ExecutionWatchdog] = None,
        output_limiter: Optional[OutputSizeLimiter] = None,
        retry_manager: Optional[BoundedRetryManager] = None,
        event_bus: Optional[Any] = None,
        max_output_chars: int = 16000,
    ) -> None:
        self.policy_registry = policy_registry or ToolPolicyRegistry()
        self.cancellation_manager = cancellation_manager or CancellationManager()
        self.watchdog = watchdog or ExecutionWatchdog()
        self.output_limiter = output_limiter or OutputSizeLimiter(default_max_chars=max_output_chars)
        self.retry_manager = retry_manager or BoundedRetryManager()
        self.event_bus = event_bus

        self._records: Dict[str, ToolExecutionRecord] = {}
        self._terminal_executions: Set[str] = set()
        self._late_ignored_count: int = 0
        self._lock = threading.RLock()

    @property
    def late_ignored_count(self) -> int:
        with self._lock:
            return self._late_ignored_count

    def get_record(self, execution_id: str) -> Optional[ToolExecutionRecord]:
        with self._lock:
            return self._records.get(execution_id)

    def get_recent_records(self, limit: int = 100) -> List[ToolExecutionRecord]:
        with self._lock:
            vals = list(self._records.values())
        return vals[-limit:]

    def cancel_all(self, reason: str = "Orchestrator cancelled") -> int:
        return self.cancellation_manager.cancel_all(reason=reason)

    def execute_managed(
        self,
        tool_name: str,
        handler: Callable[..., Any],
        arguments: Dict[str, Any],
        caller_role: str = "main",
        caller_agent: str = "orchestrator",
        timeout_override: Optional[float] = None,
        max_output_chars_override: Optional[int] = None,
        external_cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[str, ToolExecutionRecord]:
        """Execute managed."""
        policy = self.policy_registry.get_policy(tool_name, arguments)
        eff_timeout = (
            float(timeout_override)
            if (timeout_override is not None and timeout_override > 0)
            else policy.timeout_seconds
        )
        eff_max_chars = (
            max_output_chars_override
            if (max_output_chars_override is not None and max_output_chars_override > 0)
            else (policy.max_output_chars or self.output_limiter.default_max_chars)
        )

        exec_id = f"exec_{tool_name}_{uuid.uuid4().hex[:8]}"
        start_wall = time.time()
        start_perf = time.perf_counter()

        record = ToolExecutionRecord(
            execution_id=exec_id,
            tool_name=tool_name,
            caller_role=caller_role,
            caller_agent=caller_agent,
            status=ExecutionLifecycleStatus.RUNNING,
            started_at=start_wall,
        )
        with self._lock:
            self._records[exec_id] = record

        max_attempts = 1 + (policy.max_retries if policy.idempotent else 0)
        last_result_str = ""

        for attempt in range(max_attempts):
            record.attempts = attempt + 1
            if external_cancel_event is not None and external_cancel_event.is_set():
                record.status = ExecutionLifecycleStatus.CANCELLED
                record.cancelled = True
                record.error = "External cancellation requested before tool attempt"
                last_result_str = f"Hata: '{tool_name}' aracı görev iptal edildiği için çalıştırılmadı."
                break

            attempt_exec_id = f"{exec_id}_a{attempt + 1}"
            token = self.cancellation_manager.create_token(execution_id=attempt_exec_id, tool_name=tool_name)
            heartbeat = HeartbeatMonitor(
                execution_id=attempt_exec_id,
                tool_name=tool_name,
                event_bus=self.event_bus,
            )
            wd_entry = self.watchdog.watch(
                execution_id=attempt_exec_id,
                tool_name=tool_name,
                token=token,
                heartbeat=heartbeat,
                max_duration_seconds=eff_timeout,
                stall_timeout_seconds=policy.stall_timeout_seconds,
            )

            ext_stop = threading.Event()
            if external_cancel_event is not None:
                def _bridge_external_cancel(ev=external_cancel_event, tk=token, st=ext_stop) -> None:
                    while not st.wait(timeout=0.03):
                        if ev.is_set():
                            tk.cancel(reason="External task/orchestrator cancellation")
                            break

                threading.Thread(target=_bridge_external_cancel, daemon=True).start()

            attempt_timed_out = False
            attempt_cancelled = False
            attempt_exception = False

            try:
                raw_res = self._invoke_single_attempt(
                    tool_name=tool_name,
                    handler=handler,
                    arguments=arguments,
                    token=token,
                    heartbeat=heartbeat,
                    timeout_seconds=eff_timeout,
                    exec_id=exec_id,
                )
                last_result_str = str(raw_res) if raw_res is not None else ""
            except ToolCancelledError as cancel_exc:
                if wd_entry.intervened:
                    attempt_timed_out = True
                    last_result_str = (
                        f"Hata: '{tool_name}' aracı zaman aşımına uğradı veya yanıt vermeyi kesti "
                        f"({eff_timeout:.1f}sn sınırı / {wd_entry.intervention_reason}). İşlem güvenli şekilde sonlandırıldı."
                    )
                    record.error = wd_entry.intervention_reason
                else:
                    attempt_cancelled = True
                    last_result_str = f"Hata: '{tool_name}' aracı iptal edildi: {cancel_exc}"
                    record.error = str(cancel_exc)
            except concurrent.futures.TimeoutError:
                attempt_timed_out = True
                token.cancel(reason=f"Timeout after {eff_timeout:.2f}s")
                reason_detail = wd_entry.intervention_reason or f"{eff_timeout:.1f}s zaman aşımı"
                last_result_str = (
                    f"Hata: '{tool_name}' aracı {eff_timeout:.1f} saniye içinde tamamlanamadı "
                    f"(Zaman aşımı / {reason_detail}). Çalışan alt süreçler ve kaynaklar temizlendi."
                )
                record.error = f"Timeout after {eff_timeout:.2f}s"
            except TypeError as type_exc:
                attempt_exception = True
                last_result_str = f"Hata: '{tool_name}' aracı eksik veya hatalı parametre ile çağrıldı: {type_exc}"
                record.error = str(type_exc)
            except Exception as exc:
                attempt_exception = True
                last_result_str = f"Hata: '{tool_name}' aracı çalışırken beklenmeyen bir istisna oluştu: {exc}"
                record.error = str(exc)
            finally:
                ext_stop.set()
                self.watchdog.unwatch(attempt_exec_id)
                self.cancellation_manager.release_token(attempt_exec_id)
                if wd_entry.intervened:
                    record.watchdog_intervened = True

            if attempt_cancelled or (external_cancel_event is not None and external_cancel_event.is_set()):
                record.status = ExecutionLifecycleStatus.CANCELLED
                record.cancelled = True
                break

            should_retry = self.retry_manager.should_retry(
                policy=policy,
                attempt=attempt,
                error_or_result=last_result_str,
                is_exception=(attempt_exception or attempt_timed_out),
            )
            if should_retry:
                delay = self.retry_manager.compute_delay(policy, attempt)
                time.sleep(delay)
                continue

            if attempt_timed_out:
                record.status = ExecutionLifecycleStatus.TIMED_OUT
                record.timed_out = True
            elif attempt_exception or last_result_str.startswith("Hata:"):
                record.status = ExecutionLifecycleStatus.FAILED
            else:
                record.status = ExecutionLifecycleStatus.COMPLETED
            break

        with self._lock:
            self._terminal_executions.add(exec_id)

        limited_str, was_truncated = self.output_limiter.limit(
            last_result_str,
            max_chars=eff_max_chars,
            tool_name=tool_name,
        )
        record.output_truncated = was_truncated
        record.completed_at = time.time()
        record.duration_seconds = time.perf_counter() - start_perf

        self._emit_lifecycle_telemetry(record)
        return limited_str, record

    def _invoke_single_attempt(
        self,
        tool_name: str,
        handler: Callable[..., Any],
        arguments: Dict[str, Any],
        token: CancellationToken,
        heartbeat: HeartbeatMonitor,
        timeout_seconds: float,
        exec_id: str,
    ) -> str:
        """Invoke single attempt (coroutine)."""
        filtered_args = self._filter_handler_arguments(handler, arguments)
        is_coro_fn = inspect.iscoroutinefunction(handler)

        if is_coro_fn:
            return self._run_async_handler(
                handler=handler,
                filtered_args=filtered_args,
                token=token,
                heartbeat=heartbeat,
                timeout_seconds=timeout_seconds,
                exec_id=exec_id,
            )

        def _sync_worker() -> str:
            _ACTIVE_EXECUTION_TLS.token = token
            _ACTIVE_EXECUTION_TLS.heartbeat = heartbeat
            try:
                token.check_cancelled()
                res = handler(**filtered_args)
                if inspect.iscoroutine(res):
                    return asyncio.run(asyncio.wait_for(res, timeout=timeout_seconds))
                with self._lock:
                    if exec_id in self._terminal_executions or token.is_cancelled:
                        self._late_ignored_count += 1
                        rec = self._records.get(exec_id)
                        if rec:
                            rec.late_response_ignored = True
                        raise ToolCancelledError("Late tool response after timeout/cancellation ignored")
                return str(res) if res is not None else ""
            finally:
                _ACTIVE_EXECUTION_TLS.token = None
                _ACTIVE_EXECUTION_TLS.heartbeat = None

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"evren_tool_{tool_name[:12]}",
        )
        fut = executor.submit(_sync_worker)
        try:
            deadline = time.perf_counter() + timeout_seconds
            while True:
                if token.is_cancelled:
                    raise ToolCancelledError(token.reason or "Cancelled")
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise concurrent.futures.TimeoutError()
                wait_slice = min(0.05, remaining)
                try:
                    return fut.result(timeout=wait_slice)
                except concurrent.futures.TimeoutError:
                    if time.perf_counter() >= deadline:
                        raise
                    continue
        finally:
            if not fut.done():
                def _mark_late_if_completed(f: concurrent.futures.Future, eid: str = exec_id) -> None:
                    try:
                        if not f.cancelled() and f.exception() is None:
                            with self._lock:
                                if eid in self._terminal_executions:
                                    self._late_ignored_count += 1
                                    rec = self._records.get(eid)
                                    if rec:
                                        rec.late_response_ignored = True
                    except Exception:
                        pass

                fut.add_done_callback(_mark_late_if_completed)
            executor.shutdown(wait=False, cancel_futures=True)

    def _run_async_handler(
        self,
        handler: Callable[..., Any],
        filtered_args: Dict[str, Any],
        token: CancellationToken,
        heartbeat: HeartbeatMonitor,
        timeout_seconds: float,
        exec_id: str,
    ) -> str:
        """Run async handler."""
        def _async_runner() -> str:
            _ACTIVE_EXECUTION_TLS.token = token
            _ACTIVE_EXECUTION_TLS.heartbeat = heartbeat
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def _wrapped() -> str:
                    coro = handler(**filtered_args)
                    task = asyncio.create_task(coro)
                    token.register_async_task(task)
                    try:
                        res = await asyncio.wait_for(task, timeout=timeout_seconds)
                        with self._lock:
                            if exec_id in self._terminal_executions or token.is_cancelled:
                                self._late_ignored_count += 1
                                rec = self._records.get(exec_id)
                                if rec:
                                    rec.late_response_ignored = True
                                raise ToolCancelledError("Late async tool response ignored")
                        return str(res) if res is not None else ""
                    except asyncio.TimeoutError:
                        task.cancel()
                        raise concurrent.futures.TimeoutError()
                    except asyncio.CancelledError:
                        raise ToolCancelledError(token.reason or "Async task cancelled")

                return loop.run_until_complete(_wrapped())
            finally:
                try:
                    pending = asyncio.all_tasks(loop)
                    for p in pending:
                        p.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                loop.close()
                _ACTIVE_EXECUTION_TLS.token = None
                _ACTIVE_EXECUTION_TLS.heartbeat = None

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="evren_async_tool")
        fut = executor.submit(_async_runner)
        try:
            return fut.result(timeout=timeout_seconds + 0.5)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _filter_handler_arguments(handler: Callable[..., Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
        sig = inspect.signature(handler)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_kw:
            return {k: v for k, v in arguments.items() if k in sig.parameters}
        return dict(arguments)

    def _emit_lifecycle_telemetry(self, record: ToolExecutionRecord) -> None:
        if not self.event_bus:
            return
        try:
            from evren_code.core.events import EventType
            if record.timed_out and hasattr(EventType, "TOOL_EXECUTION_TIMEOUT"):
                self.event_bus.emit(
                    EventType.TOOL_EXECUTION_TIMEOUT,
                    execution_id=record.execution_id,
                    tool_name=record.tool_name,
                    duration_seconds=record.duration_seconds,
                    watchdog_intervened=record.watchdog_intervened,
                    error=record.error,
                )
            elif record.cancelled and hasattr(EventType, "TOOL_EXECUTION_CANCELLED"):
                self.event_bus.emit(
                    EventType.TOOL_EXECUTION_CANCELLED,
                    execution_id=record.execution_id,
                    tool_name=record.tool_name,
                    duration_seconds=record.duration_seconds,
                    error=record.error,
                )
        except Exception:
            pass
