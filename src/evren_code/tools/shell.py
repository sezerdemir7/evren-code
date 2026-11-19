"""Terminal / Shell Command Execution and Background Process Manager (`run_command`, `get_background_output`, `stop_background_process`).

Includes timeout protection, dynamic build command timeouts, Unix/Windows command normalization (`rm -rf` etc.),
output truncation, working directory management, and background daemon / dev server process management.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

from evren_code.config import AppConfig
from evren_code.tools.filesystem import resolve_safe_path
from evren_code.tools.registry import ToolRegistry


COMMON_SECRET_KEYS = (
    "EVREN_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "DATABASE_URL",
)

COMPILED_SECRET_PATTERNS = [
    # PEM / OpenSSH / PGP Private Key blocks
    (re.compile(r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z0-9_-]+ )?PRIVATE KEY-----"), "***MASKED_PRIVATE_KEY_BLOCK***"),
    # Database and service connection credentials (postgres, mysql, redis, etc.)
    (re.compile(r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|mssql):\/\/)([^:\s\/]+):([^@\s]+)(@)"), r"\1\2:***MASKED_PASSWORD***\4"),
    # JSON / Structured secret keys (client_secret, api_key, etc.)
    (re.compile(r'(?i)("(?:client_secret|api_key|access_token|secret_key|refresh_token|private_key|auth_token)"\s*:\s*)"[^"]+"'), r'\1"***MASKED_SECRET***"'),
    (re.compile(r"(?i)('(?:client_secret|api_key|access_token|secret_key|refresh_token|private_key|auth_token)'\s*:\s*)'[^']+'"), r"\1'***MASKED_SECRET***'"),
    # Bearer and Token patterns
    (re.compile(r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{20,}"), r"\1***MASKED_TOKEN***"),
    (re.compile(r"(?i)(token[\s:=]+)[a-zA-Z0-9_\-\.]{20,}"), r"\1***MASKED_TOKEN***"),
    (re.compile(r"evren_llm_[a-zA-Z0-9_\-]{16,}"), r"evren_llm_***MASKED_API_KEY***"),
    (re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"), r"sk-***MASKED_API_KEY***"),
    (re.compile(r"(?i)(password[\s:=]+)(['\"][^'\"]+['\"]|\S+)"), r"\1***MASKED_PASSWORD***"),
]


def mask_sensitive_data(text: str, custom_secrets: Optional[List[str]] = None) -> str:
    """Masks API keys, tokens, and secret environment variables in command outputs (stdout/stderr)."""
    if not text:
        return text

    masked = text

    # 1. Mask values of known critical environment variables
    for env_var in COMMON_SECRET_KEYS:
        val = os.environ.get(env_var)
        if val and len(val) >= 6 and val in masked:
            masked = masked.replace(val, f"***MASKED_{env_var}***")

    # Sensitive values containing 'KEY', 'TOKEN', 'SECRET', 'PASSWORD' in all environment variables
    for k, v in os.environ.items():
        if any(term in k.upper() for term in ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "AUTH")):
            if v and len(v) >= 8 and v in masked:
                masked = masked.replace(v, "***MASKED_SECRET***")

    if custom_secrets:
        for s in custom_secrets:
            if s and len(s) >= 6 and s in masked:
                masked = masked.replace(s, "***MASKED_SECRET***")

    # 2. Regex-based pattern masking
    for pattern, replacement in COMPILED_SECRET_PATTERNS:
        masked = pattern.sub(replacement, masked)

    return masked


def kill_process_tree(pid: int) -> None:
    """Forcefully terminates the entire process tree (child processes) for the given PID."""
    if not pid or pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
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


CommandInterceptor = Callable[[str, Optional[int]], tuple[str, Optional[int]]]


class CommandInterceptorRegistry:
    """Extensible interceptor registry for transforming commands or adapting timeouts before execution."""

    def __init__(self) -> None:
        self._interceptors: List[CommandInterceptor] = []

    def register(self, interceptor: CommandInterceptor) -> None:
        """Registers a new command interceptor/transform hook."""
        self._interceptors.append(interceptor)

    def process(self, command: str, timeout: Optional[int]) -> tuple[str, Optional[int]]:
        """Processes registered interceptors sequentially on command and timeout."""
        current_cmd = command
        current_timeout = timeout
        for interceptor in self._interceptors:
            try:
                current_cmd, current_timeout = interceptor(current_cmd, current_timeout)
            except Exception:
                pass
        return current_cmd, current_timeout


default_command_interceptor_registry = CommandInterceptorRegistry()


class PersistentShellSession:
    """Stateful shell session tracking working directory (cwd) and environment across agent steps."""

    def __init__(self, workspace_root: Path, default_timeout: int = 60) -> None:
        self.workspace_root = workspace_root.resolve()
        self.current_cwd = self.workspace_root
        self.default_timeout = default_timeout
        self.env: Dict[str, str] = os.environ.copy()
        self._lock = threading.RLock()

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> tuple[int, str, str]:
        with self._lock:
            # Resolve target or active working directory
            target_cwd = (
                Path(cwd).resolve() if cwd else self.current_cwd
            )
            if not target_cwd.is_dir():
                target_cwd = self.workspace_root

            eff_timeout = timeout or self.default_timeout
            cleaned_cmd = command.strip()

            # Pre-intercept cd directory changes
            cd_match = re.match(r"^cd\s+[\"']?([^\"'&|;]+)[\"']?\s*$", cleaned_cmd, re.IGNORECASE)
            if cd_match:
                rel_dir = cd_match.group(1).strip()
                if rel_dir == "~":
                    candidate = Path.home()
                else:
                    candidate = (target_cwd / rel_dir).resolve()
                if candidate.is_dir():
                    self.current_cwd = candidate
                    return 0, f"Working directory changed: {self.current_cwd}", ""
                else:
                    return 1, "", f"Error: Directory '{rel_dir}' not found."

            # Append state marker for cwd tracking
            state_marker = f"__EVREN_STATE_{uuid.uuid4().hex[:8]}__"
            if os.name == "nt":
                instrumented_cmd = f"{cleaned_cmd} & echo {state_marker} & cd"
                use_ps = _should_use_powershell(cleaned_cmd) and not _is_windows_cmd_syntax(cleaned_cmd)
                if not use_ps:
                    normalized = normalize_windows_command(instrumented_cmd, use_cmd=True)
                    args = ["cmd.exe", "/d", "/s", "/c", normalized]
                else:
                    ps_cmd = f"{cleaned_cmd}; Write-Output '{state_marker}'; (Get-Location).Path"
                    normalized = normalize_windows_command(ps_cmd, use_cmd=False)
                    args = [
                        "powershell.exe",
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        normalized,
                    ]
            else:
                instrumented_cmd = f"{cleaned_cmd}\necho '{state_marker}'\npwd"
                args = instrumented_cmd

            rc, stdout, stderr = execute_command_with_tree_kill(
                args,
                cwd=str(target_cwd),
                timeout=eff_timeout,
                shell=(os.name != "nt"),
                env=self.env,
            )

            # Parse new working directory from stdout
            final_stdout = stdout
            if state_marker in stdout:
                parts = stdout.split(state_marker)
                final_stdout = parts[0].rstrip("\r\n")
                if len(parts) > 1:
                    new_cwd_candidate = parts[1].strip().splitlines()
                    if new_cwd_candidate:
                        last_line = new_cwd_candidate[-1].strip()
                        if last_line and os.path.isdir(last_line):
                            self.current_cwd = Path(last_line).resolve()

            return rc, final_stdout, stderr

    def close(self) -> None:
        """Cleans up session and resets to workspace root."""
        with self._lock:
            self.current_cwd = self.workspace_root


def execute_command_with_tree_kill(
    args: List[str] | str,
    cwd: str,
    timeout: Optional[int] = None,
    shell: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> tuple[int, str, str]:
    """Executes command and returns (returncode, stdout, stderr) with tree-kill safety."""
    from evren_code.core.tool_reliability import (
        get_current_cancellation_token,
        register_active_subprocess_pid,
        report_tool_progress,
        unregister_active_subprocess_pid,
    )

    proc = subprocess.Popen(
        args,
        cwd=cwd,
        shell=shell,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    register_active_subprocess_pid(proc.pid)

    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    def _drain_stream(stream, chunks: List[str]) -> None:
        if not stream:
            return
        line_cnt = 0
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                chunks.append(line)
                line_cnt += 1
                if line_cnt % 10 == 1:
                    report_tool_progress("subprocess stream output", completed_units=line_cnt)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_drain_stream, args=(proc.stdout, stdout_chunks), daemon=True)
    t_err = threading.Thread(target=_drain_stream, args=(proc.stderr, stderr_chunks), daemon=True)
    t_out.start()
    t_err.start()

    try:
        rc = proc.wait(timeout=timeout)
        tok = get_current_cancellation_token()
        if tok is not None and tok.is_cancelled:
            kill_process_tree(proc.pid)
            raise subprocess.TimeoutExpired(
                cmd=args,
                timeout=timeout or 0,
                output="".join(stdout_chunks),
                stderr="".join(stderr_chunks),
            )
        # Main process exited; allow brief window to drain remaining pipe data
        t_out.join(timeout=0.25)
        t_err.join(timeout=0.25)
        return rc, "".join(stdout_chunks), "".join(stderr_chunks)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc.pid)
        try:
            proc.kill()
        except Exception:
            pass
        t_out.join(timeout=0.3)
        t_err.join(timeout=0.3)
        raise subprocess.TimeoutExpired(
            cmd=args,
            timeout=timeout or 0,
            output="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )
    except Exception:
        kill_process_tree(proc.pid)
        try:
            proc.kill()
        except Exception:
            pass
        raise
    finally:
        unregister_active_subprocess_pid(proc.pid)


@dataclass
class BackgroundProcessEntry:
    """Holds state and ring buffer for a background shell process."""

    process_id: str
    command: str
    cwd: str
    process: subprocess.Popen
    started_at: float = field(default_factory=time.time)
    output_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=500))
    lock: threading.Lock = field(default_factory=threading.Lock)


class BackgroundProcessManager:
    """Manages long-running background processes (e.g. `npm run dev`, `uvicorn`, `pytest --watch`)."""

    def __init__(self) -> None:
        self._processes: Dict[str, BackgroundProcessEntry] = {}
        self._lock = threading.Lock()

    def start(self, args: List[str] | str, cwd: str, command_display: str, shell: bool = False) -> BackgroundProcessEntry:
        proc_id = f"bg_{uuid.uuid4().hex[:6]}"
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        entry = BackgroundProcessEntry(
            process_id=proc_id,
            command=command_display,
            cwd=cwd,
            process=proc,
        )
        with self._lock:
            self._processes[proc_id] = entry

        reader_thread = threading.Thread(
            target=self._stream_reader,
            args=(entry,),
            daemon=True,
        )
        reader_thread.start()
        return entry

    @staticmethod
    def _stream_reader(entry: BackgroundProcessEntry) -> None:
        stream = entry.process.stdout
        if not stream:
            return
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                with entry.lock:
                    entry.output_lines.append(line.rstrip("\r\n"))
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def get_output(self, process_id: str, tail_lines: int = 80) -> str:
        with self._lock:
            entry = self._processes.get(process_id)
        if not entry:
            return f"Error: Background process '{process_id}' not found."

        rc = entry.process.poll()
        status_str = "Running / Çalışıyor (Running)" if rc is None else f"Terminated / Sonlandı (Exit Code: {rc})"
        elapsed = time.time() - entry.started_at
        with entry.lock:
            lines = list(entry.output_lines)[-max(1, tail_lines):]
        log_text = mask_sensitive_data("\n".join(lines) if lines else "(No output generated yet / Henüz çıktı üretilmedi)")
        return (
            f"Process ID: {entry.process_id} | Status: {status_str} | Elapsed: {elapsed:.1f}s\n"
            f"Command: `{entry.command}`\n"
            f"--- RECENT OUTPUT ({len(lines)} lines / satır) ---\n{log_text}"
        )

    def stop(self, process_id: str) -> str:
        with self._lock:
            entry = self._processes.get(process_id)
        if not entry:
            return f"Error: Background process '{process_id}' not found."

        rc = entry.process.poll()
        if rc is None:
            try:
                kill_process_tree(entry.process.pid)
                entry.process.terminate()
                try:
                    entry.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    entry.process.kill()
            except Exception as exc:
                return f"Error: Failed to stop '{process_id}': {exc}"

        return f"Success: Background process '{process_id}' (`{entry.command}`) stopped (Başarılı: '{process_id}' arka plan süreci durduruldu)."

    def list_processes(self) -> List[BackgroundProcessEntry]:
        with self._lock:
            return list(self._processes.values())

    def stop_all(self) -> None:
        for pid in list(self._processes.keys()):
            self.stop(pid)


def normalize_windows_command(command: str, use_cmd: bool) -> str:
    """Converts Unix-style commands (`rm -rf`, `touch`) to CMD or PowerShell equivalents on Windows."""
    def _rm_replacer(match: re.Match) -> str:
        raw_targets = match.group(1).strip()
        try:
            targets = shlex.split(raw_targets)
        except Exception:
            targets = raw_targets.split()
        if not targets:
            return ""
        if use_cmd:
            parts = [
                f"(if exist {t} rd /s /q {t} 2>nul & if exist {t} del /f /q {t} 2>nul)"
                for t in targets
            ]
            return " & ".join(parts)
        else:
            ps_items = ", ".join(
                [f'"{t.strip(chr(34) + chr(39))}"' for t in targets]
            )
            return f"Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -Path {ps_items}"

    # rm -rf / rm -r / rm -f yakalama
    cmd = re.sub(r"\brm\s+-[rRfF]+\s+([^;&|]+)", _rm_replacer, command)

    if use_cmd:
        cmd = re.sub(r"\btouch\s+([^\s&|;]+)", r"type nul > \1", cmd)
        cmd = re.sub(r"2>\s*/dev/null", "2>nul", cmd)
        cmd = re.sub(r">\s*/dev/null", ">nul", cmd)
        cmd = re.sub(r"\bcat\s+([^\s;&|]+)", r"type \1", cmd)
    else:
        cmd = re.sub(
            r"\btouch\s+([^\s&|;]+)",
            r"if (-not (Test-Path \1)) { New-Item -ItemType File -Force \1 }",
            cmd,
        )
        cmd = re.sub(r"\bInvoke-WebRequest(?!\s+-UseBasicParsing)\b", "Invoke-WebRequest -UseBasicParsing", cmd)
        if re.search(r"^\s*dir\s+/[a-zA-Z]", cmd, re.IGNORECASE):
            cmd = f'cmd.exe /c "{cmd}"'

        cmd = re.sub(r"2>\s*/dev/null", "2>$null", cmd)
        cmd = re.sub(r">\s*/dev/null", ">$null", cmd)

        cmd = re.sub(r"\bls\s+-[a-zA-Z]+\b", "Get-ChildItem -Force", cmd)

        cmd = re.sub(
            r"\bwc\s+-l\s+([^\s;&|]+)",
            r"(Get-Content \1 | Measure-Object -Line).Lines",
            cmd,
        )

        cmd = re.sub(
            r"(?<!\|)\s*\btail\s+(?:-n\s*|-)?(\d+)\s+([^\s;&|]+)",
            r"Get-Content -Tail \1 \2",
            cmd,
        )

        cmd = re.sub(
            r"(?<!\|)\s*\bhead\s+(?:-n\s*|-)?(\d+)\s+([^\s;&|]+)",
            r"Get-Content -Head \1 \2",
            cmd,
        )

        cmd = re.sub(
            r"\bcat\s+([^\s;&|]+)",
            r"Get-Content \1",
            cmd,
        )

        cmd = re.sub(
            r"\|\s*tail\s+(?:-n\s*|-)?(\d+)\b",
            r"| Select-Object -Last \1",
            cmd,
        )
        cmd = re.sub(
            r"\|\s*head\s+(?:-n\s*|-)?(\d+)\b",
            r"| Select-Object -First \1",
            cmd,
        )
    return cmd


def calculate_effective_timeout(
    command: str, config: AppConfig, explicit_timeout: Optional[int] = None
) -> int:
    """Calculate effective timeout."""
    if explicit_timeout and explicit_timeout > 0:
        return explicit_timeout

    cmd_lower = command.lower().strip()
    long_running_keywords = (
        "npm install",
        "npm i ",
        "npm ci",
        "yarn",
        "pnpm",
        "pip install",
        "cargo build",
        "cargo run",
        "cargo test",
        "npm run build",
        "next build",
        "npm test",
        "npm run test",
        "pytest",
        "dotnet build",
        "mvn ",
        "gradle ",
        "go build",
        "docker build",
        "cmake --build",
        "make ",
    )
    if any(kw in cmd_lower for kw in long_running_keywords):
        return max(
            config.tools.command_timeout_seconds,
            getattr(config.tools, "build_command_timeout_seconds", 90),
        )
    return config.tools.command_timeout_seconds


def _is_windows_cmd_syntax(cleaned_cmd: str) -> bool:
    """Is windows cmd syntax."""
    stripped = cleaned_cmd.strip()
    lower = stripped.lower()
    if (
        "&&" in stripped
        or "||" in stripped
        or "2>nul" in lower
        or ">nul" in lower
        or lower.startswith(("rd ", "del ", "rmdir ", "copy ", "move ", "cls", "mklink "))
    ):
        return True
    if lower.startswith("dir ") and re.search(r"(?:^|\s)/[sbaowpq]\b", lower):
        return True
    return False


def _should_use_powershell(cleaned_cmd: str) -> bool:
    """Should use powershell."""
    stripped = cleaned_cmd.strip()
    if stripped.startswith("&"):
        return True
    if "$" in stripped:
        return True
    if any(sym in stripped for sym in ("{", "}", ";")):
        return True

    ps_patterns = (
        r"\b(?:Get|Set|New|Remove|Test|Select|Measure|Format|Out|Start|Stop|Restart)-[a-zA-Z]+",
        r"\b(?:Where-Object|ForEach-Object|Select-String|Measure-Object|Test-Path)\b",
        r"\b(?:SilentlyContinue|ErrorAction|Out-Null|Out-String)\b",
    )
    for p in ps_patterns:
        if re.search(p, stripped, re.IGNORECASE):
            return True
    return False


def _build_shell_invocation(cleaned_cmd: str) -> tuple[List[str] | str, bool]:
    """Build shell invocation."""
    if os.name == "nt":
        if _is_windows_cmd_syntax(cleaned_cmd) or not _should_use_powershell(cleaned_cmd):
            normalized_cmd = normalize_windows_command(cleaned_cmd, use_cmd=True)
            return ["cmd.exe", "/d", "/s", "/c", normalized_cmd], False
        normalized_cmd = normalize_windows_command(cleaned_cmd, use_cmd=False)
        return [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            normalized_cmd,
        ], False
    return cleaned_cmd, True


def register_shell_tools(
    registry: ToolRegistry,
    config: AppConfig,
    bg_manager: Optional[BackgroundProcessManager] = None,
    interceptor_registry: Optional[CommandInterceptorRegistry] = None,
    persistent_session: Optional[PersistentShellSession] = None,
) -> BackgroundProcessManager:
    """Register shell tools."""
    manager = bg_manager or BackgroundProcessManager()
    active_interceptors = interceptor_registry or default_command_interceptor_registry
    shell_session = persistent_session or PersistentShellSession(config.workspace_root)

    def run_command(
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
        background: bool = False,
    ) -> str:
        try:
            work_dir = (
                resolve_safe_path(config.workspace_root, cwd)
                if cwd
                else config.workspace_root
            )
        except PermissionError as p_err:
            return f"Security Error: {p_err} (Güvenlik Hatası: {p_err})"

        if not work_dir.is_dir():
            return f"Error: Working directory '{cwd}' not found."

        cleaned_cmd = command.strip()
        # Pre-process command and timeout through registered hooks/interceptors
        cleaned_cmd, timeout = active_interceptors.process(cleaned_cmd, timeout)

        # If cwd is specified and command attempts to 'cd' into same directory (e.g. cwd='backend' and cmd='cd backend && ...')
        # avoid duplicate directory path errors on Windows:
        if cwd:
            dir_name = Path(cwd).name
            cd_prefix_pattern = rf"^cd\s+[\"']?{re.escape(dir_name)}[\"']?\s*&&\s*"
            cleaned_cmd = re.sub(cd_prefix_pattern, "", cleaned_cmd, flags=re.IGNORECASE)

        if background:
            try:
                inv_args, use_shell = _build_shell_invocation(cleaned_cmd)
                entry = manager.start(
                    args=inv_args,
                    cwd=str(work_dir),
                    command_display=cleaned_cmd,
                    shell=use_shell,
                )
                time.sleep(0.25)
                rc = entry.process.poll()
                status_note = "Running / Çalışıyor (Running)" if rc is None else f"Terminated immediately (Exit Code: {rc})"
                with entry.lock:
                    initial_lines = list(entry.output_lines)
                preview = mask_sensitive_data("\n".join(initial_lines[:20]) if initial_lines else "(No output yet / Henüz çıktı yok)")
                return (
                    f"Success: Command started in background. (Başarılı: Komut arka planda başlatıldı)\n"
                    f"process_id: {entry.process_id}\n"
                    f"Status: {status_note}\n"
                    f"Initial Output Preview:\n{preview}\n\n"
                    f"[Hint]: To inspect outputs, use `get_background_output(process_id='{entry.process_id}')`. "
                    f"To terminate, use `stop_background_process(process_id='{entry.process_id}')`."
                )
            except Exception as exc:
                return f"Error: Failed to start background process: {exc}"

        effective_timeout = calculate_effective_timeout(cleaned_cmd, config, timeout)

        try:
            if shell_session:
                rc, stdout, stderr = shell_session.execute(
                    cleaned_cmd,
                    cwd=str(work_dir) if cwd else None,
                    timeout=effective_timeout,
                )
            elif os.name == "nt":
                use_ps = _should_use_powershell(cleaned_cmd) and not _is_windows_cmd_syntax(cleaned_cmd)
                if not use_ps:
                    normalized_cmd = normalize_windows_command(cleaned_cmd, use_cmd=True)
                    rc, stdout, stderr = execute_command_with_tree_kill(
                        ["cmd.exe", "/d", "/s", "/c", normalized_cmd],
                        cwd=str(work_dir),
                        timeout=effective_timeout,
                    )
                else:
                    normalized_cmd = normalize_windows_command(cleaned_cmd, use_cmd=False)
                    rc, stdout, stderr = execute_command_with_tree_kill(
                        [
                            "powershell.exe",
                            "-NoProfile",
                            "-NonInteractive",
                            "-ExecutionPolicy",
                            "Bypass",
                            "-Command",
                            normalized_cmd,
                        ],
                        cwd=str(work_dir),
                        timeout=effective_timeout,
                    )
                    # If PowerShell 5.1 failed due to '&&' syntax, auto-fallback to cmd.exe
                    if (
                        rc != 0
                        and "is not a valid statement separator" in (stderr or "")
                    ):
                        cmd_fallback = normalize_windows_command(cleaned_cmd, use_cmd=True)
                        rc, stdout, stderr = execute_command_with_tree_kill(
                            ["cmd.exe", "/d", "/s", "/c", cmd_fallback],
                            cwd=str(work_dir),
                            timeout=effective_timeout,
                        )
            else:
                rc, stdout, stderr = execute_command_with_tree_kill(
                    cleaned_cmd,
                    shell=True,
                    cwd=str(work_dir),
                    timeout=effective_timeout,
                )
        except subprocess.TimeoutExpired:
            return (
                f"Error: Command did not complete within {effective_timeout} seconds (Timeout) "
                f": `{cleaned_cmd}`). "
                f"Tavsiye: Ağ gerektiren ağır derlemeler yerine `tsc --noEmit` gibi hızlı tip kontrollerini tercih edin, "
                f"veya uzun süren arka plan süreçleri için `background=True` parametresini kullanın."
            )
        except Exception as exc:
            return f"Error: Command execution failed: {exc}"

        stdout_clean = (stdout or "").strip()
        stderr_clean = (stderr or "").strip()
        combined_parts = [f"Exit Code / Çıkış Kodu (Exit Code): {rc}"]
        if stdout_clean:
            combined_parts.append(f"--- STDOUT ---\n{stdout_clean}")
        if stderr_clean:
            combined_parts.append(f"--- STDERR ---\n{stderr_clean}")
        if not stdout_clean and not stderr_clean:
            combined_parts.append("(Command produced no output / Komut çıktı üretmedi)")

        full_output = mask_sensitive_data("\n".join(combined_parts))
        if rc != 0 and stderr:
            if "is not recognized as an internal or external command" in stderr or "tanınmıyor" in stderr:
                full_output += (
                    "\n\n[System Hint / Sistem İpucu]: The command is not recognized or not in PATH. "
                    "Try project wrapper scripts or npm/python equivalents instead of repeating."
                )
            if os.name == "nt":
                linux_kw = [
                    w for w in ["ls", "wc", "tail", "head", "grep", "cat", "touch", "rm", "export", "which"]
                    if re.search(rf"\b{w}\b", command)
                ]
                if linux_kw:
                    full_output += (
                        f"\n\n💡 [Environment Hint - Windows PowerShell / Ortam İpucu]: Working shell is Windows PowerShell. "
                        f"The Unix command ({', '.join(linux_kw)}) might not behave identically under PowerShell. "
                        "Please use PowerShell equivalents:\n"
                        "  • `Get-ChildItem` instead of `ls`\n"
                        "  • `Get-Content` (or tool `read_file`) instead of `cat`\n"
                        "  • `Select-String` instead of `grep`\n"
                        "  • `(Get-Content file | Measure-Object -Line).Lines` instead of `wc -l`\n"
                        "  • `Get-Content file -Tail 20` instead of `tail -n 20`"
                    )
        max_chars = config.tools.max_output_chars
        if len(full_output) > max_chars:
            head_chars = min(1500, max_chars // 3)
            tail_chars = max(1000, max_chars - head_chars - 160)
            head = full_output[:head_chars]
            tail = full_output[-tail_chars:] if tail_chars > 0 else ""
            cut_count = len(full_output) - (len(head) + len(tail))
            full_output = (
                f"{head}\n\n"
                f"... [Output truncated from {len(full_output)} to {max_chars} characters / Çıktı özetlendi; ortadaki {cut_count} karakterlik log kırpıldı. "
                f"Head and tail error traces preserved] ...\n\n"
                f"{tail}"
            )
        return full_output

    def get_background_output(process_id: str, tail_lines: int = 80) -> str:
        return manager.get_output(process_id=process_id, tail_lines=tail_lines)

    def stop_background_process(process_id: str) -> str:
        return manager.stop(process_id=process_id)

    registry.register(
        name="run_command",
        description=(
            "Executes a shell command in the workspace (e.g. tests, package management, git, builds). "
            "Returns stdout, stderr, and return code. For long-running servers (e.g. npm run dev, uvicorn), "
            "set background=True to launch without blocking."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "cwd": {"type": "string", "description": "Optional working directory relative to workspace."},
                "timeout": {"type": "integer", "description": "Optional execution timeout in seconds."},
                "background": {
                    "type": "boolean",
                    "description": "If True, runs the process in the background and immediately returns a process_id.",
                },
            },
            "required": ["command"],
        },
        handler=run_command,
    )

    registry.register(
        name="get_background_output",
        description="Inspects the execution status and recent stdout/stderr output of a background process.",
        parameters={
            "type": "object",
            "properties": {
                "process_id": {"type": "string", "description": "Background process ID (e.g. 'bg_a1b2c3')."},
                "tail_lines": {"type": "integer", "description": "Number of recent lines to retrieve (default: 80)."},
            },
            "required": ["process_id"],
        },
        handler=get_background_output,
        requires_approval=False,
    )

    registry.register(
        name="stop_background_process",
        description="Gracefully terminates a running background process by its process_id.",
        parameters={
            "type": "object",
            "properties": {
                "process_id": {"type": "string", "description": "Background process ID to stop."},
            },
            "required": ["process_id"],
        },
        handler=stop_background_process,
        requires_approval=False,
    )

    return manager
