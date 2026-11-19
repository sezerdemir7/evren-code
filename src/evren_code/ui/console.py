"""Decoupled Rich Terminal User Interface (`ConsoleUI` v0.2.0).

- Live spinner animations (`Status`) providing real-time feedback during thinking and tool execution.
- Boxed panels, syntax-highlighted badges, and intelligent unified diff previews.
- Interactive approval prompts with fast-path defaults (`[y/N/t]`).
"""

from __future__ import annotations

import getpass
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from evren_code.config import ModelSpec
from evren_code.core.events import (
    AgentEvent,
    ApprovalDelegate,
    ApprovalRequest,
    EventBus,
    EventType,
)


class ConsoleUI(ApprovalDelegate):
    """Terminal user interface and event subscriber built on Rich."""

    def __init__(
        self,
        event_bus: EventBus,
        on_trust_all: Optional[Callable[[], None]] = None,
        console: Optional[Console] = None,
    ) -> None:
        self.console = console or Console()
        self.event_bus = event_bus
        self._on_trust_all = on_trust_all
        self._in_thinking = False
        self._in_text = False
        self._has_printed_agent_header = False
        self._line_needs_newline = False
        self._status: Optional[Status] = None
        self._idle_timer: Optional[threading.Timer] = None
        self._trust_all_active: bool = False
        self._ui_lock = threading.RLock()
        self.event_bus.subscribe(self.on_event)

    def _cancel_idle_timer(self) -> None:
        """Cancel idle timer (idle)."""
        if self._idle_timer is not None:
            try:
                self._idle_timer.cancel()
            except Exception:
                pass
            self._idle_timer = None

    def _start_status(self, text: str, spinner: str = "dots") -> None:
        """Start status."""
        if self._in_thinking or self._in_text or self._line_needs_newline:
            self._flush_stream_line()
        if self._status is not None:
            self._status.update(text, spinner=spinner)
        else:
            self._status = self.console.status(text, spinner=spinner)
            self._status.start()

    def _stop_status(self) -> None:
        """Stop status."""
        if self._status is not None:
            try:
                self._status.stop()
            except Exception:
                pass
            self._status = None

    def _flush_stream_line(self) -> None:
        """Flush stream line (newline basar)."""
        self._cancel_idle_timer()
        if self._in_thinking or self._in_text or self._line_needs_newline:
            self.console.print()
            self._in_thinking = False
            self._in_text = False
            self._line_needs_newline = False

    def _close_stream_line(self) -> None:
        """Close stream line."""
        self._cancel_idle_timer()
        self._stop_status()
        self._flush_stream_line()

    def on_event(self, event: AgentEvent) -> None:
        """On event."""
        with self._ui_lock:
            self._handle_event_locked(event)

    def _handle_event_locked(self, event: AgentEvent) -> None:
        etype = event.type
        data = event.data
        self._cancel_idle_timer()

        if etype == EventType.STEP_STARTED:
            self._close_stream_line()
            self._has_printed_agent_header = False
            self._line_needs_newline = False
            step = data.get("step", 1)
            model_name = data.get("model_name", "EVREN")
            if step > 1:
                self.console.print(f"[dim]↻ Adım {step} ({model_name})...[/dim]")
            self._start_status(f"[bold cyan]{model_name}[/bold cyan] analiz ediyor...", spinner="dots")

        elif etype == EventType.THINKING_DELTA:
            self._stop_status()
            delta = data.get("delta", "")
            if not delta:
                return
            if not self._in_thinking:
                if self._in_text or self._line_needs_newline:
                    self.console.print()
                    self._in_text = False
                    self._line_needs_newline = False
                self.console.print("[dim magenta]💭 Düşünce:[/dim magenta] ", end="")
                self._in_thinking = True
            self._line_needs_newline = True
            self.console.print(Text(delta, style="dim italic"), end="")

        elif etype == EventType.TEXT_DELTA:
            self._stop_status()
            delta = data.get("delta", "")
            if not delta:
                return
            if self._in_thinking:
                self.console.print()
                self._in_thinking = False
            if not self._has_printed_agent_header:
                if not delta.strip():
                    return
                self.console.print("\n[bold cyan]● Evren-Code:[/bold cyan]\n", end="")
                self._has_printed_agent_header = True
            self._in_text = True
            self._line_needs_newline = True
            self.console.print(Text(delta), end="")

        elif etype == EventType.TOOL_CALL_PREPARING:
            self._flush_stream_line()
            tname = data.get("tool_name") or "araç"
            bytes_rcv = data.get("bytes_received", 0)
            if bytes_rcv > 0:
                size_str = f" [dim]({bytes_rcv / 1024:.1f} KB hazırlanıyor)[/dim]"
            else:
                size_str = " [dim](hazırlanıyor)[/dim]"
            self._start_status(f"[bold yellow]⚡ {tname}[/bold yellow]{size_str}...", spinner="dots")

        elif etype == EventType.TOOL_CALL_DETECTED:
            self._close_stream_line()
            tname = data.get("tool_name", "")
            args = data.get("arguments", {})
            summary = self._summarize_args(tname, args)
            self.console.print(f"[bold yellow]⚡ {tname}[/bold yellow] [dim]({summary})[/dim]")

        elif etype == EventType.TOOL_EXECUTION_STARTED:
            tname = data.get("tool_name", "")
            self._start_status(f"[bold yellow]⚡ {tname}[/bold yellow] çalıştırılıyor...", spinner="bouncingBar")

        elif etype in (EventType.TOOL_EXECUTION_COMPLETED, EventType.TOOL_EXECUTION_TIMEOUT, EventType.TOOL_EXECUTION_CANCELLED):
            self._close_stream_line()
            approved = data.get("approved", True)
            tname = data.get("tool_name", "")
            result = str(data.get("result", ""))
            lines = result.splitlines()
            first_line = lines[0] if lines else ""
            extra_info = f" (+{len(lines) - 1} satır)" if len(lines) > 1 else ""
            if etype == EventType.TOOL_EXECUTION_TIMEOUT:
                dur = data.get("duration_seconds", 0.0)
                self.console.print(f"  [bold red]⏱ Zaman Aşımı:[/bold red] `{tname}` ({dur:.1f}s)")
            elif etype == EventType.TOOL_EXECUTION_CANCELLED:
                self.console.print(f"  [bold red]✖ İptal Edildi:[/bold red] `{tname}`")
            elif not approved:
                self.console.print(f"  [bold red]✖ İptal Edildi:[/bold red] {first_line}")
            elif result.startswith("Hata:"):
                self.console.print(f"  [bold red]⚠ Hata:[/bold red] {first_line}")
            else:
                self.console.print(f"  [bold green]✔[/bold green] [dim]{first_line[:110]}{extra_info}[/dim]")

        elif etype == EventType.SUBAGENT_STARTED:
            self._close_stream_line()
            role = data.get("role", "")
            mname = data.get("model_name", "")
            task = data.get("task", "")
            task_info = f" - {task}" if task else ""
            self.console.print(
                f"[bold blue]🧠 Uzman Alt-Ajan Devrede:[/bold blue] {mname} ([italic]{role}[/italic]){task_info}"
            )

        elif etype == EventType.SUBAGENT_COMPLETED:
            self._close_stream_line()
            role = data.get("role", "")
            self.console.print(
                f"[bold blue]✔ Alt-Ajan Görevini Tamamladı:[/bold blue] [italic]{role}[/italic]"
            )

        elif etype == EventType.SUBAGENT_HEARTBEAT:
            role = data.get("role", "")
            step = data.get("step", 0)
            target = data.get("target", "")
            summary = data.get("latest_summary", "")
            elapsed = data.get("elapsed_seconds", 0)
            status_text = f"[bold cyan]{role.capitalize()}[/bold cyan] (Adım {step}, {elapsed}s): {summary}"
            if target:
                status_text += f" [dim]({target})[/dim]"
            self._start_status(status_text, spinner="dots")

        elif etype == EventType.PLAN_UPDATED:
            self._close_stream_line()
            title = data.get("title", "")
            completed = data.get("completed", 0)
            total = data.get("total", 0)
            self.console.print(
                f"[bold cyan]📋 Görev Planı Güncellendi:[/bold cyan] {title} "
                f"([green]{completed}/{total}[/green] tamamlandı)"
            )

        elif etype == EventType.CONTEXT_COMPACTED:
            self._close_stream_line()
            count = data.get("compacted_messages", 0)
            tokens = data.get("estimated_tokens", 0)
            self.console.print(
                f"[dim cyan]📦 Bağlam sıkıştırıldı: {count} eski mesaj özetlendi (~{tokens} token aktif).[/dim cyan]"
            )

        elif etype == EventType.MODEL_SWITCHED:
            self._close_stream_line()
            mname = data.get("model_name", "")
            mid = data.get("model_id", "")
            self.console.print(
                f"[bold green]⇄ Aktif Model Değiştirildi:[/bold green] {mname} ([cyan]{mid}[/cyan])"
            )

        elif etype == EventType.INFO:
            self._close_stream_line()
            msg = data.get("message", "")
            if not msg:
                return
            self.console.print(f"[bold cyan][Bilgi][/bold cyan] {msg}")

        elif etype == EventType.WARNING:
            self._close_stream_line()
            self.console.print(f"[bold yellow]⚠ Uyarı:[/bold yellow] {data.get('message', '')}")

        elif etype == EventType.ERROR:
            self._close_stream_line()
            self.console.print(f"[bold red]✖ Hata:[/bold red] {data.get('message', '')}")

        elif etype == EventType.TURN_COMPLETED:
            self._close_stream_line()
            self._has_printed_agent_header = False
            self._line_needs_newline = False

    @staticmethod
    def _summarize_args(tool_name: str, args: dict) -> str:
        if tool_name in ("read_file", "write_file", "edit_file", "multi_edit_file", "list_directory"):
            return f"path='{args.get('path', '.')}'"
        if tool_name == "get_project_tree":
            return f"path='{args.get('path', '.')}', max_depth={args.get('max_depth', 3)}"
        if tool_name == "read_multiple_files":
            paths = args.get("paths") or []
            return f"files={len(paths)} adet"
        if tool_name == "write_multiple_files":
            files = args.get("files") or []
            return f"{len(files)} dosya"
        if tool_name == "git_info":
            return f"action='{args.get('action', 'status')}'"
        if tool_name == "fetch_url":
            return f"url='{args.get('url', '')}'"
        if tool_name == "run_command":
            return f"command='{args.get('command', '')}'"
        if tool_name == "search_code":
            return f"query='{args.get('query', '')}'"
        if tool_name == "find_files":
            return f"pattern='{args.get('pattern', '*')}'"
        if tool_name in ("delegate_subagent", "consult_specialist"):
            return f"role='{args.get('role', '')}'"
        return ", ".join(f"{k}={repr(v)[:30]}" for k, v in args.items())

    def request_approval(self, request: ApprovalRequest) -> bool:
        """Request approval."""
        if getattr(self, "_trust_all_active", False):
            return True

        self._close_stream_line()

        if request.diff_preview:
            syntax = Syntax(
                request.diff_preview,
                "diff",
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )
            self.console.print(
                Panel(
                    syntax,
                    title=f"[bold yellow]Değişiklik Önizlemesi ({request.tool_name})[/bold yellow]",
                    border_style="yellow",
                )
            )
        elif request.tool_name == "run_command":
            cmd = request.arguments.get("command", "")
            cwd = request.arguments.get("cwd", ".")
            self.console.print(
                Panel(
                    f"[bold white]{cmd}[/bold white]\n[dim]Dizin: {cwd}[/dim]",
                    title="[bold yellow]Komut Çalıştırma Onayı (run_command)[/bold yellow]",
                    border_style="yellow",
                )
            )

        while True:
            try:
                ans = self.console.input(
                    "[bold]Bu işleme izin veriyor musunuz?[/bold] "
                    "([bold green]Enter / e = Evet[/bold green] · [red]h = Hayır[/red] · [cyan]t = Oturumda güven[/cyan]) [E/h/t]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return False

            if ans in ("", "e", "evet", "y", "yes"):
                return True
            if ans in ("t", "tum", "tümü", "all"):
                self._trust_all_active = True
                if self._on_trust_all:
                    self._on_trust_all()
                self.console.print(
                    "[cyan]🔓 Güven modu aktif: Bu oturum boyunca onay sorulmayacak.[/cyan]"
                )
                return True
            if ans in ("h", "hayir", "hayır", "n", "no"):
                return False

    def ask_user_question(self, question: str, options: Optional[list[str]] = None) -> str:
        """Ask user question."""
        self._close_stream_line()
        body_lines = [f"[bold white]{question}[/bold white]"]
        if options:
            body_lines.append("")
            for idx, opt in enumerate(options, start=1):
                body_lines.append(f"  [bold cyan]{idx}.[/bold cyan] {opt}")

        self.console.print(
            Panel(
                "\n".join(body_lines),
                title="[bold magenta]❓ Ajan Sorusu (ask_user)[/bold magenta]",
                border_style="magenta",
            )
        )
        try:
            ans = self.console.input("[bold magenta]Yanıtınız (Numara veya metin): [/bold magenta]").strip()
            if options and ans.isdigit():
                choice_idx = int(ans) - 1
                if 0 <= choice_idx < len(options):
                    return options[choice_idx]
            return ans or "(Kullanıcı boş yanıt verdi)"
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return "(Kullanıcı soruyu iptal etti)"

    def print_banner(self, active_model: ModelSpec, workspace: Path) -> None:
        """Print banner."""
        caps = []
        if active_model.tools:
            caps.append("[green]🔧 Hybrid Tools[/green]")
        else:
            caps.append("[yellow]🔧 XML Tools[/yellow]")
        if active_model.thinking:
            caps.append("[magenta]💭 Thinking[/magenta]")
        if active_model.supports_vision:
            caps.append("[cyan]👁 Vision[/cyan]")

        info = (
            f"[bold cyan]⚡ EVREN-CODE v0.2.0[/bold cyan] — [bold]Otonom Kıdemli Yazılım Mühendisi[/bold]\n\n"
            f"  [bold]Aktif Model :[/bold] [bold white]{active_model.name}[/bold white] ([green]{active_model.id}[/green])\n"
            f"  [bold]Yetenekler  :[/bold] {' · '.join(caps)}\n"
            f"  [bold]Çalışma Alanı:[/bold] [dim]{workspace}[/dim]\n\n"
            f"[dim]Komutlar: /help · /model · /undo · /status · /compact · /clear · /exit[/dim]"
        )
        self.console.print(Panel(info, border_style="cyan", padding=(1, 2)))

    def print_models_table(
        self,
        catalog: Dict[str, ModelSpec],
        roles: Dict[str, str],
        active_id: str,
    ) -> None:
        """Print models table."""
        table = Table(title="EVREN AI Kullanılabilir Modeller ve Yetenekler", border_style="cyan")
        table.add_column("Durum", justify="center")
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Model Adı", style="bold")
        table.add_column("Atanan Roller", style="magenta")
        table.add_column("Bağlam (Context)", justify="right")
        table.add_column("Yetenekler")

        role_map: Dict[str, list] = {}
        for rname, mid in roles.items():
            role_map.setdefault(mid, []).append(rname)

        for mid, spec in catalog.items():
            marker = "✔ AKTİF" if mid == active_id else ""
            assigned = ", ".join(role_map.get(mid, [])) or "-"
            caps = []
            if spec.tools:
                caps.append("Tools")
            if spec.thinking:
                caps.append("Thinking")
            if spec.supports_vision:
                caps.append("Vision")
            table.add_row(
                f"[green]{marker}[/green]",
                mid,
                spec.name,
                assigned,
                f"{spec.context_length // 1000}K",
                ", ".join(caps) or "Chat",
            )
        self.console.print(table)

    def prompt_api_key(self) -> str:
        """Prompt api key."""
        self.console.print(
            Panel(
                "[bold]EVREN LLM API Kurulum Sihirbazı[/bold]\n"
                "Anahtarınızı EVREN platformunda [cyan]LLM Çıkarım -> API Anahtarı Oluştur[/cyan] "
                "sayfasından alabilirsiniz (`evren_llm_...` ile başlar).",
                border_style="cyan",
            )
        )
        return getpass.getpass("EVREN API Anahtarı (evren_llm_...): ")

    def prompt_terms_acceptance(self, terms_text: str, terms_file: Path) -> bool:
        """Prompt terms acceptance."""
        preview = terms_text[:1500] + ("\n...\n[Tam metin dosyaya kaydedildi]" if len(terms_text) > 1500 else "")
        self.console.print(
            Panel(
                preview,
                title=f"[bold yellow]EVREN Kullanım Şartları ({terms_file})[/bold yellow]",
                border_style="yellow",
            )
        )
        ans = self.console.input(
            "[bold yellow]Kullanım şartlarını okudunuz ve kabul ediyor musunuz? [e/H]: [/bold yellow]"
        ).strip().lower()
        return ans in ("e", "evet", "y", "yes")

    def prompt_resume_session(self, session_data: Dict[str, Any]) -> bool:
        """Prompt resume session."""
        rel_time = session_data.get("relative_time", "yakın zamanda")
        msg_count = len(session_data.get("messages", []))
        model = session_data.get("model", "EVREN Auto")
        last_prompt = session_data.get("last_prompt", "").strip()
        plan_data = session_data.get("plan")

        info_lines = [
            f"[bold]Son İşlem:[/bold] [green]{rel_time}[/green] | [bold]Model:[/bold] {model} | [bold]Geçmiş:[/bold] {msg_count} mesaj"
        ]

        if plan_data and plan_data.get("steps"):
            title = plan_data.get("title", "Görev Planı")
            steps = plan_data.get("steps", [])
            completed = sum(1 for s in steps if s.get("status") in ("completed", "PlanStepStatus.COMPLETED"))
            info_lines.append(f"[bold]Aktif Plan:[/bold] [cyan]{title}[/cyan] ({completed}/{len(steps)} tamamlandı)")

        if last_prompt:
            info_lines.append(f"[bold]Son İstek:[/bold] [italic dim]{last_prompt[:120]}...[/italic dim]")

        content = "\n".join(info_lines)
        self.console.print(
            Panel(
                content,
                title="[bold cyan]🔄 Önceki Yarım Kalan Oturum Tespit Edildi[/bold cyan]",
                border_style="cyan",
                subtitle="[dim]Kaldığınız yerden hemen devam edebilir veya sıfırdan başlayabilirsiniz[/dim]",
            )
        )

        try:
            ans = self.console.input(
                "[bold green]Kaldığınız yerden devam etmek ister misiniz? [E/h]: [/bold green]"
            ).strip().lower()
            if not ans or ans in ("e", "evet", "y", "yes"):
                return True
            return False
        except (KeyboardInterrupt, EOFError):
            self.console.print()
            return False
