"""Evren-Code Command Line Interface (CLI v0.2.0) and Interactive REPL Loop."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from evren_code.config import AppConfig, load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.events import EventBus
from evren_code.core.evren_auth import EvrenAuthError, EvrenAuthManager
from evren_code.core.llm_client import EvrenAPIError, EvrenLLMClient
from evren_code.core.path_setup import ensure_cli_on_path
from evren_code.core.session import SessionManager
from evren_code.core.updater import run_self_update
from evren_code.ui.console import ConsoleUI


SLASH_COMMANDS = [
    "/help",
    "/model",
    "/auth",
    "/undo",
    "/status",
    "/compact",
    "/clear",
    "/exit",
]


class SafeFileHistory(FileHistory):
    """Prevents inline API keys or `/auth <secret>` inputs from being stored in disk history."""

    def __init__(self, filename: str, key_prefix: str = "evren_llm_") -> None:
        super().__init__(filename)
        self._key_prefix = key_prefix

    def store_string(self, string: str) -> None:
        stripped = string.strip()
        lower = stripped.lower()
        if (
            re.match(r"^/(?:auth|key|apikey)\s+(?!status\b)\S+", lower)
            or (self._key_prefix and self._key_prefix in stripped)
        ):
            return
        super().store_string(string)


class SlashCommandCompleter(Completer):
    """Slash command completer implementation."""

    def __init__(
        self,
        commands: List[str],
        models: List[str],
        session_mgr: Optional[SessionManager] = None,
    ) -> None:
        self.commands = commands
        self.models = models
        self.session_mgr = session_mgr

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ):
        text_before_cursor = document.text_before_cursor
        stripped = text_before_cursor.lstrip()

        if not stripped.startswith("/"):
            return

        parts = stripped.split()

        if len(parts) <= 1 and not text_before_cursor.endswith(" "):
            prefix = parts[0].lower() if parts else "/"
            for cmd in self.commands:
                if cmd.lower().startswith(prefix):
                    yield Completion(cmd, start_position=-len(prefix))
            return

        if len(parts) >= 1:
            cmd = parts[0].lower()
            if cmd == "/model":
                current_word = document.get_word_before_cursor()
                for m in self.models:
                    if m.lower().startswith(current_word.lower()):
                        yield Completion(m, start_position=-len(current_word))
            elif cmd in ("/auth", "/key", "/apikey"):
                current_word = document.get_word_before_cursor()
                for sub in ("status",):
                    if sub.startswith(current_word.lower()):
                        yield Completion(sub, start_position=-len(current_word))
            elif cmd == "/undo":
                current_word = document.get_word_before_cursor()
                for sub in ("turn", "all"):
                    if sub.startswith(current_word.lower()):
                        yield Completion(sub, start_position=-len(current_word))


def run_onboarding_flow(
    auth_mgr: EvrenAuthManager,
    ui: ConsoleUI,
    force_new_key: bool = False,
    provided_key: Optional[str] = None,
) -> bool:
    """Run onboarding flow."""
    if provided_key is not None:
        try:
            key = auth_mgr.validate_key_format(provided_key)
        except EvrenAuthError as exc:
            ui.console.print(f"[bold red]Kurulum Hatası:[/bold red] {exc}")
            return False
    else:
        key = None if force_new_key else auth_mgr.load_stored_key()

    if not key:
        try:
            raw_key = ui.prompt_api_key()
            key = auth_mgr.validate_key_format(raw_key)
        except (EvrenAuthError, KeyboardInterrupt, EOFError) as exc:
            ui.console.print(f"[bold red]Kurulum Hatası:[/bold red] {exc}")
            return False

    ui.console.print("[dim]EVREN API anahtarı doğrulanıyor ve modeller keşfediliyor...[/dim]")
    try:
        discovered = auth_mgr.verify_and_discover_models(
            api_key=key,
            terms_acceptance_callback=ui.prompt_terms_acceptance,
        )
        saved_path = auth_mgr.save_key(key)
        ui.console.print(
            f"[bold green]✔ Bağlantı Başarılı![/bold green] Anahtar kaydedildi: [dim]{saved_path}[/dim] "
            f"({len(discovered)} model aktif)."
        )
        return True
    except EvrenAuthError as exc:
        ui.console.print(f"[bold red]✖ EVREN Doğrulama Hatası:[/bold red] {exc}")
        return False
    except Exception as exc:
        if force_new_key or provided_key is not None:
            ui.console.print(
                f"[bold red]✖ Bağlantı Hatası:[/bold red] Yeni API anahtarı doğrulanamadı ({exc}). Mevcut anahtar korundu."
            )
            return False
        auth_mgr.load_models_cache()
        ui.console.print(
            f"[yellow]⚠ Canlı model listesi alınamadı ({exc}); varsayılan katalog kullanılacak.[/yellow]"
        )
        return True


def _has_dotenv_api_key(workspace: Path) -> bool:
    """Check whether `.env` in workspace or current working directory defines `EVREN_API_KEY`."""
    for base in {workspace, Path.cwd()}:
        env_file = base / ".env"
        if env_file.is_file():
            try:
                content = env_file.read_text(encoding="utf-8", errors="ignore")
                if re.search(r"^\s*EVREN_API_KEY\s*=", content, flags=re.MULTILINE):
                    return True
            except OSError:
                pass
    return False


def handle_slash_command(
    raw_cmd: str,
    orchestrator: AgentOrchestrator,
    auth_mgr: EvrenAuthManager,
    session_mgr: SessionManager,
    ui: ConsoleUI,
) -> bool:
    """Handle slash command."""
    parts = raw_cmd.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit", "/q"):
        session_mgr.auto_save_workspace_session(
            orchestrator.context,
            orchestrator.llm_client.active_model.id,
            orchestrator.plan_manager,
        )
        ui.console.print("[dim]Oturum kaydedildi. Evren-Code kapatılıyor. İyi çalışmalar![/dim]")
        return False

    if cmd == "/help":
        help_text = (
            "[bold cyan]Evren-Code Temel Komutları:[/bold cyan]\n"
            "  [bold green]/model [id|rol][/bold green] : Aktif modeli görüntüler veya değiştirir (örn: `/model coder` veya `/model deepseek-v4.1-flash`)\n"
            "  [bold green]/auth [status][/bold green]  : Oturumdan çıkmadan EVREN API anahtarını güvenli şekilde günceller veya durumunu gösterir (`/key` takma adı da desteklenir)\n"
            "  [bold green]/undo [turn][/bold green]    : Son dosya değişikliğini veya tüm turu (`/undo turn`, Shadow Git) anında geri alır\n"
            "  [bold green]/status[/bold green]         : Aktif model, token kullanımı ve çalışma alanı durumunu gösterir\n"
            "  [bold green]/compact[/bold green]        : Konuşma geçmişini özetleyerek bağlamda yer açar\n"
            "  [bold green]/clear[/bold green]          : Konuşma hafızasını sıfırlar ve yeni bir göreve başlar\n"
            "  [bold green]/help[/bold green]           : Bu yardım kılavuzunu gösterir\n"
            "  [bold green]/exit[/bold green]           : Oturumu güvenle kaydedip çıkar\n\n"
            "[dim]💡 Diğer tüm işler için komut aramanıza gerek yoktur. "
            "Doğrudan Türkçe ne yapmak istediğinizi söyleyin (örn: 'mimariyi planla', 'testleri doğrula', 'hataları düzelt', 'dosyaları göster'). "
            "Ajan gerekli araçları arka planda otomatik olarak çalıştırır.[/dim]"
        )
        ui.console.print(help_text)
        return True

    if cmd in ("/auth", "/key", "/apikey"):
        if arg.lower() == "status":
            current_key = orchestrator.config.api.api_key or auth_mgr.config.api.api_key
            masked = auth_mgr.mask_key(current_key)
            source = auth_mgr.get_key_source()
            ui.console.print(
                f"[bold]EVREN API Anahtarı:[/bold] [cyan]{masked}[/cyan] | "
                f"[bold]Kaynak:[/bold] [dim]{source}[/dim] | "
                f"[bold]Sunucu:[/bold] [dim]{orchestrator.config.api.base_url}[/dim]\n"
                "[dim]Anahtarı yenilemek için argümansız `/auth` komutunu çalıştırın.[/dim]"
            )
            return True

        had_env_key = bool(os.environ.get("EVREN_API_KEY"))
        if arg:
            ui.console.print(
                "[yellow]⚠ Güvenlik İpucu:[/yellow] API anahtarınızı ekranda görünmeden girmek için "
                "bir sonraki sefer argümansız `/auth` komutunu kullanmanız önerilir."
            )

        ok = run_onboarding_flow(
            auth_mgr,
            ui,
            force_new_key=True,
            provided_key=arg if arg else None,
        )
        if ok and auth_mgr.config.api.api_key:
            new_key = auth_mgr.config.api.api_key
            orchestrator.config.api.api_key = new_key
            orchestrator.llm_client.config.api.api_key = new_key
            orchestrator.config.models.catalog.update(auth_mgr.config.models.catalog)
            orchestrator.llm_client._model_cooldowns.clear()
            if had_env_key:
                os.environ["EVREN_API_KEY"] = new_key
            if _has_dotenv_api_key(orchestrator.config.workspace_root):
                ui.console.print(
                    "[yellow]⚠ Not:[/yellow] Çalışma alanınızdaki `.env` dosyasında `EVREN_API_KEY` tanımlı görünüyor. "
                    "Sonraki oturumlarda eski anahtarın okunmaması için `.env` dosyasını da güncellemeyi unutmayın."
                )
        return True

    if cmd == "/model":
        if not arg:
            ui.console.print(
                f"Mevcut model: [bold green]{orchestrator.llm_client.active_model.id}[/bold green]. "
                "Değiştirmek için: `/model <model_id veya rol>`"
            )
        else:
            orchestrator.llm_client.set_model(arg)
        return True

    if cmd == "/undo":
        shadow_git = getattr(orchestrator, "shadow_git", None)
        if arg.lower() in ("turn", "all", "tur", "tümü"):
            msg = orchestrator.undo_manager.revert_turn(shadow_git=shadow_git)
        elif orchestrator.undo_manager.can_undo:
            msg = orchestrator.undo_manager.undo_last()
        elif shadow_git and getattr(shadow_git, "last_checkpoint_commit", None):
            msg = orchestrator.undo_manager.revert_turn(shadow_git=shadow_git)
        else:
            msg = orchestrator.undo_manager.undo_last()
        ui.console.print(f"[bold cyan]{msg}[/bold cyan]")
        return True

    if cmd == "/compact":
        count = orchestrator.compact_context()
        if count == 0:
            ui.console.print("[dim]Sıkıştırılacak yeterli uzunlukta geçmiş bulunmuyor.[/dim]")
        return True

    if cmd == "/status":
        spec = orchestrator.llm_client.active_model
        tokens = orchestrator.context.estimate_tokens()
        msgs = orchestrator.context.history_length
        undos = orchestrator.undo_manager.history_count
        shadow_git = getattr(orchestrator, "shadow_git", None)
        has_shadow_cp = bool(shadow_git and getattr(shadow_git, "last_checkpoint_commit", None))
        shadow_tag = " + ShadowGit Checkpoint" if has_shadow_cp else ""
        ui.console.print(
            f"[bold]Model:[/bold] {spec.name} ({spec.id}) | "
            f"[bold]Mesaj:[/bold] {msgs} | "
            f"[bold]Tahmini Token:[/bold] ~{tokens} / {spec.context_length} | "
            f"[bold]Geri Alınabilir İşlem (/undo):[/bold] {undos}{shadow_tag}"
        )
        return True

    if cmd == "/clear":
        orchestrator.context.clear()
        orchestrator.plan_manager.clear()
        orchestrator.logger.clear_logs()
        session_mgr.clear_workspace_session()
        ui.console.print("[green]✔ Konuşma geçmişi, aktif plan, log kayıtları ve oturum temizlendi.[/green]")
        return True

    if cmd == "/run":
        if not arg:
            ui.console.print("[yellow]Kullanım: `/run <komut>` (Örn: `/run pytest`)[/yellow]")
            return True
        out = orchestrator.tool_executor.execute(
            tool_name="run_command",
            arguments={"command": arg, "cwd": "."},
            caller_role="main",
            caller_agent="cli_user",
            is_trust_mode=orchestrator.trust_mode,
        )
        ui.console.print(out)
        return True

    rendered_skill = orchestrator.skill_manager.render(cmd, arg)
    if rendered_skill:
        try:
            orchestrator.run_turn(rendered_skill)
        except EvrenAPIError as exc:
            ui.console.print(f"[bold red]EVREN API Hatası:[/bold red] {exc}")
        return True

    ui.console.print(f"[yellow]Bilinmeyen komut: {cmd}. Yardım için `/help` yazın.[/yellow]")
    return True


def create_prompt_key_bindings() -> KeyBindings:
    """Configures bracketed multiline paste and Alt+Enter newline bindings for the interactive REPL."""
    kb = KeyBindings()

    @kb.add(Keys.BracketedPaste)
    def _on_bracketed_paste(event) -> None:
        raw_pasted = event.data.replace("\r\n", "\n").replace("\r", "\n")
        event.current_buffer.insert_text(raw_pasted)

    @kb.add("escape", "enter")
    def _on_alt_enter(event) -> None:
        event.current_buffer.insert_text("\n")

    return kb


def expand_prompt_mentions(user_input: str, workspace: Path) -> str:
    """Expands `@filepath` and `@clipboard` / `@pano` (image or text) mentions inline."""
    from evren_code.tools.vision import get_clipboard_text, save_clipboard_image

    def _mention_replacer(match: re.Match) -> str:
        token = match.group(1)
        clean_tok = token.strip().lower()
        if clean_tok in ("clipboard", "pano"):
            clip_img = save_clipboard_image(workspace)
            if clip_img is not None:
                try:
                    rel_img = clip_img.relative_to(workspace).as_posix()
                except ValueError:
                    rel_img = clip_img.as_posix()
                return (
                    f"\n[Panodan Görsel Kaydedildi: `{rel_img}` — "
                    f"Lütfen `analyze_image(image_path='{rel_img}')` aracıyla bu görseli incele]\n"
                )
            clip_txt = get_clipboard_text()
            if clip_txt:
                return f"\n[@clipboard]:\n```\n{clip_txt}\n```\n"
            return match.group(0)

        rel = token.replace("\\", "/")
        tgt = workspace / rel
        if tgt.is_file():
            try:
                txt = tgt.read_text(encoding="utf-8", errors="replace")
                return f"\n[@{rel}]:\n```\n{txt}\n```\n"
            except OSError:
                pass
        return match.group(0)

    return re.sub(r"@([a-zA-Z0-9_\-\./\\]+)", _mention_replacer, user_input)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evren",
        description="Autonomous Software Engineering Agent for EVREN AI Platform",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Initial model ID or role (e.g. deepseek-v4.1-flash, glm-5.3, qwen3.8-flash-next, architect).",
    )
    parser.add_argument(
        "--role",
        help="Specialist role to handle the task (architect, coder, reviewer, tester, orchestrator).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Autonomous headless subagent mode without TTY prompts (for Claude/Antigravity integration).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output structured machine-readable JSON to stdout.",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Launch Model Context Protocol (MCP) stdio server.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Send direct raw chat completion request to EVREN LLM API without agent loop or file edits.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume previous interrupted session and active plan in this workspace.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run EVREN API key setup and terms acceptance wizard.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update Evren-Code source code and dependencies.",
    )
    parser.add_argument(
        "--trust",
        action="store_true",
        help="Automatically approve file edits and command executions.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable trajectory logging (.evren/logs/) for production end-user mode.",
    )
    parser.add_argument(
        "-C",
        "--cwd",
        help="Target workspace root directory.",
    )
    parser.add_argument(
        "--agent-help",
        "--instructions",
        action="store_true",
        dest="agent_help",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="One-shot prompt task to execute (starts interactive REPL mode if omitted).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0].lower() in ("agent-help", "agent-info", "instructions", "spec"):
        from evren_code.core.agent_spec import get_agent_instructions
        as_json = "--json" in raw_args
        sys.stdout.write(get_agent_instructions(as_json=as_json) + "\n")
        return 0

    if raw_args and raw_args[0].lower() == "ask":
        raw_args = ["--llm"] + raw_args[1:]

    parser = build_arg_parser()
    args = parser.parse_args(raw_args)

    if args.agent_help:
        from evren_code.core.agent_spec import get_agent_instructions
        sys.stdout.write(get_agent_instructions(as_json=bool(args.json_output)) + "\n")
        return 0

    if args.mcp or (args.prompt and len(args.prompt) == 1 and args.prompt[0].lower() == "mcp"):
        from evren_code.mcp_server import run_mcp_server
        return run_mcp_server()

    is_headless = bool(args.headless or args.json_output)
    if is_headless:
        args.trust = True

    if is_headless and not args.prompt:
        if args.json_output:
            sys.stdout.write(json.dumps({
                "status": "error",
                "error": "Headless/JSON modunda çalıştırılacak bir görev (prompt) belirtilmelidir.",
            }, ensure_ascii=False) + "\n")
        else:
            sys.stderr.write("Hata: --headless modunda çalıştırılacak bir görev (prompt) belirtilmelidir.\n")
        return 1

    workspace = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    config: AppConfig = load_config(workspace_root=workspace)
    if args.no_log:
        config.telemetry.enabled = False

    event_bus = EventBus()
    orchestrator_ref: List[AgentOrchestrator] = []

    def _enable_trust() -> None:
        ui._trust_all_active = True
        if orchestrator_ref:
            orchestrator_ref[0].trust_mode = True

    if args.json_output:
        from rich.console import Console
        ui_console = Console(stderr=True, quiet=True)
        ui = ConsoleUI(event_bus=event_bus, on_trust_all=_enable_trust, console=ui_console)
    else:
        ui = ConsoleUI(event_bus=event_bus, on_trust_all=_enable_trust)

    auth_mgr = EvrenAuthManager(config=config)
    session_mgr = SessionManager(config=config)
    auth_mgr.load_models_cache()

    if args.update:
        ui.console.print("[dim]Evren-Code güncelleniyor...[/dim]")
        res = run_self_update(install_deps=True)
        style = "green" if res.success else "red"
        ui.console.print(f"[bold {style}]{res.summary}[/bold {style}]\n[dim]{res.details}[/dim]")
        return 0 if res.success else 1

    if args.setup:
        path_res = ensure_cli_on_path()
        if path_res.added_to_path:
            ui.console.print(f"[bold green]✔ {path_res.message}[/bold green]")
        elif path_res.already_in_path:
            ui.console.print(f"[dim]✔ {path_res.message}[/dim]")
        ok = run_onboarding_flow(auth_mgr, ui, force_new_key=True)
        return 0 if ok else 1

    is_direct_llm = bool(args.llm or (args.prompt and len(args.prompt) > 0 and args.prompt[0].lower() == "ask"))
    if is_direct_llm and args.prompt and args.prompt[0].lower() == "ask":
        args.prompt = args.prompt[1:]

    if is_direct_llm:
        if not args.prompt:
            if args.json_output:
                sys.stdout.write(json.dumps({
                    "status": "error",
                    "error": "LLM sorgusu için bir metin/istem belirtilmelidir.",
                }, ensure_ascii=False) + "\n")
            else:
                sys.stderr.write("Hata: LLM sorgusu için bir metin/istem belirtilmelidir (Örn: evren ask 'sorunuz').\n")
            return 1

        if not auth_mgr.load_stored_key():
            ok = run_onboarding_flow(auth_mgr, ui, force_new_key=False)
            if not ok:
                return 1

        llm_client = EvrenLLMClient(config=config, event_bus=event_bus)
        if args.role:
            llm_client.set_model(args.role)
        if args.model:
            llm_client.set_model(args.model)

        query_text = " ".join(args.prompt)
        try:
            resp = llm_client.chat(
                messages=[{"role": "user", "content": query_text}],
                tools_schema=None,
                stream=not args.json_output,
                silent=bool(args.json_output),
            )
            if args.json_output:
                payload = {
                    "status": "success",
                    "model": llm_client.active_model.id,
                    "content": resp.content,
                    "thinking": resp.thinking,
                    "usage": resp.usage,
                    "finish_reason": resp.finish_reason,
                }
                sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                sys.stdout.flush()
            else:
                if resp.content and not ui._in_text:
                    ui.console.print(resp.content)
            return 0
        except EvrenAPIError as exc:
            if args.json_output:
                sys.stdout.write(json.dumps({
                    "status": "error",
                    "error_type": "EvrenAPIError",
                    "error": str(exc),
                }, ensure_ascii=False, indent=2) + "\n")
                sys.stdout.flush()
            else:
                ui.console.print(f"[bold red]EVREN API Hatası:[/bold red] {exc}")
            return 1
        except Exception as exc:
            if args.json_output:
                sys.stdout.write(json.dumps({
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }, ensure_ascii=False, indent=2) + "\n")
                sys.stdout.flush()
            else:
                ui.console.print(f"[bold red]Beklenmeyen Hata:[/bold red] {exc}")
            return 1

    if not auth_mgr.load_stored_key():
        ok = run_onboarding_flow(auth_mgr, ui, force_new_key=False)
        if not ok:
            return 1

    llm_client = EvrenLLMClient(config=config, event_bus=event_bus)
    if args.role:
        llm_client.set_model(args.role)
    if args.model:
        llm_client.set_model(args.model)

    orchestrator = AgentOrchestrator(
        config=config,
        llm_client=llm_client,
        event_bus=event_bus,
        approval_delegate=ui,
    )
    orchestrator.trust_mode = bool(args.trust)
    if args.trust:
        ui._trust_all_active = True
    orchestrator_ref.append(orchestrator)

    last_session = session_mgr.get_last_workspace_session()
    should_resume = False

    if args.resume and last_session:
        should_resume = True
    elif last_session and not args.prompt and not args.setup and not args.update and not is_headless:
        should_resume = ui.prompt_resume_session(last_session)

    if should_resume and last_session:
        session_mgr.restore_session_to_orchestrator(orchestrator, last_session)
        rel_t = last_session.get("relative_time", "önceki")
        msg_cnt = len(last_session.get("messages", []))
        ui.console.print(
            f"[bold green]✔ Kaldığınız Yerden Devam Ediliyor![/bold green] "
            f"({msg_cnt} mesaj yüklendi, son işlem: {rel_t})."
        )
        if orchestrator.plan_manager.current_plan:
            ui.console.print(orchestrator.plan_manager.render_table())

    if args.prompt:
        task_text = " ".join(args.prompt)
        try:
            turn_output = orchestrator.run_turn(task_text, stream=not args.json_output)
            if args.json_output:
                modified = []
                if hasattr(orchestrator, "undo_manager") and orchestrator.undo_manager._stack:
                    for item in orchestrator.undo_manager._stack:
                        p = item.get("path") if isinstance(item, dict) else getattr(item, "file_path", None)
                        if p:
                            try:
                                rel = Path(p).relative_to(workspace)
                                modified.append(str(rel).replace("\\", "/"))
                            except Exception:
                                modified.append(str(p).replace("\\", "/"))

                result_payload = {
                    "status": "success",
                    "task": task_text,
                    "model_used": llm_client.active_model.id,
                    "role": args.role or "default",
                    "workspace": str(workspace),
                    "summary": turn_output,
                    "modified_files": sorted(list(set(modified))),
                }
                sys.stdout.write(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n")
                sys.stdout.flush()
            return 0
        except EvrenAPIError as exc:
            if args.json_output:
                sys.stdout.write(json.dumps({
                    "status": "error",
                    "error_type": "EvrenAPIError",
                    "error": str(exc),
                }, ensure_ascii=False, indent=2) + "\n")
                sys.stdout.flush()
            else:
                ui.console.print(f"[bold red]EVREN API Hatası:[/bold red] {exc}")
            return 1
        except Exception as exc:
            if args.json_output:
                sys.stdout.write(json.dumps({
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }, ensure_ascii=False, indent=2) + "\n")
                sys.stdout.flush()
            else:
                ui.console.print(f"[bold red]Beklenmeyen Hata:[/bold red] {exc}")
            return 1

    ui.print_banner(llm_client.active_model, workspace)
    hist_dir = Path.home() / ".evren-code"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_file = str(hist_dir / "history.txt")

    completer = SlashCommandCompleter(
        commands=list(set(SLASH_COMMANDS + orchestrator.skill_manager.get_command_list())),
        models=list(config.models.catalog.keys()) + list(config.models.roles.keys()),
        session_mgr=session_mgr,
    )
    try:
        session: Optional[PromptSession[str]] = PromptSession(
            history=SafeFileHistory(hist_file, key_prefix=config.api.key_prefix),
            completer=completer,
            key_bindings=create_prompt_key_bindings(),
        )
    except Exception:
        session = None

    while True:
        model_spec = llm_client.active_model
        model_tag = model_spec.name
        model_id = model_spec.id
        workspace_name = workspace.name

        def _get_toolbar() -> HTML:
            undo_cnt = orchestrator.undo_manager.history_count
            trust_icon = "🔓" if orchestrator.trust_mode else "🔒"
            token_est = orchestrator.context.estimate_tokens()
            return HTML(
                f" <ansidarkgray><b>[EVREN]</b></ansidarkgray> "
                f"<ansigray>Model:</ansigray> <b><ansicyan>{model_id}</ansicyan></b> "
                f"<ansidarkgray>│</ansidarkgray> <ansigray>Tokens:</ansigray> <b>~{token_est}</b> "
                f"<ansidarkgray>│</ansidarkgray> <ansigray>Undo:</ansigray> <b>{undo_cnt}</b> "
                f"<ansidarkgray>│</ansidarkgray> <ansigray>Mod:</ansigray> <b>{trust_icon}</b> "
                f"<ansidarkgray>│</ansidarkgray> <b><ansigreen>/help</ansigreen></b> "
                f"<ansidarkgray>│</ansidarkgray> <b><ansigreen>/exit</ansigreen></b> "
            )

        prompt_html = HTML(
            f"\n<b><ansicyan>{model_tag}</ansicyan></b> "
            f"<ansigray>({workspace_name})</ansigray> "
            f"<b><ansicyan>❯</ansicyan></b> "
        )

        try:
            if session is not None:
                user_input = session.prompt(prompt_html, bottom_toolbar=_get_toolbar).strip()
            else:
                user_input = input(f"\n[{model_tag}] ({workspace_name}) ❯ ").strip()
        except KeyboardInterrupt:
            ui.console.print("\n[dim](Çıkmak için /exit yazın)[/dim]")
            continue
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.startswith("!"):
            raw_shell_cmd = user_input[1:].strip()
            if raw_shell_cmd:
                ui.console.print(f"[dim]$ {raw_shell_cmd}[/dim]")
                out = orchestrator.tool_executor.execute(
                    tool_name="run_command",
                    arguments={"command": raw_shell_cmd, "cwd": "."},
                    caller_role="main",
                    caller_agent="cli_quick_shell",
                    is_trust_mode=orchestrator.trust_mode,
                )
                ui.console.print(out)
            continue

        if user_input.startswith("/"):
            keep_going = handle_slash_command(
                user_input, orchestrator, auth_mgr, session_mgr, ui
            )
            if not keep_going:
                break
            continue

        line_cnt = len(user_input.splitlines())
        if line_cnt > 1:
            ui.console.print(f"[dim cyan]📋 {line_cnt} satırlık çoklu girdi alındı.[/dim cyan]")

        effective_input = expand_prompt_mentions(user_input, workspace)

        try:
            turn_output = orchestrator.run_turn(effective_input)
            ui._close_stream_line()
            if turn_output and not ui._in_text:
                if "## 🏆 Çoklu Ajan Görev Raporu" in str(turn_output) and not getattr(ui, "_last_turn_had_output", False):
                    ui.console.print(f"\n{turn_output}\n")
        except EvrenAPIError as exc:
            orchestrator.last_turn_error = str(exc)
            ui._close_stream_line()
            orchestrator._auto_save_session()
            ui.console.print(f"[bold red]EVREN API Hatası:[/bold red] {exc}")
        except KeyboardInterrupt:
            ui._close_stream_line()
            orchestrator._auto_save_session()
            ui.console.print("\n[yellow]⚠ İşlem kullanıcı tarafından durduruldu.[/yellow]")
        except Exception as exc:
            orchestrator.last_turn_error = str(exc)
            ui._close_stream_line()
            orchestrator._auto_save_session()
            ui.console.print(f"[bold red]Beklenmeyen Çalışma Zamanı Hatası:[/bold red] {exc}")

    try:
        session_mgr.auto_save_workspace_session(
            orchestrator.context,
            orchestrator.llm_client.active_model.id,
            orchestrator.plan_manager,
        )
    finally:
        orchestrator.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
