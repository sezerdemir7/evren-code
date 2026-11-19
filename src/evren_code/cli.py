"""Evren-Code Command Line Interface (CLI v0.2.0) and Interactive REPL Loop."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, InMemoryHistory

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
    "/undo",
    "/status",
    "/compact",
    "/clear",
    "/exit",
]


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
            elif cmd == "/load" and self.session_mgr:
                current_word = document.get_word_before_cursor()
                sessions = [s["name"] for s in self.session_mgr.list_sessions()]
                for sname in sessions:
                    if sname.lower().startswith(current_word.lower()):
                        yield Completion(sname, start_position=-len(current_word))
            elif cmd == "/multi":
                subcmds = ["status", "plan", "agents", "tasks", "graph", "conflicts", "history", "report", "cancel", "resume"]
                current_word = document.get_word_before_cursor()
                for sc in subcmds:
                    if sc.lower().startswith(current_word.lower()):
                        yield Completion(sc, start_position=-len(current_word))


def run_onboarding_flow(
    auth_mgr: EvrenAuthManager,
    ui: ConsoleUI,
    force_new_key: bool = False,
) -> bool:
    """Run onboarding flow."""
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
        auth_mgr.load_models_cache()
        ui.console.print(
            f"[yellow]⚠ Canlı model listesi alınamadı ({exc}); varsayılan katalog kullanılacak.[/yellow]"
        )
        return True


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
            "  [bold green]/undo[/bold green]           : Ajanın yaptığı son dosya değişikliğini anında geri alır (Snapshot)\n"
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

    if cmd == "/models":
        ui.print_models_table(
            catalog=orchestrator.config.models.catalog,
            roles=orchestrator.config.models.roles,
            active_id=orchestrator.llm_client.active_model.id,
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

    if cmd == "/architect":
        if not arg:
            ui.console.print("[yellow]Lütfen mimari görev açıklaması girin: `/architect <görev>`[/yellow]")
        else:
            try:
                orchestrator.run_architect_workflow(arg)
            except EvrenAPIError as exc:
                ui.console.print(f"[bold red]API Hatası:[/bold red] {exc}")
        return True

    if cmd in ("/overview", "/project"):
        ui.console.print("[dim]Proje zekası (Repository Intelligence) taranıyor...[/dim]")
        tool_to_call = "inspect_project" if arg else "get_project_overview"
        args = {"query": arg} if arg else {}
        out = orchestrator.tool_executor.execute(
            tool_name=tool_to_call,
            arguments=args,
            caller_role="main",
            caller_agent="cli_user",
            is_trust_mode=orchestrator.trust_mode,
        )
        ui.console.print(out)
        return True

    if cmd == "/repomap":
        target_dir = arg or "."
        ui.console.print("[dim]Semantik kod haritası (AST + PageRank Repo-Map) taranıyor...[/dim]")
        map_out = orchestrator.tool_executor.execute(
            tool_name="get_repo_map",
            arguments={"directory": target_dir},
            caller_role="main",
            caller_agent="cli_user",
            is_trust_mode=orchestrator.trust_mode,
        )
        ui.console.print(map_out)
        return True

    if cmd == "/heal":
        ui.console.print("[dim]Proje doğrulama ve otonom test kontrolü başlatılıyor...[/dim]")
        task = None
        if arg:
            from evren_code.core.self_healing import VerificationTask
            task = VerificationTask(command=arg, cwd=orchestrator.config.workspace_root, description=arg)
        def _exec(c: str, w: Optional[str] = None) -> str:
            return orchestrator.tool_executor.execute(
                tool_name="run_command",
                arguments={"command": c, "cwd": w or "."},
                caller_role="tester",
                caller_agent="cli_heal",
                is_trust_mode=orchestrator.trust_mode,
            )

        ok, cmd_run, out = orchestrator.self_healing_manager.execute_verification(_exec, task)
        if ok:
            ui.console.print(f"[bold green]✔ Doğrulama Başarılı:[/bold green] `{cmd_run}` sorunsuz geçti.")
        else:
            ui.console.print(
                f"[bold red]✖ Doğrulama Başarısız:[/bold red] `{cmd_run}`\n"
                f"[dim]{(out or '')[:500]}[/dim]\n"
                "[bold cyan]Ajan otonom onarım turu başlatıyor...[/bold cyan]"
            )
            orchestrator.run_turn(
                f"[Self-Healing]: '{cmd_run}' çalıştırıldı ve şu hatayı verdi:\n"
                f"```\n{(out or '')[:2500]}\n```\n"
                "Lütfen ilgili dosyaları okuyup hatayı düzeltecek cerrahi değişikliği uygulayın."
            )
        return True

    if cmd == "/verify":
        sub_cmd = arg.strip().split(maxsplit=1)[0].lower() if arg.strip() else ""
        sub_arg = arg.strip().split(maxsplit=1)[1].strip() if len(arg.strip().split(maxsplit=1)) > 1 else ""

        if sub_cmd == "status":
            rep = orchestrator.verification_engine.last_report or orchestrator.verification_engine.history_store.get_latest_report()
            if not rep:
                ui.console.print("[yellow]Henüz yürütülmüş bir doğrulama bulunamadı.[/yellow]")
            else:
                ui.console.print(f"[bold cyan]Son Doğrulama Durumu:[/bold cyan] {rep.final_status.value} ({rep.timestamp[:19]})")
                ui.console.print(orchestrator.verification_engine.format_report_markdown(rep))
            return True

        if sub_cmd == "history":
            hist_md = orchestrator.tool_executor.execute(
                tool_name="get_verification_history",
                arguments={"limit": 10},
                caller_role="main",
                caller_agent="cli_user",
                is_trust_mode=orchestrator.trust_mode,
            )
            ui.console.print(hist_md)
            return True

        if sub_cmd == "report":
            rep_md = orchestrator.tool_executor.execute(
                tool_name="get_verification_report",
                arguments={"report_name": sub_arg} if sub_arg else {},
                caller_role="main",
                caller_agent="cli_user",
                is_trust_mode=orchestrator.trust_mode,
            )
            ui.console.print(rep_md)
            return True

        ui.console.print("[dim]Faz 2.3 Doğrulama ve Etki Analizi Döngüsü çalıştırılıyor...[/dim]")
        target_files = []
        if sub_cmd == "retry":
            last_rep = orchestrator.verification_engine.last_report or orchestrator.verification_engine.history_store.get_latest_report()
            if last_rep and last_rep.changed_files:
                target_files = last_rep.changed_files
        elif arg:
            target_files = [f.strip() for f in arg.split(",") if f.strip()]

        report_md = orchestrator.tool_executor.execute(
            tool_name="run_verification",
            arguments={
                "changed_files": ",".join(target_files),
                "task_description": f"CLI /verify yürütmesi ({', '.join(target_files) if target_files else 'genel'})",
            },
            caller_role="main",
            caller_agent="cli_user",
            is_trust_mode=orchestrator.trust_mode,
        )
        ui.console.print(report_md)
        return True

    if cmd == "/multi":
        coordinator = getattr(orchestrator, "multi_agent_coordinator", None)
        if not coordinator:
            ui.console.print("[red]Çoklu ajan koordinatörü aktif değil.[/red]")
            return True

        sub_parts = arg.split(maxsplit=1)
        sub_cmd = sub_parts[0].lower() if sub_parts else "status"
        sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

        if sub_cmd == "status":
            graph = coordinator.current_graph
            if not graph:
                ui.console.print("[dim]Şu anda aktif bir çoklu ajan yürütmesi yok. Yeni başlatmak için: `/multi plan <hedef>`[/dim]")
            else:
                ui.console.print(graph.to_ascii_tree())
            return True

        if sub_cmd == "graph":
            graph = coordinator.current_graph
            if not graph:
                ui.console.print("[dim]Aktif görev çizgesi bulunamadı.[/dim]")
            else:
                ui.console.print(f"[bold cyan]Görev Çizgesi (Topolojik Sıralama):[/bold cyan]\n{graph.to_ascii_tree()}")
            return True

        if sub_cmd == "tasks":
            graph = coordinator.current_graph
            if not graph:
                ui.console.print("[dim]Aktif görev listesi yok.[/dim]")
            else:
                from rich.table import Table
                table = Table(title="Çoklu Ajan Görev Listesi")
                table.add_column("ID", style="bold cyan")
                table.add_column("Rol", style="magenta")
                table.add_column("Durum", style="yellow")
                table.add_column("Başlık")
                table.add_column("Atanan Dosyalar", style="dim")
                for t in graph.tasks.values():
                    table.add_row(
                        t.task_id,
                        t.role.value,
                        t.status.value,
                        t.title,
                        ", ".join(t.assigned_files) if t.assigned_files else "-"
                    )
                ui.console.print(table)
            return True

        if sub_cmd == "agents":
            agents = coordinator.agent_registry.list_agents()
            if not agents:
                ui.console.print("[dim]Henüz havuza kayıtlı uzman ajan yok.[/dim]")
            else:
                from rich.table import Table
                table = Table(title="Aktif Uzman Ajanlar")
                table.add_column("Ajan ID", style="bold cyan")
                table.add_column("Rol", style="magenta")
                table.add_column("Durum", style="green")
                table.add_column("Model ID", style="dim")
                for a in agents:
                    table.add_row(a["agent_id"], a["role"], a["status"], a["model"])
                ui.console.print(table)
            return True

        if sub_cmd == "conflicts":
            conflicts = coordinator.last_conflicts
            if not conflicts:
                ui.console.print("[bold green]✔ Görevler arasında herhangi bir dosya çakışması tespit edilmedi.[/bold green]")
            else:
                for c in conflicts:
                    status = "[green]ÇÖZÜLDÜ[/green]" if c.resolved else "[red]ÇÖZÜLEMEDİ[/red]"
                    ui.console.print(f"- {c.conflict_id} ({c.task_a_id} <-> {c.task_b_id}) [{status}]: {c.details}")
            return True

        if sub_cmd == "plan":
            if not sub_arg:
                ui.console.print("[yellow]Kullanım: `/multi plan <hedef>` (Örn: `/multi plan Yeni auth modülü ekle`)[/yellow]")
                return True
            ui.console.print(f"[bold cyan]Çoklu Ajan Planı Oluşturuluyor ve Yürütülüyor:[/bold cyan] {sub_arg}")
            report = coordinator.run_plan(goal=sub_arg)
            ui.console.print(report.summary)
            return True

        if sub_cmd == "cancel":
            ok = coordinator.cancel_active_run()
            if ok:
                ui.console.print("[yellow]Aktif çoklu ajan çalışması iptal edildi.[/yellow]")
            else:
                ui.console.print("[dim]İptal edilecek aktif çalışma bulunamadı.[/dim]")
            return True

        if sub_cmd == "history":
            runs = coordinator.persistence.list_runs()
            if not runs:
                ui.console.print("[dim]Kayıtlı çoklu ajan çalışması bulunamadı.[/dim]")
            else:
                from rich.table import Table
                table = Table(title="Çoklu Ajan Çalışma Geçmişi (.evren/multi_agent/runs/)")
                table.add_column("Run ID", style="bold cyan")
                table.add_column("Görev Sayısı", justify="right")
                table.add_column("Dizin", style="dim")
                for r in runs:
                    table.add_row(r.get("run_id", ""), str(r.get("total_tasks", 0)), r.get("path", ""))
                ui.console.print(table)
            return True

        if sub_cmd == "report":
            target_run = sub_arg or coordinator.active_run_id
            if not target_run:
                ui.console.print("[yellow]Görüntülenecek Run ID belirtin veya önce bir çalışma yürütün.[/yellow]")
                return True
            try:
                _, rep, _ = coordinator.persistence.load_checkpoint(target_run)
                if rep:
                    ui.console.print(rep.summary)
                else:
                    ui.console.print(f"[yellow]'{target_run}' için rapor bulunamadı.[/yellow]")
            except Exception as exc:
                ui.console.print(f"[red]Rapor yükleme hatası: {exc}[/red]")
            return True

        if sub_cmd == "resume":
            target_run = sub_arg or coordinator.active_run_id
            if not target_run:
                ui.console.print("[yellow]Devam edilecek Run ID belirtin: `/multi resume <run_id>` (Liste için: `/multi history`)[/yellow]")
                return True
            ui.console.print(f"[bold green]'{target_run}' çalışmasına kaldığı yerden devam ediliyor...[/bold green]")
            try:
                report = coordinator.resume_run(target_run)
                ui.console.print(report.summary)
            except Exception as exc:
                ui.console.print(f"[red]Devam etme hatası: {exc}[/red]")
            return True

        ui.console.print(
            "[bold cyan]Çoklu Ajan Komutları:[/bold cyan]\n"
            "  • `/multi plan <hedef>`: Hedefi DAG çizgesine ayrıştırır ve paralel uzman ajanlarla yürütür.\n"
            "  • `/multi status`: Aktif yürütme durumunu ve ağacı gösterir.\n"
            "  • `/multi graph`: Görev bağımlılık çizgesini gösterir.\n"
            "  • `/multi tasks`: Görevlerin ayrıntılı listesini ve kapsamlarını listeler.\n"
            "  • `/multi agents`: Havuzdaki uzman ajanları ve durumlarını gösterir.\n"
            "  • `/multi conflicts`: Dosya kapsamı çakışmalarını ve çözümlerini listeler.\n"
            "  • `/multi history`: Kayıtlı çalışma oturumlarını gösterir.\n"
            "  • `/multi report [run_id]`: Çalışmanın nihai provens raporunu görüntüler.\n"
            "  • `/multi cancel`: Devam eden çalışmayı iptal eder.\n"
            "  • `/multi resume [run_id]`: Yarıda kalan çalışmaya devam eder."
        )
        return True

    if cmd == "/sprint":
        sprint_mgr = getattr(orchestrator, "sprint_manager", None)
        if not sprint_mgr:
            ui.console.print("[red]Sprint yöneticisi aktif değil.[/red]")
            return True

        sub_parts = arg.split(maxsplit=1)
        sub_cmd = sub_parts[0].lower() if sub_parts else "status"
        sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

        if sub_cmd in ("status", "progress"):
            target_sid = sub_arg or (sprint_mgr.active_sprint.id if sprint_mgr.active_sprint else None)
            if not target_sid:
                ui.console.print("[dim]Şu anda aktif bir sprint yok. Yeni planlamak için: `/sprint plan <hedef>`[/dim]")
            else:
                ui.console.print(sprint_mgr.get_status_dashboard(target_sid))
            return True

        if sub_cmd in ("plan", "create"):
            if not sub_arg:
                ui.console.print("[yellow]Kullanım: `/sprint plan <mühendislik hedefi>`[/yellow]")
                return True
            ui.console.print(f"[bold cyan]Sprint Planlanıyor:[/bold cyan] {sub_arg}")
            sp = sprint_mgr.create_and_plan(objective=sub_arg)
            ui.console.print(sprint_mgr.get_status_dashboard(sp.id))
            ui.console.print(f"\n[green]Sprint planlandı! Başlatmak için: `/sprint approve {sp.id}` ardından `/sprint start`[/green]")
            return True

        if sub_cmd == "approve":
            target_sid = sub_arg or (sprint_mgr.active_sprint.id if sprint_mgr.active_sprint else None)
            if not target_sid:
                ui.console.print("[yellow]Onaylanacak sprint ID belirtin: `/sprint approve <sprint_id>`[/yellow]")
                return True
            ok = sprint_mgr.approve_sprint(target_sid)
            if ok:
                ui.console.print(f"[bold green]✔ Sprint '{target_sid}' kullanıcı tarafından onaylandı. Başlatmak için: `/sprint start`[/bold green]")
            return True

        if sub_cmd == "start":
            target_sid = sub_arg or (sprint_mgr.active_sprint.id if sprint_mgr.active_sprint else None)
            ui.console.print(f"[bold cyan]Otonom Sprint Yürütmesi Başlatılıyor...[/bold cyan]")
            try:
                res_sp = sprint_mgr.start_sprint(target_sid)
                ui.console.print(sprint_mgr.get_status_dashboard(res_sp.id))
                ui.console.print(f"\n[bold green]Sprint tamamlandı! Durum: {res_sp.status.value}[/bold green]")
            except Exception as exc:
                ui.console.print(f"[bold red]Sprint Yürütme Hatası:[/bold red] {exc}")
            return True

        if sub_cmd == "epics":
            target_sid = sub_arg or (sprint_mgr.active_sprint.id if sprint_mgr.active_sprint else None)
            sp = sprint_mgr.get_sprint(target_sid) if target_sid else None
            if not sp:
                ui.console.print("[dim]Aktif sprint bulunamadı.[/dim]")
            else:
                from rich.table import Table
                table = Table(title=f"Sprint Epikleri ({sp.id})")
                table.add_column("Epik ID", style="bold cyan")
                table.add_column("Başlık")
                table.add_column("Durum", style="magenta")
                table.add_column("Görevler", justify="right")
                for e in sp.epics.values():
                    table.add_row(e.id, e.title, e.status.value, str(len(e.task_ids)))
                ui.console.print(table)
            return True

        if sub_cmd == "tasks":
            target_sid = sub_arg or (sprint_mgr.active_sprint.id if sprint_mgr.active_sprint else None)
            sp = sprint_mgr.get_sprint(target_sid) if target_sid else None
            if not sp:
                ui.console.print("[dim]Aktif sprint bulunamadı.[/dim]")
            else:
                from rich.table import Table
                table = Table(title=f"Sprint Görevleri ({sp.id})")
                table.add_column("Görev ID", style="bold cyan")
                table.add_column("Rol", style="magenta")
                table.add_column("Durum", style="yellow")
                table.add_column("Başlık")
                table.add_column("Atanan Dosyalar", style="dim")
                for t in sp.tasks.values():
                    table.add_row(t.task_id, t.role.value, t.status.value, t.title, ", ".join(t.assigned_files) if t.assigned_files else "-")
                ui.console.print(table)
            return True

        if sub_cmd == "acceptance":
            target_sid = sub_arg or (sprint_mgr.active_sprint.id if sprint_mgr.active_sprint else None)
            sp = sprint_mgr.get_sprint(target_sid) if target_sid else None
            if not sp:
                ui.console.print("[dim]Aktif sprint bulunamadı.[/dim]")
            else:
                from rich.table import Table
                table = Table(title=f"Kabul Kriterleri ve Kanıtlar ({sp.id})")
                table.add_column("Kriter ID", style="bold cyan")
                table.add_column("Yöntem", style="dim")
                table.add_column("Durum", style="green")
                table.add_column("Açıklama")
                table.add_column("Doğrulama Sonucu", style="dim")
                for c in sp.acceptance_criteria.values():
                    table.add_row(c.id, c.method.value, c.status.value, c.description, c.verification_result[:60] if c.verification_result else "-")
                ui.console.print(table)
            return True

        if sub_cmd == "report":
            target_sid = sub_arg or (sprint_mgr.active_sprint.id if sprint_mgr.active_sprint else None)
            if not target_sid:
                ui.console.print("[yellow]Görüntülenecek Sprint ID belirtin.[/yellow]")
            else:
                md_rep = sprint_mgr.get_markdown_report(target_sid)
                ui.console.print(md_rep)
            return True

        if sub_cmd in ("history", "list"):
            sprints = sprint_mgr.list_sprints()
            if not sprints:
                ui.console.print("[dim]Henüz kayıtlı sprint bulunmuyor.[/dim]")
            else:
                from rich.table import Table
                table = Table(title="Kayıtlı Sprintler (.evren/sprints/)")
                table.add_column("Sprint ID", style="bold cyan")
                table.add_column("Durum", style="magenta")
                table.add_column("İterasyon", justify="right")
                table.add_column("Görevler", justify="right")
                table.add_column("Başlık")
                for s in sprints:
                    table.add_row(s["sprint_id"], s["status"], str(s["iteration"]), str(s["tasks_count"]), s["title"])
                ui.console.print(table)
            return True

        if sub_cmd == "pause":
            target_sid = sub_arg or (sprint_mgr.active_sprint.id if sprint_mgr.active_sprint else None)
            ok = sprint_mgr.pause_sprint(target_sid)
            if ok:
                ui.console.print(f"[yellow]Sprint '{target_sid}' duraklatıldı.[/yellow]")
            else:
                ui.console.print("[dim]Duraklatılacak çalışan sprint bulunamadı.[/dim]")
            return True

        if sub_cmd == "resume":
            target_sid = sub_arg or (sprint_mgr.active_sprint.id if sprint_mgr.active_sprint else None)
            ui.console.print(f"[bold green]Sprint '{target_sid}' devam ettiriliyor...[/bold green]")
            try:
                res_sp = sprint_mgr.resume_sprint(target_sid)
                ui.console.print(sprint_mgr.get_status_dashboard(res_sp.id))
            except Exception as exc:
                ui.console.print(f"[bold red]Devam ettirme hatası:[/bold red] {exc}")
            return True

        if sub_cmd == "cancel":
            target_sid = sub_arg or (sprint_mgr.active_sprint.id if sprint_mgr.active_sprint else None)
            ok = sprint_mgr.cancel_sprint(target_sid)
            if ok:
                ui.console.print(f"[yellow]Sprint '{target_sid}' iptal edildi.[/yellow]")
            else:
                ui.console.print("[dim]İptal edilecek sprint bulunamadı.[/dim]")
            return True

        ui.console.print(
            "[bold cyan]Otonom Sprint Komutları (Phase 2.5):[/bold cyan]\n"
            "  • `/sprint plan <hedef>`: Yeni sprint planlar (epik, görev, kabul kriteri).\n"
            "  • `/sprint approve [sprint_id]`: Planlanan sprinti onaylar.\n"
            "  • `/sprint start [sprint_id]`: Onaylanan sprinti otonom yürütmeye başlatır.\n"
            "  • `/sprint status [sprint_id]`: Sprint ilerleme panosunu (dashboard) görüntüler.\n"
            "  • `/sprint epics [sprint_id]`: Epik listesini gösterir.\n"
            "  • `/sprint tasks [sprint_id]`: Görev listesini ve rollerini gösterir.\n"
            "  • `/sprint acceptance [sprint_id]`: Kanıta dayalı kabul kriterlerini listeler.\n"
            "  • `/sprint report [sprint_id]`: Kapsamlı Markdown teslimat raporunu görüntüler.\n"
            "  • `/sprint history`: Kayıtlı tüm sprintleri listeler.\n"
            "  • `/sprint pause`: Çalışan sprinti duraklatır.\n"
            "  • `/sprint resume [sprint_id]`: Duraklatılmış sprinti devam ettirir.\n"
            "  • `/sprint cancel`: Sprinti iptal eder."
        )
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

    if cmd == "/plan":
        if not arg:
            table = orchestrator.plan_manager.render_table()
            ui.console.print(table)
        else:
            ui.console.print(f"[bold cyan]Plan Oluşturuluyor:[/bold cyan] {arg}")
            orchestrator.run_turn(
                f"Lütfen şu görev için önce 'create_task_plan' aracını çağırarak somut 3-6 adımlık bir plan oluştur: {arg}"
            )
        return True

    if cmd in ("/subagent", "/subagents", "/delegate"):
        parts = arg.split(maxsplit=1)
        if not parts:
            ui.console.print(
                "[bold cyan]Kullanım:[/bold cyan] `/subagent <researcher|tester|reviewer> <görev açıklaması>`\n"
                "  • [bold]researcher:[/bold] Kod tabanını ve dokümanları salt-okunur derinlemesine araştırır.\n"
                "  • [bold]tester:[/bold] Testleri ve derlemeyi koşturur, hata triyajı yapar.\n"
                "  • [bold]reviewer:[/bold] Git farklarını ve kod kalitesini denetler."
            )
            return True
        role = parts[0].lower()
        sub_task = parts[1] if len(parts) > 1 else "Genel inceleme ve durum raporu çıkar."
        res = orchestrator.subagent_manager.delegate(role, sub_task)
        ui.console.print(res)
        return True

    if cmd in ("/search", "/semantic"):
        if not arg:
            ui.console.print("[yellow]Kullanım: `/search <doğal dil veya teknik kavram>` (Örn: `/search auth password hash`)[/yellow]")
            return True
        out = orchestrator.tool_executor.execute(
            tool_name="semantic_search_code",
            arguments={"query": arg},
            caller_role="main",
            caller_agent="cli_user",
            is_trust_mode=orchestrator.trust_mode,
        )
        ui.console.print(out)
        return True

    if cmd == "/tree":
        target_dir = arg or "."
        tree_out = orchestrator.tool_executor.execute(
            tool_name="get_project_tree",
            arguments={"path": target_dir},
            caller_role="main",
            caller_agent="cli_user",
            is_trust_mode=orchestrator.trust_mode,
        )
        ui.console.print(tree_out)
        return True

    if cmd == "/undo":
        msg = orchestrator.undo_manager.undo_last()
        ui.console.print(f"[bold cyan]{msg}[/bold cyan]")
        return True

    if cmd == "/save":
        session_name = arg or orchestrator.config.workspace_root.name
        saved_file = session_mgr.save_session(
            name=session_name,
            context=orchestrator.context,
            active_model_id=orchestrator.llm_client.active_model.id,
            plan_manager=orchestrator.plan_manager,
        )
        ui.console.print(
            f"[green]✔ Oturum kaydedildi:[/green] [bold]{saved_file.stem}[/bold] ([dim]{saved_file}[/dim])"
        )
        return True

    if cmd == "/load":
        if not arg:
            ui.console.print("[yellow]Yüklenecek oturum adını belirtin: `/load <oturum_adi>` (Liste için `/sessions`)[/yellow]")
            return True
        try:
            payload = session_mgr.load_session(arg, orchestrator.context)
            session_mgr.restore_session_to_orchestrator(orchestrator, payload)
            ui.console.print(
                f"[green]✔ '{arg}' oturumu geri yüklendi ({len(payload.get('messages', []))} mesaj).[/green]"
            )
            if orchestrator.plan_manager.current_plan:
                ui.console.print(orchestrator.plan_manager.render_table())
        except FileNotFoundError as exc:
            ui.console.print(f"[red]{exc}[/red]")
        return True

    if cmd in ("/resume", "/continue"):
        last_sess = session_mgr.get_last_workspace_session()
        if not last_sess:
            ui.console.print("[dim]Bu çalışma alanında devam edilecek kayıtlı oturum bulunamadı.[/dim]")
        else:
            session_mgr.restore_session_to_orchestrator(orchestrator, last_sess)
            rel_t = last_sess.get("relative_time", "önceki")
            ui.console.print(
                f"[green]✔ Son oturum başarıyla yüklendi ({len(last_sess.get('messages', []))} mesaj, {rel_t}).[/green]"
            )
            if orchestrator.plan_manager.current_plan:
                ui.console.print(orchestrator.plan_manager.render_table())
        return True

    if cmd == "/sessions":
        sessions = session_mgr.list_sessions()
        if not sessions:
            ui.console.print("[dim]Henüz kaydedilmiş bir oturum bulunmuyor (`/save <isim>` ile kaydedebilirsiniz).[/dim]")
        else:
            for s in sessions:
                ui.console.print(
                    f"  • [bold cyan]{s['name']}[/bold cyan] | Model: {s['model']} | "
                    f"Mesaj: {s['messages']} | Tarih: {s['saved_at']} | Dizin: [dim]{s['workspace']}[/dim]"
                )
        return True

    if cmd == "/compact":
        count = orchestrator.compact_context()
        if count == 0:
            ui.console.print("[dim]Sıkıştırılacak yeterli uzunlukta geçmiş bulunmuyor.[/dim]")
        return True

    if cmd == "/trust":
        orchestrator.trust_mode = not orchestrator.trust_mode
        ui._trust_all_active = orchestrator.trust_mode
        state = "AKTİF (Onay sorulmayacak)" if orchestrator.trust_mode else "KAPALI (Değişikliklerde onay sorulacak)"
        ui.console.print(f"[bold cyan]Güven Modu:[/bold cyan] {state}")
        return True

    if cmd == "/status":
        spec = orchestrator.llm_client.active_model
        tokens = orchestrator.context.estimate_tokens()
        msgs = orchestrator.context.history_length
        undos = orchestrator.undo_manager.history_count
        ui.console.print(
            f"[bold]Model:[/bold] {spec.name} ({spec.id}) | "
            f"[bold]Mesaj:[/bold] {msgs} | "
            f"[bold]Tahmini Token:[/bold] ~{tokens} / {spec.context_length} | "
            f"[bold]Geri Alınabilir İşlem (/undo):[/bold] {undos}"
        )
        return True

    if cmd in ("/logs", "/log"):
        logger = orchestrator.logger
        if not logger.enabled or not logger.log_directory:
            ui.console.print("[dim]Loglama devre dışı.[/dim]")
        elif arg == "clear":
            logger.clear_logs()
            ui.console.print("[bold green]✔ Log kayıtları (.evren/logs/) temizlendi.[/bold green]")
        else:
            ui.console.print(
                f"[bold cyan]📁 Ajan Geliştirme Günlükleri:[/bold cyan]\n"
                f"  • JSONL Yörünge: [green]{logger.jsonl_path}[/green]\n"
                f"  • Markdown Günlük: [green]{logger.markdown_path}[/green]\n"
                f"[dim]Temizlemek için: `/logs clear`[/dim]"
            )
        return True

    if cmd == "/update":
        ui.console.print("[dim]Evren-Code güncelleniyor...[/dim]")
        res = run_self_update(install_deps=False)
        style = "green" if res.success else "red"
        ui.console.print(f"[bold {style}]{res.summary}[/bold {style}]\n[dim]{res.details}[/dim]")
        return True

    if cmd == "/setup":
        run_onboarding_flow(auth_mgr, ui, force_new_key=True)
        return True

    if cmd == "/clear":
        orchestrator.context.clear()
        orchestrator.plan_manager.clear()
        orchestrator.logger.clear_logs()
        session_mgr.clear_workspace_session()
        ui.console.print("[green]✔ Konuşma geçmişi, aktif plan, log kayıtları ve oturum temizlendi.[/green]")
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
                        p = item.get("path")
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
            history=FileHistory(hist_file),
            completer=completer,
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

        import re
        def _mention_replacer(match: re.Match) -> str:
            rel = match.group(1).replace("\\", "/")
            tgt = workspace / rel
            if tgt.is_file():
                try:
                    txt = tgt.read_text(encoding="utf-8", errors="replace")
                    return f"\n[@{rel}]:\n```\n{txt}\n```\n"
                except OSError:
                    pass
            return match.group(0)

        effective_input = re.sub(r"@([a-zA-Z0-9_\-\./\\]+)", _mention_replacer, user_input)

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
