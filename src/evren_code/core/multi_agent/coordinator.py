"""Multi-Agent Coordinator module."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evren_code.config import AppConfig
from evren_code.core.events import EventBus
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.core.multi_agent.agent_registry import AgentRegistry
from evren_code.core.multi_agent.communication import AgentMessageBus
from evren_code.core.multi_agent.conflict_resolver import ConflictResolver
from evren_code.core.multi_agent.context_sharing import ContextSharingEngine
from evren_code.core.multi_agent.file_ownership import FileOwnershipManager
from evren_code.core.multi_agent.models import (
    AgentRole,
    AgentTaskResult,
    ConflictReport,
    EngineeringTask,
    MultiAgentRunConfig,
    OrchestrationReport,
    TaskStatus,
)
from evren_code.core.multi_agent.persistence import MultiAgentPersistence
from evren_code.core.multi_agent.result_aggregator import ResultAggregator
from evren_code.core.multi_agent.scheduler import MultiAgentScheduler
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.repo_intelligence.engine import RepoIntelligenceEngine
from evren_code.core.security import ToolExecutor
from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine
from evren_code.core.verification.engine import VerificationEngine
from evren_code.tools.registry import ToolRegistry


class MultiAgentCoordinator:
    """Multi agent coordinator implementation."""

    def __init__(
        self,
        config: AppConfig,
        llm_client: EvrenLLMClient,
        tool_executor: ToolExecutor,
        registry: ToolRegistry,
        repo_intelligence: Optional[RepoIntelligenceEngine] = None,
        symbol_engine: Optional[CodebaseSymbolEngine] = None,
        verification_engine: Optional[VerificationEngine] = None,
        event_bus: Optional[EventBus] = None,
        multi_config: Optional[MultiAgentRunConfig] = None,
        shadow_git: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.registry = registry
        self.repo_intelligence = repo_intelligence
        self.symbol_engine = symbol_engine
        self.verification_engine = verification_engine
        self.event_bus = event_bus or EventBus()
        self.multi_config = multi_config or MultiAgentRunConfig()
        self.shadow_git = shadow_git

        self.file_ownership = FileOwnershipManager(workspace_root=config.workspace_root)
        self.agent_registry = AgentRegistry(
            config=config,
            llm_client=llm_client,
            tool_executor=tool_executor,
            registry=registry,
            file_ownership=self.file_ownership,
            event_bus=self.event_bus,
        )
        self.message_bus = AgentMessageBus()
        self.context_sharing = ContextSharingEngine(
            repo_intelligence=repo_intelligence,
            symbol_engine=symbol_engine,
        )
        self.result_aggregator = ResultAggregator(
            agent_registry=self.agent_registry,
            verification_engine=verification_engine,
        )
        self.conflict_resolver = ConflictResolver()
        self.persistence = MultiAgentPersistence(
            workspace_root=config.workspace_root,
            base_dirname=self.multi_config.persistence_dir,
        )

        self.current_graph: Optional[TaskGraph] = None
        self.current_scheduler: Optional[MultiAgentScheduler] = None
        self.active_run_id: Optional[str] = None
        self.last_report: Optional[OrchestrationReport] = None
        self.last_conflicts: List[ConflictReport] = []

    @staticmethod
    def _normalize_role(role_raw: Any) -> AgentRole:
        """Normalize role."""
        if isinstance(role_raw, AgentRole):
            return role_raw
        r = str(role_raw or "").strip().lower()
        if not r:
            return AgentRole.DEVELOPER
        for member in AgentRole:
            if member.value == r:
                return member
        if any(w in r for w in ("dev", "coder", "frontend", "backend", "fullstack", "engineer")):
            return AgentRole.DEVELOPER
        if any(w in r for w in ("test", "qa", "verifier")):
            return AgentRole.TESTER
        if any(w in r for w in ("review", "audit")):
            return AgentRole.REVIEWER
        if any(w in r for w in ("research", "investigat", "search")):
            return AgentRole.RESEARCHER
        if any(w in r for w in ("arch", "design")):
            return AgentRole.ARCHITECT
        if any(w in r for w in ("plan",)):
            return AgentRole.PLANNER
        return AgentRole.DEVELOPER

    def plan_dag_from_goal(self, goal: str) -> TaskGraph:
        """Plan dag from goal."""
        planner = self.agent_registry.get_or_create_agent(AgentRole.PLANNER)

        structure_info = ""
        try:
            ws = self.config.workspace_root
            if self.repo_intelligence:
                mod_dict = self.repo_intelligence.get_project_modules()
                if mod_dict:
                    structure_info += f"- Proje Modülleri: {', '.join(mod_dict.keys())}\n"
            tree_lines = []
            for item in sorted(ws.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if item.name.startswith(".") or item.name in ("node_modules", "build", "dist", "__pycache__", ".git"):
                    continue
                if item.is_dir():
                    tree_lines.append(f"📁 {item.name}/")
                    try:
                        for sub in sorted(item.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:10]:
                            if not sub.name.startswith("."):
                                prefix = "📁 " if sub.is_dir() else "📄 "
                                tree_lines.append(f"  └── {prefix}{sub.name}")
                    except Exception:
                        pass
                else:
                    tree_lines.append(f"📄 {item.name}")
                if len(tree_lines) >= 35:
                    break
            if tree_lines:
                structure_info += "- Dosya ve Modül Ağacı (2 Seviye):\n  " + "\n  ".join(tree_lines[:35]) + "\n"
        except Exception:
            pass
        structure_block = f"\nMEVCUT PROJE YAPISI:\n{structure_info}\n" if structure_info else ""

        prompt = (
            f"Hedef: {goal}\n"
            f"{structure_block}\n"
            "Sen uzman bir Baş Mimar (Lead Architect) ajanısın.\n"
            "Bu hedefi analiz et. Proje alanını (Oyun Geliştirme, Mobil Uygulama, Full-Stack Web, "
            "Sistem/CLI/DevOps, Veri/ML vb.) dikkate alarak görevi EN DOĞAL VE MANTIKLI SAYIDA "
            "uzmanlaşmış bağımsız alt göreve ayrıştır (sayıyı yapay olarak 2-4 ile sınırlama; işin doğasına ve kapsamına göre 2, 4, 6 veya daha fazla olabilir).\n\n"
            "ÖNEMLİ KURALLAR:\n"
            "1. HIZLI HAREKET ET: Proje yapısını anlamak için gerekirse EN FAZLA 1-2 keşif aracı (örn: `get_project_tree` veya `get_repo_map`) kullan. Kod dosyalarını derinlemesine okuyarak adımlarını tüketme!\n"
            "2. GÖREV ATAMALARI: 'developer' rolündeki görevler için düzenlenecek somut dosya yollarını 'assigned_files' listesine ekle.\n"
            "3. PARALEL ÇALIŞMA: Bağımsız görevler paralel yürütülebilsin (dependencies boş olsun).\n"
            "4. SÖZLEŞMELER: İlişkili işlerde üst görevin 'provides_contracts' alanında yayınladığı sözleşmeyi alt görevin 'consumes_contracts' alanında belirt.\n"
            "5. NİHAİ ÇIKTI: Yanıtında MUTLAKA geçerli bir JSON listesi (```json [ ... ] ``` kod bloğu içinde) döndür.\n\n"
            "Format:\n"
            "```json\n"
            "[\n"
            "  {\n"
            '    "task_id": "task_1",\n'
            '    "title": "Kısa başlık",\n'
            '    "description": "Detaylı açıklama",\n'
            '    "domain": "gameplay|hud_ui|physics|mobile|backend|frontend|api|database|system|audio|core",\n'
            '    "role": "architect|developer|tester|reviewer|researcher",\n'
            '    "dependencies": [],\n'
            '    "provides_contracts": ["PlayerStatsModel", "CombatEvent"],\n'
            '    "consumes_contracts": [],\n'
            '    "assigned_files": ["src/player.py"],\n'
            '    "read_only_files": [],\n'
            '    "acceptance_criteria": ["Kriter 1", "Kriter 2"]\n'
            "  }\n"
            "]\n"
            "```"
        )

        plan_task = EngineeringTask(
            task_id="plan_init",
            title=f"Planlama: {goal[:50]}",
            description=prompt,
            role=AgentRole.PLANNER,
        )
        res = planner.execute_task(plan_task)

        graph = TaskGraph()
        parsed_tasks = self._parse_json_tasks(res.output)
        if not parsed_tasks:
            fallback_task = EngineeringTask(
                task_id="task_1",
                title="Görevi İncele ve Uygula",
                description=(
                    f"Hedef: {goal}\n\n"
                    "UYGULAMA VE EYLEM TALİMATI:\n"
                    "1. İlk 2-3 adımda gerekli dosyaları tespit et ve oku (`read_file`, `search_code`).\n"
                    "2. Ardından VAKİT KAYBETMEDEN `edit_file` veya `write_file` araçlarıyla gereken düzenlemeleri uygula.\n"
                    "3. Adım sınırına takılmamak için salt okuma yaparak zaman kaybetme; somut kod değişiklikleri üret!\n"
                    "4. Tüm değişiklikleri somut olarak tamamlayıp doğrula ve nihai yanıtında özetle."
                ),
                role=AgentRole.DEVELOPER,
                domain="general",
                assigned_files=["*"],
                acceptance_criteria=[
                    "Hedefe uygun kod değişiklikleri ilgili dosyalarda eksiksiz uygulanmalı",
                    "Değişiklikler temiz, hatasız ve doğrulanmış olmalı",
                ],
            )
            graph.add_task(fallback_task)
        else:
            for tdict in parsed_tasks:
                raw_id = tdict.get("task_id") or tdict.get("id") or tdict.get("taskId") or f"task_{uuid.uuid4().hex[:4]}"
                task_id = str(raw_id)
                title = str(tdict.get("title") or tdict.get("name") or "Mühendislik Görevi")
                raw_desc = tdict.get("description") or tdict.get("notes") or tdict.get("details") or tdict.get("instruction") or title
                description = str(raw_desc)
                role_enum = self._normalize_role(tdict.get("role") or tdict.get("agent") or tdict.get("role_name"))

                raw_deps = tdict.get("dependencies") or tdict.get("depends_on") or tdict.get("deps") or []
                dependencies = [str(d) for d in raw_deps if str(d).strip()]

                raw_files = tdict.get("assigned_files") or tdict.get("files_scope") or tdict.get("target_files") or tdict.get("files") or []
                assigned_files = [str(f) for f in raw_files if str(f).strip()]
                task_domain = str(tdict.get("domain", "general"))
                if role_enum == AgentRole.DEVELOPER and not assigned_files:
                    if len(parsed_tasks) == 1:
                        assigned_files = ["*"]
                    else:
                        inferred = re.findall(r"(?:[\w\-\.]+(?:/|\\)[\w\-\.\/]+\.[a-zA-Z0-9]+)", description)
                        if inferred:
                            assigned_files = [f.replace("\\", "/") for f in inferred]
                        elif task_domain and task_domain not in ("general", "core"):
                            assigned_files = [f"{task_domain}/*"]

                desc_lower = description.lower()
                has_concrete_impl = bool(assigned_files) or any(
                    w in desc_lower for w in ("oluştur", "yaz", "çalıştır", "implement", "create", "write", "run", "execute", "build", "betik")
                )
                if has_concrete_impl and role_enum in (AgentRole.ARCHITECT, AgentRole.RESEARCHER):
                    role_enum = AgentRole.DEVELOPER

                raw_read_files = tdict.get("read_only_files") or tdict.get("context_files") or []
                read_only_files = [str(f) for f in raw_read_files if str(f).strip()]

                # Kabul Kriterleri
                raw_crit = tdict.get("acceptance_criteria") or tdict.get("criteria") or []
                acceptance_criteria = [str(c) for c in raw_crit if str(c).strip()]

                raw_provides = (
                    tdict.get("provides_contracts")
                    or tdict.get("contracts_provided")
                    or tdict.get("provides")
                    or []
                )
                raw_consumes = (
                    tdict.get("consumes_contracts")
                    or tdict.get("contracts_consumed")
                    or tdict.get("consumes")
                    or []
                )
                provides_contracts = [str(c).strip() for c in raw_provides if str(c).strip()]
                consumes_contracts = [str(c).strip() for c in raw_consumes if str(c).strip()]

                task = EngineeringTask(
                    task_id=task_id,
                    title=title,
                    description=description,
                    role=role_enum,
                    dependencies=dependencies,
                    assigned_files=assigned_files,
                    read_only_files=read_only_files,
                    acceptance_criteria=acceptance_criteria,
                    domain=task_domain,
                    provides_contracts=provides_contracts,
                    consumes_contracts=consumes_contracts,
                )
                graph.add_task(task)

            def _clean_contract_name(name: str) -> str:
                return re.sub(r"[\s_\-]+", "", name).lower()

            contract_providers: Dict[str, str] = {}
            for t in graph.tasks.values():
                for c_name in getattr(t, "provides_contracts", []):
                    c_clean = _clean_contract_name(str(c_name))
                    if c_clean:
                        contract_providers[c_clean] = t.task_id

            for t in graph.tasks.values():
                for c_name in getattr(t, "consumes_contracts", []):
                    c_clean = _clean_contract_name(str(c_name))
                    provider_id = contract_providers.get(c_clean)
                    if provider_id and provider_id != t.task_id and provider_id not in t.dependencies:
                        t.dependencies.append(provider_id)

        self.last_conflicts = self.conflict_resolver.resolve_conflicts(graph)
        self.current_graph = graph
        return graph

    def _parse_json_tasks(self, text: str) -> List[Dict[str, Any]]:
        """Parse json tasks."""
        if not text or not text.strip():
            return []

        code_block_matches = re.findall(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", text, re.DOTALL | re.IGNORECASE)
        for block in code_block_matches:
            block = block.strip()
            try:
                parsed = json.loads(block)
                tasks = self._extract_task_list(parsed)
                if tasks:
                    return tasks
            except Exception:
                pass
            try:
                from evren_code.core.parser import repair_and_parse_json
                parsed = repair_and_parse_json(block)
                tasks = self._extract_task_list(parsed)
                if tasks:
                    return tasks
            except Exception:
                pass

        arr_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if arr_match:
            candidate = arr_match.group(0).strip()
            try:
                parsed = json.loads(candidate)
                tasks = self._extract_task_list(parsed)
                if tasks:
                    return tasks
            except Exception:
                pass
            try:
                from evren_code.core.parser import repair_and_parse_json
                parsed = repair_and_parse_json(candidate)
                tasks = self._extract_task_list(parsed)
                if tasks:
                    return tasks
            except Exception:
                pass

        obj_match = re.search(r"\{\s*\"(?:tasks|plan|subtasks|items)\"\s*:\s*\[.*\]\s*\}", text, re.DOTALL)
        if obj_match:
            candidate = obj_match.group(0).strip()
            try:
                parsed = json.loads(candidate)
                tasks = self._extract_task_list(parsed)
                if tasks:
                    return tasks
            except Exception:
                pass
            try:
                from evren_code.core.parser import repair_and_parse_json
                parsed = repair_and_parse_json(candidate)
                tasks = self._extract_task_list(parsed)
                if tasks:
                    return tasks
            except Exception:
                pass

        try:
            from evren_code.core.parser import repair_and_parse_json
            parsed = repair_and_parse_json(text.strip())
            tasks = self._extract_task_list(parsed)
            if tasks:
                return tasks
        except Exception:
            pass

        return []

    def _extract_task_list(self, parsed: Any) -> List[Dict[str, Any]]:
        """Extract task list."""
        if isinstance(parsed, list):
            valid_items = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or item.get("name") or "").strip()
                desc = str(item.get("description") or item.get("notes") or item.get("details") or "").strip()
                tid = str(item.get("task_id") or item.get("id") or "").strip()
                if (title and len(title) >= 3 and title.lower() != "mühendislik görevi") or (desc and len(desc) >= 5):
                    valid_items.append(item)
                elif tid and (title or desc):
                    valid_items.append(item)
            return valid_items
        if isinstance(parsed, dict):
            for key in ("tasks", "plan", "subtasks", "items", "value"):
                if key in parsed and isinstance(parsed[key], list):
                    return self._extract_task_list(parsed[key])
        return []

    def run_plan(
        self,
        goal: str,
        graph: Optional[TaskGraph] = None,
        on_update: Optional[Callable[[EngineeringTask], None]] = None,
        run_id: Optional[str] = None,
    ) -> OrchestrationReport:
        """Run plan."""
        active_id = run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        self.active_run_id = active_id
        start_time = time.time()

        if graph is None:
            graph = self.plan_dag_from_goal(goal)
        else:
            self.current_graph = graph
            self.last_conflicts = self.conflict_resolver.resolve_conflicts(graph)

        self.persistence.log_event(active_id, f"Çoklu ajan yürütmesi başlatıldı. Hedef: {goal}")
        self.persistence.save_checkpoint(active_id, graph, metadata={"goal": goal, "stage": "started"})

        self.current_scheduler = MultiAgentScheduler(
            task_graph=graph,
            agent_registry=self.agent_registry,
            file_ownership=self.file_ownership,
            context_sharing=self.context_sharing,
            result_aggregator=self.result_aggregator,
            message_bus=self.message_bus,
            config=self.multi_config,
            event_bus=self.event_bus,
            shadow_git=self.shadow_git,
        )

        def _step_callback(task: EngineeringTask) -> None:
            self.persistence.log_event(
                active_id,
                f"Görev güncellemesi: {task.task_id} -> {task.status.value}"
            )
            self.persistence.save_checkpoint(active_id, graph, metadata={"goal": goal, "stage": "in_progress"})
            if on_update:
                on_update(task)

        ok = self.current_scheduler.execute_all(on_task_update=_step_callback)

        report = self.result_aggregator.build_orchestration_report(
            run_id=active_id,
            goal=goal,
            task_graph=graph,
            start_time=start_time,
        )
        report.conflicts_detected = list(self.last_conflicts)

        # Phase 2.3 VerificationEngine entegrasyonu
        if self.multi_config.verify_with_engine and report.modified_files:
            v_ok = self.result_aggregator.verify_with_engine(report.modified_files, goal)
            if not v_ok:
                report.success = False
                report.summary += "\n\n[UYARI]: Faz 2.3 Doğrulama motoru bazı testlerin başarısız olduğunu bildirdi. Süreç başarısız olarak işaretlendi."

        self.last_report = report
        self.persistence.save_checkpoint(active_id, graph, report=report, metadata={"goal": goal, "stage": "completed"})
        self.persistence.log_event(active_id, f"Yürütme tamamlandı. Başarılı: {report.success}")

        return report

    def cancel_active_run(self) -> bool:
        """Cancel active run."""
        if self.current_scheduler:
            self.current_scheduler.cancel()
            if self.active_run_id and self.current_graph:
                self.persistence.save_checkpoint(
                    self.active_run_id,
                    self.current_graph,
                    metadata={"stage": "cancelled"}
                )
                self.persistence.log_event(self.active_run_id, "Yürütme kullanıcı tarafından iptal edildi.")
            return True
        return False

    def resume_run(self, run_id: str) -> OrchestrationReport:
        """Resume run."""
        graph, old_rep, meta = self.persistence.load_checkpoint(run_id)
        self.active_run_id = run_id
        self.current_graph = graph
        goal = meta.get("goal", "Devam edilen görev")

        self.file_ownership.reset()

        for t in graph.tasks.values():
            if t.status in (TaskStatus.RUNNING, TaskStatus.ASSIGNED):
                if t.result and t.result.success:
                    t.status = TaskStatus.COMPLETED
                else:
                    disk_modified = False
                    for af in t.assigned_files:
                        p = (self.config.workspace_root / af).resolve()
                        if p.exists() and p.stat().st_mtime >= t.created_at:
                            disk_modified = True
                            break

                    if disk_modified:
                        t.status = TaskStatus.FAILED
                        t.result = AgentTaskResult(
                            task_id=t.task_id,
                            agent_id=t.assigned_agent_id or "unknown",
                            role=t.role,
                            success=False,
                            output="Kurtarma uzlaştırması: Görev yarıda kesildi ve dosyalar diskte değiştirilmişti; çifte yazma engellendi.",
                            error="Interrupted task with partially modified files on disk",
                        )
                    else:
                        t.status = TaskStatus.PENDING

        return self.run_plan(goal=goal, graph=graph, run_id=run_id)
