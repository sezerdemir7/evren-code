"""Task Planning and Progress Tracking Engine (Plan-Before-Execute & Task Checklist).

Orchestrates multi-step engineering tasks:
1. Creates an actionable execution plan (`create_task_plan`),
2. Updates progress and notes step-by-step (`update_plan_step`),
3. Injects the active plan into system context to maintain focus on the objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple

from rich.table import Table

from evren_code.core.events import EventBus, EventType


class ComplexityLevel(str, Enum):
    """Task complexity levels."""
    SIMPLE = "simple"      # Direct execution, minimal verification, lower step budget
    MEDIUM = "medium"      # Focused plan, implementation, and unit testing
    COMPLEX = "complex"    # Multi-phase plan, phased implementation, comprehensive verification and delegation


class ExecutionState(str, Enum):
    """Formal agent runtime state machine."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_FOR_TOOL = "waiting_for_tool"
    RECOVERING = "recovering"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class AcceptanceCriterion:
    """Individual acceptance criterion."""
    description: str
    is_met: bool = False
    evidence: str = ""
    is_mandatory: bool = True


@dataclass
class TaskContract:
    """Task Contract formalizing user intent into explicit goals and criteria."""
    goal: str
    acceptance_criteria: List[AcceptanceCriterion] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)
    allowed_supporting_actions: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)
    verification_strategy: str = "targeted_check"
    state: ExecutionState = ExecutionState.PLANNING

    def is_completed(self) -> bool:
        """Verifies whether all acceptance criteria in the contract are satisfied."""
        return bool(self.acceptance_criteria and all(c.is_met for c in self.acceptance_criteria))

    @property
    def is_all_mandatory_met(self) -> bool:
        return all(c.is_met for c in self.acceptance_criteria if c.is_mandatory)

    def mark_criterion(self, pattern_or_desc: str, evidence: str = "") -> bool:
        p_clean = pattern_or_desc.lower()
        for c in self.acceptance_criteria:
            if p_clean in c.description.lower() or c.description.lower() in p_clean:
                c.is_met = True
                c.evidence = evidence
                return True
        return False

    def render_markdown(self) -> str:
        lines = [f"### 🎯 Task Contract / Görev Kontratı: {self.goal}", f"**Status / Durum:** `{self.state.value}`\n"]
        lines.append("**Acceptance Criteria / Kabul Kriterleri:**")
        for c in self.acceptance_criteria:
            box = "[x]" if c.is_met else "[ ]"
            ev = f" _(Evidence / Kanıt: {c.evidence})_" if c.evidence else ""
            lines.append(f"- {box} {c.description}{ev}")
        if self.forbidden_actions:
            lines.append("\n**Forbidden Actions / Kapsam Dışı İşlemler:**")
            for fa in self.forbidden_actions:
                lines.append(f"- ⛔ {fa}")
        return "\n".join(lines)


class TaskContractEngine:
    """Engine parsing user intent into a formal TaskContract."""

    @classmethod
    def create_contract(cls, prompt: str, assessment: Optional[ComplexityAssessment] = None) -> TaskContract:
        p_clean = prompt.strip()
        p_lower = p_clean.lower()

        goal_summary = p_clean.split("\n")[0][:120]

        criteria: List[AcceptanceCriterion] = []
        required_actions: List[str] = []
        allowed_supporting: List[str] = ["read_file", "read_multiple_files", "search_code", "find_files"]
        forbidden: List[str] = [
            "Modifying files outside workspace",
            "Modifying Windows PATH or system registry",
        ]

        # 1. Frontend / UI Task
        if any(term in p_lower for term in ("sayfa", "page", "ekran", "ui", "form", "view", "component", "bileşen")):
            criteria.append(AcceptanceCriterion("Create or update requested page/component files"))
            criteria.append(AcceptanceCriterion("Verify clean compilation or type check"))
            required_actions.extend(["write_file", "edit_file"])
            allowed_supporting.extend(["run_command", "lsp_get_diagnostics"])
            forbidden.extend([
                "Unauthorized user registration via live API",
                "Unauthorized workspace creation",
                "Unsolicited database seeding",
            ])
            strat = "build_and_compile"

        # 2. Testing Task
        elif any(term in p_lower for term in ("test ekle", "unit test", "test yaz", "add test")):
            criteria.append(AcceptanceCriterion("Author test files covering target logic"))
            criteria.append(AcceptanceCriterion("Verify all tests pass cleanly"))
            required_actions.extend(["write_file", "run_command"])
            allowed_supporting.extend(["edit_file", "read_file"])
            strat = "run_unit_test"

        # 3. Bug Fix
        elif any(term in p_lower for term in ("düzelt", "fix", "hata", "bug", "sorun", "çöz")):
            criteria.append(AcceptanceCriterion("Surgically fix defective code lines"))
            criteria.append(AcceptanceCriterion("Verify syntax and compilation pass cleanly"))
            required_actions.extend(["edit_file", "write_file"])
            strat = "compile_or_syntax"

        # 4. Typo / Micro-edit
        elif any(term in p_lower for term in ("typo", "yazım hatası", "küçük düzeltme", "format")):
            criteria.append(AcceptanceCriterion("Correct typographical or formatting error"))
            required_actions.extend(["edit_file"])
            strat = "syntax_only"

        # 5. General Code Task
        else:
            criteria.append(AcceptanceCriterion("Implement requested code modification"))
            criteria.append(AcceptanceCriterion("Pass syntax and compilation verification"))
            required_actions.extend(["edit_file", "write_file"])
            strat = "targeted_check"

        return TaskContract(
            goal=goal_summary,
            acceptance_criteria=criteria,
            required_actions=required_actions,
            allowed_supporting_actions=allowed_supporting,
            forbidden_actions=forbidden,
            verification_strategy=strat,
            state=ExecutionState.PLANNING,
        )

    @classmethod
    def validate_action(
        cls,
        contract: TaskContract,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Validates tool execution against task contract bounds."""
        if tool_name in ("run_command", "http_request"):
            cmd = str(arguments.get("command", "") or arguments.get("url", ""))
            for f_act in contract.forbidden_actions:
                if any(k in f_act.lower() for k in ("canlı api", "live api")) and any(ep in cmd for ep in ("/auth/register", "/workspaces")):
                    return False, f"Task contract violation (Görev sözleşmesi ihlali): {f_act}"
                if any(k in f_act.lower() for k in ("tohum", "seed")) and ("seed" in cmd.lower() or "tohumla" in cmd.lower()):
                    return False, f"Task contract violation (Görev sözleşmesi ihlali): {f_act}"

        return True, ""


@dataclass
class ComplexityAssessment:
    """Complexity assessment implementation."""
    level: ComplexityLevel
    max_turns: int
    allow_delegation: bool
    requires_formal_plan: bool
    reason: str
    is_urgent: bool = False
    strict_scope_enforcement: bool = True


class TaskComplexityClassifier:
    """Task complexity classifier implementation."""

    SIMPLE_KEYWORDS = [
        "typo", "yazım hatası", "küçük düzeltme", "yorum satırı", "format",
        "nedir", "nerede", "açıkla", "nasıl çalışır", "dosyasını oku", "kontrol et",
        "quick fix", "tek satır", "syntax fix"
    ]

    URGENT_KEYWORDS = [
        "hemen", "hızlı", "hızlıca", "hızlı bir şekilde", "quick", "asap",
        "çabuk", "acil", "sadece", "bitane", "bi tane"
    ]

    MEDIUM_KEYWORDS = [
        "unit test", "test ekle", "add test", "test yaz", "implement", "geliştir", "özellik",
        "feature", "endpoint", "fix bug", "hata çöz", "edge case", "servis",
        "service", "fonksiyon ekle", "component", "bileşen", "sayfa", "page",
        "ekran", "screen", "frontend", "backend", "form", "ui", "ux", "buton", "button",
        "stil", "style", "css", "tailwind", "düzenle", "güncelle", "ekle", "düzelt",
        "kur", "bağla", "çöz", "entegre", "ayarla", "hazırla", "yaz"
    ]

    COMPLEX_KEYWORDS = [
        "mimari", "architecture", "refactor", "yeniden yapılandır", "audit",
        "denetim", "multi-agent", "bütün testleri", "tüm sistemi", "uçtan uca",
        "baştan sona", "entegrasyon", "migration", "göç", "kapsamlı", "optimize et",
        "modern", "modernize", "temizle", "temiz hale", "modern hale", "kullanışlı",
        "ekranlar", "sayfalar", "bileşenler", "screens", "pages", "tasarım sistemi", "design system", "redesign",
        "paket", "package", "faz", "phase", "modül", "module", "tüm", "bütün", "hepsi",
        "altyapı", "infrastructure", "pipeline", "sistem", "fullstack", "full stack"
    ]

    @classmethod
    def classify(cls, prompt: str, touched_files_count: int = 0) -> ComplexityAssessment:
        p_lower = prompt.lower().strip()
        word_count = len(p_lower.split())
        is_urgent = any(uk in p_lower for uk in cls.URGENT_KEYWORDS)

        for kw in cls.COMPLEX_KEYWORDS:
            if kw in p_lower:
                max_t = 25 if is_urgent else 40
                return ComplexityAssessment(
                    level=ComplexityLevel.COMPLEX,
                    max_turns=max_t,
                    allow_delegation=not is_urgent,
                    requires_formal_plan=True,
                    reason=f"İstem içinde karmaşık sistem/mimari/paket anahtar kelimesi tespit edildi: '{kw}'" + (" (Acil/hızlı mod devrede)" if is_urgent else ""),
                    is_urgent=is_urgent,
                    strict_scope_enforcement=True,
                )

        if touched_files_count >= 4 or word_count > 60:
            max_t = 25 if is_urgent else 40
            return ComplexityAssessment(
                level=ComplexityLevel.COMPLEX,
                max_turns=max_t,
                allow_delegation=not is_urgent,
                requires_formal_plan=True,
                reason="Çok sayıda dosya veya uzun/detaylı gereksinim listesi tespit edildi." + (" (Acil/hızlı mod devrede)" if is_urgent else ""),
                is_urgent=is_urgent,
                strict_scope_enforcement=True,
            )

        for kw in cls.SIMPLE_KEYWORDS:
            if kw in p_lower:
                return ComplexityAssessment(
                    level=ComplexityLevel.SIMPLE,
                    max_turns=8,
                    allow_delegation=False,
                    requires_formal_plan=False,
                    reason=f"Hızlı/doğrudan işlem kategorisi tespit edildi: '{kw}'",
                    is_urgent=is_urgent,
                    strict_scope_enforcement=True,
                )

        for kw in cls.MEDIUM_KEYWORDS:
            if kw in p_lower:
                max_t = 12 if is_urgent else 20
                return ComplexityAssessment(
                    level=ComplexityLevel.MEDIUM,
                    max_turns=max_t,
                    allow_delegation=False,
                    requires_formal_plan=True,
                    reason=f"Standart geliştirme veya test odağı tespit edildi: '{kw}'" + (" (Acil/hızlı mod: bütçe kısıtlandı)" if is_urgent else ""),
                    is_urgent=is_urgent,
                    strict_scope_enforcement=True,
                )

        dev_action_terms = (
            "ekran", "sayfa", "ui", "ux", "frontend", "backend", "modern", "temiz", "tasarım",
            "paket", "package", "faz", "phase", "modül", "module", "sistem", "altyapı",
            "geliştir", "entegre", "bağla", "kur"
        )
        has_dev_action = any(term in p_lower for term in dev_action_terms)

        if word_count < 12 and touched_files_count <= 1 and not has_dev_action:
            return ComplexityAssessment(
                level=ComplexityLevel.SIMPLE,
                max_turns=8,
                allow_delegation=False,
                requires_formal_plan=False,
                reason="Kısa ve tekil dosya/açıklama odaklı istem.",
                is_urgent=is_urgent,
                strict_scope_enforcement=True,
            )

        max_t = 12 if is_urgent else 20
        return ComplexityAssessment(
            level=ComplexityLevel.MEDIUM,
            max_turns=max_t,
            allow_delegation=not is_urgent,
            requires_formal_plan=True,
            reason="Standart geliştirme, hata çözme veya modül görevi." + (" (Acil mod)" if is_urgent else ""),
            is_urgent=is_urgent,
            strict_scope_enforcement=True,
        )


@dataclass
class ExecutionBudget:
    """Execution budget implementation."""
    max_turns: int = 30
    max_model_calls: int = 60
    current_turns: int = 0
    current_model_calls: int = 0

    @property
    def is_exhausted(self) -> bool:
        return (self.current_turns >= self.max_turns) or (self.current_model_calls >= self.max_model_calls)

    def is_exceeded(self) -> bool:
        return self.is_exhausted

    def record_turn(self) -> None:
        self.current_turns += 1

    def record_model_call(self) -> None:
        self.current_model_calls += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_turns": self.max_turns,
            "max_model_calls": self.max_model_calls,
            "current_turns": self.current_turns,
            "current_model_calls": self.current_model_calls,
            "exhausted": self.is_exhausted,
        }


class PlanStepStatus(str, Enum):
    PENDING = "pending"          # [ ]
    IN_PROGRESS = "in_progress"  # [/]
    COMPLETED = "completed"      # [x]
    FAILED = "failed"            # [!]
    SKIPPED = "skipped"          # [-]


@dataclass
class PlanStep:
    """Plan step implementation."""

    id: int
    description: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    notes: str = ""

    @property
    def icon(self) -> str:
        if self.status == PlanStepStatus.COMPLETED:
            return "[x]"
        if self.status == PlanStepStatus.IN_PROGRESS:
            return "[/]"
        if self.status == PlanStepStatus.FAILED:
            return "[!]"
        if self.status == PlanStepStatus.SKIPPED:
            return "[-]"
        return "[ ]"


@dataclass
class TaskPlan:
    """Task plan implementation."""

    title: str
    steps: List[PlanStep] = field(default_factory=list)
    active: bool = True

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == PlanStepStatus.COMPLETED)

    @property
    def in_progress_step(self) -> Optional[PlanStep]:
        for s in self.steps:
            if s.status == PlanStepStatus.IN_PROGRESS:
                return s
        return None

    @property
    def is_all_completed(self) -> bool:
        """Is all completed (completed)."""
        if not self.steps:
            return True
        return all(s.status in (PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED) for s in self.steps)

    @property
    def remaining_steps(self) -> List[PlanStep]:
        """Remaining steps (pending, in_progress, failed)."""
        return [s for s in self.steps if s.status not in (PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED)]

    def render_markdown(self) -> str:
        """Render markdown."""
        if not self.steps or not self.active:
            return ""

        lines = [
            f"### Aktif Görev Planı: {self.title} ({self.completed_steps}/{self.total_steps} Tamamlandı)"
        ]
        for step in self.steps:
            status_desc = ""
            if step.status == PlanStepStatus.IN_PROGRESS:
                status_desc = " *(Şu Anda İşleniyor)*"
            elif step.status == PlanStepStatus.COMPLETED:
                status_desc = " *(Tamamlandı)*"
            elif step.status == PlanStepStatus.FAILED:
                status_desc = " *(Başarısız)*"
            elif step.status == PlanStepStatus.SKIPPED:
                status_desc = " *(Atlandı)*"

            note_str = f" - Not: {step.notes}" if step.notes else ""
            lines.append(f"- {step.icon} {step.id}. {step.description}{status_desc}{note_str}")

        lines.append(
            "\n*Kural: Bir adıma başladığında 'update_plan_step(status=\"in_progress\")' "
            "ve bitirdiğinde 'update_plan_step(status=\"completed\")' çağırarak planı güncel tut.*"
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return {
            "title": self.title,
            "active": self.active,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "status": s.status.value,
                    "notes": s.notes,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskPlan:
        """From dict."""
        steps: List[PlanStep] = []
        for s in data.get("steps", []):
            st_val = s.get("status", "pending")
            try:
                matched_status = PlanStepStatus(st_val)
            except Exception:
                matched_status = PlanStepStatus.PENDING
            steps.append(
                PlanStep(
                    id=s.get("id", 1),
                    description=s.get("description", ""),
                    status=matched_status,
                    notes=s.get("notes", ""),
                )
            )
        return cls(
            title=data.get("title", "Mühendislik Görev Planı"),
            steps=steps,
            active=data.get("active", True),
        )


class PlanManager:
    """Plan manager implementation."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus or EventBus()
        self._current_plan: Optional[TaskPlan] = None

    @property
    def current_plan(self) -> Optional[TaskPlan]:
        return self._current_plan

    def is_plan_finished(self) -> bool:
        """Is plan finished."""
        if not self._current_plan or not self._current_plan.active:
            return True
        return self._current_plan.is_all_completed

    def create_plan(self, title: str, steps: Any, preserve_completed: bool = True) -> TaskPlan:
        """Create plan."""
        existing_completed = {}
        if preserve_completed and self._current_plan:
            for s in self._current_plan.steps:
                if s.status in (PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED):
                    existing_completed[s.description.lower().strip()] = s

        if isinstance(steps, str):
            raw_str = steps.strip()
            parsed_ok = False
            try:
                import json
                parsed_s = json.loads(raw_str)
                if isinstance(parsed_s, list):
                    steps = parsed_s
                    parsed_ok = True
                elif isinstance(parsed_s, dict) and "steps" in parsed_s and isinstance(parsed_s["steps"], list):
                    steps = parsed_s["steps"]
                    parsed_ok = True
            except Exception:
                try:
                    from evren_code.core.parser import repair_and_parse_json
                    rep_s = repair_and_parse_json(raw_str)
                    if isinstance(rep_s, dict):
                        if isinstance(rep_s.get("value"), list):
                            steps = rep_s["value"]
                            parsed_ok = True
                        elif isinstance(rep_s.get("steps"), list):
                            steps = rep_s["steps"]
                            parsed_ok = True
                except Exception:
                    pass

            if not parsed_ok:
                inner = raw_str.strip("[]")
                if '", "' in inner or '","' in inner:
                    parts = [p.strip().strip("\"'") for p in re.split(r'"\s*,\s*"', inner) if p.strip().strip("\"'")]
                    if len(parts) > 1:
                        steps = parts
                        parsed_ok = True
                if not parsed_ok and "\n" in raw_str:
                    lines = [re.sub(r"^\s*(?:[-*•]|\d+[\.\)])\s*", "", ln).strip() for ln in raw_str.splitlines() if ln.strip()]
                    if len(lines) > 1:
                        steps = lines
                        parsed_ok = True
                if not parsed_ok:
                    steps = [raw_str]
        elif not isinstance(steps, list):
            steps = [steps] if steps else []

        flattened_items: List[Any] = []
        for item in steps:
            if isinstance(item, dict):
                std_keys = {"step", "description", "text", "task", "title", "status", "notes", "id"}
                extra_vals = [
                    str(v).strip()
                    for k, v in item.items()
                    if k not in std_keys and str(k).strip() != "" and isinstance(v, str) and len(str(v).strip()) > 3
                ]
                primary_text = (
                    item.get("step")
                    or item.get("description")
                    or item.get("text")
                    or item.get("task")
                    or item.get("title")
                )
                if primary_text and extra_vals:
                    flattened_items.append({"step": str(primary_text).strip(), "status": item.get("status"), "notes": item.get("notes")})
                    for ev in extra_vals:
                        flattened_items.append(ev)
                elif not primary_text and extra_vals:
                    for ev in extra_vals:
                        flattened_items.append(ev)
                else:
                    flattened_items.append(item)
            else:
                flattened_items.append(item)

        parsed_steps: List[PlanStep] = []
        for idx, desc in enumerate(flattened_items, start=1):
            initial_status = PlanStepStatus.PENDING
            step_notes = ""
            if isinstance(desc, dict):
                raw_status = str(desc.get("status") or "").lower().strip()
                if "progress" in raw_status or "devam" in raw_status:
                    initial_status = PlanStepStatus.IN_PROGRESS
                elif "comp" in raw_status or "tamam" in raw_status:
                    initial_status = PlanStepStatus.COMPLETED
                elif "fail" in raw_status or "hata" in raw_status:
                    initial_status = PlanStepStatus.FAILED
                elif "skip" in raw_status or "atla" in raw_status:
                    initial_status = PlanStepStatus.SKIPPED
                step_notes = str(desc.get("notes") or "")
                text = (
                    desc.get("step")
                    or desc.get("description")
                    or desc.get("text")
                    or desc.get("task")
                    or desc.get("title")
                    or ""
                )
                cleaned = str(text).strip()
            else:
                cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", str(desc).strip())

            if not cleaned:
                continue
            if cleaned.lower() in existing_completed:
                prev = existing_completed[cleaned.lower()]
                parsed_steps.append(PlanStep(id=idx, description=cleaned, status=prev.status, notes=prev.notes))
            else:
                parsed_steps.append(PlanStep(id=idx, description=cleaned, status=initial_status, notes=step_notes))

        plan = TaskPlan(title=title.strip() or "Mühendislik Görev Planı", steps=parsed_steps)
        self._current_plan = plan
        self.event_bus.emit(
            EventType.PLAN_UPDATED,
            title=plan.title,
            completed=plan.completed_steps,
            total=plan.total_steps,
            markdown=plan.render_markdown(),
        )
        return plan

    def update_step(
        self,
        step_id: int,
        status: str,
        notes: str = "",
    ) -> bool:
        """Update step."""
        if not self._current_plan:
            return False

        status_lower = status.strip().lower()
        matched_status: Optional[PlanStepStatus] = None
        for s in PlanStepStatus:
            if s.value == status_lower:
                matched_status = s
                break

        if matched_status is None:
            if "progress" in status_lower or "devam" in status_lower:
                matched_status = PlanStepStatus.IN_PROGRESS
            elif "complete" in status_lower or "tamam" in status_lower or "done" in status_lower:
                matched_status = PlanStepStatus.COMPLETED
            elif "fail" in status_lower or "hata" in status_lower:
                matched_status = PlanStepStatus.FAILED
            elif "skip" in status_lower or "atla" in status_lower:
                matched_status = PlanStepStatus.SKIPPED
            else:
                matched_status = PlanStepStatus.PENDING

        for step in self._current_plan.steps:
            if step.id == step_id:
                step.status = matched_status
                if notes:
                    step.notes = notes.strip()

                self.event_bus.emit(
                    EventType.PLAN_UPDATED,
                    title=self._current_plan.title,
                    completed=self._current_plan.completed_steps,
                    total=self._current_plan.total_steps,
                    markdown=self._current_plan.render_markdown(),
                )
                return True

        return False

    def clear(self) -> None:
        """Clear."""
        self._current_plan = None

    def to_dict(self) -> Optional[Dict[str, Any]]:
        """To dict."""
        return self._current_plan.to_dict() if self._current_plan else None

    def restore_from_dict(self, data: Optional[Dict[str, Any]]) -> None:
        """Restore from dict."""
        if data:
            self._current_plan = TaskPlan.from_dict(data)
            self.event_bus.emit(
                EventType.PLAN_UPDATED,
                title=self._current_plan.title,
                completed=self._current_plan.completed_steps,
                total=self._current_plan.total_steps,
                markdown=self._current_plan.render_markdown(),
            )
        else:
            self._current_plan = None

    def render_table(self) -> Table:
        """Render table."""
        table = Table(
            title=f"📋 Görev Planı: {self._current_plan.title if self._current_plan else 'Aktif Plan Yok'}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Durum", width=12)
        table.add_column("Açıklama")
        table.add_column("Not", style="dim")

        if not self._current_plan or not self._current_plan.steps:
            table.add_row("-", "Boş", "Şu anda aktif bir görev planı bulunmuyor.", "")
            return table

        for s in self._current_plan.steps:
            if s.status == PlanStepStatus.COMPLETED:
                st_text = "[bold green]✔ Tamamlandı[/bold green]"
            elif s.status == PlanStepStatus.IN_PROGRESS:
                st_text = "[bold yellow]⏳ İşleniyor[/bold yellow]"
            elif s.status == PlanStepStatus.FAILED:
                st_text = "[bold red]✖ Başarısız[/bold red]"
            elif s.status == PlanStepStatus.SKIPPED:
                st_text = "[dim]– Atlandı[/dim]"
            else:
                st_text = "[dim]Bekliyor[/dim]"

            table.add_row(str(s.id), st_text, s.description, s.notes or "")
        return table


@dataclass
class GoalGateResult:
    """Goal gate result implementation."""
    is_satisfied: bool
    reason: str = ""
    suggested_action: str = ""  # "stop_and_report", "continue"


class GoalGate:
    """Goal gate implementation."""

    BUILD_SUCCESS_PATTERNS = [
        r"compiled successfully",
        r"next\.js build succeeded",
        r"build successful",
        r"tsc exit:\s*0",
        r"built in\s+\d+",
        r"tests? passed",
        r"ok\s*\(\d+\s+tests?\)",
        r"çıkış kodu \(exit code\):\s*0",
    ]

    @classmethod
    def evaluate(
        cls,
        prompt: str,
        modified_files: List[str],
        recent_commands_output: str,
        assessment: Optional[ComplexityAssessment] = None,
        contract: Optional[TaskContract] = None,
    ) -> GoalGateResult:
        p_lower = prompt.lower()
        out_lower = (recent_commands_output or "").lower()

        if contract is not None:
            for crit in contract.acceptance_criteria:
                if not crit.is_met:
                    if any(kw in crit.description.lower() for kw in ("oluştur", "create", "yaz", "dosya")):
                        if modified_files:
                            crit.is_met = True
                    elif any(kw in crit.description.lower() for kw in ("derle", "build", "tsc", "test", "geç", "compil", "check", "verify")):
                        if any(re.search(pat, out_lower, re.IGNORECASE) for pat in cls.BUILD_SUCCESS_PATTERNS):
                            crit.is_met = True

            if contract.is_completed():
                return GoalGateResult(
                    is_satisfied=True,
                    reason=(
                        f"All acceptance criteria ({len(contract.acceptance_criteria)}/{len(contract.acceptance_criteria)}) "
                        f"in task contract satisfied (Görev sözleşmesindeki tüm kabul kriterleri sağlandı): {contract.goal}"
                    ),
                    suggested_action="stop_and_report",
                )

        # 2. UI / Page Heuristic
        is_ui_or_page_request = any(term in p_lower for term in ("sayfa", "page", "ekran", "ui", "form", "register", "login", "bileşen", "component"))
        has_modified_relevant_file = any(
            any(ext in mf.lower() for ext in (".tsx", ".jsx", ".vue", ".svelte", "page.", "component.", "service.", "schemas."))
            for mf in modified_files
        )

        has_build_success = any(
            re.search(pat, out_lower, re.IGNORECASE) for pat in cls.BUILD_SUCCESS_PATTERNS
        )

        if is_ui_or_page_request and has_modified_relevant_file and has_build_success:
            unsolicited = not any(term in p_lower for term in ("seed", "tohumla", "demo hesap", "kullanıcı oluştur", "curl ile test"))
            if unsolicited:
                return GoalGateResult(
                    is_satisfied=True,
                    reason=(
                        "Target page/component files created and build/tsc verification passed cleanly. "
                        "Definition of Done (DoD) satisfied (DoD kriterleri sağlandı)."
                    ),
                    suggested_action="stop_and_report",
                )

        return GoalGateResult(is_satisfied=False)


class ScopeGuard:
    """Guard preventing unauthorized scope expansion (Scope Creep)."""

    UNSOLICITED_PATTERNS = [
        (r"(?:\b(?:curl(?:\.exe)?|Invoke-RestMethod|Invoke-WebRequest)\b|https?://[^\s\"']*).*(?:\/api\/v\d+)?\/auth\/register", "Unsolicited live API user registration"),
        (r"(?:\b(?:curl(?:\.exe)?|Invoke-RestMethod|Invoke-WebRequest)\b|https?://[^\s\"']*).*(?:\/api\/v\d+)?\/workspaces\b", "Unsolicited live API workspace creation"),
        (r"\b(?:seed|tohumla|dummy_data)\b", "Unsolicited database seeding"),
    ]

    @classmethod
    def check_unsolicited_action(cls, prompt: str, command: str) -> Tuple[bool, str]:
        p_lower = prompt.lower()
        allow_keywords = ["demo hesap", "kullanıcı oluştur", "kayıt ol", "api test", "tohumla", "seed", "curl ile test"]
        if any(kw in p_lower for kw in allow_keywords):
            return False, ""

        for pattern, desc in cls.UNSOLICITED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return True, (
                    f"Scope Creep Guard / Barrier (Kapsam Aşımı Engeli): {desc}. "
                    "The user only requested codebase engineering changes. "
                    "Unauthorized data manipulation, registration, or database seeding via live API is prohibited. "
                    "Complete verification and deliver your final response."
                )
        return False, ""
