"""SprintPlanner module."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from evren_code.core.llm_client import EvrenLLMClient
from evren_code.core.multi_agent.models import AgentRole, TaskStatus
from evren_code.core.repo_intelligence.engine import RepoIntelligenceEngine
from evren_code.core.sprint.decomposer import SprintDecomposer
from evren_code.core.sprint.exceptions import SprintPlanningError, SprintValidationError
from evren_code.core.sprint.models import (
    AcceptanceCriterion,
    AcceptanceMethod,
    AcceptanceStatus,
    EpicStatus,
    Sprint,
    SprintConfig,
    SprintEpic,
    SprintMetadata,
    SprintStatus,
    SprintTask,
)
from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine

logger = logging.getLogger(__name__)


class SprintPlanner:
    """Sprint planner implementation."""

    def __init__(
        self,
        llm_client: Optional[EvrenLLMClient] = None,
        repo_intelligence: Optional[RepoIntelligenceEngine] = None,
        symbol_engine: Optional[CodebaseSymbolEngine] = None,
    ) -> None:
        self.llm_client = llm_client
        self.repo_intelligence = repo_intelligence
        self.symbol_engine = symbol_engine

    def plan_sprint(
        self,
        objective: str,
        config: Optional[SprintConfig] = None,
        title: Optional[str] = None,
        scope: Optional[List[str]] = None,
        out_of_scope: Optional[List[str]] = None,
        sprint_id: Optional[str] = None,
    ) -> Sprint:
        """Plan sprint."""
        cfg = config or SprintConfig()
        sid = sprint_id or f"sprint_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        sprint_title = title or f"Sprint: {objective[:40]}..."

        meta = SprintMetadata(
            sprint_id=sid,
            title=sprint_title,
            objective=objective,
            description=f"Otonom sprint planı: {objective}",
            scope=list(scope or []),
            out_of_scope=list(out_of_scope or []),
        )

        decomposer = SprintDecomposer(max_tasks=cfg.max_tasks_per_sprint)

        prompt = self._build_planning_prompt(objective, meta, cfg)

        raw_response = ""
        if self.llm_client:
            try:
                system_instruction = (
                    "Sen kıdemli bir yazılım mimarı ve otonom mühendislik yöneticisisin. "
                    "Verilen yazılım geliştirme hedefini mantıksal epiklere, bağımsız çalışabilecek "
                    "veya sıralı görevlere (tasks) ve somut kanıtlara dayalı kabul kriterlerine dönüştürürsün."
                )
                raw_response = self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt},
                    ]
                )
            except Exception as e:
                logger.warning(f"LLM ile sprint planlama hatası: {e}. Fallback plan kullanılacak.")

        if not raw_response or not raw_response.strip():
            epics, tasks, criteria = self._generate_fallback_plan(objective, sid)
        else:
            try:
                epics, tasks, criteria = decomposer.parse_decomposition_response(raw_response)
            except Exception as pe:
                logger.warning(f"Sprint JSON ayrıştırma başarısız: {pe}. Fallback plan devreye giriyor.")
                epics, tasks, criteria = self._generate_fallback_plan(objective, sid)

        errors = decomposer.validate_decomposition(epics, tasks, criteria)
        if errors:
            logger.warning(f"Sprint planında doğrulama hataları tespit edildi: {errors}. Düzeltme uygulanıyor.")
            epics, tasks, criteria = self._repair_plan(epics, tasks, criteria, errors)
            remaining_errors = decomposer.validate_decomposition(epics, tasks, criteria)
            if remaining_errors:
                raise SprintValidationError(
                    f"Sprint planı geçerlilik kontrolünü geçemedi: {'; '.join(remaining_errors)}"
                )

        sprint = Sprint(
            metadata=meta,
            config=cfg,
            epics=epics,
            tasks=tasks,
            acceptance_criteria=criteria,
            status=SprintStatus.PLANNED,
        )
        return sprint

    def plan_remediation(
        self,
        sprint: Sprint,
        failed_tasks: List[SprintTask],
        failed_criteria: List[AcceptanceCriterion],
        iteration: int,
    ) -> List[SprintTask]:
        """Plan remediation."""
        remediation_tasks: List[SprintTask] = []
        fail_summary = "\n".join(
            [f"- Görev {t.task_id} ({t.title}): {t.status.value}" for t in failed_tasks]
            + [f"- Kriter {c.id} ({c.description}): {c.verification_result}" for c in failed_criteria]
        )

        prompt = (
            f"Sprint '{sprint.id}' için {iteration}. iterasyonda şu başarısızlıklar tespit edildi:\n"
            f"{fail_summary}\n\n"
            "Bu problemleri çözmek ve kabul kriterlerini sağlamak için 1-3 adet odaklanmış düzeltme görevi (remediation tasks) oluştur.\n"
            "Çıktıyı 'tasks' anahtarı içeren standart JSON formatında ver."
        )

        raw_response = ""
        if self.llm_client:
            try:
                raw_response = self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": "Kıdemli sorun giderici ve yazılım mimarısın."},
                        {"role": "user", "content": prompt},
                    ]
                )
            except Exception as e:
                logger.warning(f"Remediation planlama LLM hatası: {e}")

        decomposer = SprintDecomposer(max_tasks=5)
        parsed_tasks: Dict[str, SprintTask] = {}
        if raw_response:
            try:
                _, parsed_tasks, _ = decomposer.parse_decomposition_response(raw_response)
            except Exception:
                pass

        origin_id = None
        if failed_tasks:
            origin_id = failed_tasks[0].origin_task_id or failed_tasks[0].task_id
        elif failed_criteria and failed_criteria[0].associated_task_ids:
            origin_id = failed_criteria[0].associated_task_ids[0]

        if not parsed_tasks:
            fix_id = f"fix_iter_{iteration}_{uuid.uuid4().hex[:4]}"
            default_epic = next(iter(sprint.epics.keys())) if sprint.epics else "epic_main"
            fallback_fix = SprintTask(
                task_id=fix_id,
                epic_id=default_epic,
                origin_task_id=origin_id,
                title=f"İterasyon {iteration} Düzeltme ve Onarım",
                description=f"Başarısızlıkları gider:\n{fail_summary}",
                role=AgentRole.DEVELOPER,
                dependencies=[t.task_id for t in failed_tasks],
                assigned_files=list({f for t in failed_tasks for f in t.assigned_files}),
                acceptance_criteria=[c.id for c in failed_criteria],
                estimated_complexity="medium",
                priority=1,
                status=TaskStatus.PENDING,
            )
            remediation_tasks.append(fallback_fix)
        else:
            for t in parsed_tasks.values():
                t.epic_id = next(iter(sprint.epics.keys())) if sprint.epics else "epic_main"
                if not t.origin_task_id and origin_id:
                    t.origin_task_id = origin_id
                remediation_tasks.append(t)

        return remediation_tasks

    def _build_planning_prompt(
        self,
        objective: str,
        meta: SprintMetadata,
        cfg: SprintConfig,
    ) -> str:
        """Build planning prompt."""
        context_parts = []
        if self.repo_intelligence:
            try:
                summary = self.repo_intelligence.get_context_summary()
                if summary:
                    context_parts.append(f"Depo Mimarisi Özeti:\n{summary}")
            except Exception:
                pass

        repo_context = "\n".join(context_parts) if context_parts else "Mevcut depo bağlamı hazır."

        return f"""Aşağıdaki mühendislik hedefini profesyonel bir sprint planına dönüştür:

HEDEF: {objective}
KAPSAM İÇİ: {', '.join(meta.scope) if meta.scope else 'Hedefin gerektirdiği tüm bileşenler'}
KAPSAM DIŞI: {', '.join(meta.out_of_scope) if meta.out_of_scope else 'İlişkisiz mimari değişiklikler'}

DEPO BAĞLAMI:
{repo_context}

Gereksinimler:
1. 1-3 adet Epik tanımla (epics).
2. 2-8 adet Görev tanımla (tasks). Her görevin rolü (developer, tester, reviewer, architect, researcher), bağımlılıkları ve atanmış dosyaları olsun.
3. Somut kanıtlara dayalı Kabul Kriterleri (acceptance_criteria) belirle:
   - method türleri: file_exists, symbol_exists, build_success, test_result, review_approval, manual_approval
   - required_evidence: doğrulanacak dosya yolu, test komutu veya sembol adı.

Çıktıyı YALNIZCA geçerli bir JSON objesi olarak şu yapıda ver:
{{
  "epics": [
    {{"id": "epic_1", "title": "Başlık", "description": "Açıklama"}}
  ],
  "tasks": [
    {{
      "task_id": "task_1",
      "epic_id": "epic_1",
      "title": "Görev Başlığı",
      "description": "Detaylı açıklama",
      "role": "developer",
      "dependencies": [],
      "assigned_files": ["src/dosya.py"],
      "read_only_files": [],
      "acceptance_criteria": ["crit_1"],
      "estimated_complexity": "medium",
      "priority": 1
    }}
  ],
  "acceptance_criteria": [
    {{
      "id": "crit_1",
      "description": "Testlerin geçmesi",
      "method": "test_result",
      "required_evidence": "pytest tests/",
      "associated_task_ids": ["task_1"]
    }}
  ]
}}
"""

    def _generate_fallback_plan(
        self,
        objective: str,
        sprint_id: str,
    ) -> Tuple[Dict[str, SprintEpic], Dict[str, SprintTask], Dict[str, AcceptanceCriterion]]:
        """Generate fallback plan."""
        epic_id = "epic_core"
        epic = SprintEpic(
            id=epic_id,
            title="Ana Mühendislik Hedefi",
            description=objective,
            task_ids=["task_dev", "task_test", "task_review"],
            status=EpicStatus.PENDING,
        )

        task_dev = SprintTask(
            task_id="task_dev",
            epic_id=epic_id,
            title="Geliştirme ve Uygulama",
            description=f"Hedefi gerçekleştir: {objective}",
            role=AgentRole.DEVELOPER,
            dependencies=[],
            assigned_files=[],
            acceptance_criteria=["crit_build", "crit_test"],
            status=TaskStatus.PENDING,
        )

        task_test = SprintTask(
            task_id="task_test",
            epic_id=epic_id,
            title="Test ve Doğrulama",
            description="Birim ve entegrasyon testlerini hazırla ve çalıştır.",
            role=AgentRole.TESTER,
            dependencies=["task_dev"],
            assigned_files=[],
            acceptance_criteria=["crit_test"],
            status=TaskStatus.PENDING,
        )

        task_review = SprintTask(
            task_id="task_review",
            epic_id=epic_id,
            title="Kod Gözden Geçirme",
            description="Uygulanan değişikliklerin mimariye ve standartlara uygunluğunu onayla.",
            role=AgentRole.REVIEWER,
            dependencies=["task_test"],
            assigned_files=[],
            acceptance_criteria=["crit_review"],
            status=TaskStatus.PENDING,
        )

        crit_build = AcceptanceCriterion(
            id="crit_build",
            description="Derleme ve statik analiz sorunsuz tamamlanmalı",
            method=AcceptanceMethod.BUILD_SUCCESS,
            required_evidence="",
            associated_task_ids=["task_dev"],
        )

        crit_test = AcceptanceCriterion(
            id="crit_test",
            description="Test paketi başarıyla sonuçlanmalı",
            method=AcceptanceMethod.TEST_RESULT,
            required_evidence="",
            associated_task_ids=["task_dev", "task_test"],
        )

        crit_review = AcceptanceCriterion(
            id="crit_review",
            description="Reviewer onayı alınmalı",
            method=AcceptanceMethod.REVIEW_APPROVAL,
            required_evidence="",
            associated_task_ids=["task_review"],
        )

        epics = {epic.id: epic}
        tasks = {t.task_id: t for t in [task_dev, task_test, task_review]}
        criteria = {c.id: c for c in [crit_build, crit_test, crit_review]}

        return epics, tasks, criteria

    def _repair_plan(
        self,
        epics: Dict[str, SprintEpic],
        tasks: Dict[str, SprintTask],
        criteria: Dict[str, AcceptanceCriterion],
        errors: List[str],
    ) -> Tuple[Dict[str, SprintEpic], Dict[str, SprintTask], Dict[str, AcceptanceCriterion]]:
        """Repair plan."""
        # Eksik epik varsa tamamla
        if not epics:
            default_epic = SprintEpic(id="epic_default", title="Ana Epik", description="Otomatik oluşturuldu")
            epics[default_epic.id] = default_epic

        default_eid = next(iter(epics.keys()))
        for t in tasks.values():
            if not t.epic_id or t.epic_id not in epics:
                t.epic_id = default_eid
            if t.task_id not in epics[t.epic_id].task_ids:
                epics[t.epic_id].task_ids.append(t.task_id)

        all_ids = set(tasks.keys())
        for t in tasks.values():
            valid_deps = [dep for dep in t.dependencies if dep in all_ids and dep != t.task_id]
            t.dependencies = valid_deps

        for c in criteria.values():
            c.associated_task_ids = [tid for tid in c.associated_task_ids if tid in all_ids]

        return epics, tasks, criteria
