"""Decomposer module."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from evren_code.core.multi_agent.models import AgentRole, TaskStatus
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.sprint.exceptions import SprintPlanningError, SprintValidationError
from evren_code.core.sprint.models import (
    AcceptanceCriterion,
    AcceptanceMethod,
    AcceptanceStatus,
    EpicStatus,
    SprintEpic,
    SprintTask,
)

logger = logging.getLogger(__name__)


class SprintDecomposer:
    """Sprint decomposer implementation."""

    def __init__(self, max_tasks: int = 30) -> None:
        self.max_tasks = max_tasks

    def parse_decomposition_response(
        self,
        raw_text: str,
        default_epic_title: str = "Ana Geliştirme",
    ) -> Tuple[Dict[str, SprintEpic], Dict[str, SprintTask], Dict[str, AcceptanceCriterion]]:
        """Parse decomposition response."""
        payload = self._extract_json_payload(raw_text)
        if not payload:
            raise SprintPlanningError("Planlama yanıtından geçerli bir JSON yapısı ayrıştırılamadı.")

        epics_dict: Dict[str, SprintEpic] = {}
        tasks_dict: Dict[str, SprintTask] = {}
        criteria_dict: Dict[str, AcceptanceCriterion] = {}

        # 1. Epikler
        raw_epics = payload.get("epics", [])
        if isinstance(raw_epics, list):
            for e in raw_epics:
                if not isinstance(e, dict):
                    continue
                eid = str(e.get("id") or f"epic_{uuid.uuid4().hex[:4]}")
                title = str(e.get("title") or "Genel Epik")
                desc = str(e.get("description") or "")
                epics_dict[eid] = SprintEpic(
                    id=eid,
                    title=title,
                    description=desc,
                    status=EpicStatus.PENDING,
                )

        if not epics_dict:
            default_id = "epic_main"
            epics_dict[default_id] = SprintEpic(
                id=default_id,
                title=default_epic_title,
                description="Sprint ana hedefleri",
                status=EpicStatus.PENDING,
            )

        default_epic_id = next(iter(epics_dict.keys()))

        raw_tasks = payload.get("tasks", [])
        if isinstance(raw_tasks, list):
            for t in raw_tasks:
                if not isinstance(t, dict):
                    continue
                tid = str(t.get("task_id") or t.get("id") or f"task_{uuid.uuid4().hex[:4]}")
                epic_id = str(t.get("epic_id") or default_epic_id)
                if epic_id not in epics_dict:
                    epic_id = default_epic_id

                role_str = str(t.get("role", "developer")).lower()
                try:
                    role_enum = AgentRole(role_str)
                except ValueError:
                    role_enum = AgentRole.DEVELOPER

                task = SprintTask(
                    task_id=tid,
                    epic_id=epic_id,
                    title=str(t.get("title") or "Mühendislik Görevi"),
                    description=str(t.get("description") or ""),
                    role=role_enum,
                    dependencies=list(t.get("dependencies", [])),
                    assigned_files=list(t.get("assigned_files", [])),
                    read_only_files=list(t.get("read_only_files", [])),
                    acceptance_criteria=list(t.get("acceptance_criteria", [])),
                    estimated_complexity=str(t.get("estimated_complexity", "medium")),
                    priority=int(t.get("priority", 1)),
                    status=TaskStatus.PENDING,
                )
                tasks_dict[tid] = task
                if tid not in epics_dict[epic_id].task_ids:
                    epics_dict[epic_id].task_ids.append(tid)

        # 3. Kabul Kriterleri (Acceptance Criteria)
        raw_criteria = payload.get("acceptance_criteria", [])
        if isinstance(raw_criteria, list):
            for c in raw_criteria:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("id") or f"crit_{uuid.uuid4().hex[:4]}")
                method_str = str(c.get("method", "test_result")).lower()
                try:
                    method_enum = AcceptanceMethod(method_str)
                except ValueError:
                    method_enum = AcceptanceMethod.TEST_RESULT

                criterion = AcceptanceCriterion(
                    id=cid,
                    description=str(c.get("description") or "Kabul kriteri doğrulaması"),
                    method=method_enum,
                    required_evidence=str(c.get("required_evidence") or ""),
                    status=AcceptanceStatus.PENDING,
                    associated_task_ids=list(c.get("associated_task_ids", [])),
                )
                criteria_dict[cid] = criterion

        for tid, t in tasks_dict.items():
            for crit_ref in t.acceptance_criteria:
                if crit_ref in criteria_dict:
                    if tid not in criteria_dict[crit_ref].associated_task_ids:
                        criteria_dict[crit_ref].associated_task_ids.append(tid)

        return epics_dict, tasks_dict, criteria_dict

    def validate_decomposition(
        self,
        epics: Dict[str, SprintEpic],
        tasks: Dict[str, SprintTask],
        criteria: Dict[str, AcceptanceCriterion],
    ) -> List[str]:
        """Validate decomposition."""
        errors: List[str] = []

        if not tasks:
            errors.append("Sprint en az bir görev içermelidir.")
            return errors

        if len(tasks) > self.max_tasks:
            errors.append(
                f"Görev sayısı ({len(tasks)}) izin verilen maksimum sınırı ({self.max_tasks}) aşıyor."
            )

        for tid, task in tasks.items():
            if not task.epic_id or task.epic_id not in epics:
                errors.append(f"Görev '{tid}' geçersiz bir epik kimliğine ('{task.epic_id}') sahip.")

        temp_graph = TaskGraph()
        for t in tasks.values():
            temp_graph.add_task(t.to_engineering_task())

        is_valid_dag, dag_errors = temp_graph.validate_dag()
        if not is_valid_dag:
            errors.extend([f"DAG Hatası: {e}" for e in dag_errors])

        all_task_ids = set(tasks.keys())
        for cid, crit in criteria.items():
            for tid in crit.associated_task_ids:
                if tid not in all_task_ids:
                    errors.append(f"Kabul kriteri '{cid}' var olmayan '{tid}' görevine bağlı.")

        return errors

    def _extract_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract json payload."""
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except Exception:
                pass

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return None
