"""Adaptive Multi-Agent Delegation Engine module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from evren_code.core.multi_agent.file_ownership import FileOwnershipManager
from evren_code.core.multi_agent.models import AgentRole, EngineeringTask, TaskStatus
from evren_code.core.planner import ComplexityAssessment, ComplexityLevel


@dataclass
class DelegationDecision:
    """Delegation decision implementation."""
    should_delegate: bool
    recommended_roles: List[str] = field(default_factory=list)
    estimated_token_cost: int = 0
    estimated_benefit_score: float = 0.0  # 0.0 - 1.0
    reason: str = ""


class AdaptiveDelegationEngine:
    """Adaptive delegation engine implementation."""

    def __init__(
        self,
        workspace_root: Path,
        file_ownership: Optional[FileOwnershipManager] = None,
        max_discussion_rounds: int = 2,
    ) -> None:
        self.workspace_root = workspace_root
        self.file_ownership = file_ownership or FileOwnershipManager(workspace_root=workspace_root)
        self.max_discussion_rounds = max_discussion_rounds
        self.discussion_rounds_count: int = 0

    def evaluate_delegation(
        self,
        assessment: ComplexityAssessment,
        prompt: str,
        token_budget_left: int = 50000,
    ) -> DelegationDecision:
        """Evaluate delegation."""
        if assessment.level == ComplexityLevel.SIMPLE:
            return DelegationDecision(
                should_delegate=False,
                recommended_roles=[],
                estimated_token_cost=0,
                estimated_benefit_score=0.1,
                reason="Basit görev (Simple): Tek ajan doğrudan yürütmeli; delegasyon ek maliyet ve gecikme yaratır.",
            )

        if assessment.level == ComplexityLevel.MEDIUM:
            p_low = prompt.lower()
            roles = []
            if any(k in p_low for k in ("test", "unit", "doğrula", "qa", "broken", "kırılan")):
                roles.append("tester")
            if any(k in p_low for k in ("review", "güvenlik", "incele", "denetle", "security")):
                roles.append("reviewer")

            if roles:
                est_cost = len(roles) * 3500
                benefit = 0.75
                should = token_budget_left > est_cost
                return DelegationDecision(
                    should_delegate=should,
                    recommended_roles=roles,
                    estimated_token_cost=est_cost,
                    estimated_benefit_score=benefit,
                    reason=f"Orta karmaşıklık (Medium): Hedefe yönelik {', '.join(roles)} uzman desteği önerildi.",
                )
            return DelegationDecision(
                should_delegate=False,
                recommended_roles=[],
                estimated_token_cost=0,
                estimated_benefit_score=0.4,
                reason="Orta karmaşıklık (Medium): Ana ajan tek başına çözebilir, delegasyona gerek yok.",
            )

        est_cost = 14000
        benefit = 0.95
        if token_budget_left < est_cost:
            return DelegationDecision(
                should_delegate=False,
                recommended_roles=["developer"],
                estimated_token_cost=est_cost,
                estimated_benefit_score=benefit,
                reason=f"Kalan token bütçesi ({token_budget_left}) çoklu ajan maliyetini ({est_cost}) karşılamıyor; tek ajan modunda devam ediliyor.",
            )

        return DelegationDecision(
            should_delegate=True,
            recommended_roles=["architect", "developer", "tester", "reviewer"],
            estimated_token_cost=est_cost,
            estimated_benefit_score=benefit,
            reason="Karmaşık görev (Complex): Mimari analiz, geliştirme, test ve kod inceleme için çoklu ajan işbölümü önerildi.",
        )

    def acquire_file_ownership(
        self,
        agent_id: str,
        file_paths: List[str],
        exclusive: bool = True,
    ) -> Tuple[bool, List[str]]:
        """Acquire file ownership."""
        blocked_files: List[str] = []
        task_id = f"task_{agent_id}"

        for fp in file_paths:
            try:
                ok, err = self.file_ownership.acquire_locks(
                    task_id=task_id,
                    agent_id=agent_id,
                    write_files=[fp] if exclusive else [],
                    read_files=[fp] if not exclusive else [],
                )
                if not ok:
                    blocked_files.append(fp)
            except Exception:
                blocked_files.append(fp)

        if blocked_files:
            self.file_ownership.release_locks(task_id=task_id)
            return False, blocked_files

        return True, []

    def release_file_ownership(self, agent_id: str, file_paths: Optional[List[str]] = None) -> None:
        """Release file ownership."""
        task_id = f"task_{agent_id}"
        self.file_ownership.release_locks(task_id=task_id)

    def validate_subagent_output(
        self,
        role: str,
        output: str,
        assigned_files: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Validate subagent output."""
        if not output or not output.strip():
            return False, f"Alt-ajan ({role}) boş çıktı üretti."

        out_lower = output.lower()
        if "hata:" in out_lower and "başarılı" not in out_lower:
            return False, f"Alt-ajan ({role}) çıktısında çözülmemiş hata raporlandı: {output[:150]}"

        if any(term in out_lower for term in ("c:\\windows", "program files", "ollama\\mvn.cmd", "setx path")):
            return False, f"Alt-ajan ({role}) güvenlik sınırlarını aşan işlem bildirdi. Çıktı reddedildi."

        return True, "Alt-ajan çıktısı doğrulandı."

    def check_debate_limit(self) -> bool:
        """Check debate limit."""
        self.discussion_rounds_count += 1
        return self.discussion_rounds_count > self.max_discussion_rounds
