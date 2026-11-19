"""Corrector module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine
from evren_code.core.verification.models import (
    CorrectionAttempt,
    FailureCategory,
    TestFailure,
)
from evren_code.core.verification.validator import ChangeValidator


class CourseCorrectionEngine:
    """Course correction engine implementation."""

    def __init__(
        self,
        workspace_root: Path,
        symbol_engine: Optional[CodebaseSymbolEngine] = None,
        validator: Optional[ChangeValidator] = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.symbol_engine = symbol_engine
        self.validator = validator or ChangeValidator(self.workspace_root)

    def formulate_correction_plan(
        self,
        failure: TestFailure,
        changed_files: List[str],
        attempt_number: int = 1,
    ) -> Dict[str, Any]:
        """Formulate correction plan."""
        clf = failure.classification
        category = clf.category if clf else FailureCategory.UNKNOWN

        symbol_context: List[str] = []
        if self.symbol_engine and failure.file_path:
            try:
                syms = self.symbol_engine.graph.get_file_symbols(failure.file_path)
                for s in syms[:5]:
                    symbol_context.append(f"- Sembol `{s.name}` ({s.symbol_type.value}) satır {s.start_line}-{s.end_line}")
            except Exception:
                pass

        prompt_lines = [
            f"### [OTONOM DÜZELTME DÖNGÜSÜ - DENEME #{attempt_number}]",
            f"**Başarısız Test / Adım:** `{failure.test_name}`",
            f"**Hata Kategorisi:** {category.value} (Güven: {clf.confidence if clf else 0.5})",
            f"**Hata Açıklaması:** {failure.failure_message.strip()}",
        ]

        if failure.file_path:
            prompt_lines.append(f"**İlgili Dosya:** `{failure.file_path}`" + (f":{failure.line_number}" if failure.line_number else ""))

        if clf and clf.evidence:
            prompt_lines.append(f"**Kanıt:** {clf.evidence}")

        if symbol_context:
            prompt_lines.append("\n**İlgili Sembol Bağlamı:**\n" + "\n".join(symbol_context))

        prompt_lines.extend([
            "\n**KRİTİK KURALLAR:**",
            "1. TESTLERİ SİLMEYİN VEYA ZAYIFLATMAYIN (@skip, assert silme yasaktır).",
            "2. Yalnızca hatanın kaynağı olan koda cerrahi (minimal) müdahale yapın.",
            "3. Kapsam dışındaki dosyalara dokunmayın.",
            f"4. Tavsiye Edilen Aksiyon: {clf.suggested_action if clf else 'Kök nedeni inceleyin ve düzeltin.'}",
        ])

        plan_desc = f"{category.value} hatasını gidermek için cerrahi kod düzeltmesi."
        return {
            "attempt_number": attempt_number,
            "failure_test": failure.test_name,
            "category": category.value,
            "correction_prompt": "\n".join(prompt_lines),
            "plan_description": plan_desc,
            "target_files": list(set(changed_files + ([failure.file_path] if failure.file_path else []))),
        }

    def post_correction_update(self, modified_files: List[str]) -> None:
        """Post correction update."""
        if not self.symbol_engine:
            return

        for mf in modified_files:
            try:
                self.symbol_engine.incremental_index(force=True)
                break
            except Exception:
                pass
