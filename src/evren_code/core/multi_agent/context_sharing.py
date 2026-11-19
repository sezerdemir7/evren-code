"""Inter-Agent Selective Context Sharing (Task Context Sharing Engine)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from evren_code.core.multi_agent.models import EngineeringTask
from evren_code.core.multi_agent.task_graph import TaskGraph
from evren_code.core.repo_intelligence.engine import RepoIntelligenceEngine
from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine


def _normalize_contract_key(key: str) -> str:
    """Normalize contract key."""
    return re.sub(r"[\s_\-]+", "", key).lower()


def _find_contract_in_result(consumed_name: str, result: Any) -> str:
    """Find contract in result."""
    if not result:
        return ""
    norm = _normalize_contract_key(consumed_name)

    if result.artifacts:
        if consumed_name in result.artifacts:
            return str(result.artifacts[consumed_name])
        for k, v in result.artifacts.items():
            if _normalize_contract_key(str(k)) == norm:
                return str(v)

    output = result.output or ""
    if not output:
        return ""

    header_pattern = rf"(?:#{{1,4}}\s*.*{re.escape(consumed_name)}.*?\n)([\s\S]*?)(?=\n#{{1,4}}|\Z)"
    match = re.search(header_pattern, output, re.IGNORECASE)
    if match and len(match.group(1).strip()) > 10:
        return match.group(0).strip()[:1800]

    code_blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", output)
    for cb in code_blocks:
        if norm in _normalize_contract_key(cb):
            return f"```{cb.strip()}```"

    return output[:1200]


class ContextSharingEngine:
    """Context sharing engine implementation."""

    def __init__(
        self,
        repo_intelligence: Optional[RepoIntelligenceEngine] = None,
        symbol_engine: Optional[CodebaseSymbolEngine] = None,
    ) -> None:
        self.repo_intelligence = repo_intelligence
        self.symbol_engine = symbol_engine

    def prepare_task_context(
        self,
        task: EngineeringTask,
        task_graph: TaskGraph,
    ) -> Dict[str, Any]:
        """Prepare task context."""
        context: Dict[str, Any] = {
            "task_id": task.task_id,
            "dependency_artifacts": {},
            "symbol_summaries": {},
            "module_info": {},
        }

        for dep_id in task.dependencies:
            dep_task = task_graph.get_task(dep_id)
            if dep_task and dep_task.result:
                context["dependency_artifacts"][dep_id] = (
                    f"Başarı: {dep_task.result.success}\n"
                    f"Değiştirilen Dosyalar: {', '.join(dep_task.result.modified_files)}\n"
                    f"Çıktı Özeti: {dep_task.result.output[:1500]}"
                )

        context["contracts"] = {}
        for raw_consumed in getattr(task, "consumes_contracts", []):
            consumed_name = str(raw_consumed).strip()
            if not consumed_name:
                continue
            norm_consumed = _normalize_contract_key(consumed_name)
            for dep_id in task.dependencies:
                dep_task = task_graph.get_task(dep_id)
                if not dep_task:
                    continue
                provided_raw = getattr(dep_task, "provides_contracts", [])
                provided_norm_map = {_normalize_contract_key(str(p)): str(p) for p in provided_raw if str(p).strip()}
                if norm_consumed in provided_norm_map:
                    contract_val = ""
                    if dep_task.result and dep_task.result.modified_files and self.symbol_engine:
                        ast_snippets = []
                        for mf in dep_task.result.modified_files[:3]:
                            try:
                                symbols = self.symbol_engine.get_file_symbols(mf)
                                if symbols:
                                    s_names = [f"{s.kind} {s.name}" for s in symbols if s.name]
                                    if s_names:
                                        ast_snippets.append(f"// {mf} Sembolleri: {', '.join(s_names[:10])}")
                            except Exception:
                                pass
                        if ast_snippets:
                            contract_val = "\n".join(ast_snippets)

                    if not contract_val and dep_task.result:
                        contract_val = _find_contract_in_result(consumed_name, dep_task.result)
                    context["contracts"][consumed_name] = {
                        "provider_task_id": dep_id,
                        "provider_title": dep_task.title,
                        "domain": getattr(dep_task, "domain", "general"),
                        "contract_data": contract_val or f"[Sözleşme '{consumed_name}' upstream görev tarafından tanımlandı]",
                    }
                    break

        all_files = list(task.assigned_files) + list(task.read_only_files)
        if self.symbol_engine and all_files:
            for fpath in all_files[:5]:
                try:
                    symbols = self.symbol_engine.get_file_symbols(fpath)
                    if symbols:
                        sym_names = [s.name for s in symbols[:15]]
                        context["symbol_summaries"][fpath] = sym_names
                except Exception:
                    pass

        if self.repo_intelligence and all_files:
            for fpath in all_files[:3]:
                try:
                    overview = self.repo_intelligence.get_overview()
                    if overview:
                        context["module_info"]["project_name"] = overview.project_name
                        context["module_info"]["primary_language"] = overview.primary_language
                except Exception:
                    pass

        return context
