"""Tool Registry module."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


ToolHandler = Callable[..., str]
PreviewBuilder = Callable[[Dict[str, Any]], Optional[str]]


CORE_CODING_TOOLS: Set[str] = {
    "read_file",
    "read_multiple_files",
    "write_file",
    "append_file",
    "edit_file",
    "multi_edit_file",
    "write_multiple_files",
    "get_project_tree",
    "list_directory",
    "find_files",
    "search_code",
    "semantic_search_code",
    "get_repo_map",
    "get_project_overview",
    "skeletonize_code",
    "search_symbols",
    "analyze_change_impact",
    "find_related_tests",
    "run_verification",
    "git_info",
    "run_command",
    "get_background_output",
    "stop_background_process",
    "validate_web_contract",
    "fetch_url",
    "http_request",
    "create_task_plan",
    "update_plan_step",
    "get_task_plan",
    "lsp_get_diagnostics",
    "ask_user",
    "remember_project_fact",
    "get_project_memory",
}


@dataclass
class ToolDefinition:
    """Tool definition implementation."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: ToolHandler
    requires_approval: bool = False
    preview_builder: Optional[PreviewBuilder] = None

    def to_openai_schema(self) -> Dict[str, Any]:
        """To openai schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Tool registry implementation."""

    def __init__(self, require_approval_for: Optional[List[str]] = None) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._approval_list = set(require_approval_for or [])

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: ToolHandler,
        requires_approval: Optional[bool] = None,
        preview_builder: Optional[PreviewBuilder] = None,
    ) -> None:
        """Register."""
        need_approval = (
            requires_approval
            if requires_approval is not None
            else (name in self._approval_list)
        )
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            requires_approval=need_approval,
            preview_builder=preview_builder,
        )

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def select_active_tools(
        self,
        context_text: str = "",
        has_active_background: bool = False,
        sticky_set: Optional[Set[str]] = None,
        complexity_level: Optional[str] = None,
    ) -> Set[str]:
        """Select active tools."""
        active = set(sticky_set) if sticky_set else set(CORE_CODING_TOOLS)
        active.update(CORE_CODING_TOOLS)
        active.add("delegate_subagent")
        active.add("lsp_find_definition")
        active.add("lsp_find_references")

        c_lower = context_text.lower()
        comp_lower = (complexity_level or "").lower()
        is_non_trivial = comp_lower in ("medium", "complex")

        if is_non_trivial or any(
            w in c_lower
            for w in ("sembol", "symbol", "etki", "impact", "referans", "caller", "callee", "refactor", "hata", "bug", "fix", "düzelt", "test")
        ):
            active.update({
                "search_symbols",
                "find_symbol_definition",
                "find_symbol_references",
                "analyze_change_impact",
                "find_related_tests",
                "get_file_symbols",
                "index_codebase",
            })

        if is_non_trivial or any(
            w in c_lower
            for w in ("mimari", "architecture", "overview", "genel bakış", "repo_map", "modül", "inspect", "analiz", "incele", "proje")
        ):
            active.update({
                "get_repo_map",
                "get_project_overview",
                "get_project_modules",
                "get_module_details",
                "inspect_project",
                "get_important_files",
                "refresh_project_map",
            })

        if any(w in c_lower for w in ("görsel", "resim", "ekran görüntüsü", "screenshot", "image", "png", "jpg", "tasarım")):
            active.update({
                "analyze_image",
                "capture_web_screenshot",
                "inspect_web_page_visually",
            })

        if any(w in c_lower for w in ("profil", "performans", "iskelet", "skeleton", "hotspot", "churn", "bottleneck")):
            active.update({
                "skeletonize_code",
                "profile_code_performance",
            })

        if any(w in c_lower for w in ("doğrula", "verify", "verification", "kabul kriteri")):
            active.update({
                "run_verification",
                "get_verification_report",
                "get_verification_history",
            })

        # 6. Semantik Arama
        if any(w in c_lower for w in ("semantik", "anlamsal", "benzer kod")):
            active.add("semantic_search_code")

        if has_active_background:
            active.update({"get_background_output", "stop_background_process"})

        known_specialized = {
            "search_symbols", "find_symbol_definition", "find_symbol_references",
            "analyze_change_impact", "find_related_tests", "get_file_symbols", "index_codebase",
            "get_repo_map", "get_project_overview", "get_project_modules", "get_module_details",
            "inspect_project", "get_important_files", "refresh_project_map",
            "analyze_image", "capture_web_screenshot", "inspect_web_page_visually",
            "skeletonize_code", "profile_code_performance",
            "run_verification", "get_verification_report", "get_verification_history",
            "semantic_search_code",
        }
        for name in self._tools:
            if name not in known_specialized:
                active.add(name)

        return active

    def get_openai_schemas(
        self,
        tool_names: Optional[Set[str] | List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get openai schemas."""
        if tool_names is not None:
            names_set = set(tool_names)
            tools = [tool for name, tool in self._tools.items() if name in names_set]
        else:
            tools = list(self._tools.values())
        tools.sort(key=lambda t: t.name)
        return [tool.to_openai_schema() for tool in tools]

    def build_fallback_prompt_instructions(
        self,
        tool_names: Optional[Set[str] | List[str]] = None,
    ) -> str:
        """Generates fallback XML tool call instructions for models without native tool support."""
        lines = [
            "## Tool Calling Format (XML Fallback)",
            "To invoke a tool, strictly output the call within the XML block as shown below:",
            "<tool_call>",
            '{"name": "tool_name", "arguments": {"param1": "value1"}}',
            "</tool_call>",
            "",
            "### Available Tools:",
        ]
        names_set = set(tool_names) if tool_names is not None else None
        target_tools = [tool for name, tool in self._tools.items() if (names_set is None or name in names_set)]
        for tool in target_tools:
            params_str = json.dumps(tool.parameters, ensure_ascii=False)
            lines.append(f"- **{tool.name}**: {tool.description}\n  Schema: `{params_str}`")
        return "\n".join(lines)

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute."""
        tool = self._tools.get(name)
        if not tool:
            available = ", ".join(self._tools.keys())
            return f"Hata: '{name}' adında bir araç bulunamadı. Mevcut araçlar: {available}"

        if "_raw_unparsed" in arguments:
            unparsed = str(arguments["_raw_unparsed"])
            hint = ""
            if name == "write_multiple_files":
                hint = "\nİpucu: Toplu dosya üretimi token tavanına takılmış olabilir. Lütfen dosyaları tek tek `write_file` ile parça parça oluşturun."
            return (
                f"Hata: '{name}' aracı için gönderilen JSON argümanları ayrıştırılamadı: "
                f"{unparsed}. Lütfen geçerli bir JSON nesnesi gönderin.{hint}"
            )

        try:
            sig = inspect.signature(tool.handler)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if not has_var_kw:
                filtered_args = {
                    k: v for k, v in arguments.items() if k in sig.parameters
                }
            else:
                filtered_args = arguments
            return tool.handler(**filtered_args)
        except TypeError as exc:
            return f"Hata: '{name}' aracı eksik veya hatalı parametre ile çağrıldı: {exc}"
        except Exception as exc:
            return f"Hata: '{name}' aracı çalışırken beklenmeyen bir istisna oluştu: {exc}"
