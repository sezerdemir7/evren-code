"""Tools module."""

from __future__ import annotations

import json
from typing import Optional

from evren_code.core.repo_intelligence.engine import RepoIntelligenceEngine
from evren_code.tools.registry import ToolRegistry


def register_repo_intelligence_tools(
    registry: ToolRegistry,
    engine: RepoIntelligenceEngine,
) -> None:
    """Register repo intelligence tools."""

    def get_project_overview() -> str:
        return engine.get_overview()

    def get_project_modules() -> str:
        pmap = engine.get_project_map()
        if not pmap.submodules:
            if pmap.root_module:
                rm = pmap.root_module
                return (
                    f"Tekil Proje: `{rm.name}`\n"
                    f"- Diller: {', '.join(rm.languages)}\n"
                    f"- Frameworkler: {', '.join(rm.frameworks)}\n"
                    f"- Derleme Sistemi: {rm.build_system}\n"
                    f"- Manifestler: {', '.join(rm.dependency_manifests)}"
                )
            return "Projede belirgin bir modül yapısı tespit edilemedi."

        lines = [f"### Tespit Edilen Modüller ({len(pmap.submodules)} adet):"]
        for idx, sm in enumerate(pmap.submodules, 1):
            langs = ", ".join(sm.languages) if sm.languages else "Belirtilmemiş"
            fws = f" ({', '.join(sm.frameworks)})" if sm.frameworks else ""
            lines.append(f"{idx}. `{sm.path}/` [{sm.module_type}]: {langs}{fws} - {sm.build_system}")
            if sm.test_commands:
                lines.append(f"   Test: `{sm.test_commands[0].command}`")
        return "\n".join(lines)

    def get_module_details(module_path: str) -> str:
        return engine.get_module_details(module_path=module_path)

    def inspect_project(query: Optional[str] = None) -> str:
        if not query or not query.strip():
            return engine.get_overview()
        return engine.query_targeted(query=query)

    def get_important_files(category: Optional[str] = None) -> str:
        files = engine.get_important_files(category=category)
        if not files:
            cat_msg = f"'{category}' kategorisinde " if category else ""
            return f"Projede {cat_msg}önemli bir dosya bulunamadı."

        lines = [f"### Kritik Proje Dosyaları ({len(files)} adet):"]
        for f in files[:40]:
            lines.append(f"- `{f.path}` [{f.category}]: {f.description} ({f.size_bytes} bayt)")
        if len(files) > 40:
            lines.append(f"... ve {len(files) - 40} dosya daha")
        return "\n".join(lines)

    def refresh_project_map(force: bool = False) -> str:
        _, diff = engine.refresh(force=force)
        mode = diff.get("mode", "incremental")
        if mode == "full_force":
            return "✔ Proje haritası tam yeniden tarama (force rescan) ile güncellendi ve önbelleğe kaydedildi."
        elif mode == "initial_scan":
            return "✔ İlk proje taraması tamamlandı ve `.evren/project_map.json` içine kaydedildi."
        else:
            add_c = diff.get("added_count", 0)
            mod_c = diff.get("modified_count", 0)
            del_c = diff.get("deleted_count", 0)
            reanalyzed = diff.get("reanalyzed", False)
            return (
                f"✔ Artımlı proje taraması tamamlandı. "
                f"Eklenen: {add_c}, Değişen: {mod_c}, Silinen: {del_c}. "
                f"({'Kritik konfigürasyon değişikliği nedeniyle modül analizi yenilendi' if reanalyzed else 'Harita güncel'})."
            )

    registry.register(
        name="get_project_overview",
        description="Returns a lightweight intelligence report summarizing tech stack, architecture (monorepo/single), modules, and build/test commands.",
        parameters={"type": "object", "properties": {}},
        handler=get_project_overview,
        requires_approval=False,
    )

    registry.register(
        name="get_project_modules",
        description="Lists independent sub-services and modules across the workspace.",
        parameters={"type": "object", "properties": {}},
        handler=get_project_modules,
        requires_approval=False,
    )

    registry.register(
        name="get_module_details",
        description="Retrieves languages, dependencies, entry points, and commands for a specific module (e.g. 'backend', 'frontend').",
        parameters={
            "type": "object",
            "properties": {
                "module_path": {
                    "type": "string",
                    "description": "Relative module directory path (e.g. 'frontend' or 'backend').",
                }
            },
            "required": ["module_path"],
        },
        handler=get_module_details,
        requires_approval=False,
    )

    registry.register(
        name="inspect_project",
        description="Queries repo intelligence engine for relevant modules, key files, and commands. Returns general overview if query is empty.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional search term, technology, module name, or file type (e.g. 'auth', 'database', 'spring', 'test').",
                }
            },
        },
        handler=inspect_project,
        requires_approval=False,
    )

    registry.register(
        name="get_important_files",
        description="Lists key project files (READMEs, manifests, dockerfiles, CI/CD, configs, entry points) with optional category filter.",
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter category ('readme', 'dependency_manifest', 'build_config', 'test_config', 'entry_point', 'main_config').",
                }
            },
        },
        handler=get_important_files,
        requires_approval=False,
    )

    registry.register(
        name="refresh_project_map",
        description="Re-scans or incrementally updates the project map cache.",
        parameters={
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "If True, ignores existing cache and performs a full scan from scratch.",
                }
            },
        },
        handler=refresh_project_map,
        requires_approval=False,
    )
