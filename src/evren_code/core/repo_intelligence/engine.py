"""Engine module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evren_code.config import AppConfig
from evren_code.core.repo_intelligence.detector import ProjectDetector
from evren_code.core.repo_intelligence.models import (
    DevCommand,
    ImportantFile,
    ModuleInfo,
    ProjectMap,
    RepoMetadata,
    ScanLimits,
)
from evren_code.core.repo_intelligence.persistence import ProjectMapStore
from evren_code.core.repo_intelligence.scanner import RepoScanner, ScanResult


class RepoIntelligenceEngine:
    """Repo intelligence engine implementation."""

    def __init__(
        self,
        workspace_root: Path,
        config: Optional[AppConfig] = None,
        limits: Optional[ScanLimits] = None,
    ):
        self.workspace_root = workspace_root.resolve()
        self.config = config
        self.limits = limits or ScanLimits()
        self.store = ProjectMapStore(self.workspace_root)
        self.detector = ProjectDetector(self.workspace_root)
        self._cached_map: Optional[ProjectMap] = None

    def get_scanner(self) -> RepoScanner:
        custom_ignores = []
        if self.config and hasattr(self.config, "tools") and hasattr(self.config.tools, "ignored_dirs"):
            custom_ignores.extend(self.config.tools.ignored_dirs)
        return RepoScanner(self.workspace_root, limits=self.limits, custom_ignores=custom_ignores)

    def get_project_map(self, force_refresh: bool = False) -> ProjectMap:
        """Get project map."""
        if not force_refresh and self._cached_map is not None:
            return self._cached_map

        if not force_refresh:
            stored = self.store.load()
            if stored is not None:
                self._cached_map = stored
                return stored

        return self.rescan_and_analyze()

    def rescan_and_analyze(self) -> ProjectMap:
        """Rescan and analyze."""
        scanner = self.get_scanner()
        scan_result = scanner.scan()

        root_module, submodules, metadata = self.detector.detect_project_structure(scan_result)
        important_files = self.detector.classify_important_files(scan_result)

        all_commands: List[DevCommand] = []
        if root_module:
            all_commands.extend(root_module.test_commands)
            all_commands.extend(root_module.build_commands)
            all_commands.extend(root_module.dev_commands)
        for sm in submodules:
            all_commands.extend(sm.test_commands)
            all_commands.extend(sm.build_commands)
            all_commands.extend(sm.dev_commands)

        seen_cmds = set()
        deduped_commands: List[DevCommand] = []
        for cmd in all_commands:
            key = (cmd.name, cmd.command)
            if key not in seen_cmds:
                seen_cmds.add(key)
                deduped_commands.append(cmd)

        file_hashes: Dict[str, str] = {}
        for rel_p, finfo in scan_result.files.items():
            file_hashes[rel_p] = f"{finfo.mtime}:{finfo.size_bytes}"

        pmap = ProjectMap(
            metadata=metadata,
            root_module=root_module,
            submodules=submodules,
            directory_tree=sorted(scan_result.directories),
            important_files=important_files,
            dev_commands=deduped_commands,
            file_hashes=file_hashes,
        )

        self._cached_map = pmap
        self.store.save(pmap)
        return pmap

    def refresh(self, force: bool = False) -> Tuple[ProjectMap, Dict[str, Any]]:
        """Refresh."""
        if force:
            pmap = self.rescan_and_analyze()
            return pmap, {"mode": "full_force", "reanalyzed": True}

        cached = self.store.load()
        if not cached:
            pmap = self.rescan_and_analyze()
            return pmap, {"mode": "initial_scan", "reanalyzed": True}

        scanner = self.get_scanner()
        current_scan = scanner.scan()

        added, modified, deleted, requires_reanalysis = self.store.detect_changes(cached, current_scan)

        diff_summary = {
            "mode": "incremental",
            "added_count": len(added),
            "modified_count": len(modified),
            "deleted_count": len(deleted),
            "reanalyzed": requires_reanalysis,
            "added": added[:10],
            "modified": modified[:10],
            "deleted": deleted[:10],
        }

        if not added and not modified and not deleted:
            self._cached_map = cached
            return cached, diff_summary

        if requires_reanalysis:
            pmap = self.rescan_and_analyze()
            return pmap, diff_summary

        new_hashes: Dict[str, str] = {}
        for rel_p, finfo in current_scan.files.items():
            new_hashes[rel_p] = f"{finfo.mtime}:{finfo.size_bytes}"

        cached.file_hashes = new_hashes
        cached.metadata.total_files_scanned = current_scan.total_files_scanned
        cached.metadata.total_dirs_scanned = current_scan.total_dirs_scanned
        cached.metadata.total_bytes_scanned = current_scan.total_bytes_scanned
        cached.metadata.is_partial = current_scan.is_partial
        cached.metadata.partial_reasons = current_scan.partial_reasons
        cached.directory_tree = sorted(current_scan.directories)
        cached.important_files = self.detector.classify_important_files(current_scan)

        self._cached_map = cached
        self.store.save(cached)
        return cached, diff_summary

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def get_overview(self) -> str:
        """Get overview."""
        pmap = self.get_project_map()
        meta = pmap.metadata

        lines: List[str] = ["## Proje Mimarisi ve Teknoloji Özeti (Repository Intelligence)"]
        arch_type = "Çoklu Modül / Monorepo" if meta.is_monorepo else "Tekil Proje"
        lines.append(f"- **Proje Adı:** `{meta.project_name}` ({arch_type})")
        if meta.detected_languages:
            lines.append(f"- **Programlama Dilleri:** {', '.join(meta.detected_languages)}")
        if meta.detected_frameworks:
            lines.append(f"- **Kütüphane / Frameworkler:** {', '.join(meta.detected_frameworks)}")

        if pmap.submodules:
            lines.append("- **Tespit Edilen Modüller:**")
            for sm in pmap.submodules:
                sm_tech = []
                if sm.languages:
                    sm_tech.append("/".join(sm.languages))
                if sm.frameworks:
                    sm_tech.append(", ".join(sm.frameworks))
                tech_str = f" ({'; '.join(sm_tech)})" if sm_tech else ""
                lines.append(f"  - `{sm.path}/` [{sm.module_type}]: {sm.build_system}{tech_str}")
        elif pmap.root_module:
            rm = pmap.root_module
            lines.append(f"- **Derleme Sistemi:** {rm.build_system}")
            if rm.entry_points:
                lines.append(f"- **Giriş Noktaları:** {', '.join(rm.entry_points[:3])}")

        test_cmds = [c.command for c in pmap.dev_commands if c.name == "test"]
        build_cmds = [c.command for c in pmap.dev_commands if c.name == "build"]
        dev_cmds = [c.command for c in pmap.dev_commands if c.name in ("dev", "start")]

        cmd_items = []
        if test_cmds:
            cmd_items.append(f"Test: `{test_cmds[0]}`")
        if build_cmds:
            cmd_items.append(f"Build: `{build_cmds[0]}`")
        if dev_cmds:
            cmd_items.append(f"Dev: `{dev_cmds[0]}`")
        if cmd_items:
            lines.append(f"- **Geliştirme Komutları:** {' | '.join(cmd_items)}")

        cfg_files = [
            f.path for f in pmap.important_files
            if f.category in ("dependency_manifest", "main_config", "build_config")
        ][:6]
        if cfg_files:
            lines.append(f"- **Ana Yapılandırma Dosyaları:** {', '.join([f'`{p}`' for p in cfg_files])}")

        if meta.is_partial:
            lines.append(f"- **Tarama Durumu:** Kısmi tarama yapıldı ({'; '.join(meta.partial_reasons)})")

        return "\n".join(lines)

    def get_module_details(self, module_path: str) -> str:
        """Get module details."""
        pmap = self.get_project_map()
        target_path = module_path.strip().replace("\\", "/").rstrip("/") or "."

        matched: Optional[ModuleInfo] = None
        if pmap.root_module and pmap.root_module.path == target_path:
            matched = pmap.root_module
        else:
            for sm in pmap.submodules:
                if sm.path == target_path or sm.name == target_path:
                    matched = sm
                    break

        if not matched:
            available = [m.path for m in pmap.submodules]
            if pmap.root_module:
                available.insert(0, pmap.root_module.path)
            return f"Hata: '{module_path}' yoluyla eşleşen bir modül bulunamadı. Mevcut modüller: {available}"

        lines = [f"### Modül Detayları: `{matched.name}` (`{matched.path}`)", f"- **Modül Türü:** {matched.module_type}"]
        if matched.languages:
            lines.append(f"- **Diller:** {', '.join(matched.languages)}")
        if matched.frameworks:
            lines.append(f"- **Frameworkler:** {', '.join(matched.frameworks)}")
        lines.append(f"- **Derleme Sistemi:** {matched.build_system}")

        if matched.dependency_manifests:
            lines.append(f"- **Bağımlılık Tanımları:** {', '.join(matched.dependency_manifests)}")
        if matched.entry_points:
            lines.append(f"- **Giriş Noktaları:** {', '.join(matched.entry_points)}")

        if matched.test_commands:
            lines.append("- **Test Komutları:**")
            for c in matched.test_commands:
                lines.append(f"  - `{c.command}` ({c.source})")

        if matched.build_commands:
            lines.append("- **Derleme Komutları:**")
            for c in matched.build_commands:
                lines.append(f"  - `{c.command}` ({c.source})")

        if matched.dev_commands:
            lines.append("- **Geliştirme / Sunucu Komutları:**")
            for c in matched.dev_commands:
                lines.append(f"  - `{c.command}` ({c.source})")

        return "\n".join(lines)

    def query_targeted(self, query: str) -> str:
        """Query targeted."""
        pmap = self.get_project_map()
        q_low = query.strip().lower()
        if not q_low:
            return self.get_overview()

        results: List[str] = [f"### Hedefli Proje Araması: '{query}'"]

        candidate_modules = list(pmap.submodules)
        if pmap.root_module:
            candidate_modules.append(pmap.root_module)

        matched_mods = []
        for sm in candidate_modules:
            if (
                q_low in sm.name.lower()
                or q_low in sm.path.lower()
                or any(q_low in l.lower() for l in sm.languages)
                or any(q_low in f.lower() for f in sm.frameworks)
            ):
                matched_mods.append(sm)

        if matched_mods:
            results.append("- **Eşleşen Modüller:**")
            for sm in matched_mods:
                p_label = f"`{sm.path}/`" if sm.path != "." else f"`{sm.name}` (Kök)"
                results.append(f"  - {p_label} ({', '.join(sm.languages + sm.frameworks)})")

        matched_files = [
            f for f in pmap.important_files
            if q_low in f.path.lower() or q_low in f.category.lower() or q_low in f.description.lower()
        ][:15]

        if matched_files:
            results.append("- **İlgili Kritik Dosyalar:**")
            for f in matched_files:
                results.append(f"  - `{f.path}` [{f.category}]: {f.description}")

        matched_cmds = [
            c for c in pmap.dev_commands
            if q_low in c.name.lower() or q_low in c.command.lower() or q_low in c.source.lower()
        ]
        if matched_cmds:
            results.append("- **İlgili Komutlar:**")
            for c in matched_cmds:
                results.append(f"  - `{c.command}` [{c.name}]: {c.source}")

        if len(results) == 1:
            return f"'{query}' aramasıyla eşleşen modül, kritik dosya veya komut bulunamadı. Genel proje özeti için `get_project_overview` çağırabilirsiniz."

        return "\n".join(results)

    def get_important_files(self, category: Optional[str] = None) -> List[ImportantFile]:
        """Get important files."""
        pmap = self.get_project_map()
        if not category:
            return list(pmap.important_files)
        cat_low = category.strip().lower()
        return [f for f in pmap.important_files if f.category.lower() == cat_low]

    def measure_footprint(self) -> Dict[str, Any]:
        """Measure footprint."""
        overview = self.get_overview()
        overview_chars = len(overview)
        overview_est_tokens = max(1, overview_chars // 4)

        pmap = self.get_project_map()
        modules_details = {}
        for sm in pmap.submodules:
            det = self.get_module_details(sm.path)
            modules_details[sm.path] = {
                "chars": len(det),
                "est_tokens": max(1, len(det) // 4),
            }

        full_json = json.dumps(pmap.to_dict(), ensure_ascii=False)

        return {
            "level1_overview_chars": overview_chars,
            "level1_overview_est_tokens": overview_est_tokens,
            "submodule_count": len(pmap.submodules),
            "modules_details": modules_details,
            "total_files_scanned": pmap.metadata.total_files_scanned,
            "total_dirs_scanned": pmap.metadata.total_dirs_scanned,
            "full_json_size_bytes": len(full_json.encode("utf-8")),
        }
