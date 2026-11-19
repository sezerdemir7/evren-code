"""Persistence module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from evren_code.core.repo_intelligence.models import ProjectMap
from evren_code.core.repo_intelligence.scanner import ScanResult


class ProjectMapStore:
    """Project map store implementation."""

    CACHE_FILE_NAME = "project_map.json"
    CURRENT_VERSION = "2.1"

    KEY_MANIFEST_NAMES = {
        "package.json", "tsconfig.json", "pnpm-workspace.yaml", "pnpm-lock.yaml", "yarn.lock",
        "pyproject.toml", "requirements.txt", "setup.py", "Pipfile", "poetry.lock", "uv.lock",
        "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
        "Cargo.toml", "go.mod", "composer.json", "pubspec.yaml",
    }

    def __init__(self, workspace_root: Path, cache_path: Optional[Path] = None):
        self.workspace_root = workspace_root.resolve()
        self.cache_file = cache_path or (self.workspace_root / ".evren" / self.CACHE_FILE_NAME)

    def load(self) -> Optional[ProjectMap]:
        """Load."""
        if not self.cache_file.is_file():
            return None

        try:
            raw = self.cache_file.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None

            meta = data.get("metadata", {})
            if meta.get("scan_version") != self.CURRENT_VERSION:
                return None

            cached_root = meta.get("workspace_root", "")
            if cached_root and Path(cached_root).resolve() != self.workspace_root:
                return None

            return ProjectMap.from_dict(data)
        except Exception:
            return None

    def save(self, project_map: ProjectMap) -> bool:
        """Save."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = project_map.to_dict()
            content = json.dumps(data, indent=2, ensure_ascii=False)
            self.cache_file.write_text(content, encoding="utf-8")
            return True
        except OSError:
            return False

    def detect_changes(
        self,
        cached_map: ProjectMap,
        current_scan: ScanResult,
    ) -> Tuple[List[str], List[str], List[str], bool]:
        """Detect changes."""
        cached_hashes = cached_map.file_hashes or {}
        current_hashes: Dict[str, str] = {}
        for rel_p, finfo in current_scan.files.items():
            current_hashes[rel_p] = f"{finfo.mtime}:{finfo.size_bytes}"

        cached_set = set(cached_hashes.keys())
        current_set = set(current_hashes.keys())

        added_files = sorted(list(current_set - cached_set))
        deleted_files = sorted(list(cached_set - current_set))
        modified_files: List[str] = []

        common_files = cached_set.intersection(current_set)
        for f in sorted(list(common_files)):
            if cached_hashes[f] != current_hashes[f]:
                modified_files.append(f)

        requires_reanalysis = False
        all_changed = set(added_files + deleted_files + modified_files)
        for cf in all_changed:
            name = Path(cf).name
            if name in self.KEY_MANIFEST_NAMES:
                requires_reanalysis = True
                break

        return added_files, modified_files, deleted_files, requires_reanalysis

    def clear(self) -> bool:
        """Clear."""
        try:
            if self.cache_file.is_file():
                self.cache_file.unlink()
            return True
        except OSError:
            return False
