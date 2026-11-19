"""Models module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScanLimits:
    """Scan limits implementation."""

    max_depth: int = 8
    max_files: int = 5000
    max_file_size_bytes: int = 500_000  # 500 KB limit for reading manifest/config files
    max_total_scan_bytes: int = 50_000_000  # 50 MB total read limit across files


@dataclass
class DevCommand:
    """Dev command implementation."""

    name: str  # "test", "build", "dev", "lint", "format"
    command: str  # "pytest", "npm test", "./gradlew test"
    source: str  # "package.json:scripts.test", "pyproject.toml:pytest", "ecosystem_convention"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DevCommand:
        return cls(
            name=data.get("name", ""),
            command=data.get("command", ""),
            source=data.get("source", ""),
            description=data.get("description", ""),
        )


@dataclass
class ImportantFile:
    """Important file implementation."""

    path: str
    category: str  # "readme", "dependency_manifest", "build_config", "test_config", "env_example", "docker_config", "ci_cd_config", "entry_point", "main_config"
    description: str = ""
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImportantFile:
        return cls(
            path=data.get("path", ""),
            category=data.get("category", ""),
            description=data.get("description", ""),
            size_bytes=data.get("size_bytes", 0),
        )


@dataclass
class ModuleInfo:
    """Module info implementation."""

    name: str
    path: str
    module_type: str  # "root", "backend", "frontend", "library", "service", "cli", "unknown"
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    build_system: str = "unknown"  # "maven", "gradle", "npm", "pnpm", "yarn", "poetry", "pip", "cargo", "go", "dotnet"
    dependency_manifests: List[str] = field(default_factory=list)  # ["package.json", "requirements.txt"]
    entry_points: List[str] = field(default_factory=list)  # ["src/main.py", "src/index.tsx"]
    test_commands: List[DevCommand] = field(default_factory=list)
    build_commands: List[DevCommand] = field(default_factory=list)
    dev_commands: List[DevCommand] = field(default_factory=list)
    important_files: List[ImportantFile] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "module_type": self.module_type,
            "languages": list(self.languages),
            "frameworks": list(self.frameworks),
            "build_system": self.build_system,
            "dependency_manifests": list(self.dependency_manifests),
            "entry_points": list(self.entry_points),
            "test_commands": [c.to_dict() for c in self.test_commands],
            "build_commands": [c.to_dict() for c in self.build_commands],
            "dev_commands": [c.to_dict() for c in self.dev_commands],
            "important_files": [f.to_dict() for f in self.important_files],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModuleInfo:
        return cls(
            name=data.get("name", ""),
            path=data.get("path", "."),
            module_type=data.get("module_type", "unknown"),
            languages=data.get("languages", []),
            frameworks=data.get("frameworks", []),
            build_system=data.get("build_system", "unknown"),
            dependency_manifests=data.get("dependency_manifests", []),
            entry_points=data.get("entry_points", []),
            test_commands=[DevCommand.from_dict(c) for c in data.get("test_commands", [])],
            build_commands=[DevCommand.from_dict(c) for c in data.get("build_commands", [])],
            dev_commands=[DevCommand.from_dict(c) for c in data.get("dev_commands", [])],
            important_files=[ImportantFile.from_dict(f) for f in data.get("important_files", [])],
        )


@dataclass
class RepoMetadata:
    """Repo metadata implementation."""

    workspace_root: str
    project_name: str
    detected_languages: List[str] = field(default_factory=list)
    detected_frameworks: List[str] = field(default_factory=list)
    project_type: str = "unknown"  # "single_project", "monorepo", "multi_module", "unknown"
    is_monorepo: bool = False
    scan_timestamp: str = ""
    scan_version: str = "2.1"
    is_partial: bool = False
    partial_reasons: List[str] = field(default_factory=list)
    total_files_scanned: int = 0
    total_dirs_scanned: int = 0
    total_bytes_scanned: int = 0
    ignored_files_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RepoMetadata:
        return cls(
            workspace_root=data.get("workspace_root", ""),
            project_name=data.get("project_name", ""),
            detected_languages=data.get("detected_languages", []),
            detected_frameworks=data.get("detected_frameworks", []),
            project_type=data.get("project_type", "unknown"),
            is_monorepo=bool(data.get("is_monorepo", False)),
            scan_timestamp=data.get("scan_timestamp", ""),
            scan_version=data.get("scan_version", "2.1"),
            is_partial=bool(data.get("is_partial", False)),
            partial_reasons=data.get("partial_reasons", []),
            total_files_scanned=data.get("total_files_scanned", 0),
            total_dirs_scanned=data.get("total_dirs_scanned", 0),
            total_bytes_scanned=data.get("total_bytes_scanned", 0),
            ignored_files_count=data.get("ignored_files_count", 0),
        )


@dataclass
class ProjectMap:
    """Project map implementation."""

    metadata: RepoMetadata
    root_module: Optional[ModuleInfo] = None
    submodules: List[ModuleInfo] = field(default_factory=list)
    directory_tree: List[str] = field(default_factory=list)
    important_files: List[ImportantFile] = field(default_factory=list)
    dev_commands: List[DevCommand] = field(default_factory=list)
    file_hashes: Dict[str, str] = field(default_factory=dict)  # rel_path -> "mtime:size" or sha256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "root_module": self.root_module.to_dict() if self.root_module else None,
            "submodules": [m.to_dict() for m in self.submodules],
            "directory_tree": list(self.directory_tree),
            "important_files": [f.to_dict() for f in self.important_files],
            "dev_commands": [c.to_dict() for c in self.dev_commands],
            "file_hashes": dict(self.file_hashes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectMap:
        meta = RepoMetadata.from_dict(data.get("metadata", {}))
        root_mod = ModuleInfo.from_dict(data["root_module"]) if data.get("root_module") else None
        subs = [ModuleInfo.from_dict(m) for m in data.get("submodules", [])]
        tree = data.get("directory_tree", [])
        imp_files = [ImportantFile.from_dict(f) for f in data.get("important_files", [])]
        commands = [DevCommand.from_dict(c) for c in data.get("dev_commands", [])]
        hashes = data.get("file_hashes", {})
        return cls(
            metadata=meta,
            root_module=root_mod,
            submodules=subs,
            directory_tree=tree,
            important_files=imp_files,
            dev_commands=commands,
            file_hashes=hashes,
        )
