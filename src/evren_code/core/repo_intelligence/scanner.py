"""Scanner module."""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from evren_code.core.repo_intelligence.models import ScanLimits


DEFAULT_IGNORED_DIRS: Set[str] = {
    ".git",
    ".idea",
    ".vscode",
    ".evren",
    "node_modules",
    "target",
    "build",
    "dist",
    "coverage",
    ".next",
    ".nuxt",
    ".turbo",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    ".env",
    "vendor",
    "bin",
    "obj",
    ".gradle",
    ".mvn",
}

BINARY_EXTENSIONS: Set[str] = {
    ".exe", ".dll", ".so", ".dylib", ".class", ".jar", ".war", ".ear",
    ".pyc", ".pyo", ".pyd", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".bmp", ".tiff",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".iso", ".bin", ".wasm", ".db", ".sqlite", ".sqlite3",
    ".pkl", ".npy", ".npz", ".onnx", ".pt", ".pth",
    ".woff", ".woff2", ".ttf", ".eot",
}


class GitIgnoreRule:
    """Git ignore rule implementation."""

    def __init__(self, pattern: str, base_dir: Path, workspace_root: Path):
        self.raw_pattern = pattern.strip()
        self.base_dir = base_dir
        self.workspace_root = workspace_root
        self.is_negation = False
        self.dir_only = False

        pat = self.raw_pattern
        if pat.startswith("!"):
            self.is_negation = True
            pat = pat[1:]

        if pat.endswith("/"):
            self.dir_only = True
            pat = pat[:-1]

        self.anchored = "/" in pat.lstrip("/")
        if pat.startswith("/"):
            pat = pat[1:]

        self.pattern = pat
        # Base dir relative to workspace
        try:
            self.rel_base = self.base_dir.relative_to(self.workspace_root).as_posix()
            if self.rel_base == ".":
                self.rel_base = ""
        except ValueError:
            self.rel_base = ""

    def matches(self, rel_path: str, is_dir: bool) -> bool:
        """Matches."""
        if self.dir_only and not is_dir:
            return False

        if self.rel_base:
            if not rel_path.startswith(self.rel_base + "/") and rel_path != self.rel_base:
                return False
            check_path = rel_path[len(self.rel_base) + 1 :] if rel_path != self.rel_base else ""
        else:
            check_path = rel_path

        if not check_path:
            return False

        # Anchored vs unanchored
        if not self.anchored:
            name = check_path.split("/")[-1]
            if fnmatch.fnmatch(name, self.pattern):
                return True
            for part in check_path.split("/"):
                if fnmatch.fnmatch(part, self.pattern):
                    return True

        if fnmatch.fnmatch(check_path, self.pattern):
            return True

        if "**" in self.pattern:
            regex_pat = self.pattern.replace(".", r"\.").replace("**", ".*").replace("*", "[^/]*").replace("?", "[^/]")
            if re.fullmatch(regex_pat, check_path):
                return True

        return False


class GitIgnoreMatcher:
    """Git ignore matcher implementation."""

    def __init__(self, workspace_root: Path, custom_ignores: Optional[List[str]] = None):
        self.workspace_root = workspace_root
        self.rules: List[GitIgnoreRule] = []
        if custom_ignores:
            for pat in custom_ignores:
                if pat and not pat.startswith("#"):
                    self.rules.append(GitIgnoreRule(pat, workspace_root, workspace_root))

    def load_gitignore(self, gitignore_path: Path) -> None:
        if not gitignore_path.is_file():
            return
        try:
            content = gitignore_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        base_dir = gitignore_path.parent
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self.rules.append(GitIgnoreRule(line, base_dir, self.workspace_root))

    def is_ignored(self, rel_path: str, is_dir: bool = False) -> bool:
        """Is ignored."""
        ignored = False
        for rule in self.rules:
            if rule.matches(rel_path, is_dir):
                ignored = not rule.is_negation
        return ignored


@dataclass
class ScannedFileInfo:
    """Scanned file info implementation."""

    path: str  # POSIX relative path
    abs_path: Path
    size_bytes: int
    mtime: float
    is_binary: bool = False


@dataclass
class ScanResult:
    """Scan result implementation."""

    files: Dict[str, ScannedFileInfo] = field(default_factory=dict)  # rel_path -> ScannedFileInfo
    directories: List[str] = field(default_factory=list)  # rel_dir_paths
    total_files_scanned: int = 0
    total_dirs_scanned: int = 0
    total_bytes_scanned: int = 0
    ignored_files_count: int = 0
    is_partial: bool = False
    partial_reasons: List[str] = field(default_factory=list)


class RepoScanner:
    """Repo scanner implementation."""

    def __init__(
        self,
        workspace_root: Path,
        limits: Optional[ScanLimits] = None,
        custom_ignores: Optional[List[str]] = None,
        follow_symlinks: bool = False,
    ):
        self.workspace_root = workspace_root.resolve()
        self.limits = limits or ScanLimits()
        self.custom_ignores = list(custom_ignores or [])
        self.follow_symlinks = follow_symlinks
        self.gitignore_matcher = GitIgnoreMatcher(self.workspace_root, self.custom_ignores)

    def _is_binary_file(self, file_path: Path) -> bool:
        """Is binary file (binary)."""
        if file_path.suffix.lower() in BINARY_EXTENSIONS:
            return True
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return True
        except (OSError, PermissionError):
            pass
        return False

    def scan(self) -> ScanResult:
        """Scan."""
        result = ScanResult()
        if not self.workspace_root.is_dir():
            result.is_partial = True
            result.partial_reasons.append(f"Çalışma kök dizini mevcut değil veya dizin değil: '{self.workspace_root}'")
            return result

        root_gitignore = self.workspace_root / ".gitignore"
        if root_gitignore.is_file():
            self.gitignore_matcher.load_gitignore(root_gitignore)

        visited_dirs: Set[str] = set()

        def _traverse(current_dir: Path, current_depth: int) -> None:
            if result.is_partial and len(result.partial_reasons) > 0:
                # Kota tamamen dolduysa dur
                if result.total_files_scanned >= self.limits.max_files:
                    return
                if result.total_bytes_scanned >= self.limits.max_total_scan_bytes:
                    return

            if current_depth > self.limits.max_depth:
                result.is_partial = True
                msg = f"Maksimum tarama derinliğine ({self.limits.max_depth}) ulaşıldı: '{current_dir}'"
                if msg not in result.partial_reasons:
                    result.partial_reasons.append(msg)
                return

            try:
                real_cur = current_dir.resolve()
                try:
                    real_cur.relative_to(self.workspace_root)
                except ValueError:
                    return

                dir_id = str(real_cur.as_posix())
                if dir_id in visited_dirs:
                    return
                visited_dirs.add(dir_id)

                entries = sorted(current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except (PermissionError, OSError) as exc:
                result.partial_reasons.append(f"Dizin okuma hatası / erişim engellendi ('{current_dir}'): {exc}")
                return

            sub_gitignore = current_dir / ".gitignore"
            if sub_gitignore.is_file() and sub_gitignore != root_gitignore:
                self.gitignore_matcher.load_gitignore(sub_gitignore)

            subdirs: List[Path] = []
            files: List[Path] = []

            for entry in entries:
                name = entry.name
                if entry.is_dir() and (name in DEFAULT_IGNORED_DIRS or name.startswith(".")):
                    result.ignored_files_count += 1
                    continue

                if entry.is_symlink() and not self.follow_symlinks:
                    result.ignored_files_count += 1
                    continue

                try:
                    rel_p = entry.relative_to(self.workspace_root).as_posix()
                except ValueError:
                    continue

                if self.gitignore_matcher.is_ignored(rel_p, is_dir=entry.is_dir()):
                    result.ignored_files_count += 1
                    continue

                if entry.is_dir():
                    subdirs.append(entry)
                else:
                    files.append(entry)

            # Dizinleri kaydet
            try:
                rel_dir = current_dir.relative_to(self.workspace_root).as_posix()
                if rel_dir != ".":
                    result.directories.append(rel_dir)
                    result.total_dirs_scanned += 1
            except ValueError:
                pass

            for f in files:
                if result.total_files_scanned >= self.limits.max_files:
                    result.is_partial = True
                    msg = f"Maksimum dosya sayısı sınırına ({self.limits.max_files}) ulaşıldı."
                    if msg not in result.partial_reasons:
                        result.partial_reasons.append(msg)
                    break

                if result.total_bytes_scanned >= self.limits.max_total_scan_bytes:
                    result.is_partial = True
                    msg = f"Maksimum toplam tarama bayt sınırına ({self.limits.max_total_scan_bytes} bayt) ulaşıldı."
                    if msg not in result.partial_reasons:
                        result.partial_reasons.append(msg)
                    break

                try:
                    rel_f = f.relative_to(self.workspace_root).as_posix()
                    stat = f.stat()
                    size = stat.st_size
                    mtime = stat.st_mtime
                    is_bin = self._is_binary_file(f)

                    result.files[rel_f] = ScannedFileInfo(
                        path=rel_f,
                        abs_path=f,
                        size_bytes=size,
                        mtime=mtime,
                        is_binary=is_bin,
                    )
                    result.total_files_scanned += 1
                    result.total_bytes_scanned += size
                except (PermissionError, OSError) as exc:
                    result.partial_reasons.append(f"Dosya okunamadı: '{f.name}': {exc}")

            # Alt dizinleri gez
            for sd in subdirs:
                _traverse(sd, current_depth + 1)

        _traverse(self.workspace_root, 1)
        return result
