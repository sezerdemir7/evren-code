"""Project Discovery, Directory Tree Mapping, and Code Search Tools (v0.2.0):
`get_project_tree`, `list_directory`, `find_files`, `search_code`.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from evren_code.config import AppConfig
from evren_code.tools.filesystem import resolve_safe_path
from evren_code.tools.registry import ToolRegistry

_TREE_CACHE: Dict[Tuple[str, int], str] = {}


def invalidate_search_cache() -> None:
    """Invalidates the cached project tree and discovery maps when filesystem changes."""
    _TREE_CACHE.clear()


def find_matching_directories(
    workspace_root: Path,
    target_str: str,
    ignored_set: set,
    max_results: int = 3,
) -> List[str]:
    """Scans workspace sub-folders to discover matching directory paths when requested directory is not found."""
    clean = target_str.replace("\\", "/").strip("/").lower()
    if not clean:
        return []

    # 1. Truncate path segments to check closest existing parent directory
    parts = clean.split("/")
    for i in range(len(parts) - 1, 0, -1):
        parent_candidate = "/".join(parts[:i])
        candidate_path = workspace_root / parent_candidate
        if candidate_path.is_dir():
            try:
                rel = candidate_path.relative_to(workspace_root).as_posix()
                return [rel]
            except ValueError:
                pass

    matches: List[str] = []
    try:
        for root, dirs, _ in os.walk(workspace_root):
            dirs[:] = [d for d in dirs if d not in ignored_set and not d.startswith(".")]
            for d in dirs:
                full_p = Path(root) / d
                try:
                    rel_p = full_p.relative_to(workspace_root).as_posix()
                    rel_lower = rel_p.lower()
                    if (
                        rel_lower == clean
                        or rel_lower.endswith("/" + clean)
                        or clean in rel_lower
                        or (len(rel_lower) > 8 and rel_lower in clean)
                    ):
                        if rel_p not in matches:
                            matches.append(rel_p)
                            if len(matches) >= max_results:
                                return matches
                except ValueError:
                    pass
    except Exception:
        pass
    return matches


def _try_ripgrep_search(
    root_dir: Path,
    query: str,
    file_pattern: Optional[str],
    max_results: int,
) -> Optional[List[str]]:
    """Executes high-performance code search using 'rg' (ripgrep) if installed on the host system."""
    rg_bin = shutil.which("rg")
    if not rg_bin:
        return None
    try:
        from evren_code.tools.shell import execute_command_with_tree_kill

        args = [
            rg_bin,
            "--no-heading",
            "--line-number",
            "--color=never",
            "-m",
            str(max_results),
        ]
        if file_pattern:
            args.extend(["-g", file_pattern])
        args.extend([query, str(root_dir)])
        rc, stdout_str, _stderr_str = execute_command_with_tree_kill(
            args=args,
            cwd=str(root_dir),
            timeout=15,
            shell=False,
        )
        if rc in (0, 1):
            return [line.strip() for line in stdout_str.splitlines() if line.strip()][:max_results]
    except Exception:
        pass
    return None


def register_search_tools(registry: ToolRegistry, config: AppConfig) -> None:
    """Registers project tree, search, and directory listing tools."""

    ignored_set = set(config.tools.ignored_dirs)
    ignored_set.update({
        "Library", "Temp", "Obj", "Build", "Builds", "Logs", "MemoryCaptures", "bin", "obj"
    })
    gi_path = config.workspace_root / ".gitignore"
    if gi_path.is_file():
        try:
            for gline in gi_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                gline = gline.strip()
                if gline and not gline.startswith("#"):
                    clean_pattern = gline.rstrip("/")
                    if not any(ch in clean_pattern for ch in "*?["):
                        ignored_set.add(clean_pattern)
        except Exception:
            pass

    def get_project_tree(
        path: str = ".",
        max_depth: Optional[int] = None,
    ) -> str:
        """Returns the hierarchical directory and file tree of the specified folder."""
        target = resolve_safe_path(config.workspace_root, path)
        if not target.exists():
            return f"Error: Directory '{path}' not found."
        if not target.is_dir():
            return f"Error: '{path}' is not a directory."

        depth_limit = min(
            config.tools.max_tree_depth,
            max(1, int(max_depth) if max_depth is not None else config.tools.default_tree_depth),
        )

        cache_key = (str(target.resolve()), depth_limit)
        if cache_key in _TREE_CACHE:
            return _TREE_CACHE[cache_key]

        max_entries = config.tools.max_tree_entries
        lines: List[str] = [f"{target.name}/"]
        count = 0

        def _walk(current: Path, prefix: str, depth: int) -> None:
            nonlocal count
            if depth > depth_limit or count >= max_entries:
                return
            try:
                children = [
                    p
                    for p in sorted(
                        current.iterdir(),
                        key=lambda item: (not item.is_dir(), item.name.lower()),
                    )
                    if p.name not in ignored_set
                ]
            except OSError:
                return

            total = len(children)
            for idx, child in enumerate(children):
                if count >= max_entries:
                    lines.append(f"{prefix}... ({max_entries} max entries limit reached / girdi sınırı aşıldı)")
                    return
                is_last = idx == total - 1
                connector = "└── " if is_last else "├── "
                suffix = "/" if child.is_dir() else ""
                lines.append(f"{prefix}{connector}{child.name}{suffix}")
                count += 1
                if child.is_dir() and depth < depth_limit:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    _walk(child, next_prefix, depth + 1)

        _walk(target, "", 1)
        res_tree = (
            f"Project Tree / Proje Ağacı (Directory / Dizin: {path}, Max Depth: {depth_limit}, Shown: {count}):\n"
            + "\n".join(lines)
        )
        _TREE_CACHE[cache_key] = res_tree
        return res_tree

    def list_directory(path: str = ".") -> str:
        target = resolve_safe_path(config.workspace_root, path)
        if not target.exists():
            similars = find_matching_directories(config.workspace_root, path, ignored_set)
            err_msg = f"Error: Directory '{path}' not found."
            if similars:
                err_msg += "\nHint: Found the following existing directories in workspace / İpucu: Alt dizinlerde şu mevcut klasörler bulundu:\n" + "\n".join(f"  • {s}" for s in similars)
                err_msg += "\nPlease call `list_directory(path='...')` specifying one of these valid paths."
            return err_msg
        if not target.is_dir():
            return f"Error: '{path}' is not a directory."

        entries: List[str] = []
        for idx, item in enumerate(
            sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        ):
            if idx >= config.tools.max_directory_entries:
                entries.append(f"... ({config.tools.max_directory_entries} max entries limit reached / girdi sınırı aşıldı)")
                break
            if item.name in ignored_set:
                continue
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            size_info = f" ({item.stat().st_size} B)" if item.is_file() else ""
            entries.append(f"{prefix:8s} {item.name}{size_info}")

        if not entries:
            return f"Directory '{path}' is empty (Dizin boş)."
        return f"Directory Contents / Dizin İçeriği ({path}):\n" + "\n".join(entries)

    def find_files(pattern: str = "*", directory: str = ".") -> str:
        root_dir = resolve_safe_path(config.workspace_root, directory)
        if not root_dir.is_dir():
            similars = find_matching_directories(config.workspace_root, directory, ignored_set)
            err_msg = f"Error: Directory '{directory}' not found."
            if similars:
                err_msg += "\nHint: Found the following existing directories in workspace:\n" + "\n".join(f"  • {s}" for s in similars)
                err_msg += "\nPlease call `find_files` specifying one of these valid paths (Lütfen 'directory' parametresine bu alt yollardan birini verin)."
            return err_msg

        matches: List[str] = []
        max_results = config.tools.max_search_results

        from evren_code.core.tool_reliability import (
            get_current_cancellation_token,
            report_tool_progress,
        )

        tok = get_current_cancellation_token()
        scanned_dirs = 0
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if tok is not None and tok.is_cancelled:
                break
            scanned_dirs += 1
            if scanned_dirs % 25 == 1:
                report_tool_progress("search_code scanning directories", completed_units=scanned_dirs)
            dirnames[:] = [d for d in dirnames if d not in ignored_set]
            for fname in filenames:
                if fnmatch.fnmatch(fname.lower(), pattern.lower()):
                    full_path = Path(dirpath) / fname
                    try:
                        rel_path = full_path.relative_to(config.workspace_root).as_posix()
                    except ValueError:
                        rel_path = full_path.as_posix()
                    matches.append(rel_path)
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break

        if not matches:
            ext_counts: Dict[str, int] = {}
            try:
                for dp, dn, fn in os.walk(config.workspace_root):
                    dn[:] = [d for d in dn if d not in ignored_set and not d.startswith(".")]
                    for f in fn:
                        ext = Path(f).suffix.lower()
                        if ext and len(ext) <= 6:
                            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            except Exception:
                pass

            top_exts = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            hint = f"No files found matching '{pattern}' ('{pattern}' desenine uyan dosya bulunamadı)."
            if top_exts:
                ext_str = ", ".join(f"{ext} ({cnt} files / dosya)" for ext, cnt in top_exts)
                hint += (
                    f"\nHint: Most common file extensions in this project (İpucu: Bu projede en yaygın dosya uzantıları): {ext_str}."
                    f"\nPlease search using a pattern matching the tech stack or inspect structure with 'get_project_tree'."
                )
            return hint
        return f"Found Files / Bulunan Dosyalar ({len(matches)} files):\n" + "\n".join(matches)

    def search_code(
        query: str,
        directory: str = ".",
        file_pattern: Optional[str] = None,
    ) -> str:
        target_path = resolve_safe_path(config.workspace_root, directory)
        try:
            regex = re.compile(query, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(query), re.IGNORECASE)

        results: List[str] = []
        max_results = config.tools.max_search_results

        # If directory directly targets a file, search inside that file
        if target_path.is_file():
            try:
                if target_path.stat().st_size > config.tools.max_file_size_bytes:
                    return f"Error: '{directory}' exceeds maximum file size limit."
                text = target_path.read_text(encoding="utf-8", errors="ignore")
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        try:
                            rel = target_path.relative_to(config.workspace_root).as_posix()
                        except ValueError:
                            rel = target_path.as_posix()
                        results.append(f"{rel}:{line_no}: {line.strip()[:160]}")
                        if len(results) >= max_results:
                            break
                if not results:
                    return f"No matches found for '{query}' in '{directory}' ('{query}' araması '{directory}' dosyasında bulunamadı)."
                return f"Search Results / Arama Sonuçları ({len(results)} matches):\n" + "\n".join(results)
            except OSError as err:
                return f"Error: Could not read '{directory}': {err}"

        if not target_path.is_dir():
            return f"Error: Directory or file '{directory}' not found."

        root_dir = target_path

        # 1. Ripgrep (rg) integration
        rg_matches = _try_ripgrep_search(root_dir, query, file_pattern, max_results)
        if rg_matches is not None:
            if not rg_matches:
                return f"No matches found for query '{query}' ('{query}' araması için eşleşme bulunamadı)."
            return f"Search Results / Arama Sonuçları ({len(rg_matches)} matches):\n" + "\n".join(rg_matches)

        from evren_code.core.tool_reliability import (
            get_current_cancellation_token,
            report_tool_progress,
        )

        tok = get_current_cancellation_token()
        scanned_dirs = 0
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if tok is not None and tok.is_cancelled:
                break
            scanned_dirs += 1
            if scanned_dirs % 25 == 1:
                report_tool_progress("search_code scanning directories", completed_units=scanned_dirs)
            dirnames[:] = [d for d in dirnames if d not in ignored_set]
            for fname in filenames:
                if file_pattern and not fnmatch.fnmatch(fname.lower(), file_pattern.lower()):
                    continue
                fpath = Path(dirpath) / fname
                try:
                    if fpath.stat().st_size > config.tools.max_file_size_bytes:
                        continue
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue

                file_match_count = 0
                max_per_file = getattr(config.tools, "max_search_matches_per_file", 6)
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        try:
                            rel = fpath.relative_to(config.workspace_root).as_posix()
                        except ValueError:
                            rel = fpath.as_posix()
                        results.append(f"{rel}:{line_no}: {line.strip()[:160]}")
                        file_match_count += 1
                        if file_match_count >= max_per_file or len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        if not results:
            return f"No matches found for query '{query}' ('{query}' araması için eşleşme bulunamadı)."
        return f"Search Results / Arama Sonuçları ({len(results)} matches):\n" + "\n".join(results)

    registry.register(
        name="get_project_tree",
        description=(
            "Recursively renders a visual directory and file tree structure up to 3-6 levels deep. "
            "Use this as the FIRST exploratory tool to understand project layout instead of multiple 'list_directory' calls."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Root directory path to generate tree from (default: '.')."},
                "max_depth": {"type": "integer", "description": "Maximum folder depth (default: 3, max: 6)."},
            },
            "required": [],
        },
        handler=get_project_tree,
        requires_approval=False,
    )

    registry.register(
        name="list_directory",
        description="Lists files and sub-folders within a single directory along with file sizes.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list (default: '.')."},
            },
            "required": [],
        },
        handler=list_directory,
        requires_approval=False,
    )

    registry.register(
        name="find_files",
        description="Finds files by filename or glob pattern (e.g. '*.tsx', '*auth*') across the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Filename glob pattern (e.g. '*.py')."},
                "directory": {"type": "string", "description": "Optional sub-directory to restrict search."},
            },
            "required": ["pattern"],
        },
        handler=find_files,
        requires_approval=False,
    )

    registry.register(
        name="search_code",
        description="Searches codebase file contents for a keyword or regex pattern, returning file paths and line matches.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text or regular expression."},
                "directory": {"type": "string", "description": "Directory path to search in (default: '.')."},
                "file_pattern": {"type": "string", "description": "Optional glob filter for target file extensions (e.g. '*.py')."},
            },
            "required": ["query"],
        },
        handler=search_code,
        requires_approval=False,
    )
