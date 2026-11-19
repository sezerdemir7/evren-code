"""Semantic Code Map and Dependency Graph Module (`repo_map.py`).

Analyzes classes, function signatures, interfaces, and type definitions in large repositories
using AST (Abstract Syntax Tree) parsing and PageRank-style heuristics.
Compresses even 100k+ line codebases into 1000-2000 tokens to provide the model with a
high-level "bird's-eye semantic map".
"""

from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from evren_code.config import AppConfig
from evren_code.tools.filesystem import resolve_safe_path
from evren_code.tools.registry import ToolRegistry


@dataclass
class CodeTag:
    """A single class, function, interface, or type definition tag."""

    kind: str  # "class", "def", "interface", "type", "func", "struct"
    name: str
    signature: str = ""
    line: int = 1


class RepoMapGenerator:
    """Engine that scans the project, executes AST parsing, and applies semantic ranking."""

    SUPPORTED_EXTS = {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".go",
        ".rs",
        ".cs",
        ".java",
        ".c",
        ".cpp",
        ".h",
    }

    def __init__(
        self,
        workspace_root: Path,
        ignored_dirs: Set[str],
        max_tokens: int = 1500,
        max_files: int = 120,
    ) -> None:
        self.workspace_root = workspace_root
        self.ignored_dirs = ignored_dirs
        self.max_tokens = max_tokens
        self.max_files = max_files

    def extract_python_tags(self, file_path: Path, code: str) -> List[CodeTag]:
        """Extracts class and function definitions from Python code via AST."""
        tags: List[CodeTag] = []
        try:
            tree = ast.parse(code, filename=str(file_path))
        except (SyntaxError, ValueError):
            return self._regex_fallback_tags(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases] if hasattr(ast, "unparse") else []
                base_str = f"({', '.join(bases)})" if bases else ""
                tags.append(CodeTag(kind="class", name=node.name, signature=base_str, line=node.lineno))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("__") or node.name == "__init__":
                    args_list = [a.arg for a in node.args.args]
                    arg_str = f"({', '.join(args_list[:4])}{'...' if len(args_list) > 4 else ''})"
                    kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    tags.append(CodeTag(kind=kind, name=node.name, signature=arg_str, line=node.lineno))
        return tags

    def extract_ts_js_tags(self, code: str) -> List[CodeTag]:
        """Extracts key symbols from TypeScript / JavaScript code via regex-based pseudo-AST."""
        tags: List[CodeTag] = []

        # 1. Interfaces & Types
        for m in re.finditer(r"(?:export\s+)?interface\s+([A-Za-z0-9_]+)", code):
            tags.append(CodeTag(kind="interface", name=m.group(1)))
        for m in re.finditer(r"(?:export\s+)?type\s+([A-Za-z0-9_]+)\s*=", code):
            tags.append(CodeTag(kind="type", name=m.group(1)))

        # 2. Classes
        for m in re.finditer(r"(?:export\s+)?class\s+([A-Za-z0-9_]+)(?:\s+extends\s+([A-Za-z0-9_]+))?", code):
            ext = f" extends {m.group(2)}" if m.group(2) else ""
            tags.append(CodeTag(kind="class", name=m.group(1), signature=ext))

        # 3. Functions
        for m in re.finditer(r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)", code):
            args = m.group(2).strip()
            args_short = f"({args[:30]}...)" if len(args) > 30 else f"({args})"
            tags.append(CodeTag(kind="function", name=m.group(1), signature=args_short))

        # 4. Exported const components / functions (React arrow functions, etc.)
        for m in re.finditer(r"export\s+const\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z0-9_]+)\s*=>", code):
            tags.append(CodeTag(kind="const fn", name=m.group(1)))

        return tags

    def _regex_fallback_tags(self, code: str) -> List[CodeTag]:
        tags: List[CodeTag] = []
        for line in code.splitlines():
            line_s = line.strip()
            if line_s.startswith("class "):
                parts = line_s.split()
                if len(parts) > 1:
                    tags.append(CodeTag(kind="class", name=parts[1].split("(")[0].rstrip(":")))
            elif line_s.startswith("def ") or line_s.startswith("async def "):
                parts = line_s.split()
                if len(parts) > 1:
                    tags.append(CodeTag(kind="def", name=parts[1].split("(")[0]))
        return tags

    def extract_file_tags(self, file_path: Path) -> List[CodeTag]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        ext = file_path.suffix.lower()
        if ext == ".py":
            return self.extract_python_tags(file_path, content)
        if ext in (".ts", ".tsx", ".js", ".jsx"):
            return self.extract_ts_js_tags(content)
        return self._regex_fallback_tags(content)

    def generate_map(self, target_dir: Optional[Path] = None) -> str:
        """Scans all source files in the specified directory and produces a hierarchical Repo-Map."""
        root = target_dir or self.workspace_root
        if not root.is_dir():
            return f"Error: '{root}' is not a directory."

        file_tags: Dict[str, List[CodeTag]] = {}
        all_identifiers: Set[str] = set()

        # 1. Scan files and collect tags
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self.ignored_dirs and not d.startswith(".")]
            for fname in sorted(filenames):
                p = Path(dirpath) / fname
                if p.suffix.lower() in self.SUPPORTED_EXTS:
                    try:
                        rel_p = p.relative_to(self.workspace_root).as_posix()
                    except ValueError:
                        rel_p = p.as_posix()

                    tags = self.extract_file_tags(p)
                    if tags:
                        file_tags[rel_p] = tags
                        for t in tags:
                            all_identifiers.add(t.name)
                        file_count += 1
                        if file_count >= self.max_files:
                            break
            if file_count >= self.max_files:
                break

        if not file_tags:
            return "Semantic Repo-Map: No source code symbols (classes/functions) found in the target directory."

        # 2. Ranking Heuristic
        # Prioritize files containing central types, schemas, stores, models
        def _score_file(item):
            path, tags = item
            priority = 0
            p_lower = path.lower()
            if any(k in p_lower for k in ("type", "model", "schema", "store", "api", "core", "client")):
                priority += 5
            return priority + len(tags)

        sorted_files = sorted(file_tags.items(), key=_score_file, reverse=True)

        # 3. Format within token budget (~1500 tokens ~ 6000 chars)
        max_chars = self.max_tokens * 4
        output_lines: List[str] = [
            f"# Semantic Project Code Map (Repo-Map) - {len(file_tags)} Files Analyzed\n"
        ]
        current_len = len(output_lines[0])

        for rel_path, tags in sorted_files:
            file_header = f"{rel_path}:\n"
            if current_len + len(file_header) > max_chars:
                output_lines.append("\n... [Remaining submodules truncated due to token budget limit]")
                break

            output_lines.append(file_header)
            current_len += len(file_header)

            # Write tags in the file (up to 15 per file)
            for t in tags[:15]:
                sig = f" {t.signature}" if t.signature else ""
                tag_line = f"  | {t.kind} {t.name}{sig}\n"
                if current_len + len(tag_line) > max_chars:
                    output_lines.append("  | ... (additional symbols omitted)\n")
                    break
                output_lines.append(tag_line)
                current_len += len(tag_line)

        return "".join(output_lines)


def register_repo_map_tools(registry: ToolRegistry, config: AppConfig) -> None:
    """Registers the Repo Map tool into the ToolRegistry."""
    ignored = set(config.tools.ignored_dirs)
    generator = RepoMapGenerator(
        workspace_root=config.workspace_root,
        ignored_dirs=ignored,
        max_tokens=config.tools.repo_map.max_tokens,
        max_files=config.tools.repo_map.max_files,
    )

    def get_repo_map(directory: str = ".") -> str:
        target = resolve_safe_path(config.workspace_root, directory)
        return generator.generate_map(target)

    registry.register(
        name="get_repo_map",
        description=(
            "Generates an AST-based semantic code map (Repo Map) of the codebase within token budget. "
            "Provides a high-level architectural overview of classes, function signatures, and exports. "
            "Recommended at the start of large architectural tasks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Root directory to map (default: '.').",
                }
            },
            "required": [],
        },
        handler=get_repo_map,
        requires_approval=False,
    )
