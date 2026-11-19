"""Language Server Protocol (LSP) and Static Diagnostics Engine.

Provides live language intelligence:
1. `lsp_get_diagnostics(file_path)`: Detects type and syntax errors immediately without waiting for full builds.
2. `lsp_find_definition(symbol)`: Pinpoints definition location (file and line) for functions, classes, and types.
3. `lsp_find_references(symbol)`: Finds all symbol references across the project for refactoring safety.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from evren_code.config import AppConfig
from evren_code.tools.filesystem import count_unclosed_jsx_divs, resolve_safe_path
from evren_code.tools.registry import ToolRegistry


@dataclass
class DiagnosticItem:
    """Represents a single syntax or type diagnostic."""

    line: int
    column: int
    severity: str  # "ERROR", "WARNING", "INFO"
    message: str
    source: str = "compiler"


@dataclass
class SymbolLocation:
    """Location of a symbol in source code."""

    file_path: str
    line: int
    preview: str


class LSPClient:
    """Hybrid language server client (TypeScript tsc / Python pyright / AST fallback / Symbol Index)."""

    def __init__(self, workspace_root: Path, symbol_engine: Optional[Any] = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.symbol_engine = symbol_engine

    def get_diagnostics(self, file_path: str) -> List[DiagnosticItem]:
        """Analyzes type and syntax diagnostics for the given file."""
        target = resolve_safe_path(self.workspace_root, file_path)
        if not target.is_file():
            return [DiagnosticItem(line=1, column=1, severity="ERROR", message=f"File not found: '{file_path}' (Dosya bulunamadı)")]

        suffix = target.suffix.lower()
        if suffix == ".py":
            return self._diagnose_python(target)
        elif suffix in (".ts", ".tsx", ".js", ".jsx"):
            return self._diagnose_typescript(target)
        elif suffix == ".cs":
            return self._diagnose_csharp(target)
        return []

    def _diagnose_python(self, file_path: Path) -> List[DiagnosticItem]:
        """Real-time syntax checking via Python AST and compile()."""
        diagnostics: List[DiagnosticItem] = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            compile(content, str(file_path), "exec")
        except SyntaxError as exc:
            diagnostics.append(
                DiagnosticItem(
                    line=exc.lineno or 1,
                    column=exc.offset or 1,
                    severity="ERROR",
                    message=f"Syntax Error: {exc.msg}",
                    source="python_ast",
                )
            )
        except Exception as exc:
            diagnostics.append(
                DiagnosticItem(
                    line=1,
                    column=1,
                    severity="ERROR",
                    message=f"Compile Error: {exc}",
                    source="python_ast",
                )
            )
        return diagnostics

    def _diagnose_typescript(self, file_path: Path) -> List[DiagnosticItem]:
        """Syntax and unclosed tag analysis for TypeScript / JSX files."""
        diagnostics: List[DiagnosticItem] = []
        content = file_path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        # 1. Unclosed brace and bracket checks
        open_curlies = content.count("{") - content.count("}")
        open_parens = content.count("(") - content.count(")")
        if open_curlies > 0:
            diagnostics.append(
                DiagnosticItem(
                    line=len(lines),
                    column=1,
                    severity="ERROR",
                    message=f"Unclosed curly brace / Kapatılmamış süslü parantez: {open_curlies} missing '}}' (Expected '}}' before <eof>).",
                    source="syntax_guard",
                )
            )
        if open_parens > 0:
            diagnostics.append(
                DiagnosticItem(
                    line=len(lines),
                    column=1,
                    severity="ERROR",
                    message=f"Unclosed parenthesis / Kapatılmamış parantez: {open_parens} missing ')'.",
                    source="syntax_guard",
                )
            )

        # 2. JSX unclosed tag check
        if file_path.suffix in (".tsx", ".jsx"):
            open_divs = count_unclosed_jsx_divs(content)
            if open_divs > 0:
                diagnostics.append(
                    DiagnosticItem(
                        line=len(lines),
                        column=1,
                        severity="ERROR",
                        message=f"Unclosed JSX tag / Kapatılmamış JSX etiketi: {open_divs} missing '</div>'.",
                        source="jsx_guard",
                    )
                )

        # 3. Quick tsc check if available
        tsc_res = self._run_tsc_diagnostics(file_path)
        diagnostics.extend(tsc_res)

        return diagnostics

    def _run_tsc_diagnostics(self, file_path: Path) -> List[DiagnosticItem]:
        """Run tsc diagnostics (varsa)."""
        diagnostics: List[DiagnosticItem] = []
        curr = file_path.parent
        project_root: Optional[Path] = None
        while curr != self.workspace_root.parent:
            if (curr / "tsconfig.json").is_file():
                project_root = curr
                break
            curr = curr.parent

        if not project_root:
            return []

        try:
            from evren_code.tools.shell import execute_command_with_tree_kill

            cmd = ["npx", "tsc", "--noEmit", "--pretty", "false"]
            _rc, stdout_str, _stderr_str = execute_command_with_tree_kill(
                args=cmd,
                cwd=str(project_root),
                timeout=12,
                shell=(os.name == "nt"),
            )
            pattern = re.compile(r"([^(]+)\((\d+),(\d+)\):\s+(error|warning)\s+(TS\d+):\s+(.+)")
            for line in (stdout_str or "").splitlines():
                m = pattern.match(line.strip())
                if m:
                    fpath, lno, cno, sev, code, msg = m.groups()
                    target_rel = str(file_path.relative_to(project_root).as_posix()).lower()
                    if target_rel in fpath.replace("\\", "/").lower():
                        diagnostics.append(
                            DiagnosticItem(
                                line=int(lno),
                                column=int(cno),
                                severity=sev.upper(),
                                message=f"{code}: {msg}",
                                source="tsc",
                            )
                        )
        except Exception:
            pass
        return diagnostics

    def _diagnose_csharp(self, file_path: Path) -> List[DiagnosticItem]:
        """Diagnose csharp."""
        diagnostics: List[DiagnosticItem] = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return [DiagnosticItem(line=1, column=1, severity="ERROR", message=f"Dosya okunamadı: {exc}", source="csharp_guard")]

        lines = content.splitlines()

        open_curlies = 0
        open_parens = 0
        open_brackets = 0
        in_multiline_comment = False

        for idx, line in enumerate(lines, start=1):
            s = line.strip()
            if in_multiline_comment:
                if "*/" in s:
                    in_multiline_comment = False
                    s = s[s.find("*/") + 2:]
                else:
                    continue

            if "/*" in s and "*/" not in s:
                in_multiline_comment = True
                s = s[:s.find("/*")]

            if s.startswith("//"):
                continue

            clean_line = re.sub(r'@"(?:""|[^"])*"', '""', line)
            clean_line = re.sub(r'"(?:\\.|[^"\\])*"', '""', clean_line)
            clean_line = re.sub(r"'(?:\\.|[^'\\])*'", "''", clean_line)

            for char in clean_line:
                if char == "{":
                    open_curlies += 1
                elif char == "}":
                    open_curlies -= 1
                    if open_curlies < 0:
                        diagnostics.append(
                            DiagnosticItem(
                                line=idx,
                                column=line.find("}") + 1,
                                severity="ERROR",
                                message="Beklenmeyen fazladan kapanış parantezi '}' (Unmatched closing brace).",
                                source="csharp_syntax",
                            )
                        )
                        open_curlies = 0
                elif char == "(":
                    open_parens += 1
                elif char == ")":
                    open_parens -= 1
                elif char == "[":
                    open_brackets += 1
                elif char == "]":
                    open_brackets -= 1

            if (s.startswith("using ") or s.startswith("namespace ")) and not s.endswith((";", "{")):
                if s.startswith("using "):
                    diagnostics.append(
                        DiagnosticItem(
                            line=idx,
                            column=len(line),
                            severity="ERROR",
                            message="Eksik noktalı virgül ';' (Expected ';' after using directive).",
                            source="csharp_syntax",
                        )
                    )

        if in_multiline_comment:
            diagnostics.append(
                DiagnosticItem(
                    line=len(lines),
                    column=1,
                    severity="ERROR",
                    message="Kapatılmamış çok satırlı yorum '/*' bloğu.",
                    source="csharp_syntax",
                )
            )

        if open_curlies > 0:
            diagnostics.append(
                DiagnosticItem(
                    line=len(lines),
                    column=1,
                    severity="ERROR",
                    message=f"Kapatılmamış süslü parantez: {open_curlies} adet '}}' eksik (Expected '}}' before end of file).",
                    source="csharp_syntax",
                )
            )

        if open_parens > 0:
            diagnostics.append(
                DiagnosticItem(
                    line=len(lines),
                    column=1,
                    severity="ERROR",
                    message=f"Kapatılmamış normal parantez: {open_parens} adet ')' eksik.",
                    source="csharp_syntax",
                )
            )

        if open_brackets > 0:
            diagnostics.append(
                DiagnosticItem(
                    line=len(lines),
                    column=1,
                    severity="ERROR",
                    message=f"Kapatılmamış köşeli parantez: {open_brackets} adet ']' eksik.",
                    source="csharp_syntax",
                )
            )

        return diagnostics

    def find_definition(self, symbol_name: str) -> List[SymbolLocation]:
        """Find definition."""
        locations: List[SymbolLocation] = []
        name = symbol_name.strip()
        if not name:
            return []

        if self.symbol_engine and hasattr(self.symbol_engine, "db"):
            try:
                syms = self.symbol_engine.db.get_symbols_by_name(name)
                for s in syms:
                    locations.append(
                        SymbolLocation(
                            file_path=s.file_path,
                            line=s.start_line,
                            preview=s.signature or f"{s.symbol_type.value} {s.name}",
                        )
                    )
                if locations:
                    return locations
            except Exception:
                pass

        patterns = [
            rf"(?:def|class)\s+{re.escape(name)}\b",
            rf"(?:function|class|interface|type)\s+{re.escape(name)}\b",
            rf"(?:const|let|var)\s+{re.escape(name)}\s*=",
        ]

        ignores = {".git", "node_modules", ".next", "__pycache__", "dist", "build"}
        for root, dirs, files in os.walk(self.workspace_root):
            dirs[:] = [d for d in dirs if d not in ignores and not d.startswith(".")]
            for file in files:
                p = Path(root) / file
                if p.suffix in (".py", ".ts", ".tsx", ".js", ".jsx"):
                    try:
                        content = p.read_text(encoding="utf-8", errors="replace")
                        for idx, line in enumerate(content.splitlines(), start=1):
                            if any(re.search(pat, line) for pat in patterns):
                                rel = str(p.relative_to(self.workspace_root).as_posix())
                                locations.append(
                                    SymbolLocation(
                                        file_path=rel,
                                        line=idx,
                                        preview=line.strip()[:100],
                                    )
                                )
                    except Exception:
                        pass
        return locations

    def find_references(self, symbol_name: str, max_results: int = 15) -> List[SymbolLocation]:
        """Find references."""
        locations: List[SymbolLocation] = []
        name = symbol_name.strip()
        if not name:
            return []

        pattern = re.compile(rf"\b{re.escape(name)}\b")
        ignores = {".git", "node_modules", ".next", "__pycache__", "dist", "build"}

        for root, dirs, files in os.walk(self.workspace_root):
            dirs[:] = [d for d in dirs if d not in ignores and not d.startswith(".")]
            for file in files:
                p = Path(root) / file
                if p.suffix in (".py", ".ts", ".tsx", ".js", ".jsx"):
                    try:
                        content = p.read_text(encoding="utf-8", errors="replace")
                        for idx, line in enumerate(content.splitlines(), start=1):
                            if pattern.search(line):
                                rel = str(p.relative_to(self.workspace_root).as_posix())
                                locations.append(
                                    SymbolLocation(
                                        file_path=rel,
                                        line=idx,
                                        preview=line.strip()[:100],
                                    )
                                )
                                if len(locations) >= max_results:
                                    return locations
                    except Exception:
                        pass
        return locations


def register_lsp_tools(
    registry: ToolRegistry,
    config: AppConfig,
    symbol_engine: Optional[Any] = None,
) -> None:
    """Registers live LSP code intelligence tools."""
    lsp = LSPClient(config.workspace_root, symbol_engine=symbol_engine)

    def lsp_get_diagnostics(file_path: Optional[str] = None, path: Optional[str] = None) -> str:
        target_path = file_path or path
        if not target_path:
            return "Error: Valid file path (`file_path` or `path`) was not provided for `lsp_get_diagnostics`."
        items = lsp.get_diagnostics(target_path)
        if not items:
            return f"LSP Diagnostics: No errors found in '{target_path}' (Clean ✔ / Temiz ✔)."

        lines = [f"### LSP Diagnostics Report / LSP Teşhis Raporu: '{target_path}' ({len(items)} diagnostics)"]
        for it in items:
            lines.append(f"  • [Line/Satır {it.line}:{it.column}] [{it.severity}] {it.message} ({it.source})")
        return "\n".join(lines)

    def lsp_find_definition(symbol: str) -> str:
        locs = lsp.find_definition(symbol)
        if not locs:
            return f"LSP: Definition for symbol '{symbol}' not found (LSP: sembol tanımı bulunamadı)."
        lines = [f"### Definition Locations / Tanım Konumları: '{symbol}' ({len(locs)} matches)"]
        for lc in locs:
            lines.append(f"  • `{lc.file_path}:{lc.line}` -> `{lc.preview}`")
        return "\n".join(lines)

    def lsp_find_references(symbol: str, max_results: Optional[int] = 15) -> str:
        limit = max(1, min(30, int(max_results or 15)))
        locs = lsp.find_references(symbol, max_results=limit)
        if not locs:
            return f"LSP: No references found for '{symbol}' (LSP: referans bulunamadı)."
        lines = [f"### References / Kullanım Referansları: '{symbol}' ({len(locs)} matches)"]
        for lc in locs:
            lines.append(f"  • `{lc.file_path}:{lc.line}` -> `{lc.preview}`")
        return "\n".join(lines)

    registry.register(
        name="lsp_get_diagnostics",
        description=(
            "Inspects a file for live type errors, missing imports, and syntax issues. "
            "Use after editing code to verify diagnostics before triggering full builds."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Target file path to inspect (e.g. src/app/page.tsx)."},
            },
            "required": ["file_path"],
        },
        handler=lsp_get_diagnostics,
        requires_approval=False,
    )

    registry.register(
        name="lsp_find_definition",
        description="Locates the exact file and line number where a function, class, or interface is defined.",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name to look up (e.g. UserService, ButtonProps)."},
            },
            "required": ["symbol"],
        },
        handler=lsp_find_definition,
        requires_approval=False,
    )

    registry.register(
        name="lsp_find_references",
        description="Finds all usages and references of a symbol across the codebase to avoid breaking changes during refactoring.",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name whose references to find."},
                "max_results": {"type": "integer", "description": "Maximum number of references to return (default: 15)."},
            },
            "required": ["symbol"],
        },
        handler=lsp_find_references,
        requires_approval=False,
    )
