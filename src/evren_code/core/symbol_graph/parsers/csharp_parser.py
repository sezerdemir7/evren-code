"""Csharp parser module."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from evren_code.core.symbol_graph.models import (
    CodeSymbol,
    FileImport,
    ParserConfidence,
    RelationshipType,
    ResolutionStatus,
    SymbolReference,
    SymbolType,
)
from evren_code.core.symbol_graph.parsers.base import (
    BaseParser,
    ParseDiagnostics,
    ParsedResult,
)


class CSharpParser(BaseParser):
    """C sharp parser implementation."""

    # Using bildirimleri: using System; using System.Collections.Generic; using static System.Math;
    USING_RE = re.compile(
        r"^\s*using\s+(?:static\s+)?(?P<alias>[a-zA-Z0-9_]+\s*=\s*)?(?P<module>[a-zA-Z0-9_\.]+)\s*;",
        re.MULTILINE,
    )

    NAMESPACE_RE = re.compile(
        r"^\s*namespace\s+(?P<name>[a-zA-Z0-9_\.]+)",
        re.MULTILINE,
    )

    CONTAINER_RE = re.compile(
        r"^\s*(?:\[[^\]]+\]\s*)*(?P<visibility>public|private|protected|internal)?\s*"
        r"(?:(?:static|abstract|sealed|partial|readonly)\s+)*"
        r"(?P<kind>class|struct|interface|record|enum)\s+"
        r"(?P<name>[a-zA-Z0-9_]+)(?:<(?P<generics>[^>]+)>)?\s*"
        r"(?::\s*(?P<bases>[a-zA-Z0-9_<>,\s\.]+))?",
        re.MULTILINE,
    )

    # Metotlar: [visibility] [modifiers] ReturnType MethodName<T>(params)
    METHOD_RE = re.compile(
        r"^\s*(?:\[[^\]]+\]\s*)*(?P<visibility>public|private|protected|internal)?\s*"
        r"(?:(?:static|virtual|override|abstract|async|unsafe|partial|extern|sealed|new)\s+)*"
        r"(?P<return>[a-zA-Z0-9_<>[\]?,\s]+?)\s+"
        r"(?P<name>[a-zA-Z0-9_]+)\s*(?:<(?P<generics>[^>]+)>)?\s*"
        r"\((?P<params>[^;]*)\)(?:\s*where\s+.*)?$",
    )

    # Kurucular: [visibility] ClassName(params) [: base/this(...)]
    CONSTRUCTOR_RE = re.compile(
        r"^\s*(?P<visibility>public|private|protected|internal)?\s*"
        r"(?P<name>[a-zA-Z0-9_]+)\s*\((?P<params>[^;]*)\)"
        r"(?:\s*:\s*(?:base|this)\([^)]*\))?$",
    )

    PROPERTY_RE = re.compile(
        r"^\s*(?:\[[^\]]+\]\s*)*(?P<visibility>public|private|protected|internal)?\s*"
        r"(?:(?:static|virtual|override|abstract|sealed|new)\s+)*"
        r"(?P<type>[a-zA-Z0-9_<>[\]?,\s]+?)\s+"
        r"(?P<name>[a-zA-Z0-9_]+)\s*"
        r"(?:\{|=>)",
    )

    KEYWORDS_TO_IGNORE = {
        "if", "while", "for", "foreach", "switch", "catch", "using", "lock",
        "return", "throw", "yield", "fixed", "checked", "unchecked",
    }

    @property
    def language_name(self) -> str:
        return "csharp"

    @property
    def supported_extensions(self) -> List[str]:
        return [".cs"]

    def parse(self, file_path: str, code: str) -> ParsedResult:
        result = ParsedResult(file_path=file_path, language="csharp")
        lines = code.splitlines()
        n = len(lines)

        # 1. Using Bildirimleri
        for match in self.USING_RE.finditer(code):
            module_name = match.group("module")
            line_no = code[: match.start()].count("\n") + 1
            result.imports.append(
                FileImport(
                    file_path=file_path,
                    imported_module=module_name,
                    imported_symbols=[],
                    line_number=line_no,
                )
            )

        # 2. Namespace Tespiti
        current_namespace = ""
        ns_match = self.NAMESPACE_RE.search(code)
        if ns_match:
            current_namespace = ns_match.group("name")

        active_containers: List[Tuple[CodeSymbol, int, int]] = []  # (symbol, depth, start_line)
        current_brace_depth = 0

        i = 0
        while i < n:
            raw_line = lines[i]
            stripped = raw_line.strip()
            line_no = i + 1

            if not stripped or stripped.startswith(("//", "/*", "*", "using ", "#")):
                current_brace_depth += raw_line.count("{") - raw_line.count("}")
                self._check_closed_containers(active_containers, current_brace_depth, line_no)
                i += 1
                continue

            is_class_level = bool(active_containers and current_brace_depth == active_containers[-1][1])

            c_match = self.CONTAINER_RE.match(raw_line)
            if c_match:
                kind = c_match.group("kind").lower()
                c_name = c_match.group("name")
                c_vis = c_match.group("visibility") or "internal"
                bases_raw = c_match.group("bases") or ""

                stype = SymbolType.CLASS
                if kind == "struct":
                    stype = SymbolType.STRUCT
                elif kind == "interface":
                    stype = SymbolType.INTERFACE
                elif kind == "enum":
                    stype = SymbolType.ENUM
                elif kind == "record":
                    stype = SymbolType.RECORD

                parent_id = active_containers[-1][0].id if active_containers else None
                fq_prefix = f"{current_namespace}." if current_namespace else ""
                if active_containers:
                    fq_prefix += f"{active_containers[-1][0].name}."
                fq_name = f"{fq_prefix}{c_name}"
                sym_id = f"{file_path}::{fq_name}::{stype.value}"

                end_line = self._find_closing_brace_line(lines, i)

                container_sym = CodeSymbol(
                    id=sym_id,
                    name=c_name,
                    fully_qualified_name=fq_name,
                    symbol_type=stype,
                    language="csharp",
                    file_path=file_path,
                    start_line=line_no,
                    end_line=end_line,
                    parent_symbol_id=parent_id,
                    visibility=c_vis,
                    signature=stripped,
                    confidence=ParserConfidence.PARSER_VERIFIED,
                )
                result.symbols.append(container_sym)

                if bases_raw:
                    for b in bases_raw.split(","):
                        base_clean = b.strip().split("<")[0].strip()
                        if base_clean:
                            result.references.append(
                                SymbolReference(
                                    source_file=file_path,
                                    source_symbol_id=sym_id,
                                    target_name=base_clean,
                                    target_symbol_id=None,
                                    relationship_type=RelationshipType.EXTENDS if stype == SymbolType.CLASS else RelationshipType.IMPLEMENTS,
                                    line_number=line_no,
                                    confidence=ParserConfidence.PARSER_VERIFIED,
                                    resolution_status=ResolutionStatus.UNRESOLVED,
                                )
                            )

                container_depth = current_brace_depth + 1
                active_containers.append((container_sym, container_depth, line_no))
                current_brace_depth += raw_line.count("{") - raw_line.count("}")
                i += 1
                continue

            m_match = self.METHOD_RE.match(raw_line) if is_class_level else None
            if m_match:
                m_name = m_match.group("name")
                if m_name not in self.KEYWORDS_TO_IGNORE:
                    m_vis = m_match.group("visibility") or "private"
                    parent_sym = active_containers[-1][0] if active_containers else None
                    parent_id = parent_sym.id if parent_sym else None
                    parent_name = parent_sym.name if parent_sym else (current_namespace or "Global")
                    fq_name = f"{parent_name}.{m_name}"
                    sym_id = f"{file_path}::{fq_name}::method"

                    end_line = self._find_closing_brace_line(lines, i)
                    method_sym = CodeSymbol(
                        id=sym_id,
                        name=m_name,
                        fully_qualified_name=fq_name,
                        symbol_type=SymbolType.METHOD,
                        language="csharp",
                        file_path=file_path,
                        start_line=line_no,
                        end_line=end_line,
                        parent_symbol_id=parent_id,
                        visibility=m_vis,
                        signature=stripped,
                        confidence=ParserConfidence.PARSER_VERIFIED,
                    )
                    result.symbols.append(method_sym)

                    current_brace_depth += raw_line.count("{") - raw_line.count("}")
                    self._check_closed_containers(active_containers, current_brace_depth, line_no)
                    i += 1
                    continue

            c_ctor = self.CONSTRUCTOR_RE.match(raw_line) if is_class_level else None
            if c_ctor and active_containers and c_ctor.group("name") == active_containers[-1][0].name:
                ctor_name = c_ctor.group("name")
                ctor_vis = c_ctor.group("visibility") or "public"
                parent_sym = active_containers[-1][0]
                fq_name = f"{parent_sym.name}.{ctor_name}"
                sym_id = f"{file_path}::{fq_name}::constructor"

                end_line = self._find_closing_brace_line(lines, i)
                ctor_sym = CodeSymbol(
                    id=sym_id,
                    name=ctor_name,
                    fully_qualified_name=fq_name,
                    symbol_type=SymbolType.CONSTRUCTOR,
                    language="csharp",
                    file_path=file_path,
                    start_line=line_no,
                    end_line=end_line,
                    parent_symbol_id=parent_sym.id,
                    visibility=ctor_vis,
                    signature=stripped,
                    confidence=ParserConfidence.PARSER_VERIFIED,
                )
                result.symbols.append(ctor_sym)
                current_brace_depth += raw_line.count("{") - raw_line.count("}")
                self._check_closed_containers(active_containers, current_brace_depth, line_no)
                i += 1
                continue

            p_match = self.PROPERTY_RE.match(raw_line) if is_class_level else None
            if p_match and active_containers:
                p_name = p_match.group("name")
                if p_name not in self.KEYWORDS_TO_IGNORE:
                    p_vis = p_match.group("visibility") or "public"
                    parent_sym = active_containers[-1][0]
                    fq_name = f"{parent_sym.name}.{p_name}"
                    sym_id = f"{file_path}::{fq_name}::property"

                    end_line = self._find_closing_brace_line(lines, i)
                    prop_sym = CodeSymbol(
                        id=sym_id,
                        name=p_name,
                        fully_qualified_name=fq_name,
                        symbol_type=SymbolType.PROPERTY,
                        language="csharp",
                        file_path=file_path,
                        start_line=line_no,
                        end_line=end_line,
                        parent_symbol_id=parent_sym.id,
                        visibility=p_vis,
                        signature=stripped,
                        confidence=ParserConfidence.PARSER_VERIFIED,
                    )
                    result.symbols.append(prop_sym)

            current_brace_depth += raw_line.count("{") - raw_line.count("}")
            self._check_closed_containers(active_containers, current_brace_depth, line_no)
            i += 1

        return result

    def _find_closing_brace_line(self, lines: List[str], start_idx: int) -> int:
        """Find closing brace line."""
        n = len(lines)
        k = start_idx
        while k < n and "{" not in lines[k]:
            if lines[k].strip().endswith(";"):
                return k + 1
            k += 1

        if k >= n:
            return start_idx + 1

        depth = 0
        for idx in range(k, n):
            depth += lines[idx].count("{") - lines[idx].count("}")
            if depth <= 0:
                return idx + 1
        return n

    def _check_closed_containers(
        self,
        containers: List[Tuple[CodeSymbol, int, int]],
        current_depth: int,
        current_line: int,
    ) -> None:
        """Check closed containers."""
        while containers and current_depth < containers[-1][1]:
            sym, _, _ = containers.pop()
            if sym.end_line < current_line:
                sym.end_line = current_line
