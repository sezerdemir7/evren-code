"""Registry module."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from evren_code.core.symbol_graph.parsers.base import BaseParser, ParsedResult
from evren_code.core.symbol_graph.parsers.csharp_parser import CSharpParser
from evren_code.core.symbol_graph.parsers.go_parser import GoParser
from evren_code.core.symbol_graph.parsers.java_parser import JavaParser
from evren_code.core.symbol_graph.parsers.python_parser import PythonParser
from evren_code.core.symbol_graph.parsers.rust_parser import RustParser
from evren_code.core.symbol_graph.parsers.ts_js_parser import TypeScriptJavaScriptParser


class ParserRegistry:
    """Parser registry implementation."""

    def __init__(self) -> None:
        self._parsers_by_ext: Dict[str, BaseParser] = {}
        self._all_parsers: List[BaseParser] = [
            PythonParser(),
            JavaParser(),
            TypeScriptJavaScriptParser(),
            GoParser(),
            RustParser(),
            CSharpParser(),
        ]
        for parser in self._all_parsers:
            for ext in parser.supported_extensions:
                self._parsers_by_ext[ext.lower()] = parser

    def get_parser(self, file_path: str) -> Optional[BaseParser]:
        ext = Path(file_path).suffix.lower()
        return self._parsers_by_ext.get(ext)

    def is_supported(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self._parsers_by_ext

    def parse_file(self, file_path: str, code: str) -> ParsedResult:
        parser = self.get_parser(file_path)
        if not parser:
            res = ParsedResult(file_path=file_path, language="unsupported", is_partial=True)
            res.diagnostics.append(
                ParseDiagnostics(
                    line=1,
                    column=1,
                    message=f"Desteklenmeyen dosya türü: '{Path(file_path).suffix}'",
                    severity="warning",
                )
            )
            return res
        return parser.parse(file_path, code)
