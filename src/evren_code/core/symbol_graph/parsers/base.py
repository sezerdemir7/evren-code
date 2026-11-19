"""Base module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from evren_code.core.symbol_graph.models import CodeSymbol, FileImport, SymbolReference


@dataclass
class ParseDiagnostics:
    """Parse diagnostics implementation."""

    line: int
    column: int
    message: str
    severity: str = "error"  # "error", "warning"


@dataclass
class ParsedResult:
    """Parsed result implementation."""

    file_path: str
    language: str
    symbols: List[CodeSymbol] = field(default_factory=list)
    imports: List[FileImport] = field(default_factory=list)
    references: List[SymbolReference] = field(default_factory=list)
    diagnostics: List[ParseDiagnostics] = field(default_factory=list)
    is_partial: bool = False


class BaseParser(ABC):
    """Base parser implementation."""

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Language name."""
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Supported extensions."""
        pass

    @abstractmethod
    def parse(self, file_path: str, code: str) -> ParsedResult:
        """Parse."""
        pass
