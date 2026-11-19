"""Models module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SymbolType(str, Enum):
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    PROPERTY = "property"
    FIELD = "field"
    CONSTANT = "constant"
    VARIABLE = "variable"
    TYPE_ALIAS = "type_alias"
    RECORD = "record"
    TRAIT = "trait"


class RelationshipType(str, Enum):
    CONTAINS = "CONTAINS"
    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    EXTENDS = "EXTENDS"
    IMPLEMENTS = "IMPLEMENTS"
    REFERENCES = "REFERENCES"
    DEPENDS_ON = "DEPENDS_ON"
    DEFINES = "DEFINES"


class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


class ParserConfidence(str, Enum):
    PARSER_VERIFIED = "PARSER_VERIFIED"
    HEURISTIC_CANDIDATE = "HEURISTIC_CANDIDATE"


@dataclass
class CodeSymbol:
    """Code symbol implementation."""

    id: str
    name: str
    fully_qualified_name: str
    symbol_type: SymbolType
    language: str
    file_path: str  # POSIX relative path
    start_line: int
    end_line: int
    parent_symbol_id: Optional[str] = None
    visibility: str = "public"  # "public", "private", "protected", "package-private", "internal"
    signature: str = ""
    docstring: str = ""
    is_exported: bool = True
    confidence: ParserConfidence = ParserConfidence.PARSER_VERIFIED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["symbol_type"] = self.symbol_type.value
        d["confidence"] = self.confidence.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CodeSymbol:
        return cls(
            id=data["id"],
            name=data["name"],
            fully_qualified_name=data.get("fully_qualified_name", data["name"]),
            symbol_type=SymbolType(data["symbol_type"]),
            language=data["language"],
            file_path=data["file_path"],
            start_line=int(data["start_line"]),
            end_line=int(data["end_line"]),
            parent_symbol_id=data.get("parent_symbol_id"),
            visibility=data.get("visibility", "public"),
            signature=data.get("signature", ""),
            docstring=data.get("docstring", ""),
            is_exported=bool(data.get("is_exported", True)),
            confidence=ParserConfidence(data.get("confidence", ParserConfidence.PARSER_VERIFIED.value)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SymbolReference:
    """Symbol reference implementation."""

    source_file: str
    source_symbol_id: Optional[str]
    target_name: str
    target_symbol_id: Optional[str]
    relationship_type: RelationshipType
    line_number: int
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    confidence: ParserConfidence = ParserConfidence.PARSER_VERIFIED
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_symbol_id": self.source_symbol_id,
            "target_name": self.target_name,
            "target_symbol_id": self.target_symbol_id,
            "relationship_type": self.relationship_type.value,
            "line_number": self.line_number,
            "resolution_status": self.resolution_status.value,
            "confidence": self.confidence.value,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SymbolReference:
        return cls(
            source_file=data["source_file"],
            source_symbol_id=data.get("source_symbol_id"),
            target_name=data["target_name"],
            target_symbol_id=data.get("target_symbol_id"),
            relationship_type=RelationshipType(data["relationship_type"]),
            line_number=int(data["line_number"]),
            resolution_status=ResolutionStatus(data.get("resolution_status", ResolutionStatus.UNRESOLVED.value)),
            confidence=ParserConfidence(data.get("confidence", ParserConfidence.PARSER_VERIFIED.value)),
            evidence=data.get("evidence", ""),
        )


@dataclass
class FileImport:
    """File import implementation."""

    file_path: str
    imported_module: str
    imported_symbols: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    line_number: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileImport:
        return cls(
            file_path=data["file_path"],
            imported_module=data["imported_module"],
            imported_symbols=data.get("imported_symbols", []),
            alias=data.get("alias"),
            line_number=int(data.get("line_number", 1)),
        )


@dataclass
class ImpactReport:
    """Impact report implementation."""

    target_symbol: CodeSymbol
    direct_callers: List[Dict[str, Any]] = field(default_factory=list)
    indirect_callers: List[Dict[str, Any]] = field(default_factory=list)
    related_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    implementations: List[Dict[str, Any]] = field(default_factory=list)
    inheritance_hierarchy: List[Dict[str, Any]] = field(default_factory=list)
    imported_dependencies: List[Dict[str, Any]] = field(default_factory=list)
    related_test_files: List[Dict[str, Any]] = field(default_factory=list)
    affected_modules: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_symbol": self.target_symbol.to_dict(),
            "direct_callers": self.direct_callers,
            "indirect_callers": self.indirect_callers,
            "related_interfaces": self.related_interfaces,
            "implementations": self.implementations,
            "inheritance_hierarchy": self.inheritance_hierarchy,
            "imported_dependencies": self.imported_dependencies,
            "related_test_files": self.related_test_files,
            "affected_modules": self.affected_modules,
            "limitations": self.limitations,
        }


@dataclass
class IndexingStats:
    """Indexing stats implementation."""

    files_discovered: int = 0
    files_parsed: int = 0
    files_skipped: int = 0
    symbols_extracted: int = 0
    references_extracted: int = 0
    resolved_relationships: int = 0
    unresolved_relationships: int = 0
    parse_errors: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
