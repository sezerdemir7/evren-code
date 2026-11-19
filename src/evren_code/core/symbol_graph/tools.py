"""Tools module."""

from __future__ import annotations

from typing import Optional

from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine
from evren_code.tools.registry import ToolRegistry


def register_symbol_graph_tools(
    registry: ToolRegistry,
    engine: CodebaseSymbolEngine,
) -> None:
    """Register symbol graph tools."""

    def search_symbols(
        query: str,
        symbol_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        return engine.search_symbols(query=query, symbol_type=symbol_type, language=language)

    def find_symbol_definition(name_or_id: str) -> str:
        return engine.find_symbol_definition(name_or_id=name_or_id)

    def find_symbol_references(symbol_name_or_id: str) -> str:
        return engine.find_symbol_references(symbol_name_or_id=symbol_name_or_id)

    def analyze_change_impact(symbol_name_or_id: str) -> str:
        return engine.analyze_change_impact(symbol_name_or_id=symbol_name_or_id)

    def find_related_tests(target: str) -> str:
        return engine.find_related_tests(target=target)

    def get_file_symbols(file_path: str) -> str:
        syms = engine.graph.get_file_symbols(file_path)
        if not syms:
            return f"'{file_path}' dosyasında indekslenmiş bir sembol bulunamadı."
        lines = [f"### `{file_path}` Dosyasındaki Semboller ({len(syms)} adet):"]
        for s in syms:
            lines.append(f"- `{s.name}` [{s.symbol_type.value}] (Satır {s.start_line}-{s.end_line}): `{s.signature}`")
        return "\n".join(lines)

    def index_codebase(force: bool = False) -> str:
        stats = engine.index_codebase(force_reindex=force)
        return (
            f"✔ Kod tabanı sembol indeksleme tamamlandı ({stats.duration_seconds}s).\n"
            f"- Keşfedilen Dosya: {stats.files_discovered}\n"
            f"- Ayrıştırılan Dosya: {stats.files_parsed} (Atlanan: {stats.files_skipped})\n"
            f"- Çıkarılan Sembol: {stats.symbols_extracted}\n"
            f"- Çıkarılan Referans: {stats.references_extracted}\n"
            f"- Çözümlenen İlişkiler: {stats.resolved_relationships} (Çözülemeyen: {stats.unresolved_relationships})\n"
            f"- Hatalar: {stats.parse_errors}"
        )

    registry.register(
        name="search_symbols",
        description="Searches for functions, classes, methods, or interfaces by name, signature, or FQN using FTS5 symbol indexing.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Symbol name or search query."},
                "symbol_type": {
                    "type": "string",
                    "description": "Optional symbol type filter (e.g. 'class', 'function', 'method', 'interface').",
                },
                "language": {"type": "string", "description": "Optional language filter (e.g. 'python', 'java', 'typescript')."},
            },
            "required": ["query"],
        },
        handler=search_symbols,
        requires_approval=False,
    )

    registry.register(
        name="find_symbol_definition",
        description="Returns the source code definition, line span, and signature of a symbol (function, class, method).",
        parameters={
            "type": "object",
            "properties": {
                "name_or_id": {"type": "string", "description": "Symbol name or unique symbol ID."},
            },
            "required": ["name_or_id"],
        },
        handler=find_symbol_definition,
        requires_approval=False,
    )

    registry.register(
        name="find_symbol_references",
        description="Lists all codebase locations referencing, calling, or inheriting a given symbol.",
        parameters={
            "type": "object",
            "properties": {
                "symbol_name_or_id": {"type": "string", "description": "Symbol name or ID to find references for."},
            },
            "required": ["symbol_name_or_id"],
        },
        handler=find_symbol_references,
        requires_approval=False,
    )

    registry.register(
        name="analyze_change_impact",
        description=(
            "Generates an impact analysis report identifying direct/indirect callers, interfaces, tests, "
            "and modules that would be affected before modifying a function or class."
        ),
        parameters={
            "type": "object",
            "properties": {
                "symbol_name_or_id": {
                    "type": "string",
                    "description": "Target symbol name or ID to analyze (e.g. 'UserService.registerUser' or 'calculate_score').",
                },
            },
            "required": ["symbol_name_or_id"],
        },
        handler=analyze_change_impact,
        requires_approval=False,
    )

    registry.register(
        name="find_related_tests",
        description="Finds existing test cases (pytest, JUnit, Jest, etc.) associated with a target symbol or source file.",
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Symbol name or source file path."},
            },
            "required": ["target"],
        },
        handler=find_related_tests,
        requires_approval=False,
    )

    registry.register(
        name="get_file_symbols",
        description="Lists all classes, methods, functions, and type definitions inside a specific source file.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Source file path (e.g. 'src/services/user.py')."},
            },
            "required": ["file_path"],
        },
        handler=get_file_symbols,
        requires_approval=False,
    )

    registry.register(
        name="index_codebase",
        description="Scans workspace code files to incrementally update the symbol and dependency graph.",
        parameters={
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "description": "If True, wipes existing index and rebuilds from scratch."},
            },
        },
        handler=index_codebase,
        requires_approval=False,
    )
