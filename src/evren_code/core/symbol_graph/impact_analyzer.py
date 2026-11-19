"""Impact analyzer module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from evren_code.core.symbol_graph.graph import SymbolGraphEngine
from evren_code.core.symbol_graph.index import SymbolIndexDB
from evren_code.core.symbol_graph.models import (
    CodeSymbol,
    ImpactReport,
    RelationshipType,
    SymbolType,
)
from evren_code.core.symbol_graph.test_linker import TestLinker


class ImpactAnalyzer:
    """ImpactAnalyzer: impact analysis."""

    def __init__(self, db: SymbolIndexDB, graph: SymbolGraphEngine) -> None:
        self.db = db
        self.graph = graph
        self.test_linker = TestLinker(db)

    def analyze_symbol(
        self,
        symbol_id: str,
        max_indirect_depth: int = 3,
        caller_limit: int = 25,
    ) -> ImpactReport:
        """Analyze symbol."""
        target_symbol = self.db.get_symbol_by_id(symbol_id)
        if not target_symbol:
            syms = self.db.get_symbols_by_name(symbol_id)
            if syms:
                target_symbol = syms[0]

        if not target_symbol:
            dummy_sym = CodeSymbol(
                id=symbol_id,
                name=symbol_id,
                fully_qualified_name=symbol_id,
                symbol_type=SymbolType.FUNCTION,
                language="unknown",
                file_path="",
                start_line=1,
                end_line=1,
            )
            return ImpactReport(
                target_symbol=dummy_sym,
                limitations=[f"Sembol ID '{symbol_id}' veritabanında bulunamadı."],
            )

        direct_edges = self.graph.get_incoming_references(
            target_symbol.id,
            rel_types=[RelationshipType.CALLS, RelationshipType.REFERENCES],
            limit=caller_limit,
        )
        direct_callers: List[Dict[str, Any]] = []
        direct_caller_ids: Set[str] = set()

        for edge in direct_edges:
            src_id = edge.get("source_symbol_id")
            if src_id:
                direct_caller_ids.add(src_id)
            direct_callers.append({
                "source_file": edge["source_file"],
                "source_symbol": edge.get("source_name") or edge["source_file"],
                "source_symbol_id": src_id,
                "line": edge["line_number"],
                "evidence": edge["evidence"],
                "relationship": edge["relationship_type"],
            })

        indirect_callers: List[Dict[str, Any]] = []
        traversal = self.graph.traverse_graph(
            target_symbol.id,
            direction="incoming",
            max_depth=max_indirect_depth,
            rel_types=[RelationshipType.CALLS],
            limit=caller_limit,
        )

        for step in traversal:
            if step["depth"] > 1:
                edge = step["edge"]
                indirect_callers.append({
                    "depth": step["depth"],
                    "source_file": edge["source_file"],
                    "source_symbol": edge.get("source_name") or edge["source_file"],
                    "source_symbol_id": edge.get("source_symbol_id"),
                    "line": edge["line_number"],
                    "relationship": edge["relationship_type"],
                })

        related_interfaces: List[Dict[str, Any]] = []
        implementations: List[Dict[str, Any]] = []

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.target_name, r.target_symbol_id, r.evidence
                FROM symbol_references r
                WHERE r.source_symbol_id = ? AND r.relationship_type = 'IMPLEMENTS'
                """,
                (target_symbol.id,),
            )
            for r in cursor.fetchall():
                related_interfaces.append(dict(r))

            cursor = conn.execute(
                """
                SELECT r.source_file, r.source_symbol_id, s.name as implementor_name, r.evidence
                FROM symbol_references r
                LEFT JOIN symbols s ON s.id = r.source_symbol_id
                WHERE (r.target_symbol_id = ? OR r.target_name = ?) AND r.relationship_type = 'IMPLEMENTS'
                """,
                (target_symbol.id, target_symbol.name),
            )
            for r in cursor.fetchall():
                implementations.append(dict(r))

        inheritance_hierarchy: List[Dict[str, Any]] = []
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.target_name, r.target_symbol_id, 'SUPERCLASS' as direction
                FROM symbol_references r
                WHERE r.source_symbol_id = ? AND r.relationship_type = 'EXTENDS'
                """,
                (target_symbol.id,),
            )
            for r in cursor.fetchall():
                inheritance_hierarchy.append(dict(r))

            cursor = conn.execute(
                """
                SELECT r.source_file, r.source_symbol_id, s.name as subclass_name, 'SUBCLASS' as direction
                FROM symbol_references r
                LEFT JOIN symbols s ON s.id = r.source_symbol_id
                WHERE (r.target_symbol_id = ? OR r.target_name = ?) AND r.relationship_type = 'EXTENDS'
                """,
                (target_symbol.id, target_symbol.name),
            )
            for r in cursor.fetchall():
                inheritance_hierarchy.append(dict(r))

        imported_dependencies: List[Dict[str, Any]] = []
        out_edges = self.graph.get_outgoing_references(
            target_symbol.id,
            rel_types=[RelationshipType.CALLS, RelationshipType.REFERENCES],
            limit=15,
        )
        for edge in out_edges:
            imported_dependencies.append({
                "target_name": edge["target_name"],
                "target_file": edge.get("target_file"),
                "target_symbol_id": edge.get("target_symbol_id"),
                "relationship": edge["relationship_type"],
                "evidence": edge["evidence"],
            })

        related_tests = self.test_linker.find_tests_for_symbol(target_symbol)

        affected_modules: Set[str] = set()
        all_affected_files = [target_symbol.file_path]
        all_affected_files.extend([c["source_file"] for c in direct_callers])
        all_affected_files.extend([c["source_file"] for c in indirect_callers])
        all_affected_files.extend([t["test_file"] for t in related_tests])

        for f in all_affected_files:
            parts = f.replace("\\", "/").split("/")
            if len(parts) > 1:
                affected_modules.add(parts[0])
            else:
                affected_modules.add(".")

        limitations = [
            "Dinamik çalışma zamanı bağımlılık enjeksiyonu (Spring DI, Guice) statik olarak sınırlı çözülebilir.",
            "Reflection veya dinamik string ile çağrılan metodlar statik graf analizine yansımayabilir.",
            "Polimorfik dinamik çağrılarda (dynamic dispatch) alt sınıf implementasyonları dolaylı etki olarak değerlendirilmelidir.",
        ]

        return ImpactReport(
            target_symbol=target_symbol,
            direct_callers=direct_callers,
            indirect_callers=indirect_callers,
            related_interfaces=related_interfaces,
            implementations=implementations,
            inheritance_hierarchy=inheritance_hierarchy,
            imported_dependencies=imported_dependencies,
            related_test_files=related_tests,
            affected_modules=sorted(list(affected_modules)),
            limitations=limitations,
        )
