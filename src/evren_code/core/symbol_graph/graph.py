"""Graph module."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set

from evren_code.core.symbol_graph.index import SymbolIndexDB
from evren_code.core.symbol_graph.models import CodeSymbol, RelationshipType


class SymbolGraphEngine:
    """Symbol graph engine implementation."""

    def __init__(self, db: SymbolIndexDB) -> None:
        self.db = db

    def get_symbol(self, symbol_id: str) -> Optional[CodeSymbol]:
        return self.db.get_symbol_by_id(symbol_id)

    def get_file_symbols(self, file_path: str) -> List[CodeSymbol]:
        return self.db.get_symbols_in_file(file_path)

    def get_incoming_references(
        self,
        target_symbol_id: str,
        rel_types: Optional[List[RelationshipType]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get incoming references."""
        with self.db.get_connection() as conn:
            clauses = ["r.target_symbol_id = ?"]
            params: List[Any] = [target_symbol_id]

            if rel_types:
                placeholders = ",".join("?" for _ in rel_types)
                clauses.append(f"r.relationship_type IN ({placeholders})")
                params.extend([t.value for t in rel_types])

            params.append(limit)
            query = f"""
                SELECT r.source_file, r.source_symbol_id, r.relationship_type, r.line_number,
                       r.evidence, s.name as source_name, s.symbol_type as source_type, s.signature as source_sig
                FROM symbol_references r
                LEFT JOIN symbols s ON s.id = r.source_symbol_id
                WHERE {' AND '.join(clauses)}
                LIMIT ?
            """
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append(dict(row))
            return results

    def get_outgoing_references(
        self,
        source_symbol_id: str,
        rel_types: Optional[List[RelationshipType]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get outgoing references."""
        with self.db.get_connection() as conn:
            clauses = ["r.source_symbol_id = ?"]
            params: List[Any] = [source_symbol_id]

            if rel_types:
                placeholders = ",".join("?" for _ in rel_types)
                clauses.append(f"r.relationship_type IN ({placeholders})")
                params.extend([t.value for t in rel_types])

            params.append(limit)
            query = f"""
                SELECT r.target_name, r.target_symbol_id, r.relationship_type, r.line_number,
                       r.resolution_status, r.evidence, s.file_path as target_file, s.signature as target_sig
                FROM symbol_references r
                LEFT JOIN symbols s ON s.id = r.target_symbol_id
                WHERE {' AND '.join(clauses)}
                LIMIT ?
            """
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append(dict(row))
            return results

    def traverse_graph(
        self,
        start_symbol_id: str,
        direction: str = "incoming",
        max_depth: int = 3,
        rel_types: Optional[List[RelationshipType]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Traverse graph (cycle-safe)."""
        visited: Set[str] = {start_symbol_id}
        queue: deque = deque([(start_symbol_id, 1)])
        results: List[Dict[str, Any]] = []

        while queue and len(results) < limit:
            current_id, depth = queue.popleft()
            if depth > max_depth:
                break

            if direction == "incoming":
                edges = self.get_incoming_references(current_id, rel_types=rel_types, limit=limit)
            else:
                edges = self.get_outgoing_references(current_id, rel_types=rel_types, limit=limit)

            for edge in edges:
                next_id = edge.get("source_symbol_id") if direction == "incoming" else edge.get("target_symbol_id")
                edge_info = {
                    "depth": depth,
                    "direction": direction,
                    "edge": edge,
                    "next_symbol_id": next_id,
                }
                results.append(edge_info)

                if next_id and next_id not in visited and len(results) < limit:
                    visited.add(next_id)
                    queue.append((next_id, depth + 1))

        return results

    def find_related_files(self, file_path: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Find related files."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT target_file as related_file, 'OUTGOING' as direction
                FROM (
                    SELECT s.file_path as target_file
                    FROM symbol_references r
                    JOIN symbols s ON s.id = r.target_symbol_id
                    WHERE r.source_file = ? AND s.file_path != ?
                )
                UNION
                SELECT DISTINCT source_file as related_file, 'INCOMING' as direction
                FROM symbol_references r
                WHERE r.target_symbol_id LIKE ? AND r.source_file != ?
                LIMIT ?
                """,
                (file_path, file_path, f"{file_path}::%", file_path, limit),
            )
            return [dict(r) for r in cursor.fetchall()]
