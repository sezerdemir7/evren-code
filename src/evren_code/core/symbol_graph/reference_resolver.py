"""Reference resolver module."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from evren_code.core.symbol_graph.index import SymbolIndexDB
from evren_code.core.symbol_graph.models import (
    CodeSymbol,
    RelationshipType,
    ResolutionStatus,
    SymbolReference,
)


class ReferenceResolver:
    """Reference resolver implementation."""

    def __init__(self, db: SymbolIndexDB) -> None:
        self.db = db

    def resolve_all_unresolved(self) -> Tuple[int, int, int]:
        """Resolve all unresolved."""
        resolved_count = 0
        ambiguous_count = 0
        unresolved_count = 0

        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT id, name, fqn, file_path FROM symbols")
            symbols_by_name: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            symbols_by_fqn: Dict[str, str] = {}
            symbols_by_file: Dict[str, List[Dict[str, str]]] = defaultdict(list)

            for r in cursor.fetchall():
                entry = {"id": r["id"], "name": r["name"], "fqn": r["fqn"], "file_path": r["file_path"]}
                symbols_by_name[r["name"]].append(entry)
                symbols_by_fqn[r["fqn"]] = r["id"]
                symbols_by_file[r["file_path"]].append(entry)

            cursor = conn.execute("SELECT file_path, imported_module, imported_symbols_json FROM file_imports")
            imports_by_file: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
            for r in cursor.fetchall():
                syms = json.loads(r["imported_symbols_json"] or "[]")
                imports_by_file[r["file_path"]].append((r["imported_module"], syms))

            cursor = conn.execute(
                """
                SELECT id, source_file, source_symbol_id, target_name, relationship_type
                FROM symbol_references
                WHERE target_symbol_id IS NULL OR resolution_status = 'UNRESOLVED'
                """
            )
            refs = cursor.fetchall()

            updates: List[Tuple[Optional[str], str, int]] = []

            for ref in refs:
                ref_id = ref["id"]
                s_file = ref["source_file"]
                t_name = ref["target_name"]

                target_id: Optional[str] = None
                status = ResolutionStatus.UNRESOLVED

                file_syms = symbols_by_file.get(s_file, [])
                same_file_matches = [s for s in file_syms if s["name"] == t_name or s["fqn"].endswith(f".{t_name}")]
                if len(same_file_matches) == 1:
                    target_id = same_file_matches[0]["id"]
                    status = ResolutionStatus.RESOLVED

                if not target_id:
                    file_impor_list = imports_by_file.get(s_file, [])
                    for mod, sym_list in file_impor_list:
                        if t_name in sym_list or mod.endswith(f".{t_name}") or mod == t_name:
                            matched_by_import = [
                                s for s in symbols_by_name.get(t_name, [])
                                if mod in s["fqn"] or s["file_path"].replace("/", ".").endswith(f"{mod}.py") or s["file_path"].replace("/", ".").endswith(f"{mod}.java")
                            ]
                            if len(matched_by_import) == 1:
                                target_id = matched_by_import[0]["id"]
                                status = ResolutionStatus.RESOLVED
                                break

                if not target_id and t_name in symbols_by_fqn:
                    target_id = symbols_by_fqn[t_name]
                    status = ResolutionStatus.RESOLVED

                if not target_id:
                    global_matches = symbols_by_name.get(t_name, [])
                    if len(global_matches) == 1:
                        target_id = global_matches[0]["id"]
                        status = ResolutionStatus.RESOLVED
                    elif len(global_matches) > 1:
                        status = ResolutionStatus.AMBIGUOUS

                if status == ResolutionStatus.RESOLVED:
                    resolved_count += 1
                elif status == ResolutionStatus.AMBIGUOUS:
                    ambiguous_count += 1
                else:
                    unresolved_count += 1

                updates.append((target_id, status.value, ref_id))

            if updates:
                conn.executemany(
                    """
                    UPDATE symbol_references
                    SET target_symbol_id = ?, resolution_status = ?
                    WHERE id = ?
                    """,
                    updates,
                )
                conn.commit()

        return resolved_count, ambiguous_count, unresolved_count
