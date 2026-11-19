"""Test linker module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from evren_code.core.symbol_graph.index import SymbolIndexDB
from evren_code.core.symbol_graph.models import CodeSymbol


class TestLinker:
    """Test linker implementation."""

    def __init__(self, db: SymbolIndexDB) -> None:
        self.db = db

    def is_test_file(self, file_path: str) -> bool:
        """Is test file."""
        p_low = file_path.lower().replace("\\", "/")
        name = Path(file_path).name.lower()

        if any(seg in p_low.split("/") for seg in ("test", "tests", "__tests__", "spec", "specs")):
            return True

        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        if name.endswith("test.java") or name.endswith("tests.java") or name.endswith("testcase.java"):
            return True
        if name.endswith(".test.ts") or name.endswith(".spec.ts") or name.endswith(".test.js") or name.endswith(".spec.js"):
            return True
        if name.endswith("_test.go") or name.endswith("_test.rs"):
            return True

        return False

    def find_tests_for_symbol(self, symbol: CodeSymbol, limit: int = 20) -> List[Dict[str, Any]]:
        """Find tests for symbol."""
        tests: List[Dict[str, Any]] = []
        seen_files: Set[str] = set()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT r.source_file, r.line_number, r.evidence, s.name as test_method_name
                FROM symbol_references r
                LEFT JOIN symbols s ON s.id = r.source_symbol_id
                WHERE (r.target_symbol_id = ? OR r.target_name = ?)
                """,
                (symbol.id, symbol.name),
            )
            for row in cursor.fetchall():
                src_file = row["source_file"]
                if self.is_test_file(src_file):
                    seen_files.add(src_file)
                    tests.append({
                        "test_file": src_file,
                        "line": row["line_number"],
                        "test_method": row["test_method_name"] or "test",
                        "match_type": "CONFIRMED_REFERENCE",
                        "evidence": row["evidence"],
                    })

        s_name_low = symbol.name.lower()
        candidate_names = [
            f"{s_name_low}test",
            f"{s_name_low}tests",
            f"test_{s_name_low}",
            f"{s_name_low}_test",
            f"{s_name_low}.test",
            f"{s_name_low}.spec",
        ]

        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT DISTINCT file_path FROM indexed_files")
            for r in cursor.fetchall():
                f_path = r["file_path"]
                if f_path in seen_files:
                    continue
                if not self.is_test_file(f_path):
                    continue

                f_stem = Path(f_path).stem.lower().replace(".", "_")
                if any(c in f_stem for c in candidate_names) or s_name_low in f_stem:
                    tests.append({
                        "test_file": f_path,
                        "line": 1,
                        "test_method": None,
                        "match_type": "HEURISTIC_MATCH",
                        "evidence": f"Dosya adı '{symbol.name}' ile uyuşuyor",
                    })

        return tests[:limit]

    def find_tests_for_file(self, file_path: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Find tests for file."""
        symbols = self.db.get_symbols_in_file(file_path)
        all_tests: List[Dict[str, Any]] = []
        seen = set()

        for sym in symbols[:10]:
            for t in self.find_tests_for_symbol(sym, limit=limit):
                key = (t["test_file"], t.get("test_method"))
                if key not in seen:
                    seen.add(key)
                    all_tests.append(t)

        return all_tests[:limit]
