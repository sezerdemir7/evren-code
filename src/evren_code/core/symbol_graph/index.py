"""Index module."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from evren_code.core.symbol_graph.models import (
    CodeSymbol,
    FileImport,
    ParserConfidence,
    RelationshipType,
    ResolutionStatus,
    SymbolReference,
    SymbolType,
)


class SymbolIndexDB:
    """Symbol index db implementation."""

    DB_FILE_NAME = "symbol_graph.db"
    PARSER_VERSION = "2.2"

    def __init__(self, workspace_root: Path, db_path: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.db_path = db_path or (self.workspace_root / ".evren" / self.DB_FILE_NAME)
        self._ensure_schema()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    file_path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    language TEXT NOT NULL,
                    parser_version TEXT NOT NULL
                )
                """
            )

            # 2. Semboller Tablosu
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbols (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    fqn TEXT NOT NULL,
                    symbol_type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    parent_symbol_id TEXT,
                    visibility TEXT NOT NULL,
                    signature TEXT,
                    docstring TEXT,
                    is_exported INTEGER NOT NULL,
                    confidence TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_symbol_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols(symbol_type)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbol_references (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT NOT NULL,
                    source_symbol_id TEXT,
                    target_name TEXT NOT NULL,
                    target_symbol_id TEXT,
                    relationship_type TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    resolution_status TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    evidence TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_source ON symbol_references(source_symbol_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_target ON symbol_references(target_symbol_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_name ON symbol_references(target_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_rel ON symbol_references(relationship_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_sfile ON symbol_references(source_file)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_imports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    imported_module TEXT NOT NULL,
                    imported_symbols_json TEXT,
                    alias TEXT,
                    line_number INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_imports_file ON file_imports(file_path)")

            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                    name,
                    fqn,
                    signature,
                    docstring,
                    content='symbols',
                    content_rowid='rowid'
                )
                """
            )

            conn.commit()

    def clear(self) -> None:
        """Clear."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM indexed_files")
            conn.execute("DELETE FROM symbols")
            conn.execute("DELETE FROM symbol_references")
            conn.execute("DELETE FROM file_imports")
            conn.execute("DELETE FROM symbols_fts")
            conn.commit()

    def get_indexed_files(self) -> Dict[str, Tuple[float, str]]:
        """Get indexed files."""
        out: Dict[str, Tuple[float, str]] = {}
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT file_path, mtime, parser_version FROM indexed_files")
            for row in cursor.fetchall():
                out[row["file_path"]] = (float(row["mtime"]), row["parser_version"])
        return out

    def save_file_data(
        self,
        file_path: str,
        mtime: float,
        language: str,
        symbols: List[CodeSymbol],
        imports: List[FileImport],
        references: List[SymbolReference],
    ) -> None:
        """Save file data (transaction)."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM symbol_references WHERE source_file = ?", (file_path,))
            conn.execute("DELETE FROM file_imports WHERE file_path = ?", (file_path,))

            conn.execute(
                """
                INSERT OR REPLACE INTO indexed_files (file_path, mtime, language, parser_version)
                VALUES (?, ?, ?, ?)
                """,
                (file_path, mtime, language, self.PARSER_VERSION),
            )

            # 3. Sembolleri ekle
            for sym in symbols:
                cur = conn.execute(
                    """
                    INSERT INTO symbols (
                        id, name, fqn, symbol_type, language, file_path,
                        start_line, end_line, parent_symbol_id, visibility,
                        signature, docstring, is_exported, confidence, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sym.id,
                        sym.name,
                        sym.fully_qualified_name,
                        sym.symbol_type.value,
                        sym.language,
                        sym.file_path,
                        sym.start_line,
                        sym.end_line,
                        sym.parent_symbol_id,
                        sym.visibility,
                        sym.signature,
                        sym.docstring,
                        1 if sym.is_exported else 0,
                        sym.confidence.value,
                        json.dumps(sym.metadata, ensure_ascii=False),
                    ),
                )
                rowid = cur.lastrowid
                conn.execute(
                    """
                    INSERT INTO symbols_fts (rowid, name, fqn, signature, docstring)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (rowid, sym.name, sym.fully_qualified_name, sym.signature, sym.docstring),
                )

            for imp in imports:
                conn.execute(
                    """
                    INSERT INTO file_imports (file_path, imported_module, imported_symbols_json, alias, line_number)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        imp.file_path,
                        imp.imported_module,
                        json.dumps(imp.imported_symbols, ensure_ascii=False),
                        imp.alias,
                        imp.line_number,
                    ),
                )

            for ref in references:
                conn.execute(
                    """
                    INSERT INTO symbol_references (
                        source_file, source_symbol_id, target_name, target_symbol_id,
                        relationship_type, line_number, resolution_status, confidence, evidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ref.source_file,
                        ref.source_symbol_id,
                        ref.target_name,
                        ref.target_symbol_id,
                        ref.relationship_type.value,
                        ref.line_number,
                        ref.resolution_status.value,
                        ref.confidence.value,
                        ref.evidence,
                    ),
                )

            conn.commit()

    def remove_file(self, file_path: str) -> None:
        """Remove file."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM indexed_files WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM symbol_references WHERE source_file = ?", (file_path,))
            conn.execute("DELETE FROM file_imports WHERE file_path = ?", (file_path,))
            conn.execute(
                """
                UPDATE symbol_references 
                SET target_symbol_id = NULL, resolution_status = 'UNRESOLVED'
                WHERE target_symbol_id LIKE ?
                """,
                (f"{file_path}::%",),
            )
            conn.commit()

    def get_symbol_by_id(self, symbol_id: str) -> Optional[CodeSymbol]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM symbols WHERE id = ?", (symbol_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_symbol(row)

    def get_symbols_by_name(self, name: str) -> List[CodeSymbol]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM symbols WHERE name = ?", (name,))
            return [self._row_to_symbol(r) for r in cursor.fetchall()]

    def get_symbols_in_file(self, file_path: str) -> List[CodeSymbol]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM symbols WHERE file_path = ? ORDER BY start_line", (file_path,))
            return [self._row_to_symbol(r) for r in cursor.fetchall()]

    def search_symbols(
        self,
        query: str,
        symbol_type: Optional[str] = None,
        language: Optional[str] = None,
        limit: int = 20,
    ) -> List[CodeSymbol]:
        clean_q = query.strip()
        results: List[CodeSymbol] = []
        with self.get_connection() as conn:
            if not clean_q:
                clauses = []
                params = []
                if symbol_type:
                    clauses.append("symbol_type = ?")
                    params.append(symbol_type.lower())
                if language:
                    clauses.append("language = ?")
                    params.append(language.lower())
                where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                cursor = conn.execute(f"SELECT * FROM symbols {where_str} LIMIT ?", (*params, limit))
                return [self._row_to_symbol(r) for r in cursor.fetchall()]

            # 1. FTS5 Sorgusu
            try:
                tokens = [t.strip() for t in clean_q.split() if len(t.strip()) > 1]
                fts_query = " OR ".join(f'"{t}"*' for t in tokens) if tokens else f'"{clean_q}"*'
                cursor = conn.execute(
                    """
                    SELECT s.* FROM symbols_fts f
                    JOIN symbols s ON s.rowid = f.rowid
                    WHERE symbols_fts MATCH ?
                    LIMIT ?
                    """,
                    (fts_query, limit),
                )
                rows = cursor.fetchall()
                for r in rows:
                    sym = self._row_to_symbol(r)
                    if symbol_type and sym.symbol_type.value != symbol_type.lower():
                        continue
                    if language and sym.language.lower() != language.lower():
                        continue
                    results.append(sym)
            except Exception:
                pass

            if not results:
                like_term = f"%{clean_q}%"
                cursor = conn.execute(
                    """
                    SELECT * FROM symbols
                    WHERE name LIKE ? OR fqn LIKE ? OR signature LIKE ?
                    LIMIT ?
                    """,
                    (like_term, like_term, like_term, limit),
                )
                for r in cursor.fetchall():
                    sym = self._row_to_symbol(r)
                    if symbol_type and sym.symbol_type.value != symbol_type.lower():
                        continue
                    if language and sym.language.lower() != language.lower():
                        continue
                    results.append(sym)

        return results[:limit]

    def _row_to_symbol(self, row: sqlite3.Row) -> CodeSymbol:
        meta = json.loads(row["metadata_json"] or "{}")
        return CodeSymbol(
            id=row["id"],
            name=row["name"],
            fully_qualified_name=row["fqn"],
            symbol_type=SymbolType(row["symbol_type"]),
            language=row["language"],
            file_path=row["file_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            parent_symbol_id=row["parent_symbol_id"],
            visibility=row["visibility"],
            signature=row["signature"] or "",
            docstring=row["docstring"] or "",
            is_exported=bool(row["is_exported"]),
            confidence=ParserConfidence(row["confidence"]),
            metadata=meta,
        )
