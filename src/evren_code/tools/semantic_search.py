"""Local Semantic Code Search and RAG Engine (`semantic_search_code`, `reindex_codebase`).

Instead of blind regex or filesystem crawling on large codebases:
1. Parses Python, TypeScript, JavaScript, Go, Rust files into AST and class/function blocks.
2. Indexes code blocks, parameters, return types, and docstrings using local SQLite FTS5 / BM25.
3. Retrieves relevant snippets (file_path, start_line, end_line, snippet) for natural language queries.
"""

from __future__ import annotations

import ast
import math
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from evren_code.config import AppConfig
from evren_code.tools.filesystem import resolve_safe_path
from evren_code.tools.registry import ToolRegistry


@dataclass
class CodeChunk:
    """Represents a single indexed code chunk (function, class, interface, or type)."""

    file_path: str
    symbol_name: str
    symbol_type: str  # "class", "function", "interface", "type", "module"
    start_line: int
    end_line: int
    content: str
    docstring: str = ""
    keywords: str = ""


class SemanticCodeIndex:
    """Lightweight and high-performance local SQLite-based semantic code index."""

    def __init__(self, workspace_root: Path, db_path: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.db_path = db_path or (self.workspace_root / ".evren" / "code_index.db")
        self._ensure_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    file_path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS code_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    symbol_name TEXT NOT NULL,
                    symbol_type TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    docstring TEXT,
                    keywords TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS code_chunks_fts USING fts5(
                    symbol_name,
                    symbol_type,
                    docstring,
                    keywords,
                    content,
                    content='code_chunks',
                    content_rowid='id'
                )
                """
            )
            conn.commit()

    def clear(self) -> None:
        """Completely resets the index."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM indexed_files")
            conn.execute("DELETE FROM code_chunks")
            conn.execute("DELETE FROM code_chunks_fts")
            conn.commit()

    def index_codebase(
        self,
        max_files: int = 5000,
        ignored_dirs: Optional[List[str]] = None,
        force_reindex: bool = False,
    ) -> int:
        """Index codebase."""
        ignores = set(ignored_dirs or [".git", "node_modules", ".next", "__pycache__", "venv", ".venv", "dist", "build"])
        supported_exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs"}

        if force_reindex:
            self.clear()

        # Get mtime records of indexed files
        known_mtimes: Dict[str, float] = {}
        with self._get_connection() as conn:
            try:
                for row in conn.execute("SELECT file_path, mtime FROM indexed_files"):
                    known_mtimes[row[0]] = float(row[1])
            except Exception:
                pass

        file_count = 0
        new_chunks_count = 0
        current_files: Set[str] = set()

        for root, dirs, files in os.walk(self.workspace_root):
            dirs[:] = [d for d in dirs if d not in ignores and not d.startswith(".")]
            for file in files:
                p = Path(root) / file
                if p.suffix in supported_exts:
                    file_count += 1
                    rel_path = str(p.relative_to(self.workspace_root).as_posix())
                    current_files.add(rel_path)

                    try:
                        mtime = p.stat().st_mtime
                    except OSError:
                        continue

                    # Skip if file unmodified (incremental indexing)
                    if not force_reindex and known_mtimes.get(rel_path) == mtime:
                        continue

                    try:
                        text = p.read_text(encoding="utf-8", errors="replace")
                        if p.suffix == ".py":
                            file_chunks = self._extract_python_chunks(rel_path, text)
                        else:
                            file_chunks = self._extract_ts_chunks(rel_path, text)

                        with self._get_connection() as conn:
                            conn.execute(
                                "DELETE FROM code_chunks_fts WHERE rowid IN (SELECT id FROM code_chunks WHERE file_path = ?)",
                                (rel_path,),
                            )
                            conn.execute("DELETE FROM code_chunks WHERE file_path = ?", (rel_path,))
                            conn.execute(
                                "INSERT OR REPLACE INTO indexed_files (file_path, mtime) VALUES (?, ?)",
                                (rel_path, mtime),
                            )
                            for chunk in file_chunks:
                                cur = conn.execute(
                                    """
                                    INSERT INTO code_chunks 
                                    (file_path, symbol_name, symbol_type, start_line, end_line, content, docstring, keywords)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        chunk.file_path,
                                        chunk.symbol_name,
                                        chunk.symbol_type,
                                        chunk.start_line,
                                        chunk.end_line,
                                        chunk.content,
                                        chunk.docstring,
                                        chunk.keywords,
                                    ),
                                )
                                rowid = cur.lastrowid
                                conn.execute(
                                    """
                                    INSERT INTO code_chunks_fts (rowid, symbol_name, symbol_type, docstring, keywords, content)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        rowid,
                                        chunk.symbol_name,
                                        chunk.symbol_type,
                                        chunk.docstring,
                                        chunk.keywords,
                                        chunk.content,
                                    ),
                                )
                            conn.commit()
                            new_chunks_count += len(file_chunks)
                    except Exception:
                        pass

                    if file_count >= max_files:
                        break
            if file_count >= max_files:
                break

        deleted_files = set(known_mtimes.keys()) - current_files
        if deleted_files:
            with self._get_connection() as conn:
                for df in deleted_files:
                    conn.execute(
                        "DELETE FROM code_chunks_fts WHERE rowid IN (SELECT id FROM code_chunks WHERE file_path = ?)",
                        (df,),
                    )
                    conn.execute("DELETE FROM code_chunks WHERE file_path = ?", (df,))
                    conn.execute("DELETE FROM indexed_files WHERE file_path = ?", (df,))
                conn.commit()

        return new_chunks_count

    def _extract_python_chunks(self, file_path: str, content: str) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        lines = content.splitlines()
        try:
            tree = ast.parse(content)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = node.name
                    stype = "class" if isinstance(node, ast.ClassDef) else "function"
                    sline = node.lineno
                    eline = getattr(node, "end_lineno", sline + 10)
                    doc = ast.get_docstring(node) or ""
                    snippet = "\n".join(lines[sline - 1 : eline])
                    keywords = f"{name} {stype} {' '.join(self._tokenize_identifier(name))}"
                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            symbol_name=name,
                            symbol_type=stype,
                            start_line=sline,
                            end_line=eline,
                            content=snippet,
                            docstring=doc,
                            keywords=keywords,
                        )
                    )
        except Exception:
            # Fallback regex
            chunks.extend(self._extract_regex_chunks(file_path, content))
        return chunks

    def _extract_ts_chunks(self, file_path: str, content: str) -> List[CodeChunk]:
        return self._extract_regex_chunks(file_path, content)

    def _extract_regex_chunks(self, file_path: str, content: str) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        lines = content.splitlines()
        patterns = [
            (r"(?:export\s+)?class\s+([A-Za-z0-9_]+)", "class"),
            (r"(?:export\s+)?interface\s+([A-Za-z0-9_]+)", "interface"),
            (r"(?:export\s+)?type\s+([A-Za-z0-9_]+)", "type"),
            (r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)", "function"),
            (r"(?:export\s+)?const\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", "function"),
        ]

        for i, line in enumerate(lines, start=1):
            for pat, stype in patterns:
                m = re.search(pat, line)
                if m:
                    name = m.group(1)
                    sline = i
                    eline = min(len(lines), sline + 25)
                    snippet = "\n".join(lines[sline - 1 : eline])
                    keywords = f"{name} {stype} {' '.join(self._tokenize_identifier(name))}"
                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            symbol_name=name,
                            symbol_type=stype,
                            start_line=sline,
                            end_line=eline,
                            content=snippet,
                            keywords=keywords,
                        )
                    )
                    break
        return chunks

    @staticmethod
    def _tokenize_identifier(name: str) -> List[str]:
        """Tokenize identifier."""
        s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s1)
        tokens = re.split(r"[_\s\-]+", s2.lower())
        return [t for t in tokens if len(t) > 1]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search."""
        tokens = [t.strip().lower() for t in re.split(r"[\s,\.\-_]+", query) if len(t.strip()) > 1]
        if not tokens:
            return []

        fts_query = " OR ".join(f'"{t}"*' for t in tokens)

        results: List[Dict[str, Any]] = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute(
                    """
                    SELECT c.file_path, c.symbol_name, c.symbol_type, c.start_line, c.end_line, c.content, c.docstring,
                           rank
                    FROM code_chunks_fts f
                    JOIN code_chunks c ON c.id = f.rowid
                    WHERE code_chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, top_k),
                )
                for row in cursor.fetchall():
                    results.append(
                        {
                            "file_path": row["file_path"],
                            "symbol_name": row["symbol_name"],
                            "symbol_type": row["symbol_type"],
                            "start_line": row["start_line"],
                            "end_line": row["end_line"],
                            "docstring": row["docstring"] or "",
                            "snippet": row["content"][:600],
                            "score": round(float(row["rank"]), 4),
                        }
                    )
            except Exception:
                like_clauses = " OR ".join(["content LIKE ?" for _ in tokens])
                params = [f"%{t}%" for t in tokens] + [top_k]
                cursor = conn.execute(
                    f"""
                    SELECT file_path, symbol_name, symbol_type, start_line, end_line, content, docstring
                    FROM code_chunks
                    WHERE {like_clauses}
                    LIMIT ?
                    """,
                    params,
                )
                for row in cursor.fetchall():
                    results.append(
                        {
                            "file_path": row["file_path"],
                            "symbol_name": row["symbol_name"],
                            "symbol_type": row["symbol_type"],
                            "start_line": row["start_line"],
                            "end_line": row["end_line"],
                            "docstring": row["docstring"] or "",
                            "snippet": row["content"][:600],
                            "score": 1.0,
                        }
                    )

        return results


def register_semantic_search_tools(registry: ToolRegistry, config: AppConfig) -> None:
    """Registers semantic code search and indexing tools."""
    index = SemanticCodeIndex(config.workspace_root)

    def semantic_search_code(query: str, top_k: Optional[int] = 5) -> str:
        # If index is empty, automatically build initial index
        with index._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM code_chunks")
            count = cur.fetchone()[0]
        if count == 0:
            index.index_codebase(ignored_dirs=config.tools.ignored_dirs)

        limit = max(1, min(10, int(top_k or 5)))
        matches = index.search(query, top_k=limit)
        if not matches:
            return f"Semantic Search: No matching code chunks found for '{query}' (Semantik Arama: kod parçası bulunamadı)."

        output_parts = [f"### Semantic Code Search Results / Semantik Kod Arama Sonuçları: '{query}' ({len(matches)} results)"]
        for idx, m in enumerate(matches, start=1):
            header = f"{idx}. [{m['symbol_type'].upper()}] {m['symbol_name']} -> `{m['file_path']}:{m['start_line']}-{m['end_line']}`"
            doc = f"   Docstring / Açıklama: {m['docstring']}\n" if m['docstring'] else ""
            code_block = f"```\n{m['snippet']}\n```"
            output_parts.append(f"{header}\n{doc}{code_block}")

        return "\n\n".join(output_parts)

    def reindex_codebase() -> str:
        count = index.index_codebase(ignored_dirs=config.tools.ignored_dirs)
        return f"Success: Codebase reindexed (Başarılı: Kod tabanı yeniden indekslendi - Toplam {count} semantik parça)."

    registry.register(
        name="semantic_search_code",
        description=(
            "Performs semantic natural language search over codebase functions, classes, and types (SQLite FTS5). "
            "Use when you do not know exact symbol or file names (e.g. 'auth password hashing', 'data table component')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query or technical concept description."},
                "top_k": {"type": "integer", "description": "Maximum number of matches to return (default: 5)."},
            },
            "required": ["query"],
        },
        handler=semantic_search_code,
        requires_approval=False,
    )

    registry.register(
        name="reindex_codebase",
        description="Re-scans the workspace to update the local semantic search index (SQLite FTS5).",
        parameters={"type": "object", "properties": {}},
        handler=reindex_codebase,
        requires_approval=False,
    )
