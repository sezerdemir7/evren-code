"""Engine module."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evren_code.config import AppConfig
from evren_code.core.repo_intelligence.scanner import RepoScanner, ScanLimits
from evren_code.core.symbol_graph.graph import SymbolGraphEngine
from evren_code.core.symbol_graph.impact_analyzer import ImpactAnalyzer
from evren_code.core.symbol_graph.index import SymbolIndexDB
from evren_code.core.symbol_graph.models import (
    CodeSymbol,
    ImpactReport,
    IndexingStats,
    RelationshipType,
    SymbolType,
)
from evren_code.core.symbol_graph.parsers.registry import ParserRegistry
from evren_code.core.symbol_graph.reference_resolver import ReferenceResolver
from evren_code.core.symbol_graph.spring_boot import SpringBootAnalyzer
from evren_code.core.symbol_graph.test_linker import TestLinker


class CodebaseSymbolEngine:
    """Codebase symbol engine implementation."""

    def __init__(
        self,
        workspace_root: Path,
        config: Optional[AppConfig] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.config = config
        self.db = SymbolIndexDB(self.workspace_root, db_path=db_path)
        self.parser_registry = ParserRegistry()
        self.resolver = ReferenceResolver(self.db)
        self.graph = SymbolGraphEngine(self.db)
        self.spring_boot_analyzer = SpringBootAnalyzer(self.db)
        self.test_linker = TestLinker(self.db)
        self.impact_analyzer = ImpactAnalyzer(self.db, self.graph)

    def index_codebase(
        self,
        force_reindex: bool = False,
        max_files: int = 5000,
    ) -> IndexingStats:
        """Index codebase (incremental)."""
        t0 = time.perf_counter()
        stats = IndexingStats()

        if force_reindex:
            self.db.clear()

        custom_ignores = []
        if self.config and hasattr(self.config, "tools") and hasattr(self.config.tools, "ignored_dirs"):
            custom_ignores.extend(self.config.tools.ignored_dirs)

        scanner = RepoScanner(
            self.workspace_root,
            limits=ScanLimits(max_files=max_files),
            custom_ignores=custom_ignores,
        )
        scan_res = scanner.scan()

        known_files = self.db.get_indexed_files()
        current_supported_files: Dict[str, Any] = {}

        for rel_path, finfo in scan_res.files.items():
            if not self.parser_registry.is_supported(rel_path):
                continue
            stats.files_discovered += 1
            current_supported_files[rel_path] = finfo

            cached_info = known_files.get(rel_path)
            if (
                not force_reindex
                and cached_info is not None
                and cached_info[0] == finfo.mtime
                and cached_info[1] == self.db.PARSER_VERSION
            ):
                stats.files_skipped += 1
                continue

            try:
                text = finfo.abs_path.read_text(encoding="utf-8", errors="replace")
                parsed = self.parser_registry.parse_file(rel_path, text)
                if parsed.diagnostics and any(d.severity == "error" for d in parsed.diagnostics):
                    stats.parse_errors += 1

                self.db.save_file_data(
                    file_path=rel_path,
                    mtime=finfo.mtime,
                    language=parsed.language,
                    symbols=parsed.symbols,
                    imports=parsed.imports,
                    references=parsed.references,
                )
                stats.files_parsed += 1
                stats.symbols_extracted += len(parsed.symbols)
                stats.references_extracted += len(parsed.references)
            except Exception:
                stats.parse_errors += 1

        deleted = set(known_files.keys()) - set(current_supported_files.keys())
        for df in deleted:
            self.db.remove_file(df)

        resolved, ambiguous, unresolved = self.resolver.resolve_all_unresolved()
        stats.resolved_relationships = resolved
        stats.unresolved_relationships = unresolved + ambiguous
        stats.duration_seconds = round(time.perf_counter() - t0, 3)

        return stats

    def search_symbols(
        self,
        query: str,
        symbol_type: Optional[str] = None,
        language: Optional[str] = None,
        limit: int = 15,
    ) -> str:
        """Search symbols."""
        syms = self.db.search_symbols(query, symbol_type=symbol_type, language=language, limit=limit)
        if not syms:
            return f"'{query}' aramasıyla eşleşen bir sembol bulunamadı."

        lines = [f"### Sembol Arama Sonuçları: '{query}' ({len(syms)} adet)"]
        for s in syms:
            lines.append(
                f"- `{s.name}` [{s.symbol_type.value}] ({s.language}): `{s.file_path}:{s.start_line}`"
            )
            if s.signature:
                lines.append(f"  İmza: `{s.signature}`")
        return "\n".join(lines)

    def find_symbol_definition(self, name_or_id: str) -> str:
        """Find symbol definition."""
        sym = self.db.get_symbol_by_id(name_or_id)
        if not sym:
            matches = self.db.get_symbols_by_name(name_or_id)
            if matches:
                sym = matches[0]

        if not sym:
            return f"Hata: '{name_or_id}' adında veya kimliğinde bir sembol tanımı bulunamadı."

        lines = [
            f"### Sembol Tanımı: `{sym.name}`",
            f"- **Tür:** {sym.symbol_type.value}",
            f"- **Dil:** {sym.language}",
            f"- **Dosya:** `{sym.file_path}` (Satır: {sym.start_line} - {sym.end_line})",
            f"- **Görünürlük:** {sym.visibility}",
            f"- **İmza:** `{sym.signature}`",
            f"- **Dışa Aktarılmış (Exported):** {'Evet' if sym.is_exported else 'Hayır'}",
        ]
        if sym.docstring:
            lines.append(f"- **Dokümantasyon:** {sym.docstring.strip()[:300]}")
        return "\n".join(lines)

    def find_symbol_references(self, symbol_name_or_id: str, limit: int = 25) -> str:
        """Find symbol references."""
        sym = self.db.get_symbol_by_id(symbol_name_or_id)
        if not sym:
            matches = self.db.get_symbols_by_name(symbol_name_or_id)
            if matches:
                sym = matches[0]

        target_id = sym.id if sym else symbol_name_or_id
        edges = self.graph.get_incoming_references(target_id, limit=limit)
        if not edges and sym:
            with self.db.get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM symbol_references WHERE target_name = ? LIMIT ?",
                    (sym.name, limit),
                )
                edges = [dict(r) for r in cur.fetchall()]

        if not edges:
            return f"'{symbol_name_or_id}' sembolüne doğrudan referans veren veya çağıran bir kod bulunamadı."

        lines = [f"### '{symbol_name_or_id}' Sembolüne Gelen Referanslar ({len(edges)} adet):"]
        for e in edges:
            src = e.get("source_name") or e.get("source_file")
            lines.append(
                f"- `{e['source_file']}:{e['line_number']}` -> [{e['relationship_type']}] `{src}` ({e.get('evidence', '')})"
            )
        return "\n".join(lines)

    def analyze_change_impact(self, symbol_name_or_id: str) -> str:
        """Analyze change impact."""
        report = self.impact_analyzer.analyze_symbol(symbol_name_or_id)
        sym = report.target_symbol

        lines = [
            f"## Değişiklik Etki Analizi: `{sym.name}`",
            f"- **Hedef Sembol:** `{sym.fully_qualified_name}` [{sym.symbol_type.value}]",
            f"- **Dosya Konumu:** `{sym.file_path}:{sym.start_line}`",
            f"- **Etkilenen Modüller:** {', '.join(report.affected_modules) if report.affected_modules else '.'}",
        ]

        if report.direct_callers:
            lines.append(f"\n### 1. Doğrudan Etkilenenler / Çağıranlar ({len(report.direct_callers)} adet):")
            for c in report.direct_callers[:10]:
                lines.append(f"  - `{c['source_file']}:{c['line']}` -> `{c['source_symbol']}` ({c['evidence']})")
            if len(report.direct_callers) > 10:
                lines.append(f"  ... ve {len(report.direct_callers) - 10} adet daha")
        else:
            lines.append("\n- **Doğrudan Çağıran:** Doğrudan statik çağrı tespit edilmedi.")

        if report.indirect_callers:
            lines.append(f"\n### 2. Dolaylı Etkilenenler ({len(report.indirect_callers)} adet):")
            for c in report.indirect_callers[:8]:
                lines.append(f"  - [Derinlik {c['depth']}] `{c['source_file']}:{c['line']}` -> `{c['source_symbol']}`")
            if len(report.indirect_callers) > 8:
                lines.append(f"  ... ve {len(report.indirect_callers) - 8} adet daha")

        if report.implementations:
            lines.append(f"\n### 3. Bu Arayüzü Uygulayan Sınıflar ({len(report.implementations)} adet):")
            for imp in report.implementations[:5]:
                lines.append(f"  - `{imp['source_file']}` -> `{imp.get('implementor_name', '')}`")
        if report.related_interfaces:
            lines.append(f"\n### 4. İlişkili Arayüzler:")
            for iface in report.related_interfaces[:5]:
                lines.append(f"  - `{iface['target_name']}` ({iface['evidence']})")

        if report.related_test_files:
            lines.append(f"\n### 5. Koşulması Gereken İlgili Testler ({len(report.related_test_files)} adet):")
            for t in report.related_test_files[:8]:
                m_label = f" -> `{t['test_method']}`" if t.get("test_method") else ""
                lines.append(f"  - [{t['match_type']}] `{t['test_file']}`{m_label}")
        else:
            lines.append("\n- **İlgili Test:** Bu sembolle doğrudan ilişkili bir test tespit edilemedi.")

        if report.limitations:
            lines.append("\n> [!NOTE]\n> **Statik Analiz Sınırları:** " + " ".join(report.limitations))

        return "\n".join(lines)

    def find_related_tests(self, target: str) -> str:
        """Find related tests."""
        if "/" in target or target.endswith((".py", ".java", ".ts", ".js", ".go", ".rs")):
            tests = self.test_linker.find_tests_for_file(target)
        else:
            syms = self.db.get_symbols_by_name(target)
            if syms:
                tests = self.test_linker.find_tests_for_symbol(syms[0])
            else:
                return f"'{target}' için sembol veya dosya bulunamadı."

        if not tests:
            return f"'{target}' için ilişkili test dosyası bulunamadı."

        lines = [f"### İlgili Test Dosyaları: '{target}' ({len(tests)} adet):"]
        for t in tests:
            m_label = f" -> `{t['test_method']}`" if t.get("test_method") else ""
            lines.append(f"- [{t['match_type']}] `{t['test_file']}`{m_label} ({t['evidence']})")
        return "\n".join(lines)
