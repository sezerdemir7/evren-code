"""Go parser module."""

from __future__ import annotations

from typing import List, Optional

import tree_sitter
import tree_sitter_go

from evren_code.core.symbol_graph.models import (
    CodeSymbol,
    FileImport,
    ParserConfidence,
    RelationshipType,
    ResolutionStatus,
    SymbolReference,
    SymbolType,
)
from evren_code.core.symbol_graph.parsers.base import (
    BaseParser,
    ParseDiagnostics,
    ParsedResult,
)


class GoParser(BaseParser):
    """Go parser implementation."""

    def __init__(self) -> None:
        self.language = tree_sitter.Language(tree_sitter_go.language())
        self.parser = tree_sitter.Parser(self.language)

    @property
    def language_name(self) -> str:
        return "go"

    @property
    def supported_extensions(self) -> List[str]:
        return [".go"]

    def parse(self, file_path: str, code: str) -> ParsedResult:
        result = ParsedResult(file_path=file_path, language="go")
        code_bytes = code.encode("utf-8", errors="replace")

        try:
            tree = self.parser.parse(code_bytes)
        except Exception as exc:
            result.is_partial = True
            result.diagnostics.append(
                ParseDiagnostics(line=1, column=1, message=f"Go ayrıştırma hatası: {exc}")
            )
            return result

        def _node_text(node: tree_sitter.Node) -> str:
            return code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

        package_name = ""

        for child in tree.root_node.children:
            if child.type == "package_clause":
                for sub in child.children:
                    if sub.type == "package_identifier":
                        package_name = _node_text(sub).strip()
            elif child.type == "import_declaration":
                for sub in child.children:
                    if sub.type == "import_spec":
                        path_node = sub.child_by_field_name("path") or sub
                        mod = _node_text(path_node).strip("\"'`")
                        result.imports.append(
                            FileImport(file_path=file_path, imported_module=mod, line_number=sub.start_point[0] + 1)
                        )

        def _find_calls(curr: tree_sitter.Node, source_id: str) -> None:
            if curr.type == "call_expression":
                fn_node = curr.child_by_field_name("function")
                if fn_node:
                    c_name = _node_text(fn_node).strip()
                    if "." in c_name:
                        c_name = c_name.split(".")[-1]
                    if c_name:
                        result.references.append(
                            SymbolReference(
                                source_file=file_path,
                                source_symbol_id=source_id,
                                target_name=c_name,
                                target_symbol_id=None,
                                relationship_type=RelationshipType.CALLS,
                                line_number=curr.start_point[0] + 1,
                                resolution_status=ResolutionStatus.UNRESOLVED,
                                evidence=f"go call {c_name}()",
                            )
                        )
            for ch in curr.children:
                _find_calls(ch, source_id)

        for child in tree.root_node.children:
            if child.type == "type_declaration":
                for spec in child.children:
                    if spec.type == "type_spec":
                        name_node = spec.child_by_field_name("name")
                        type_node = spec.child_by_field_name("type")
                        if name_node and type_node:
                            t_name = _node_text(name_node).strip()
                            t_kind = type_node.type
                            stype = SymbolType.STRUCT if t_kind == "struct_type" else (
                                SymbolType.INTERFACE if t_kind == "interface_type" else SymbolType.TYPE_ALIAS
                            )
                            fqn = f"{package_name}.{t_name}" if package_name else t_name
                            sym_id = f"{file_path}::{fqn}::{stype.value}"
                            is_exp = t_name[0].isupper() if t_name else False
                            result.symbols.append(
                                CodeSymbol(
                                    id=sym_id,
                                    name=t_name,
                                    fully_qualified_name=fqn,
                                    symbol_type=stype,
                                    language="go",
                                    file_path=file_path,
                                    start_line=spec.start_point[0] + 1,
                                    end_line=spec.end_point[0] + 1,
                                    visibility="public" if is_exp else "internal",
                                    signature=f"type {t_name} {stype.value}",
                                    is_exported=is_exp,
                                    confidence=ParserConfidence.PARSER_VERIFIED,
                                )
                            )

            elif child.type == "function_declaration":
                name_node = child.child_by_field_name("name")
                if name_node:
                    f_name = _node_text(name_node).strip()
                    fqn = f"{package_name}.{f_name}" if package_name else f_name
                    fn_id = f"{file_path}::{fqn}::function"
                    is_exp = f_name[0].isupper() if f_name else False
                    result.symbols.append(
                        CodeSymbol(
                            id=fn_id,
                            name=f_name,
                            fully_qualified_name=fqn,
                            symbol_type=SymbolType.FUNCTION,
                            language="go",
                            file_path=file_path,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            visibility="public" if is_exp else "internal",
                            signature=f"func {f_name}()",
                            is_exported=is_exp,
                            confidence=ParserConfidence.PARSER_VERIFIED,
                        )
                    )
                    _find_calls(child, fn_id)

            elif child.type == "method_declaration":
                name_node = child.child_by_field_name("name")
                receiver_node = child.child_by_field_name("receiver")
                receiver_type = ""
                if receiver_node:
                    rec_text = _node_text(receiver_node).strip("()")
                    receiver_type = rec_text.replace("*", "").strip().split()[-1] if rec_text else ""

                if name_node:
                    m_name = _node_text(name_node).strip()
                    fqn = f"{package_name}.{receiver_type}.{m_name}" if receiver_type else f"{package_name}.{m_name}"
                    m_id = f"{file_path}::{fqn}::method"
                    is_exp = m_name[0].isupper() if m_name else False
                    result.symbols.append(
                        CodeSymbol(
                            id=m_id,
                            name=m_name,
                            fully_qualified_name=fqn,
                            symbol_type=SymbolType.METHOD,
                            language="go",
                            file_path=file_path,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            parent_symbol_id=f"{file_path}::{package_name}.{receiver_type}::struct" if receiver_type else None,
                            visibility="public" if is_exp else "internal",
                            signature=f"func ({receiver_type}) {m_name}()",
                            is_exported=is_exp,
                            confidence=ParserConfidence.PARSER_VERIFIED,
                        )
                    )
                    _find_calls(child, m_id)

        return result
