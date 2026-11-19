"""Rust parser module."""

from __future__ import annotations

from typing import List, Optional

import tree_sitter
import tree_sitter_rust

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


class RustParser(BaseParser):
    """Rust parser implementation."""

    def __init__(self) -> None:
        self.language = tree_sitter.Language(tree_sitter_rust.language())
        self.parser = tree_sitter.Parser(self.language)

    @property
    def language_name(self) -> str:
        return "rust"

    @property
    def supported_extensions(self) -> List[str]:
        return [".rs"]

    def parse(self, file_path: str, code: str) -> ParsedResult:
        result = ParsedResult(file_path=file_path, language="rust")
        code_bytes = code.encode("utf-8", errors="replace")

        try:
            tree = self.parser.parse(code_bytes)
        except Exception as exc:
            result.is_partial = True
            result.diagnostics.append(
                ParseDiagnostics(line=1, column=1, message=f"Rust ayrıştırma hatası: {exc}")
            )
            return result

        def _node_text(node: tree_sitter.Node) -> str:
            return code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

        def _has_pub(node: tree_sitter.Node) -> bool:
            vis_node = node.child_by_field_name("visibility_modifier")
            return vis_node is not None or any(c.type == "visibility_modifier" for c in node.children)

        def _find_calls(curr: tree_sitter.Node, source_id: str) -> None:
            if curr.type == "call_expression":
                fn_node = curr.child_by_field_name("function")
                if fn_node:
                    c_name = _node_text(fn_node).strip()
                    if "::" in c_name:
                        c_name = c_name.split("::")[-1]
                    elif "." in c_name:
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
                                evidence=f"rust call {c_name}()",
                            )
                        )
            for ch in curr.children:
                _find_calls(ch, source_id)

        for child in tree.root_node.children:
            # 1. Use / Import
            if child.type == "use_declaration":
                arg_node = child.child_by_field_name("argument")
                use_text = _node_text(arg_node).strip() if arg_node else _node_text(child).strip()
                result.imports.append(
                    FileImport(file_path=file_path, imported_module=use_text, line_number=child.start_point[0] + 1)
                )

            # 2. Struct
            elif child.type == "struct_item":
                name_node = child.child_by_field_name("name")
                if name_node:
                    s_name = _node_text(name_node).strip()
                    s_id = f"{file_path}::{s_name}::struct"
                    is_pub = _has_pub(child)
                    result.symbols.append(
                        CodeSymbol(
                            id=s_id,
                            name=s_name,
                            fully_qualified_name=s_name,
                            symbol_type=SymbolType.STRUCT,
                            language="rust",
                            file_path=file_path,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            visibility="public" if is_pub else "internal",
                            signature=f"{'pub ' if is_pub else ''}struct {s_name}",
                            is_exported=is_pub,
                            confidence=ParserConfidence.PARSER_VERIFIED,
                        )
                    )

            # 3. Trait
            elif child.type == "trait_item":
                name_node = child.child_by_field_name("name")
                if name_node:
                    t_name = _node_text(name_node).strip()
                    t_id = f"{file_path}::{t_name}::trait"
                    is_pub = _has_pub(child)
                    result.symbols.append(
                        CodeSymbol(
                            id=t_id,
                            name=t_name,
                            fully_qualified_name=t_name,
                            symbol_type=SymbolType.TRAIT,
                            language="rust",
                            file_path=file_path,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            visibility="public" if is_pub else "internal",
                            signature=f"{'pub ' if is_pub else ''}trait {t_name}",
                            is_exported=is_pub,
                            confidence=ParserConfidence.PARSER_VERIFIED,
                        )
                    )

            # 4. Standalone Function
            elif child.type == "function_item":
                name_node = child.child_by_field_name("name")
                if name_node:
                    fn_name = _node_text(name_node).strip()
                    fn_id = f"{file_path}::{fn_name}::function"
                    is_pub = _has_pub(child)
                    result.symbols.append(
                        CodeSymbol(
                            id=fn_id,
                            name=fn_name,
                            fully_qualified_name=fn_name,
                            symbol_type=SymbolType.FUNCTION,
                            language="rust",
                            file_path=file_path,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            visibility="public" if is_pub else "internal",
                            signature=f"{'pub ' if is_pub else ''}fn {fn_name}()",
                            is_exported=is_pub,
                            confidence=ParserConfidence.PARSER_VERIFIED,
                        )
                    )
                    _find_calls(child, fn_id)

            # 5. Impl Block
            elif child.type == "impl_item":
                trait_node = child.child_by_field_name("trait")
                type_node = child.child_by_field_name("type")
                body_node = child.child_by_field_name("body")

                t_name = _node_text(trait_node).strip() if trait_node else ""
                type_name = _node_text(type_node).strip() if type_node else ""

                if t_name and type_name:
                    result.references.append(
                        SymbolReference(
                            source_file=file_path,
                            source_symbol_id=f"{file_path}::{type_name}::struct",
                            target_name=t_name,
                            target_symbol_id=None,
                            relationship_type=RelationshipType.IMPLEMENTS,
                            line_number=child.start_point[0] + 1,
                            resolution_status=ResolutionStatus.UNRESOLVED,
                            evidence=f"impl {t_name} for {type_name}",
                        )
                    )

                if body_node:
                    for member in body_node.children:
                        if member.type == "function_item":
                            m_name_node = member.child_by_field_name("name")
                            if m_name_node:
                                m_name = _node_text(m_name_node).strip()
                                fqn = f"{type_name}.{m_name}" if type_name else m_name
                                m_id = f"{file_path}::{fqn}::method"
                                is_pub = _has_pub(member)
                                result.symbols.append(
                                    CodeSymbol(
                                        id=m_id,
                                        name=m_name,
                                        fully_qualified_name=fqn,
                                        symbol_type=SymbolType.METHOD,
                                        language="rust",
                                        file_path=file_path,
                                        start_line=member.start_point[0] + 1,
                                        end_line=member.end_point[0] + 1,
                                        parent_symbol_id=f"{file_path}::{type_name}::struct" if type_name else None,
                                        visibility="public" if is_pub else "internal",
                                        signature=f"{'pub ' if is_pub else ''}fn {m_name}()",
                                        is_exported=is_pub,
                                        confidence=ParserConfidence.PARSER_VERIFIED,
                                        metadata={"trait": t_name, "type": type_name},
                                    )
                                )
                                _find_calls(member, m_id)

        return result
