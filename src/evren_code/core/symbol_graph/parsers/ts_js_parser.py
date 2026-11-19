"""Ts js parser module."""

from __future__ import annotations

from typing import List, Optional, Tuple

import tree_sitter
import tree_sitter_javascript
import tree_sitter_typescript

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


class TypeScriptJavaScriptParser(BaseParser):
    """Type script java script parser implementation."""

    def __init__(self) -> None:
        self.ts_lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
        self.tsx_lang = tree_sitter.Language(tree_sitter_typescript.language_tsx())
        self.js_lang = tree_sitter.Language(tree_sitter_javascript.language())

        self.ts_parser = tree_sitter.Parser(self.ts_lang)
        self.tsx_parser = tree_sitter.Parser(self.tsx_lang)
        self.js_parser = tree_sitter.Parser(self.js_lang)

    @property
    def language_name(self) -> str:
        return "typescript"

    @property
    def supported_extensions(self) -> List[str]:
        return [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]

    def _get_parser_for_path(self, path: str) -> tree_sitter.Parser:
        if path.endswith(".tsx"):
            return self.tsx_parser
        elif path.endswith(".ts"):
            return self.ts_parser
        return self.js_parser

    def parse(self, file_path: str, code: str) -> ParsedResult:
        is_ts = file_path.endswith((".ts", ".tsx"))
        lang_label = "typescript" if is_ts else "javascript"
        result = ParsedResult(file_path=file_path, language=lang_label)
        code_bytes = code.encode("utf-8", errors="replace")

        parser = self._get_parser_for_path(file_path)
        try:
            tree = parser.parse(code_bytes)
        except Exception as exc:
            result.is_partial = True
            result.diagnostics.append(
                ParseDiagnostics(line=1, column=1, message=f"TS/JS ayrıştırma hatası: {exc}")
            )
            return result

        if tree.root_node.has_error:
            result.is_partial = True
            result.diagnostics.append(
                ParseDiagnostics(
                    line=1,
                    column=1,
                    message="Sözdizimi hataları tespit edildi, kısmi AST çıkarılıyor.",
                    severity="warning",
                )
            )

        def _node_text(node: tree_sitter.Node) -> str:
            return code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

        def _find_calls_in_node(curr: tree_sitter.Node, source_id: str) -> None:
            if curr.type == "call_expression":
                fn_node = curr.children[0] if curr.children else None
                call_name = ""
                if fn_node:
                    if fn_node.type == "identifier":
                        call_name = _node_text(fn_node).strip()
                    elif fn_node.type == "member_expression":
                        # obj.method() -> method
                        prop = fn_node.child_by_field_name("property")
                        if prop:
                            call_name = _node_text(prop).strip()
                if call_name:
                    result.references.append(
                        SymbolReference(
                            source_file=file_path,
                            source_symbol_id=source_id,
                            target_name=call_name,
                            target_symbol_id=None,
                            relationship_type=RelationshipType.CALLS,
                            line_number=curr.start_point[0] + 1,
                            resolution_status=ResolutionStatus.UNRESOLVED,
                            evidence=f"call {call_name}()",
                        )
                    )
            for child in curr.children:
                _find_calls_in_node(child, source_id)

        def _visit_node(node: tree_sitter.Node, is_exported: bool = False, parent_id: Optional[str] = None, parent_fqn: str = "") -> None:
            if node.type == "export_statement":
                for child in node.children:
                    if child.type != "export":
                        _visit_node(child, is_exported=True, parent_id=parent_id, parent_fqn=parent_fqn)
                return

            # Import ifadeleri
            if node.type == "import_statement":
                module_name = ""
                imported_symbols: List[str] = []
                for child in node.children:
                    if child.type == "string":
                        module_name = _node_text(child).strip("'\"`")
                    elif child.type == "import_clause":
                        clause_text = _node_text(child)
                        for part in clause_text.replace("{", "").replace("}", "").split(","):
                            clean_part = part.strip()
                            if clean_part:
                                sym = clean_part.split(" as ")[0].strip()
                                imported_symbols.append(sym)
                if module_name:
                    result.imports.append(
                        FileImport(
                            file_path=file_path,
                            imported_module=module_name,
                            imported_symbols=imported_symbols,
                            line_number=node.start_point[0] + 1,
                        )
                    )
                    for sym in imported_symbols:
                        result.references.append(
                            SymbolReference(
                                source_file=file_path,
                                source_symbol_id=None,
                                target_name=sym,
                                target_symbol_id=None,
                                relationship_type=RelationshipType.IMPORTS,
                                line_number=node.start_point[0] + 1,
                                resolution_status=ResolutionStatus.UNRESOLVED,
                                evidence=f"import {{{sym}}} from '{module_name}'",
                            )
                        )
                return

            if node.type == "class_declaration":
                name = ""
                extends_target = ""
                implements_targets: List[str] = []
                body_node: Optional[tree_sitter.Node] = None

                for child in node.children:
                    if child.type in ("type_identifier", "identifier"):
                        name = _node_text(child).strip()
                    elif child.type == "class_heritage":
                        for clause in child.children:
                            if clause.type == "extends_clause":
                                for c_child in clause.children:
                                    if c_child.type in ("type_identifier", "identifier"):
                                        extends_target = _node_text(c_child).strip()
                            elif clause.type == "implements_clause":
                                for c_child in clause.children:
                                    if c_child.type in ("type_identifier", "identifier"):
                                        implements_targets.append(_node_text(c_child).strip())
                    elif child.type == "class_body":
                        body_node = child

                if not name:
                    return

                fqn = f"{parent_fqn}.{name}" if parent_fqn else name
                c_id = f"{file_path}::{fqn}::class"

                if extends_target:
                    result.references.append(
                        SymbolReference(
                            source_file=file_path,
                            source_symbol_id=c_id,
                            target_name=extends_target,
                            target_symbol_id=None,
                            relationship_type=RelationshipType.EXTENDS,
                            line_number=node.start_point[0] + 1,
                            resolution_status=ResolutionStatus.UNRESOLVED,
                            evidence=f"class {name} extends {extends_target}",
                        )
                    )

                for iface in implements_targets:
                    result.references.append(
                        SymbolReference(
                            source_file=file_path,
                            source_symbol_id=c_id,
                            target_name=iface,
                            target_symbol_id=None,
                            relationship_type=RelationshipType.IMPLEMENTS,
                            line_number=node.start_point[0] + 1,
                            resolution_status=ResolutionStatus.UNRESOLVED,
                            evidence=f"class {name} implements {iface}",
                        )
                    )

                sig = f"{'export ' if is_exported else ''}class {name}"
                if extends_target:
                    sig += f" extends {extends_target}"
                if implements_targets:
                    sig += f" implements {', '.join(implements_targets)}"

                result.symbols.append(
                    CodeSymbol(
                        id=c_id,
                        name=name,
                        fully_qualified_name=fqn,
                        symbol_type=SymbolType.CLASS,
                        language=lang_label,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent_symbol_id=parent_id,
                        visibility="public" if is_exported else "internal",
                        signature=sig,
                        is_exported=is_exported,
                        confidence=ParserConfidence.PARSER_VERIFIED,
                        metadata={"extends": extends_target, "implements": implements_targets},
                    )
                )

                if body_node:
                    for member in body_node.children:
                        if member.type == "method_definition":
                            m_name = ""
                            for m_child in member.children:
                                if m_child.type in ("property_identifier", "identifier"):
                                    m_name = _node_text(m_child).strip()
                            if m_name:
                                m_fqn = f"{fqn}.{m_name}"
                                is_ctor = m_name == "constructor"
                                m_stype = SymbolType.CONSTRUCTOR if is_ctor else SymbolType.METHOD
                                m_id = f"{file_path}::{m_fqn}::{m_stype.value}"
                                result.symbols.append(
                                    CodeSymbol(
                                        id=m_id,
                                        name=m_name,
                                        fully_qualified_name=m_fqn,
                                        symbol_type=m_stype,
                                        language=lang_label,
                                        file_path=file_path,
                                        start_line=member.start_point[0] + 1,
                                        end_line=member.end_point[0] + 1,
                                        parent_symbol_id=c_id,
                                        visibility="public",
                                        signature=f"{m_name}()",
                                        is_exported=is_exported,
                                        confidence=ParserConfidence.PARSER_VERIFIED,
                                    )
                                )
                                _find_calls_in_node(member, m_id)
                return

            # Interface Deklarasyonu
            if node.type == "interface_declaration":
                name = ""
                extends_list: List[str] = []
                for child in node.children:
                    if child.type in ("type_identifier", "identifier"):
                        name = _node_text(child).strip()
                    elif child.type == "extends_type_clause":
                        for c_child in child.children:
                            if c_child.type in ("type_identifier", "identifier"):
                                extends_list.append(_node_text(c_child).strip())

                if name:
                    fqn = f"{parent_fqn}.{name}" if parent_fqn else name
                    iface_id = f"{file_path}::{fqn}::interface"
                    for ext in extends_list:
                        result.references.append(
                            SymbolReference(
                                source_file=file_path,
                                source_symbol_id=iface_id,
                                target_name=ext,
                                target_symbol_id=None,
                                relationship_type=RelationshipType.EXTENDS,
                                line_number=node.start_point[0] + 1,
                                resolution_status=ResolutionStatus.UNRESOLVED,
                                evidence=f"interface {name} extends {ext}",
                            )
                        )
                    sig = f"{'export ' if is_exported else ''}interface {name}"
                    if extends_list:
                        sig += f" extends {', '.join(extends_list)}"
                    result.symbols.append(
                        CodeSymbol(
                            id=iface_id,
                            name=name,
                            fully_qualified_name=fqn,
                            symbol_type=SymbolType.INTERFACE,
                            language=lang_label,
                            file_path=file_path,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            parent_symbol_id=parent_id,
                            visibility="public" if is_exported else "internal",
                            signature=sig,
                            is_exported=is_exported,
                            confidence=ParserConfidence.PARSER_VERIFIED,
                            metadata={"extends": extends_list},
                        )
                    )
                return

            # Type Alias Deklarasyonu
            if node.type == "type_alias_declaration":
                name = ""
                for child in node.children:
                    if child.type in ("type_identifier", "identifier"):
                        name = _node_text(child).strip()
                if name:
                    fqn = f"{parent_fqn}.{name}" if parent_fqn else name
                    result.symbols.append(
                        CodeSymbol(
                            id=f"{file_path}::{fqn}::type_alias",
                            name=name,
                            fully_qualified_name=fqn,
                            symbol_type=SymbolType.TYPE_ALIAS,
                            language=lang_label,
                            file_path=file_path,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            parent_symbol_id=parent_id,
                            visibility="public" if is_exported else "internal",
                            signature=f"{'export ' if is_exported else ''}type {name}",
                            is_exported=is_exported,
                            confidence=ParserConfidence.PARSER_VERIFIED,
                        )
                    )
                return

            if node.type in ("function_declaration", "generator_function_declaration"):
                name = ""
                for child in node.children:
                    if child.type == "identifier":
                        name = _node_text(child).strip()
                if name:
                    fqn = f"{parent_fqn}.{name}" if parent_fqn else name
                    fn_id = f"{file_path}::{fqn}::function"
                    result.symbols.append(
                        CodeSymbol(
                            id=fn_id,
                            name=name,
                            fully_qualified_name=fqn,
                            symbol_type=SymbolType.FUNCTION,
                            language=lang_label,
                            file_path=file_path,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            parent_symbol_id=parent_id,
                            visibility="public" if is_exported else "internal",
                            signature=f"{'export ' if is_exported else ''}function {name}()",
                            is_exported=is_exported,
                            confidence=ParserConfidence.PARSER_VERIFIED,
                        )
                    )
                    _find_calls_in_node(node, fn_id)
                return

            if node.type in ("lexical_declaration", "variable_declaration"):
                for child in node.children:
                    if child.type == "variable_declarator":
                        v_name = ""
                        is_arrow = False
                        for sub in child.children:
                            if sub.type == "identifier":
                                v_name = _node_text(sub).strip()
                            elif sub.type == "arrow_function":
                                is_arrow = True

                        if v_name:
                            fqn = f"{parent_fqn}.{v_name}" if parent_fqn else v_name
                            stype = SymbolType.FUNCTION if is_arrow else SymbolType.VARIABLE
                            v_id = f"{file_path}::{fqn}::{stype.value}"
                            sig = f"{'export ' if is_exported else ''}const {v_name}" + (" = () => ..." if is_arrow else "")
                            result.symbols.append(
                                CodeSymbol(
                                    id=v_id,
                                    name=v_name,
                                    fully_qualified_name=fqn,
                                    symbol_type=stype,
                                    language=lang_label,
                                    file_path=file_path,
                                    start_line=child.start_point[0] + 1,
                                    end_line=child.end_point[0] + 1,
                                    parent_symbol_id=parent_id,
                                    visibility="public" if is_exported else "internal",
                                    signature=sig,
                                    is_exported=is_exported,
                                    confidence=ParserConfidence.PARSER_VERIFIED,
                                )
                            )
                            if is_arrow:
                                _find_calls_in_node(child, v_id)
                return

        for top_child in tree.root_node.children:
            _visit_node(top_child)

        return result
