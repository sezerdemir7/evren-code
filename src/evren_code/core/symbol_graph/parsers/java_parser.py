"""Java parser module."""

from __future__ import annotations

import re
from typing import List, Optional

import tree_sitter
import tree_sitter_java

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


class JavaParser(BaseParser):
    """Java parser implementation."""

    def __init__(self) -> None:
        self.language = tree_sitter.Language(tree_sitter_java.language())
        self.parser = tree_sitter.Parser(self.language)

    @property
    def language_name(self) -> str:
        return "java"

    @property
    def supported_extensions(self) -> List[str]:
        return [".java"]

    def parse(self, file_path: str, code: str) -> ParsedResult:
        result = ParsedResult(file_path=file_path, language="java")
        code_bytes = code.encode("utf-8", errors="replace")

        try:
            tree = self.parser.parse(code_bytes)
        except Exception as exc:
            result.is_partial = True
            result.diagnostics.append(
                ParseDiagnostics(line=1, column=1, message=f"Java ayrıştırma hatası: {exc}")
            )
            return result

        if tree.root_node.has_error:
            result.is_partial = True
            result.diagnostics.append(
                ParseDiagnostics(
                    line=1,
                    column=1,
                    message="Java sözdizimi hataları tespit edildi, kısmi AST çıkarılıyor.",
                    severity="warning",
                )
            )

        package_name = ""

        def _node_text(node: tree_sitter.Node) -> str:
            return code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

        for child in tree.root_node.children:
            if child.type == "package_declaration":
                for sub in child.children:
                    if sub.type in ("scoped_identifier", "identifier"):
                        package_name = _node_text(sub).strip()
            elif child.type == "import_declaration":
                import_text = ""
                for sub in child.children:
                    if sub.type in ("scoped_identifier", "identifier"):
                        import_text = _node_text(sub).strip()
                if import_text:
                    parts = import_text.split(".")
                    sym_name = parts[-1] if parts else import_text
                    result.imports.append(
                        FileImport(
                            file_path=file_path,
                            imported_module=import_text,
                            imported_symbols=[sym_name] if sym_name != "*" else [],
                            line_number=child.start_point[0] + 1,
                        )
                    )
                    result.references.append(
                        SymbolReference(
                            source_file=file_path,
                            source_symbol_id=None,
                            target_name=import_text,
                            target_symbol_id=None,
                            relationship_type=RelationshipType.IMPORTS,
                            line_number=child.start_point[0] + 1,
                            resolution_status=ResolutionStatus.UNRESOLVED,
                            evidence=f"import {import_text};",
                        )
                    )

        def _extract_modifiers_and_annotations(node: tree_sitter.Node) -> Tuple[str, List[str]]:
            vis = "package-private"
            annotations: List[str] = []
            for child in node.children:
                if child.type == "modifiers":
                    for mod_child in child.children:
                        if "annotation" in mod_child.type:
                            annotations.append(_node_text(mod_child).strip())
                        elif mod_child.type in ("public", "private", "protected"):
                            vis = mod_child.type
            return vis, annotations

        def _visit_class_or_interface(
            node: tree_sitter.Node,
            parent_fqn_prefix: str,
        ) -> None:
            name = ""
            vis, annotations = _extract_modifiers_and_annotations(node)
            stype = SymbolType.CLASS
            if node.type == "interface_declaration":
                stype = SymbolType.INTERFACE
            elif node.type == "enum_declaration":
                stype = SymbolType.ENUM
            elif node.type == "record_declaration":
                stype = SymbolType.RECORD

            superclass = ""
            interfaces: List[str] = []
            body_node: Optional[tree_sitter.Node] = None

            for child in node.children:
                if child.type == "identifier":
                    name = _node_text(child).strip()
                elif child.type == "superclass":
                    # extends SomeClass
                    for sc in child.children:
                        if sc.type in ("type_identifier", "scoped_type_identifier"):
                            superclass = _node_text(sc).strip()
                elif child.type == "super_interfaces":
                    # implements A, B
                    for iface in child.children:
                        if iface.type == "type_list":
                            for t in iface.children:
                                if t.type in ("type_identifier", "scoped_type_identifier"):
                                    interfaces.append(_node_text(t).strip())
                        elif iface.type in ("type_identifier", "scoped_type_identifier"):
                            interfaces.append(_node_text(iface).strip())
                elif child.type in ("class_body", "interface_body", "enum_body"):
                    body_node = child

            if not name:
                return

            fqn = f"{parent_fqn_prefix}.{name}" if parent_fqn_prefix else name
            c_id = f"{file_path}::{fqn}::{stype.value}"

            if superclass:
                result.references.append(
                    SymbolReference(
                        source_file=file_path,
                        source_symbol_id=c_id,
                        target_name=superclass,
                        target_symbol_id=None,
                        relationship_type=RelationshipType.EXTENDS,
                        line_number=node.start_point[0] + 1,
                        resolution_status=ResolutionStatus.UNRESOLVED,
                        evidence=f"{stype.value} {name} extends {superclass}",
                    )
                )

            for iface_name in interfaces:
                result.references.append(
                    SymbolReference(
                        source_file=file_path,
                        source_symbol_id=c_id,
                        target_name=iface_name,
                        target_symbol_id=None,
                        relationship_type=RelationshipType.IMPLEMENTS,
                        line_number=node.start_point[0] + 1,
                        resolution_status=ResolutionStatus.UNRESOLVED,
                        evidence=f"{stype.value} {name} implements {iface_name}",
                    )
                )

            sig = f"{' '.join(annotations) + ' ' if annotations else ''}{vis} {stype.value} {name}"
            if superclass:
                sig += f" extends {superclass}"
            if interfaces:
                sig += f" implements {', '.join(interfaces)}"

            sym = CodeSymbol(
                id=c_id,
                name=name,
                fully_qualified_name=fqn,
                symbol_type=stype,
                language="java",
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_symbol_id=None,
                visibility=vis,
                signature=sig,
                is_exported=vis == "public",
                confidence=ParserConfidence.PARSER_VERIFIED,
                metadata={
                    "package": package_name,
                    "annotations": annotations,
                    "superclass": superclass,
                    "interfaces": interfaces,
                },
            )
            result.symbols.append(sym)

            if body_node:
                for member in body_node.children:
                    if member.type in ("class_declaration", "interface_declaration", "enum_declaration"):
                        _visit_class_or_interface(member, fqn)
                    elif member.type in ("method_declaration", "constructor_declaration"):
                        _visit_method(member, c_id, fqn)
                    elif member.type == "field_declaration":
                        _visit_field(member, c_id, fqn)

        def _visit_method(
            node: tree_sitter.Node,
            parent_id: str,
            parent_fqn: str,
        ) -> None:
            m_name = ""
            vis, annotations = _extract_modifiers_and_annotations(node)
            return_type = ""
            params_str = "()"
            is_ctor = node.type == "constructor_declaration"

            for child in node.children:
                if child.type == "identifier":
                    m_name = _node_text(child).strip()
                elif child.type in ("type_identifier", "generic_type", "void_type", "integral_type", "floating_point_type", "boolean_type"):
                    return_type = _node_text(child).strip()
                elif child.type == "formal_parameters":
                    params_str = _node_text(child).strip().replace("\n", " ")

            if not m_name:
                return

            stype = SymbolType.CONSTRUCTOR if is_ctor else SymbolType.METHOD
            m_fqn = f"{parent_fqn}.{m_name}"
            m_id = f"{file_path}::{m_fqn}::{stype.value}"

            sig_parts = []
            if annotations:
                sig_parts.append(" ".join(annotations))
            sig_parts.append(vis)
            if return_type:
                sig_parts.append(return_type)
            sig_parts.append(f"{m_name}{params_str}")
            sig = " ".join(sig_parts)

            sym = CodeSymbol(
                id=m_id,
                name=m_name,
                fully_qualified_name=m_fqn,
                symbol_type=stype,
                language="java",
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_symbol_id=parent_id,
                visibility=vis,
                signature=sig,
                is_exported=vis == "public",
                confidence=ParserConfidence.PARSER_VERIFIED,
                metadata={"annotations": annotations, "return_type": return_type},
            )
            result.symbols.append(sym)

            def _find_calls(curr: tree_sitter.Node) -> None:
                if curr.type == "method_invocation":
                    call_name = ""
                    for c_child in curr.children:
                        if c_child.type == "identifier":
                            call_name = _node_text(c_child).strip()
                    if call_name:
                        result.references.append(
                            SymbolReference(
                                source_file=file_path,
                                source_symbol_id=m_id,
                                target_name=call_name,
                                target_symbol_id=None,
                                relationship_type=RelationshipType.CALLS,
                                line_number=curr.start_point[0] + 1,
                                resolution_status=ResolutionStatus.UNRESOLVED,
                                evidence=f"call {call_name}() in {m_name}",
                            )
                        )
                for c in curr.children:
                    _find_calls(c)

            _find_calls(node)

        def _visit_field(
            node: tree_sitter.Node,
            parent_id: str,
            parent_fqn: str,
        ) -> None:
            vis, annotations = _extract_modifiers_and_annotations(node)
            type_name = ""
            var_name = ""
            for child in node.children:
                if child.type in ("type_identifier", "scoped_type_identifier", "generic_type"):
                    type_name = _node_text(child).strip()
                elif child.type == "variable_declarator":
                    for sub in child.children:
                        if sub.type == "identifier":
                            var_name = _node_text(sub).strip()

            if var_name:
                f_id = f"{file_path}::{parent_fqn}.{var_name}::field"
                sym = CodeSymbol(
                    id=f_id,
                    name=var_name,
                    fully_qualified_name=f"{parent_fqn}.{var_name}",
                    symbol_type=SymbolType.FIELD,
                    language="java",
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent_symbol_id=parent_id,
                    visibility=vis,
                    signature=f"{' '.join(annotations) + ' ' if annotations else ''}{vis} {type_name} {var_name}",
                    is_exported=vis == "public",
                    confidence=ParserConfidence.PARSER_VERIFIED,
                    metadata={"annotations": annotations, "type": type_name},
                )
                result.symbols.append(sym)
                if type_name:
                    result.references.append(
                        SymbolReference(
                            source_file=file_path,
                            source_symbol_id=f_id,
                            target_name=type_name,
                            target_symbol_id=None,
                            relationship_type=RelationshipType.REFERENCES,
                            line_number=node.start_point[0] + 1,
                            resolution_status=ResolutionStatus.UNRESOLVED,
                            evidence=f"field type {type_name}",
                        )
                    )

        prefix = package_name if package_name else ""
        for child in tree.root_node.children:
            if child.type in ("class_declaration", "interface_declaration", "enum_declaration", "record_declaration"):
                _visit_class_or_interface(child, prefix)

        return result
