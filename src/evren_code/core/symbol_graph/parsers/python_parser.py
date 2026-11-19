"""Python parser module."""

from __future__ import annotations

import ast
from typing import List, Optional, Tuple

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


class PythonParser(BaseParser):
    """Python parser implementation."""

    @property
    def language_name(self) -> str:
        return "python"

    @property
    def supported_extensions(self) -> List[str]:
        return [".py", ".pyw"]

    def parse(self, file_path: str, code: str) -> ParsedResult:
        result = ParsedResult(file_path=file_path, language="python")
        try:
            tree = ast.parse(code, filename=file_path)
        except SyntaxError as exc:
            result.is_partial = True
            result.diagnostics.append(
                ParseDiagnostics(
                    line=exc.lineno or 1,
                    column=exc.offset or 1,
                    message=f"Sözdizimi hatası: {exc.msg}",
                )
            )
            return result
        except Exception as exc:
            result.is_partial = True
            result.diagnostics.append(
                ParseDiagnostics(line=1, column=1, message=f"Ayrıştırma hatası: {exc}")
            )
            return result

        lines = code.splitlines()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append(
                        FileImport(
                            file_path=file_path,
                            imported_module=alias.name,
                            imported_symbols=[],
                            alias=alias.asname,
                            line_number=node.lineno,
                        )
                    )
                    result.references.append(
                        SymbolReference(
                            source_file=file_path,
                            source_symbol_id=None,
                            target_name=alias.name,
                            target_symbol_id=None,
                            relationship_type=RelationshipType.IMPORTS,
                            line_number=node.lineno,
                            resolution_status=ResolutionStatus.UNRESOLVED,
                            evidence=f"import {alias.name}",
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                symbols = [alias.name for alias in node.names]
                result.imports.append(
                    FileImport(
                        file_path=file_path,
                        imported_module=mod_name,
                        imported_symbols=symbols,
                        line_number=node.lineno,
                    )
                )
                for sym in symbols:
                    result.references.append(
                        SymbolReference(
                            source_file=file_path,
                            source_symbol_id=None,
                            target_name=f"{mod_name}.{sym}" if mod_name else sym,
                            target_symbol_id=None,
                            relationship_type=RelationshipType.IMPORTS,
                            line_number=node.lineno,
                            resolution_status=ResolutionStatus.UNRESOLVED,
                            evidence=f"from {mod_name} import {sym}",
                        )
                    )

        def _extract_visibility(name: str) -> str:
            if name.startswith("__") and not name.endswith("__"):
                return "private"
            elif name.startswith("_"):
                return "protected"
            return "public"

        def _format_args(args_node: ast.arguments) -> str:
            parts = []
            for arg in args_node.args:
                annotation = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
                parts.append(f"{arg.arg}{annotation}")
            return f"({', '.join(parts)})"

        def _visit_nodes(
            parent_node: ast.AST,
            parent_symbol_id: Optional[str] = None,
            parent_fqn: str = "",
        ) -> None:
            for child in ast.iter_child_nodes(parent_node):
                if isinstance(child, ast.ClassDef):
                    c_name = child.name
                    c_fqn = f"{parent_fqn}.{c_name}" if parent_fqn else c_name
                    c_id = f"{file_path}::{c_fqn}::class"
                    doc = ast.get_docstring(child) or ""
                    vis = _extract_visibility(c_name)

                    bases = []
                    for b in child.bases:
                        b_name = ast.unparse(b)
                        bases.append(b_name)
                        result.references.append(
                            SymbolReference(
                                source_file=file_path,
                                source_symbol_id=c_id,
                                target_name=b_name,
                                target_symbol_id=None,
                                relationship_type=RelationshipType.EXTENDS,
                                line_number=child.lineno,
                                resolution_status=ResolutionStatus.UNRESOLVED,
                                evidence=f"class {c_name}({b_name})",
                            )
                        )

                    sig = f"class {c_name}({', '.join(bases)})" if bases else f"class {c_name}"
                    decorators = [ast.unparse(d) for d in child.decorator_list]

                    sym = CodeSymbol(
                        id=c_id,
                        name=c_name,
                        fully_qualified_name=c_fqn,
                        symbol_type=SymbolType.CLASS,
                        language="python",
                        file_path=file_path,
                        start_line=child.lineno,
                        end_line=getattr(child, "end_lineno", child.lineno),
                        parent_symbol_id=parent_symbol_id,
                        visibility=vis,
                        signature=sig,
                        docstring=doc,
                        is_exported=not c_name.startswith("_"),
                        confidence=ParserConfidence.PARSER_VERIFIED,
                        metadata={"decorators": decorators, "bases": bases},
                    )
                    result.symbols.append(sym)

                    _visit_nodes(child, parent_symbol_id=c_id, parent_fqn=c_fqn)

                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    f_name = child.name
                    is_method = parent_symbol_id is not None
                    stype = SymbolType.CONSTRUCTOR if f_name == "__init__" else (SymbolType.METHOD if is_method else SymbolType.FUNCTION)
                    f_fqn = f"{parent_fqn}.{f_name}" if parent_fqn else f_name
                    f_id = f"{file_path}::{f_fqn}::{stype.value}"
                    doc = ast.get_docstring(child) or ""
                    vis = _extract_visibility(f_name)
                    args_sig = _format_args(child.args)
                    ret_ann = f" -> {ast.unparse(child.returns)}" if child.returns else ""
                    kind_str = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                    sig = f"{kind_str} {f_name}{args_sig}{ret_ann}"
                    decorators = [ast.unparse(d) for d in child.decorator_list]

                    sym = CodeSymbol(
                        id=f_id,
                        name=f_name,
                        fully_qualified_name=f_fqn,
                        symbol_type=stype,
                        language="python",
                        file_path=file_path,
                        start_line=child.lineno,
                        end_line=getattr(child, "end_lineno", child.lineno),
                        parent_symbol_id=parent_symbol_id,
                        visibility=vis,
                        signature=sig,
                        docstring=doc,
                        is_exported=not f_name.startswith("_") or f_name == "__init__",
                        confidence=ParserConfidence.PARSER_VERIFIED,
                        metadata={"decorators": decorators, "is_async": isinstance(child, ast.AsyncFunctionDef)},
                    )
                    result.symbols.append(sym)

                    for sub in ast.walk(child):
                        if isinstance(sub, ast.Call):
                            call_name = ""
                            if isinstance(sub.func, ast.Name):
                                call_name = sub.func.id
                            elif isinstance(sub.func, ast.Attribute):
                                call_name = sub.func.attr
                            if call_name:
                                result.references.append(
                                    SymbolReference(
                                        source_file=file_path,
                                        source_symbol_id=f_id,
                                        target_name=call_name,
                                        target_symbol_id=None,
                                        relationship_type=RelationshipType.CALLS,
                                        line_number=getattr(sub, "lineno", child.lineno),
                                        resolution_status=ResolutionStatus.UNRESOLVED,
                                        evidence=f"call {call_name}()",
                                    )
                                )

                    _visit_nodes(child, parent_symbol_id=f_id, parent_fqn=f_fqn)

        _visit_nodes(tree)
        return result
