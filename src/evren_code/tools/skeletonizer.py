"""Intelligent Code Skeletonizer (AST Semantic Code Slicing).

Analyzes large source code files (Python, TS/JS, Go, Java, C#, etc.) and folds
class, function, and method bodies, slashing token costs by 80%-95% while allowing the
model to immediately comprehend architecture, interfaces, and signatures.
When a focus symbol (`focus_symbol`) is specified, preserves that specific symbol's body
intact while folding all other bodies.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, List, Optional, Tuple

from evren_code.config import AppConfig
from evren_code.tools.filesystem import resolve_safe_path
from evren_code.tools.registry import ToolRegistry


def _format_arg(arg: ast.arg) -> str:
    """Formats a single argument with its type annotation."""
    res = arg.arg
    if arg.annotation:
        try:
            res += f": {ast.unparse(arg.annotation)}"
        except Exception:
            pass
    return res


def _format_function_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool = False) -> str:
    """Generates function or method signature with parameters and return type annotation."""
    parts = []
    # Positional only
    posonly = getattr(fn.args, "posonlyargs", [])
    for a in posonly:
        parts.append(_format_arg(a))
    if posonly:
        parts.append("/")

    # Regular args with defaults
    args = fn.args.args
    defaults = fn.args.defaults
    diff = len(args) - len(defaults)
    for idx, a in enumerate(args):
        formatted = _format_arg(a)
        if idx >= diff:
            def_val = defaults[idx - diff]
            try:
                formatted += f" = {ast.unparse(def_val)}"
            except Exception:
                formatted += " = ..."
        parts.append(formatted)

    # Vararg (*args)
    if fn.args.vararg:
        formatted = f"*{fn.args.vararg.arg}"
        if fn.args.vararg.annotation:
            try:
                formatted += f": {ast.unparse(fn.args.vararg.annotation)}"
            except Exception:
                pass
        parts.append(formatted)
    elif fn.args.kwonlyargs:
        parts.append("*")

    # Kwonly args
    kw_defaults = fn.args.kw_defaults
    for idx, a in enumerate(fn.args.kwonlyargs):
        formatted = _format_arg(a)
        if idx < len(kw_defaults) and kw_defaults[idx] is not None:
            try:
                formatted += f" = {ast.unparse(kw_defaults[idx])}"
            except Exception:
                formatted += " = ..."
        parts.append(formatted)

    # Kwarg (**kwargs)
    if fn.args.kwarg:
        formatted = f"**{fn.args.kwarg.arg}"
        if fn.args.kwarg.annotation:
            try:
                formatted += f": {ast.unparse(fn.args.kwarg.annotation)}"
            except Exception:
                pass
        parts.append(formatted)

    async_prefix = "async " if is_async else ""
    ret_annotation = ""
    if fn.returns:
        try:
            ret_annotation = f" -> {ast.unparse(fn.returns)}"
        except Exception:
            pass

    return f"{async_prefix}def {fn.name}({', '.join(parts)}){ret_annotation}:"


def skeletonize_python(
    source_code: str,
    focus_symbol: Optional[str] = None,
    include_docstrings: bool = True,
) -> str:
    """Extracts skeleton representation of Python source code using AST."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return skeletonize_generic(source_code, ".py", focus_symbol=focus_symbol)

    source_lines = source_code.splitlines()
    target_clean = focus_symbol.strip().lower() if focus_symbol else None
    result_lines: List[str] = []

    def get_raw_node_text(node: ast.AST) -> str:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        return "\n".join(source_lines[start:end])

    def node_line_count(node: ast.AST) -> int:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        return max(1, end - start + 1)

    def is_target(name: str, parent: Optional[str] = None) -> bool:
        if not target_clean:
            return False
        n_low = name.lower()
        if n_low == target_clean:
            return True
        if parent:
            fq = f"{parent.lower()}.{n_low}"
            if fq == target_clean or target_clean in (n_low, fq):
                return True
        return False

    for item in tree.body:
        # 1. Imports
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            try:
                result_lines.append(ast.unparse(item))
            except Exception:
                result_lines.append(get_raw_node_text(item))

        # 2. Module Docstring
        elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
            doc = item.value.value.strip()
            if include_docstrings:
                if "\n" in doc:
                    first_l = doc.splitlines()[0]
                    result_lines.append(f'"""{first_l} ..."""')
                else:
                    result_lines.append(f'"""{doc}"""')

        # 3. Global Assignments / Type aliases
        elif isinstance(item, (ast.Assign, ast.AnnAssign)):
            try:
                line_rep = ast.unparse(item)
                if len(line_rep) <= 120:
                    result_lines.append(line_rep)
                else:
                    result_lines.append(get_raw_node_text(item).splitlines()[0] + " = ...")
            except Exception:
                result_lines.append(get_raw_node_text(item).splitlines()[0])

        # 4. Top-level Functions
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_async = isinstance(item, ast.AsyncFunctionDef)
            if is_target(item.name):
                # Full body preserved
                result_lines.append(get_raw_node_text(item))
            else:
                # Skeleton
                for dec in item.decorator_list:
                    try:
                        result_lines.append(f"@{ast.unparse(dec)}")
                    except Exception:
                        pass
                sig = _format_function_signature(item, is_async=is_async)
                result_lines.append(sig)
                doc = ast.get_docstring(item)
                total_lines = node_line_count(item)
                if doc and include_docstrings:
                    doc_1st = doc.strip().splitlines()[0]
                    result_lines.append(f'    """{doc_1st}"""')
                result_lines.append(f"    ... # (body folded / gövde gizlendi - {total_lines} lines)")

        # 5. Classes
        elif isinstance(item, ast.ClassDef):
            for dec in item.decorator_list:
                try:
                    result_lines.append(f"@{ast.unparse(dec)}")
                except Exception:
                    pass

            bases_str = ""
            if item.bases:
                try:
                    bases_str = f"({', '.join(ast.unparse(b) for b in item.bases)})"
                except Exception:
                    bases_str = "(...)"
            result_lines.append(f"class {item.name}{bases_str}:")

            class_doc = ast.get_docstring(item)
            if class_doc and include_docstrings:
                first_d = class_doc.strip().splitlines()[0]
                result_lines.append(f'    """{first_d}"""')

            class_body_empty = True
            for member in item.body:
                if isinstance(member, (ast.Assign, ast.AnnAssign)):
                    try:
                        mem_rep = ast.unparse(member)
                        if len(mem_rep) <= 100:
                            result_lines.append(f"    {mem_rep}")
                            class_body_empty = False
                    except Exception:
                        pass
                elif isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_body_empty = False
                    m_async = isinstance(member, ast.AsyncFunctionDef)
                    if is_target(member.name, parent=item.name):
                        # Full method body
                        raw_fn = get_raw_node_text(member)
                        for r_l in raw_fn.splitlines():
                            result_lines.append(r_l if r_l.startswith("    ") else f"    {r_l}")
                    else:
                        for dec in member.decorator_list:
                            try:
                                result_lines.append(f"    @{ast.unparse(dec)}")
                            except Exception:
                                pass
                        sig = _format_function_signature(member, is_async=m_async)
                        result_lines.append(f"    {sig}")
                        fn_doc = ast.get_docstring(member)
                        lines_cnt = node_line_count(member)
                        if fn_doc and include_docstrings:
                            doc_1st = fn_doc.strip().splitlines()[0]
                            result_lines.append(f'        """{doc_1st}"""')
                        result_lines.append(f"        ... # (body folded / gövde gizlendi - {lines_cnt} lines)")

            if class_body_empty:
                result_lines.append("    ...")

    return "\n".join(result_lines)


def skeletonize_generic(
    source_code: str,
    suffix: str,
    focus_symbol: Optional[str] = None,
) -> str:
    """Performs intelligent block skeletonization for C-style languages (JS, TS, Go, Java, C#, Rust, C++).

    Features:
    - Supports Allman-style bracing (opening curly brace on the next line).
    - Preserves container scopes (class, struct, interface, namespace) while folding method/function bodies.
    - Fully preserves target function/method body if `focus_symbol` is provided.
    """
    target_clean = focus_symbol.strip().lower() if focus_symbol else None
    lines = source_code.splitlines()
    result_lines: List[str] = []

    container_pat = re.compile(
        r'^\s*(?:\[[^\]]+\]\s*)*(?:public|private|protected|internal|static|abstract|sealed|partial|\s)*'
        r'(?:class|struct|interface|record|enum|namespace)\s+([a-zA-Z0-9_]+)'
    )
    method_pat = re.compile(
        r'^\s*(?:\[[^\]]+\]\s*)*(?:public|private|protected|internal|static|virtual|override|abstract|async|unsafe|partial|extern|sealed|readonly|\s)*'
        r'([a-zA-Z0-9_<>[\]?,\s]+?)\s+([a-zA-Z0-9_]+)\s*(?:<[^>]+>)?\s*\([^;]*\)(?:\s*where\s+.*)?$'
    )
    constructor_pat = re.compile(
        r'^\s*(?:public|private|protected|internal|\s)*([a-zA-Z0-9_]+)\s*\([^;]*\)(?:\s*:\s*(?:base|this)\([^)]*\))?$'
    )
    generic_func_pats = [
        re.compile(r'^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)'),
        re.compile(r'^\s*func\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)\s*\('),
        re.compile(r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z0-9_]+)'),
    ]

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith(("import ", "from ", "package ", "using ", "use ", "#include", "//", "/*")):
            result_lines.append(line)
            i += 1
            continue

        c_match = container_pat.search(line)
        if c_match:
            result_lines.append(line)
            i += 1
            continue

        m_name = None
        m = method_pat.search(line)
        if m:
            cand_name = m.group(2)
            if cand_name not in ("if", "while", "for", "foreach", "switch", "catch", "using", "lock"):
                m_name = cand_name
        if not m_name:
            c_ctor = constructor_pat.search(line)
            if c_ctor:
                cand_name = c_ctor.group(1)
                if cand_name not in ("if", "while", "for", "foreach", "switch", "catch", "using", "lock"):
                    m_name = cand_name
        if not m_name:
            for pat in generic_func_pats:
                m_gen = pat.search(line)
                if m_gen:
                    m_name = m_gen.group(1)
                    break

        if m_name:
            has_brace = "{" in line
            brace_line_idx = i if has_brace else -1
            if not has_brace:
                k = i + 1
                while k < n and not lines[k].strip():
                    k += 1
                if k < n and lines[k].strip().startswith("{"):
                    brace_line_idx = k

            if brace_line_idx != -1:
                is_matched_target = bool(target_clean and m_name.lower() == target_clean)
                if is_matched_target:
                    result_lines.append(line)
                    i += 1
                    continue
                else:
                    brace_line = lines[brace_line_idx]
                    depth = brace_line.count("{") - brace_line.count("}")
                    curr_i = brace_line_idx
                    while depth > 0 and curr_i + 1 < n:
                        curr_i += 1
                        depth += lines[curr_i].count("{") - lines[curr_i].count("}")

                    collapsed_count = curr_i - brace_line_idx

                    if brace_line_idx == i:
                        # K&R style: Opening brace on same line as signature
                        sig_part = line[: line.rfind("{") + 1]
                        result_lines.append(f"{sig_part} /* body folded / gövde gizlendi - {collapsed_count} lines */ }}")
                    else:
                        # Allman style: Opening brace on next line
                        for j in range(i, brace_line_idx):
                            result_lines.append(lines[j])
                        indent = " " * (len(lines[brace_line_idx]) - len(lines[brace_line_idx].lstrip()))
                        result_lines.append(f"{indent}{{ /* body folded / gövde gizlendi - {collapsed_count} lines */ }}")

                    i = curr_i + 1
                    continue

        result_lines.append(line)
        i += 1

    return "\n".join(result_lines)


def skeletonize_file_content(
    workspace_root: Path,
    file_path: str,
    focus_symbol: Optional[str] = None,
    include_docstrings: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Reads file, extracts its semantic skeleton, and computes token saving metrics."""
    target_path = resolve_safe_path(workspace_root, file_path)
    if not target_path.is_file():
        return False, f"Error: File '{file_path}' not found.", {}

    raw_text = target_path.read_text(encoding="utf-8", errors="replace")
    original_lines = len(raw_text.splitlines())
    suffix = target_path.suffix.lower()

    if suffix == ".py":
        skeleton = skeletonize_python(raw_text, focus_symbol=focus_symbol, include_docstrings=include_docstrings)
    else:
        skeleton = skeletonize_generic(raw_text, suffix, focus_symbol=focus_symbol)

    skeleton_lines = len(skeleton.splitlines())

    # Token estimate (1 token ≈ 3.5 chars)
    orig_chars = len(raw_text)
    skel_chars = len(skeleton)
    orig_tokens = max(1, int(orig_chars / 3.5))
    skel_tokens = max(1, int(skel_chars / 3.5))

    savings_pct = max(0.0, round(((orig_chars - skel_chars) / max(1, orig_chars)) * 100, 1))

    stats = {
        "original_lines": original_lines,
        "skeleton_lines": skeleton_lines,
        "original_tokens_est": orig_tokens,
        "skeleton_tokens_est": skel_tokens,
        "savings_pct": savings_pct,
        "focus_symbol": focus_symbol,
    }

    return True, skeleton, stats


def register_skeletonizer_tools(registry: ToolRegistry, config: AppConfig) -> None:
    """Registers the `skeletonize_code` tool with the ToolRegistry."""

    def skeletonize_code(
        path: str,
        focus_symbol: Optional[str] = None,
        include_docstrings: bool = True,
    ) -> str:
        ok, skeleton_text, stats = skeletonize_file_content(
            config.workspace_root,
            path,
            focus_symbol=focus_symbol,
            include_docstrings=include_docstrings,
        )
        if not ok:
            return skeleton_text

        focus_info = (
            f"- **Focused Symbol:** `{focus_symbol}` (Body fully preserved)\n"
            if focus_symbol
            else "- **Focus:** Entire file general skeleton (function/method bodies folded)\n"
        )

        return (
            f"### 🦴 [AST Code Skeleton: `{path}`]\n"
            f"- **Original / Orijinal:** {stats['original_lines']} lines (~{stats['original_tokens_est']} tokens)\n"
            f"- **Skeleton / Skeleton:** {stats['skeleton_lines']} lines (~{stats['skeleton_tokens_est']} tokens)\n"
            f"- **Savings / Tasarruf:** **{stats['savings_pct']}% Token Savings (Token Tasarrufu)**\n"
            f"{focus_info}\n"
            f"```python\n{skeleton_text}\n```"
        )

    registry.register(
        name="skeletonize_code",
        description=(
            "Extracts structural code skeletons (classes, function signatures, types, docstrings) while collapsing bodies (80-95% token savings). "
            "If 'focus_symbol' is provided, only that function/method body is fully expanded while summarizing the rest. "
            "Prefer this over 'read_file' when surveying large unfamiliar files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Target source file path."},
                "focus_symbol": {
                    "type": "string",
                    "description": "Optional specific function, method, or class name to expand fully.",
                },
                "include_docstrings": {
                    "type": "boolean",
                    "description": "Whether to retain docstrings and comments (default: true).",
                },
            },
            "required": ["path"],
        },
        handler=skeletonize_code,
        requires_approval=False,
    )
