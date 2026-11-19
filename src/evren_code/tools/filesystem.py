"""Filesystem Tools:
`read_file`, `read_multiple_files`, `write_file`, `edit_file`, `multi_edit_file`.

- Line-numbered and windowed file reading (`start_line`, `end_line`).
- Single-step batch file reading (`read_multiple_files`).
- Multi-block surgical editing on single/multiple files (`multi_edit_file`).
- Pre-change Unified Diff generation with automatic rollback snapshots via `UndoManager`.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import uuid

from evren_code.config import AppConfig
from evren_code.core.file_tracker import FileStateTracker
from evren_code.core.undo import UndoManager
from evren_code.tools.registry import ToolRegistry


def resolve_safe_path(
    workspace_root: Path,
    raw_path: str,
    allow_outside: bool = False,
    is_write: bool = False,
) -> Path:
    """Safely resolves relative or absolute paths within workspace.
    Automatically repairs typos in file extensions (e.g. 'page_tsx' -> 'page.tsx')
    and subfolder naming mismatches (e.g. 'my_frontend' -> 'my-frontend').
    
    Security: Validates containment within workspace_root via WorkspaceGuard;
    prevents directory traversal, symlink escapes, and unauthorized writes.
    """
    import re
    from evren_code.core.security import WorkspaceGuard

    cleaned = raw_path.replace("\\", "/").strip()
    exact_candidate = (
        Path(cleaned).resolve()
        if Path(cleaned).is_absolute()
        else (workspace_root / cleaned).resolve()
    )

    # Use directly if valid existing file or directory
    if exact_candidate.exists():
        return WorkspaceGuard.validate_path(
            workspace_root=workspace_root,
            target_path=exact_candidate,
            is_write=is_write,
            allow_outside=allow_outside,
            raw_input=raw_path,
        )

    # 1. Fix underscore in extension (page_tsx -> page.tsx, index_ts -> index.ts)
    fixed_cleaned = re.sub(
        r'_([a-zA-Z0-9]{2,4})$',
        lambda m: f".{m.group(1)}" if m.group(1).lower() in ("tsx", "ts", "jsx", "js", "py", "java", "json", "yml", "yaml", "css", "html", "md", "dart") else f"_{m.group(1)}",
        cleaned
    )

    candidate = Path(fixed_cleaned)
    if not candidate.is_absolute():
        parts = fixed_cleaned.split("/")
        # Map directory if hyphen or underscore mismatch exists
        if parts:
            first_seg = parts[0]
            if not (workspace_root / first_seg).exists():
                first_norm = first_seg.replace("_", "-").lower()
                try:
                    for item in workspace_root.iterdir():
                        if item.is_dir() and item.name.replace("_", "-").lower() == first_norm:
                            parts[0] = item.name
                            break
                except OSError:
                    pass
        candidate = (workspace_root / "/".join(parts)).resolve()
    else:
        candidate = candidate.resolve()

    return WorkspaceGuard.validate_path(
        workspace_root=workspace_root,
        target_path=candidate,
        is_write=is_write,
        allow_outside=allow_outside,
        raw_input=raw_path,
    )


def generate_unified_diff(
    old_text: str,
    new_text: str,
    filename: str,
) -> str:
    """Generates Unified Diff between two text strings."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=3,
    )
    return "".join(diff)


_SIMILAR_FILES_CACHE: Dict[str, Tuple[float, List[str]]] = {}
_SIMILAR_FILES_CACHE_TTL = 5.0
_SIMILAR_FILES_MAX_SCAN = 5000


def find_similar_files(
    workspace_root: Path,
    raw_path: str,
    max_results: int = 3,
) -> List[str]:
    """Finds existing files with similar names when a requested path does not exist."""
    import time as _time
    from evren_code.core.tool_reliability import get_current_cancellation_token

    raw = Path(raw_path)
    stem = raw.stem.lower()
    name = raw.name.lower()
    if not stem and not name:
        return []

    matches: List[str] = []
    ignored = {
        ".git", "node_modules", ".next", ".evren", "__pycache__",
        "venv", ".venv", "dist", "build", ".turbo", ".idea", ".vscode"
    }

    ws_key = str(workspace_root.resolve())
    now = _time.monotonic()
    cached_entry = _SIMILAR_FILES_CACHE.get(ws_key)
    all_rel_files: Optional[List[str]] = None
    if cached_entry is not None and (now - cached_entry[0]) < _SIMILAR_FILES_CACHE_TTL:
        all_rel_files = cached_entry[1]

    if all_rel_files is None:
        scanned_files: List[str] = []
        try:
            tok = get_current_cancellation_token()
            for root, dirs, files in os.walk(workspace_root):
                if tok is not None and tok.is_cancelled:
                    break
                dirs[:] = [d for d in dirs if d not in ignored and not d.startswith(".")]
                for f in files:
                    full_p = Path(root) / f
                    try:
                        rel_p = full_p.relative_to(workspace_root).as_posix()
                        scanned_files.append(rel_p)
                        if len(scanned_files) >= _SIMILAR_FILES_MAX_SCAN:
                            break
                    except ValueError:
                        pass
                if len(scanned_files) >= _SIMILAR_FILES_MAX_SCAN:
                    break
            all_rel_files = scanned_files
            _SIMILAR_FILES_CACHE[ws_key] = (now, all_rel_files)
        except Exception:
            all_rel_files = []

    for rel_p in all_rel_files:
        f_name = Path(rel_p).name
        f_lower = f_name.lower()
        f_stem = Path(f_name).stem.lower()
        if name == f_lower or stem == f_stem or (stem in f_lower and len(stem) > 3):
            if rel_p not in matches:
                matches.append(rel_p)
                if len(matches) >= max_results:
                    return matches

    return matches


def count_unclosed_jsx_divs(content: str) -> int:
    """Count unclosed jsx divs."""
    import re as _re

    cleaned = _re.sub(r"/\*[\s\S]*?\*/", "", content)
    cleaned_lines = [
        line for line in cleaned.splitlines() if not line.lstrip().startswith("//")
    ]
    text = "\n".join(cleaned_lines)

    close_count = len(_re.findall(r"</div\s*>", text))
    open_count = 0

    n = len(text)
    i = 0
    while i < n:
        idx = text.find("<div", i)
        if idx == -1:
            break
        after = idx + 4
        if after >= n:
            break
        nxt_ch = text[after]
        if nxt_ch not in (" ", "\t", "\n", "\r", ">", "/"):
            i = after
            continue

        j = after
        brace_depth = 0
        quote_ch: Optional[str] = None
        last_non_ws = "v"
        closed_tag = False

        while j < n:
            ch = text[j]
            if quote_ch is not None:
                if ch == "\\" and j + 1 < n:
                    j += 2
                    continue
                if ch == quote_ch:
                    quote_ch = None
                j += 1
                continue

            if ch in ('"', "'", "`"):
                quote_ch = ch
                last_non_ws = ch
                j += 1
                continue

            if ch == "{":
                brace_depth += 1
                last_non_ws = ch
                j += 1
                continue
            if ch == "}":
                if brace_depth > 0:
                    brace_depth -= 1
                last_non_ws = ch
                j += 1
                continue

            if brace_depth == 0 and ch == ">":
                if last_non_ws != "/":
                    open_count += 1
                closed_tag = True
                i = j + 1
                break

            if not ch.isspace():
                last_non_ws = ch
            j += 1

        if not closed_tag:
            break

    return open_count - close_count


def atomic_write(
    target_path: Path,
    content: str,
    workspace_root: Path,
    undo_manager: Optional[UndoManager] = None,
    relative_display: str = "",
    tool_name: str = "write_file",
    allow_empty: bool = False,
) -> Tuple[bool, str]:
    """Writes content to temp file with fsync and integrity validation, then atomically replaces target with os.replace."""
    import os
    import uuid
    from evren_code.core.security import WorkspaceGuard

    # 1. Workspace boundary validation
    try:
        WorkspaceGuard.validate_path(
            workspace_root=workspace_root,
            target_path=target_path,
            is_write=True,
            raw_input=relative_display or str(target_path),
        )
    except PermissionError as p_err:
        return False, str(p_err)

    # 2. Unexpected empty file check
    if not allow_empty and not content.strip() and target_path.is_file():
        try:
            old_size = target_path.stat().st_size
            if old_size > 50:
                return False, (
                    f"Reliability Error / Güvenilirlik Hatası: '{relative_display or target_path.name}' is an existing file ({old_size} bytes) "
                    "but attempted write content is empty. Write canceled to prevent accidental file reset."
                )
        except OSError:
            pass

    # 3. Undo snapshot recording
    if undo_manager is not None:
        undo_manager.record_before_write(target_path, relative_display or str(target_path), tool_name)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = target_path.parent / f".{target_path.name}.{uuid.uuid4().hex[:8]}.tmp"

    try:
        raw_bytes = content.encode("utf-8")
        with open(temp_file, "wb") as f:
            f.write(raw_bytes)
            f.flush()
            os.fsync(f.fileno())

        # 4. Temp file verification
        if not temp_file.is_file():
            return False, f"Error: Temp file ('{temp_file.name}') could not be written to disk (Geçici dosya diske yazılamadı)."

        written_size = temp_file.stat().st_size
        if written_size != len(raw_bytes):
            return False, (
                f"File Integrity Error (Dosya Bütünlük Hatası): Written size ({written_size} bytes) "
                f"does not match expected size ({len(raw_bytes)} bytes)."
            )

        # 5. Atomic replace
        os.replace(temp_file, target_path)

        # 6. Post-replace disk confirmation
        if not target_path.is_file():
            return False, f"Error: '{target_path.name}' not found after atomic replace."

        actual_size = target_path.stat().st_size
        if actual_size != len(raw_bytes):
            return False, (
                f"Verification Error: Disk file size ({actual_size} bytes) "
                f"does not match written data ({len(raw_bytes)} bytes)."
            )

        try:
            from evren_code.tools.search import invalidate_search_cache
            invalidate_search_cache()
        except Exception:
            pass

        return True, "Success / Başarılı"

    except Exception as exc:
        # Rollback in case of unexpected failure
        if undo_manager is not None and undo_manager.can_undo:
            try:
                undo_manager.undo_last()
            except Exception:
                pass
        return False, f"Atomic write error (Atomik dosya yazma hatası): {exc}"

    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass


def run_fast_inline_diagnostics(file_path: Path, content: str, lsp_manager: Any = None) -> str:
    """Run fast inline diagnostics."""
    import ast as _ast
    import json as _json
    from collections import Counter

    suffix = file_path.suffix.lower()
    issues: List[str] = []

    if suffix == ".py":
        try:
            _ast.parse(content, filename=str(file_path))
        except SyntaxError as exc:
            line_no = exc.lineno or 1
            col_offset = exc.offset or 0
            issues.append(f"Python Sözdizimi Hatası (Satır {line_no}, Sütun {col_offset}): {exc.msg}")
            c_lines = content.splitlines()
            if 0 < line_no <= len(c_lines):
                issues.append(f"    --> `{c_lines[line_no - 1].strip()}`")
    elif suffix == ".json":
        try:
            _json.loads(content)
        except Exception as exc:
            issues.append(f"JSON Sözdizimi Hatası: {exc}")
    elif suffix in (".ts", ".tsx", ".js", ".jsx"):
        lines = content.splitlines()
        open_curlies = content.count("{") - content.count("}")
        open_parens = content.count("(") - content.count(")")
        open_brackets = content.count("[") - content.count("]")
        if open_curlies > 0:
            issues.append(f"Satır {len(lines)}: {open_curlies} adet kapatılmamış '}}' eksik.")
        elif open_curlies < 0:
            issues.append(f"Satır {len(lines)}: {abs(open_curlies)} adet fazladan/açılmamış '}}' mevcut.")
        if open_parens > 0:
            issues.append(f"Satır {len(lines)}: {open_parens} adet kapatılmamış ')' eksik.")
        elif open_parens < 0:
            issues.append(f"Satır {len(lines)}: {abs(open_parens)} adet fazladan/açılmamış ')' mevcut.")
        if open_brackets > 0:
            issues.append(f"Satır {len(lines)}: {open_brackets} adet kapatılmamış ']' eksik.")
        elif open_brackets < 0:
            issues.append(f"Satır {len(lines)}: {abs(open_brackets)} adet fazladan/açılmamış ']' mevcut.")
        backticks = len(re.findall(r'(?<!\\)`', content))
        if backticks % 2 != 0:
            issues.append("JavaScript/TypeScript Sözdizimi: Kapatılmamış ters tırnak (template literal `) mevcut.")
        if suffix in (".tsx", ".jsx"):
            open_divs = count_unclosed_jsx_divs(content)
            if open_divs > 0:
                issues.append(f"Satır {len(lines)}: {open_divs} adet kapatılmamış '</div>' eksik.")
    elif suffix == ".css":
        lines = content.splitlines()
        clean_css = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        clean_css = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', clean_css)
        clean_css = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", clean_css)
        open_curlies = clean_css.count("{") - clean_css.count("}")
        if open_curlies > 0:
            issues.append(f"CSS Sözdizimi: {open_curlies} adet kapatılmamış '}}' eksik.")
        elif open_curlies < 0:
            issues.append(f"CSS Sözdizimi: {abs(open_curlies)} adet fazladan/açılmamış '}}' mevcut.")
        if content.count("/*") > content.count("*/"):
            issues.append("CSS Sözdizimi: Kapatılmamış yorum bloğu ('/*') mevcut.")
    elif suffix in (".html", ".htm"):
        if content.count("<script") > content.count("</script>"):
            issues.append("HTML Sözdizimi: Kapatılmamış '<script>' etiketi mevcut.")
        if content.count("<style") > content.count("</style>"):
            issues.append("HTML Sözdizimi: Kapatılmamış '<style>' etiketi mevcut.")
        if "<body>" in content and "</body>" not in content:
            issues.append("HTML Sözdizimi: '<body>' etiketi açılmış ancak kapatılmamış.")
        if "<html>" in content and "</html>" not in content:
            issues.append("HTML Sözdizimi: '<html>' etiketi açılmış ancak kapatılmamış.")
        if "<head>" in content and "</head>" not in content:
            issues.append("HTML Sözdizimi: '<head>' etiketi açılmış ancak kapatılmamış.")
        for tag in ("main", "header", "footer", "section", "article", "nav"):
            open_tag = f"<{tag}"
            close_tag = f"</{tag}>"
            if content.count(open_tag) > content.count(close_tag):
                diff_t = content.count(open_tag) - content.count(close_tag)
                issues.append(f"HTML Sözdizimi: {diff_t} adet kapatılmamış '<{tag}>' etiketi mevcut.")
    elif suffix == ".java":
        annotations_to_imports = {
            "@GetMapping": "org.springframework.web.bind.annotation.GetMapping",
            "@PostMapping": "org.springframework.web.bind.annotation.PostMapping",
            "@PutMapping": "org.springframework.web.bind.annotation.PutMapping",
            "@DeleteMapping": "org.springframework.web.bind.annotation.DeleteMapping",
            "@PatchMapping": "org.springframework.web.bind.annotation.PatchMapping",
            "@RequestBody": "org.springframework.web.bind.annotation.RequestBody",
            "@PathVariable": "org.springframework.web.bind.annotation.PathVariable",
            "@RequestParam": "org.springframework.web.bind.annotation.RequestParam",
            "@Transactional": "org.springframework.transaction.annotation.Transactional",
            "@Valid": "jakarta.validation.Valid",
            "@NotBlank": "jakarta.validation.constraints.NotBlank",
            "@NotNull": "jakarta.validation.constraints.NotNull",
        }
        for anno, imp in annotations_to_imports.items():
            if anno in content and imp not in content:
                pkg = ".".join(imp.split(".")[:-1])
                wildcard = f"import {pkg}.*;"
                if wildcard not in content:
                    issues.append(f"Java Eksik Import: Dosyada '{anno}' kullanılmış ancak 'import {imp};' eksik.")

        lines = content.splitlines()
        open_curlies = content.count("{") - content.count("}")
        if open_curlies > 0:
            issues.append(f"Satır {len(lines)}: {open_curlies} adet kapatılmamış '}}' eksik.")
        elif open_curlies < 0:
            issues.append(f"Satır {len(lines)}: {abs(open_curlies)} adet fazladan/açılmamış '}}' mevcut.")

    raw_lines = [ln.strip() for ln in content.splitlines()]
    consec_cnt = 1
    for idx in range(1, len(raw_lines)):
        ln = raw_lines[idx]
        if (
            len(ln) >= 8
            and ln == raw_lines[idx - 1]
            and not ln.startswith(("/*", "*", "<!--", "//", "#", "--", ";"))
        ):
            consec_cnt += 1
            if consec_cnt >= 6:
                issues.append(
                    f"Model Bozulması (Degeneration): '{ln[:40]}...' satırı aralıksız {consec_cnt} kez peş peşe tekrar ediyor."
                )
                break
        else:
            consec_cnt = 1

    if lsp_manager is not None:
        try:
            lsp_items = lsp_manager.get_diagnostics(str(file_path))
            if lsp_items:
                for it in lsp_items[:5]:
                    issues.append(f"LSP [{it.severity}] (Satır {it.line}:{it.column}): {it.message}")
        except Exception:
            pass

    if not issues:
        return ""
    return (
        "\n\n⚠ [Live Syntax Warning] [Canlı Sözdizimi/AST Uyarısı]: Syntax errors detected in file after this change:\n"
        + "\n".join(f"  • {it}" for it in issues)
        + "\nPlease fix these syntax errors using `edit_file` before proceeding to next step."
    )


_run_fast_inline_diagnostics = run_fast_inline_diagnostics


def register_filesystem_tools(
    registry: ToolRegistry,
    config: AppConfig,
    undo_manager: Optional[UndoManager] = None,
    file_tracker: Optional[FileStateTracker] = None,
    lsp_manager: Optional[Any] = None,
) -> None:
    """Register filesystem tools."""

    def read_file(
        path: Any,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        # If model passed a list/tuple to read_file, delegate to batch read
        if isinstance(path, (list, tuple)):
            return read_multiple_files(paths=list(path))

        path_lines = [l.strip().strip("\"'").strip() for l in str(path or "").splitlines() if l.strip()]
        raw_str_path = path_lines[0] if path_lines else ""
        if not raw_str_path:
            return "Error: Valid file path was not provided for `read_file`."
        target = resolve_safe_path(config.workspace_root, raw_str_path)

        # FileStateTracker: Return stub if file was already read and unchanged
        if file_tracker is not None and start_line is None and end_line is None:
            is_unchanged, stub = file_tracker.check_file(str(target))
            if is_unchanged and stub:
                return stub

        auto_resolved_note = ""
        if not target.exists():
            similars = find_similar_files(config.workspace_root, raw_str_path)
            is_bare_filename = "/" not in raw_str_path and "\\" not in raw_str_path
            exact_matches = [
                m for m in similars
                if Path(m).name.lower() == Path(raw_str_path).name.lower()
                and not any(m.lower().endswith(s) for s in (".class", ".pyc", ".o", ".dll", ".exe"))
            ]
            if is_bare_filename and exact_matches:
                auto_resolved = exact_matches[0]
                target = resolve_safe_path(config.workspace_root, auto_resolved)
                auto_resolved_note = f"[Auto Resolved Location / Otomatik Konum Çözümlemesi]: '{raw_str_path}' -> '{auto_resolved}'.\n"
            else:
                err_msg = f"Error: File '{raw_str_path}' not found."
                if similars:
                    tip = "\nHint: Existing files with similar names / İpucu: Benzer ada sahip mevcut dosyalar:\n" + "\n".join(f"  • {m}" for m in similars)
                    err_msg += tip + "\nPlease call `read_file` again specifying the correct path."
                else:
                    err_msg += (
                        "\nHint: This file does not exist yet (İpucu: Bu dosya henüz mevcut değil). "
                        "If you are creating a new module/feature, do not probe with 'read_file'; "
                        "create it directly using 'write_file' or 'write_multiple_files'."
                    )
                return err_msg
        if not target.is_file():
            return f"Error: '{raw_str_path}' is not a file."

        disallowed_binary_suffixes = {
            ".class", ".pyc", ".pyo", ".pyd", ".exe", ".dll", ".so", ".dylib", ".o", ".obj", ".bin"
        }
        if target.suffix.lower() in disallowed_binary_suffixes:
            return (
                f"Error: '{target.name}' is a compiled binary file (derlenmiş bir ikili (binary) dosyadır). "
                "Direct inspection of binary files is prohibited. "
                "Please inspect the corresponding source code files (.java, .py, .ts, etc.)."
            )

        size = target.stat().st_size
        if size > config.tools.max_file_size_bytes:
            return (
                f"Error: File '{raw_str_path}' is too large ({size} bytes). "
                f"Maximum allowed size: {config.tools.max_file_size_bytes} bytes."
            )

        content = target.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        total_lines = len(lines)

        s_idx = max(1, int(start_line)) if start_line is not None else 1
        max_span = config.tools.max_file_read_lines
        e_idx = (
            min(total_lines, int(end_line))
            if end_line is not None
            else min(total_lines, s_idx + max_span - 1)
        )

        selected = lines[s_idx - 1 : e_idx]
        max_line_chars = getattr(config.tools, "max_line_length_chars", 800)
        numbered = []
        for line_no, line in enumerate(selected, start=s_idx):
            if len(line) > max_line_chars:
                line_disp = line[:max_line_chars] + f" ... [{len(line) - max_line_chars} chars truncated]"
            else:
                line_disp = line
            numbered.append(f"{line_no:4d} | {line_disp}")

        header = auto_resolved_note + f"File: {raw_str_path} (Lines {s_idx}-{e_idx} / Satır {s_idx}-{e_idx} / Total {total_lines})"
        if e_idx < total_lines and end_line is None:
            header += f" [Note: Use start_line={e_idx + 1} to continue reading]"

        if file_tracker is not None:
            file_tracker.record_read(
                str(target),
                is_full_read=(s_idx == 1 and e_idx >= total_lines),
            )

        return header + "\n" + "\n".join(numbered)

    def read_multiple_files(
        paths: Any = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Reads multiple files in a single step with per-file timeout and error isolation."""
        if paths is None:
            paths = kwargs.get("files") or kwargs.get("file_paths") or kwargs.get("path") or kwargs.get("items")

        if isinstance(paths, str):
            if "," in paths:
                paths = [p.strip() for p in paths.split(",") if p.strip()]
            else:
                paths = [p.strip() for p in paths.splitlines() if p.strip()]
        elif isinstance(paths, (list, tuple)):
            norm_paths = []
            for p in paths:
                if isinstance(p, dict):
                    p_val = p.get("path") or p.get("file") or p.get("file_path")
                    if p_val:
                        norm_paths.append(str(p_val))
                elif p:
                    norm_paths.append(str(p))
            paths = norm_paths

        if not paths or not isinstance(paths, list):
            return "Error: `paths` parameter must be a non-empty list of file paths."

        from evren_code.core.tool_reliability import (
            BatchExecutionManager,
            get_current_cancellation_token,
             get_current_heartbeat_monitor,
        )

        max_files = config.tools.max_batch_read_files
        max_lines = config.tools.max_batch_lines_per_file
        eff_start = max(1, int(start_line)) if start_line is not None else 1
        eff_end = (
            int(end_line)
            if end_line is not None
            else (eff_start + max_lines - 1)
        )
        sliced_paths = [str(p) for p in paths[:max_files]]

        batch_mgr = BatchExecutionManager(per_item_timeout_seconds=5.0)
        batch_results = batch_mgr.execute_batch(
            items=sliced_paths,
            item_handler=lambda p: read_file(p, start_line=eff_start, end_line=eff_end),
            key_fn=str,
            cancellation_token=get_current_cancellation_token(),
            heartbeat=get_current_heartbeat_monitor(),
        )

        sections: List[str] = []
        for br in batch_results:
            if br.success:
                sections.append(br.output)
            else:
                sections.append(f"File/Dosya: {br.item_key}\n{br.output}")

        if len(paths) > max_files:
            sections.append(
                f"[Note: Maximum {max_files} files can be read at once; remaining {len(paths) - max_files} skipped]"
            )
        return f"Batch File Read / Toplu Dosya Okuma ({len(sliced_paths)} dosya):\n\n" + "\n\n---\n\n".join(sections)

    def _preview_write(args: Dict[str, Any]) -> Optional[str]:
        raw_path = str(args.get("path", ""))
        new_content = str(args.get("content", ""))
        if not raw_path:
            return None
        target = resolve_safe_path(config.workspace_root, raw_path)
        old_content = ""
        if target.is_file():
            try:
                old_content = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                old_content = ""
        return generate_unified_diff(old_content, new_content, raw_path)

    def _run_fast_inline_diagnostics(fp: Path, c: str) -> str:
        return run_fast_inline_diagnostics(fp, c, lsp_manager=lsp_manager)

    def _analyze_code_efficiency(file_path: Path, content: str) -> str:
        """Inspects code efficiency, 300 line ceiling, and copy-paste redundancy in 0ms."""
        from collections import Counter

        suffix = file_path.suffix.lower()
        if suffix not in (".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs", ".dart"):
            return ""

        lines = content.splitlines()
        line_count = len(lines)
        max_rec_lines = getattr(config.tools, "max_recommended_file_lines", 300)
        notes: List[str] = []

        if line_count > max_rec_lines:
            notes.append(
                f"File size reached {line_count} lines (recommended ceiling: {max_rec_lines} lines / önerilen tavan: {max_rec_lines} satır). "
                "In upcoming files, prefer data-driven arrays (`.map()` / loops) and modular helper functions."
            )

        # Copy-paste line redundancy analysis
        significant_lines = [
            ln.strip()
            for ln in lines
            if len(ln.strip()) >= 28
            and not ln.strip().startswith(("import ", "from ", "//", "#", "/*", "*", "export type", "return ("))
        ]
        if significant_lines:
            counts = Counter(significant_lines)
            repeated = [(s, c) for s, c in counts.items() if c >= 5]
            if repeated:
                sample_line, rep_cnt = repeated[0]
                notes.append(
                    f"Lines like `{sample_line[:45]}...` repeated {rep_cnt} times ({rep_cnt} kez tekrarlandığı tespit edildi). "
                    "Use shared components/functions or `.map()` loops instead of redundant code blocks."
                )

        if not notes:
            return ""
        return "\n💡 [Code Efficiency Note] [Kod Ekonomisi / Yalınlık Notu]: " + " ".join(notes)

    def _compact_diff_for_context(diff_text: str, max_lines: int = 24) -> str:
        """Compact diff for context (token)."""
        diff_lines = diff_text.splitlines()
        if len(diff_lines) <= max_lines:
            return diff_text
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        head = "\n".join(diff_lines[:12])
        tail = "\n".join(diff_lines[-8:])
        return (
            f"{head}\n"
            f"... [Diff bağlam tasarrufu için özetlendi: +{added}/-{removed} satır değişti] ...\n"
            f"{tail}\n"
        )

    def _format_context_window(file_content: str, edited_fragment: str, window_lines: int = 8) -> str:
        """Format context window."""
        if not edited_fragment or not file_content:
            return ""
        all_lines = file_content.splitlines()
        if not all_lines:
            return ""

        meaningful_lines = [l.strip() for l in edited_fragment.splitlines() if l.strip()]
        anchor = meaningful_lines[0] if meaningful_lines else ""

        target_idx = -1
        if anchor:
            for idx, line in enumerate(all_lines):
                if anchor in line:
                    target_idx = idx
                    break

        if target_idx == -1:
            target_idx = max(0, len(all_lines) // 2)

        start_idx = max(0, target_idx - window_lines)
        end_idx = min(len(all_lines), target_idx + len(meaningful_lines) + window_lines)

        snippet_lines = [f"{i + 1:4d} | {all_lines[i]}" for i in range(start_idx, end_idx)]
        return (
            f"\n\n[Güncellenen Bölge ve Çevre Bağlamı (Satır {start_idx + 1}-{end_idx} / Toplam {len(all_lines)})]:\n```\n"
            + "\n".join(snippet_lines)
            + "\n```"
        )

    def sanitize_runaway_repetition(text: str, max_consecutive_repeats: int = 10) -> str:
        """Sanitize runaway repetition (degeneration)."""
        lines = text.splitlines(keepends=True)
        if len(lines) < 6:
            return text
        result: List[str] = []
        i = 0
        n = len(lines)
        while i < n:
            matched_cycle = None
            for cycle_len in (1, 2, 3):
                if i + cycle_len * max_consecutive_repeats <= n:
                    cycle = [l.strip() for l in lines[i : i + cycle_len]]
                    if all(cycle):
                        repeats = 1
                        while i + (repeats + 1) * cycle_len <= n:
                            next_chunk = [
                                l.strip()
                                for l in lines[i + repeats * cycle_len : i + (repeats + 1) * cycle_len]
                            ]
                            if next_chunk == cycle:
                                repeats += 1
                            else:
                                break
                        if repeats >= max_consecutive_repeats:
                            matched_cycle = (cycle_len, repeats)
                            break
            if matched_cycle:
                c_len, reps = matched_cycle
                result.extend(lines[i : i + c_len])
                i += reps * c_len
            else:
                result.append(lines[i])
                i += 1
        return "".join(result)

    def write_file(path: str, content: str, mode: str = "write") -> str:
        path_lines = [l.strip().strip("\"'").strip() for l in str(path or "").splitlines() if l.strip()]
        raw_p = path_lines[0] if path_lines else ""
        if not raw_p:
            return "Error: Valid file path was not provided for `write_file`."
        path = raw_p

        try:
            target = resolve_safe_path(config.workspace_root, path, is_write=True)
        except PermissionError as p_err:
            return f"Security Error: {p_err} (Güvenlik Hatası: {p_err})"

        if target.exists() and target.is_dir():
            return (
                f"Error: '{path}' is a directory, not a file. "
                "Please provide a complete file path (e.g. 'src/app/page.tsx')."
            )
        existed = target.is_file()
        final_content = sanitize_runaway_repetition(content)
        if mode == "append" and existed:
            try:
                existing_text = target.read_text(encoding="utf-8", errors="replace")
                sep = "\n" if not existing_text.endswith("\n") else ""
                final_content = sanitize_runaway_repetition(existing_text + sep + content)
            except OSError as e:
                return f"Error: Could not read existing content of '{path}': {e}."

        ok, msg = atomic_write(
            target_path=target,
            content=final_content,
            workspace_root=config.workspace_root,
            undo_manager=undo_manager,
            relative_display=path,
            tool_name="write_file" if mode != "append" else "append_file",
        )
        if not ok:
            return f"Error: Could not write file '{path}': {msg}"

        if file_tracker is not None:
            file_tracker.record_write(str(target))
            file_tracker.record_write(path)

        lines = final_content.splitlines()
        line_count = len(lines)
        tail_lines = lines[-3:] if lines else []
        tail_preview = "\n".join(f"  {ln}" for ln in tail_lines)
        tail_section = f"\nFile last 3 lines / Dosya son 3 satırı:\n{tail_preview}" if tail_preview else ""

        action = "appended" if (mode == "append" and existed) else ("updated / güncellendi" if existed else "created / oluşturuldu")
        diag_note = _run_fast_inline_diagnostics(target, final_content)
        eff_note = _analyze_code_efficiency(target, final_content)

        if diag_note:
            return (
                f"⚠ [Written with Syntax/Degeneration Warning / Yazıldı ancak Sözdizimi/Bozulma Uyarısı]: '{path}' ({action} / {line_count} lines) "
                f"BUT critical syntax issues were detected:\n"
                f"{diag_note}{tail_section}{eff_note}\n"
                "Please fix these issues using `edit_file` in the next step!"
            )

        return f"Success: '{path}' written successfully ({action} / {line_count} lines) (Başarılı: '{path}' dosyası {action} ({line_count} satır yazıldı ve disk doğrulaması tamamlandı)).{tail_section}{eff_note}"

    def append_file(path: str, content: str) -> str:
        """Append file (append)."""
        return write_file(path=path, content=content, mode="append")

    def _apply_single_replacement(content: str, target: str, replacement: str) -> Optional[str]:
        if not target:
            return None
        if target == replacement or target.strip() == replacement.strip():
            return content

        norm_content = content.replace("\r\n", "\n")
        norm_target = target.replace("\r\n", "\n")
        norm_replacement = replacement.replace("\r\n", "\n")
        content_lines = norm_content.splitlines()
        target_lines = norm_target.splitlines()
        sep = "\r\n" if "\r\n" in content else "\n"

        if len(target_lines) == 1 and "\n" not in norm_target:
            t_single = target_lines[0]
            for i, cl in enumerate(content_lines):
                if cl == t_single or cl.rstrip() == t_single.rstrip():
                    rep_lines = norm_replacement.splitlines()
                    new_lines = content_lines[:i] + rep_lines + content_lines[i + 1 :]
                    return sep.join(new_lines) + (sep if content.endswith(("\r\n", "\n")) else "")

            rep_non_empty = [rl.strip() for rl in norm_replacement.splitlines() if rl.strip()]
            if len(rep_non_empty) >= 2:
                last_rep_line = rep_non_empty[-1]
                if last_rep_line.startswith(t_single.strip()):
                    for i, cl in enumerate(content_lines):
                        if cl.strip() == last_rep_line:
                            rep_lines = norm_replacement.splitlines()
                            new_lines = content_lines[:i] + rep_lines + content_lines[i + 1 :]
                            return sep.join(new_lines) + (sep if content.endswith(("\r\n", "\n")) else "")

            pos = norm_content.find(norm_target)
            if pos != -1:
                end_pos = pos + len(norm_target)
                prev_ch = norm_content[pos - 1] if pos > 0 else ""
                next_ch = norm_content[end_pos] if end_pos < len(norm_content) else ""
                starts_mid_word = bool(norm_target[0].isalnum() and prev_ch.isalnum())
                ends_mid_word = bool(norm_target[-1].isalnum() and next_ch.isalnum())
                if not starts_mid_word and not ends_mid_word:
                    res = norm_content.replace(norm_target, norm_replacement, 1)
                    return res.replace("\n", sep) if sep == "\r\n" else res
        else:
            if target in content:
                return content.replace(target, replacement, 1)
            if norm_target in norm_content:
                res = norm_content.replace(norm_target, norm_replacement, 1)
                return res.replace("\n", sep) if sep == "\r\n" else res


        if len(target_lines) <= len(content_lines) and target_lines:
            target_stripped = [l.rstrip() for l in target_lines]
            for i in range(len(content_lines) - len(target_lines) + 1):
                chunk = [content_lines[i + j].rstrip() for j in range(len(target_lines))]
                if chunk == target_stripped:
                    rep_lines = replacement.replace("\r\n", "\n").splitlines()
                    new_lines = (
                        content_lines[:i]
                        + rep_lines
                        + content_lines[i + len(target_lines) :]
                    )
                    sep = "\r\n" if "\r\n" in content else "\n"
                    return sep.join(new_lines) + (
                        sep if content.endswith(("\r\n", "\n")) else ""
                    )

            target_both_stripped = [l.strip() for l in target_lines]
            if any(target_both_stripped):
                for i in range(len(content_lines) - len(target_lines) + 1):
                    chunk_both_stripped = [content_lines[i + j].strip() for j in range(len(target_lines))]
                    if chunk_both_stripped == target_both_stripped:
                        indent_delta = 0
                        for j, tl in enumerate(target_lines):
                            if tl.strip():
                                actual_line = content_lines[i + j]
                                actual_indent = len(actual_line) - len(actual_line.lstrip(" "))
                                target_indent = len(tl) - len(tl.lstrip(" "))
                                indent_delta = actual_indent - target_indent
                                break

                        rep_raw_lines = replacement.replace("\r\n", "\n").splitlines()
                        adjusted_rep_lines: List[str] = []
                        for rl in rep_raw_lines:
                            if not rl.strip():
                                adjusted_rep_lines.append("")
                            elif indent_delta > 0:
                                adjusted_rep_lines.append((" " * indent_delta) + rl)
                            elif indent_delta < 0:
                                remove_cnt = min( abs(indent_delta), len(rl) - len(rl.lstrip(" ")) )
                                adjusted_rep_lines.append(rl[remove_cnt:])
                            else:
                                adjusted_rep_lines.append(rl)

                        new_lines = (
                            content_lines[:i]
                            + adjusted_rep_lines
                            + content_lines[i + len(target_lines) :]
                        )
                        sep = "\r\n" if "\r\n" in content else "\n"
                        return sep.join(new_lines) + (
                            sep if content.endswith(("\r\n", "\n")) else ""
                        )

            sig_target = [tl.strip() for tl in target_lines if tl.strip()]
            if len(sig_target) >= 2:
                content_sig = [(idx, cl.strip()) for idx, cl in enumerate(content_lines) if cl.strip()]
                for ci in range(len(content_sig) - len(sig_target) + 1):
                    window = [content_sig[ci + k][1] for k in range(len(sig_target))]
                    if window == sig_target:
                        start_line_idx = content_sig[ci][0]
                        end_line_idx = content_sig[ci + len(sig_target) - 1][0]
                        rep_lines = replacement.replace("\r\n", "\n").splitlines()
                        new_lines = (
                            content_lines[:start_line_idx]
                            + rep_lines
                            + content_lines[end_line_idx + 1 :]
                        )
                        sep = "\r\n" if "\r\n" in content else "\n"
                        return sep.join(new_lines) + (
                            sep if content.endswith(("\r\n", "\n")) else ""
                        )

            if len(target_lines) >= 3 and len(content_lines) >= len(target_lines):
                best_ratio = 0.0
                best_match_idx = -1
                best_window_len = len(target_lines)
                match_count = 0
                target_str_block = "\n".join(l.strip() for l in target_lines if l.strip())
                win_size = len(target_lines)

                for w in (win_size, win_size - 1, win_size + 1):
                    if w < 2 or w > len(content_lines):
                        continue
                    for i in range(len(content_lines) - w + 1):
                        cand_lines = [content_lines[i + k].strip() for k in range(w) if content_lines[i + k].strip()]
                        if not cand_lines:
                            continue
                        cand_block = "\n".join(cand_lines)
                        ratio = difflib.SequenceMatcher(None, target_str_block, cand_block).ratio()
                        if ratio >= 0.88:
                            if ratio > best_ratio + 0.03:
                                best_ratio = ratio
                                best_match_idx = i
                                best_window_len = w
                                match_count = 1
                            elif abs(ratio - best_ratio) <= 0.03:
                                match_count += 1

                if best_ratio >= 0.88 and match_count == 1 and best_match_idx != -1:
                    indent_delta = 0
                    for j in range(best_window_len):
                        if j < len(target_lines) and target_lines[j].strip():
                            act_line = content_lines[best_match_idx + j]
                            act_indent = len(act_line) - len(act_line.lstrip(" "))
                            tgt_indent = len(target_lines[j]) - len(target_lines[j].lstrip(" "))
                            indent_delta = act_indent - tgt_indent
                            break

                    rep_raw_lines = replacement.replace("\r\n", "\n").splitlines()
                    adjusted_rep_lines: List[str] = []
                    for rl in rep_raw_lines:
                        if not rl.strip():
                            adjusted_rep_lines.append("")
                        elif indent_delta > 0:
                            adjusted_rep_lines.append((" " * indent_delta) + rl)
                        elif indent_delta < 0:
                            remove_cnt = min(abs(indent_delta), len(rl) - len(rl.lstrip(" ")))
                            adjusted_rep_lines.append(rl[remove_cnt:])
                        else:
                            adjusted_rep_lines.append(rl)

                    new_lines = (
                        content_lines[:best_match_idx]
                        + adjusted_rep_lines
                        + content_lines[best_match_idx + best_window_len :]
                    )
                    sep = "\r\n" if "\r\n" in content else "\n"
                    return sep.join(new_lines) + (
                        sep if content.endswith(("\r\n", "\n")) else ""
                    )

            if 1 <= len(target_lines) <= 2 and len(content_lines) >= len(target_lines):
                def _norm_micro(s: str) -> str:
                    s_c = s.strip().replace('"', "'")
                    s_c = re.sub(r"\s*([=+\-*/:;,()\[\]{}])\s*", r"\1", s_c)
                    s_c = re.sub(r"\s+", " ", s_c)
                    return s_c

                t_norm = [_norm_micro(tl) for tl in target_lines if tl.strip()]
                if t_norm:
                    w_size = len(target_lines)
                    matched_indices: List[int] = []
                    for i in range(len(content_lines) - w_size + 1):
                        cand_norm = [_norm_micro(content_lines[i + k]) for k in range(w_size) if content_lines[i + k].strip()]
                        if cand_norm == t_norm:
                            matched_indices.append(i)

                    if len(matched_indices) == 1:
                        match_idx = matched_indices[0]
                        act_line = content_lines[match_idx]
                        act_indent = len(act_line) - len(act_line.lstrip(" "))
                        tgt_line = target_lines[0]
                        tgt_indent = len(tgt_line) - len(tgt_line.lstrip(" "))
                        indent_delta = act_indent - tgt_indent

                        rep_raw_lines = replacement.replace("\r\n", "\n").splitlines()
                        adjusted_rep_lines: List[str] = []
                        for rl in rep_raw_lines:
                            if not rl.strip():
                                adjusted_rep_lines.append("")
                            elif indent_delta > 0:
                                adjusted_rep_lines.append((" " * indent_delta) + rl)
                            elif indent_delta < 0:
                                rem = min(abs(indent_delta), len(rl) - len(rl.lstrip(" ")))
                                adjusted_rep_lines.append(rl[rem:])
                            else:
                                adjusted_rep_lines.append(rl)

                        new_lines = (
                            content_lines[:match_idx]
                            + adjusted_rep_lines
                            + content_lines[match_idx + w_size :]
                        )
                        sep = "\r\n" if "\r\n" in content else "\n"
                        return sep.join(new_lines) + (
                            sep if content.endswith(("\r\n", "\n")) else ""
                        )

                    if not matched_indices and len(target_lines) == 1:
                        target_single_str = t_norm[0]
                        best_r = 0.0
                        second_r = 0.0
                        best_i = -1
                        for i, cl in enumerate(content_lines):
                            if not cl.strip():
                                continue
                            r = difflib.SequenceMatcher(None, target_single_str, _norm_micro(cl)).ratio()
                            if r > best_r:
                                second_r = best_r
                                best_r = r
                                best_i = i
                            elif r > second_r:
                                second_r = r
                        if best_r >= 0.88 and (best_r - second_r >= 0.10) and best_i != -1:
                            act_line = content_lines[best_i]
                            act_indent = len(act_line) - len(act_line.lstrip(" "))
                            tgt_line = target_lines[0]
                            tgt_indent = len(tgt_line) - len(tgt_line.lstrip(" "))
                            indent_delta = act_indent - tgt_indent

                            rep_raw_lines = replacement.replace("\r\n", "\n").splitlines()
                            adjusted_rep_lines = []
                            for rl in rep_raw_lines:
                                if not rl.strip():
                                    adjusted_rep_lines.append("")
                                elif indent_delta > 0:
                                    adjusted_rep_lines.append((" " * indent_delta) + rl)
                                elif indent_delta < 0:
                                    rem = min(abs(indent_delta), len(rl) - len(rl.lstrip(" ")))
                                    adjusted_rep_lines.append(rl[rem:])
                                else:
                                    adjusted_rep_lines.append(rl)

                            new_lines = (
                                content_lines[:best_i]
                                + adjusted_rep_lines
                                + content_lines[best_i + 1 :]
                            )
                            sep = "\r\n" if "\r\n" in content else "\n"
                            return sep.join(new_lines) + (
                                sep if content.endswith(("\r\n", "\n")) else ""
                            )

        return None

    def _diagnose_replacement_failure(content: str, target: str) -> str:
        """Diagnose replacement failure."""
        norm_content = content.replace("\r\n", "\n")
        norm_target = target.replace("\r\n", "\n")
        content_lines = norm_content.splitlines()
        target_lines = norm_target.splitlines()
        if not target_lines:
            return "Hedef blok boş."

        def _format_window(center_idx: int, span: int = 12) -> str:
            s_i = max(0, center_idx - 3)
            e_i = min(len(content_lines), center_idx + span)
            snippet_lines = [
                f"{line_no:4d} | {content_lines[line_no - 1]}"
                for line_no in range(s_i + 1, e_i + 1)
            ]
            return (
                f"\n[File Slice] [Güncel Dosya Kesiti (Lines/Satır {s_i + 1}-{e_i})]:\n```\n"
                + "\n".join(snippet_lines)
                + "\n```"
            )

        first_target = next((l.strip() for l in target_lines if l.strip()), "")
        if not first_target:
            return "Please inspect current lines with 'read_file' and provide only the target root block."

        # Find occurrences where the first line matches
        matches = [
            idx
            for idx, cl in enumerate(content_lines)
            if cl.strip() == first_target or (len(first_target) > 8 and first_target in cl)
        ]
        if matches:
            idx = matches[0]
            for j, tl in enumerate(target_lines):
                if idx + j >= len(content_lines):
                    return (
                        f"Hint / İpucu: Block started at line {idx + 1} but exceeded end of file ({len(content_lines)} lines)."
                        + _format_window(idx, len(target_lines) + 4)
                    )
                cl = content_lines[idx + j]
                if tl.strip() != cl.strip():
                    win_start = max(1, idx)
                    win_end = min(len(content_lines), idx + len(target_lines) + 2)
                    return (
                        f"Hint / İpucu: Mismatch detected at line {idx + j + 1} (Satır {idx + j + 1}'de uyumsuzluk tespit edildi): "
                        f"Expected / Beklenen: `{tl.strip()[:60]}` | In file / Dosyadaki: `{cl.strip()[:60]}`. "
                        f"Please inspect lines {win_start}-{win_end} with 'read_file' or call 'edit_file' "
                        f"with start_line={win_start + 1}, end_line={win_end} directly."
                        + _format_window(idx, len(target_lines) + 6)
                    )

        # If first line didn't match, check if any other line matches
        for t_offset, tl in enumerate(target_lines):
            ts = tl.strip()
            if len(ts) < 8:
                continue
            sub_matches = [
                idx for idx, cl in enumerate(content_lines) if cl.strip() == ts or ts in cl
            ]
            if sub_matches:
                approx_start = max(0, sub_matches[0] - t_offset)
                return (
                    f"Hint / İpucu: Target line {t_offset + 1} (`{ts[:50]}`) matched file line {sub_matches[0] + 1}. "
                    "Please base your target on the slice below:"
                    + _format_window(approx_start, len(target_lines) + 6)
                )

        # 3. Sliding window fuzzy match
        if len(content_lines) >= 3:
            target_joined = "\n".join(tl.strip() for tl in target_lines if tl.strip())
            win_size = max(2, min(len(target_lines), len(content_lines)))
            best_r = 0.0
            best_idx = 0
            for i in range(len(content_lines) - win_size + 1):
                cand_sub = "\n".join(content_lines[i + k].strip() for k in range(win_size) if content_lines[i + k].strip())
                r = difflib.SequenceMatcher(None, target_joined, cand_sub).ratio()
                if r > best_r:
                    best_r = r
                    best_idx = i
            if best_r >= 0.45:
                return (
                    f"Hint / İpucu: Target block was not found exactly, but a {int(best_r * 100)}% match was found starting at line {best_idx + 1}. "
                    "Use the slice below as target without needing another read_file call:"
                    + _format_window(best_idx, win_size + 4)
                )

        return (
            f"Hint / İpucu: None of the lines in target block exist in this file ({len(content_lines)} total lines). "
            "Please check the path or inspect file contents with 'read_file'."
        )

    def _check_ambiguous_target(content: str, target: str) -> Optional[str]:
        """Warns if target block matches multiple identical locations in file to prevent modifying wrong line."""
        norm_content = content.replace("\r\n", "\n")
        norm_target = target.replace("\r\n", "\n").strip()
        if not norm_target:
            return None
        count = norm_content.count(norm_target)
        if count <= 1:
            return None

        match_lines: List[int] = []
        search_from = 0
        content_lines = norm_content.splitlines()
        while True:
            pos = norm_content.find(norm_target, search_from)
            if pos == -1:
                break
            line_no = norm_content.count("\n", 0, pos) + 1
            match_lines.append(line_no)
            search_from = pos + max(1, len(norm_target))

        match_previews = []
        for ln in match_lines[:5]:
            snippet = content_lines[ln - 1].strip() if 0 < ln <= len(content_lines) else ""
            match_previews.append(f"  • Line/Satır {ln}: `{snippet[:60]}`")

        lines_str = "\n".join(match_previews)
        return (
            f"Target block matches {count} different locations in file ({', '.join(f'Satır {l}' for l in match_lines[:5])}) "
            f"(Ambiguous Target / Belirsiz Hedef):\n{lines_str}\n"
            "Execution paused to avoid corrupting wrong line. Please include 2-3 lines of surrounding context or provide start_line and end_line."
        )

    def _preview_edit(args: Dict[str, Any]) -> Optional[str]:
        raw_path = str(args.get("path", ""))
        old_str = str(args.get("target", ""))
        new_str = str(args.get("replacement", ""))
        if not raw_path or not old_str:
            return None
        target = resolve_safe_path(config.workspace_root, raw_path)
        if not target.is_file():
            return None
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        updated = _apply_single_replacement(content, old_str, new_str)
        if updated is None:
            return None
        return generate_unified_diff(content, updated, raw_path)

    def parse_and_apply_patch(content: str, patch_str: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Parse and apply patch (Hunk)."""
        if not patch_str or not patch_str.strip():
            return False, None, "Yama içeriği (patch) boş."

        norm_content = content.replace("\r\n", "\n")
        content_lines = norm_content.splitlines()
        patch_lines = patch_str.replace("\r\n", "\n").splitlines()

        hunk_regex = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
        hunks: List[Tuple[Tuple[int, int], List[str]]] = []
        current_hunk_header: Optional[Tuple[int, int]] = None
        current_hunk_lines: List[str] = []

        for line in patch_lines:
            m = hunk_regex.match(line)
            if m:
                if current_hunk_header:
                    hunks.append((current_hunk_header, current_hunk_lines))
                current_hunk_header = (int(m.group(1)), int(m.group(2) or 1))
                current_hunk_lines = []
            elif current_hunk_header is not None:
                if line.startswith(("+", "-", " ", "\\")):
                    current_hunk_lines.append(line)
                elif not line.strip():
                    current_hunk_lines.append(" " + line)
                else:
                    current_hunk_lines.append(" " + line)

        if current_hunk_header:
            hunks.append((current_hunk_header, current_hunk_lines))

        if not hunks:
            loose_old = []
            loose_new = []
            for line in patch_lines:
                if line.startswith("-") and not line.startswith("---"):
                    loose_old.append(line[1:])
                elif line.startswith("+") and not line.startswith("+++"):
                    loose_new.append(line[1:])
                elif line.startswith(" "):
                    loose_old.append(line[1:])
                    loose_new.append(line[1:])
                elif not line.strip():
                    loose_old.append("")
                    loose_new.append("")

            if loose_old:
                target_str = "\n".join(loose_old)
                rep_str = "\n".join(loose_new)
                res = _apply_single_replacement(norm_content, target_str, rep_str)
                if res is not None:
                    sep = "\r\n" if "\r\n" in content else "\n"
                    res_final = res.replace("\n", sep)
                    return True, res_final, None
                return False, None, "Gevşek yama (loose diff) dosya içinde eşleşmedi."
            return False, None, "Geçerli bir Unified Diff hunk veya yama bloğu bulunamadı."

        working_lines = list(content_lines)
        offset = 0

        for idx, (header, h_lines) in enumerate(hunks, 1):
            old_start, old_len = header
            old_lines: List[str] = []
            new_lines: List[str] = []

            for hl in h_lines:
                if hl.startswith("\\"):
                    continue
                prefix = hl[0] if hl else " "
                rest = hl[1:] if len(hl) > 0 else ""
                if prefix in (" ", "-"):
                    old_lines.append(rest)
                if prefix in (" ", "+"):
                    new_lines.append(rest)

            target_pos = None
            expected_pos = max(0, old_start - 1 + offset)
            search_radius = max(25, len(working_lines))
            candidates = [expected_pos]
            for delta in range(1, search_radius):
                if expected_pos + delta < len(working_lines):
                    candidates.append(expected_pos + delta)
                if expected_pos - delta >= 0:
                    candidates.append(expected_pos - delta)

            old_stripped = [l.strip() for l in old_lines if l.strip()]

            for pos in candidates:
                if pos + len(old_lines) <= len(working_lines):
                    window = working_lines[pos : pos + len(old_lines)]
                    if window == old_lines:
                        target_pos = pos
                        break
                    if [l.rstrip() for l in window] == [l.rstrip() for l in old_lines]:
                        target_pos = pos
                        break
                    if [l.strip() for l in window] == [l.strip() for l in old_lines]:
                        target_pos = pos
                        break

            if target_pos is None and old_stripped:
                for i in range(len(working_lines) - len(old_lines) + 1):
                    chunk = working_lines[i : i + len(old_lines)]
                    if [l.strip() for l in chunk if l.strip()] == old_stripped:
                        target_pos = i
                        break

            if target_pos is None:
                return False, None, f"Hunk #{idx} (-{old_start},{old_len}) hedef satırları bulamadı."

            working_lines = working_lines[:target_pos] + new_lines + working_lines[target_pos + len(old_lines):]
            offset += len(new_lines) - len(old_lines)

        sep = "\r\n" if "\r\n" in content else "\n"
        res_text = sep.join(working_lines) + (sep if content.endswith(("\r\n", "\n")) else "")
        return True, res_text, None

    def _preview_patch(args: Dict[str, Any]) -> Optional[str]:
        raw_path = str(args.get("path", "")).strip()
        patch_str = str(args.get("patch", "")).strip()
        if not raw_path or not patch_str:
            return None
        target = resolve_safe_path(config.workspace_root, raw_path)
        if not target.is_file():
            return None
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        ok, updated, _ = parse_and_apply_patch(content, patch_str)
        if not ok or updated is None:
            return None
        return generate_unified_diff(content, updated, raw_path)

    def apply_patch(path: str, patch: str) -> str:
        orig_p = str(path or "")
        raw_p = orig_p.strip().strip("\"'")
        if not raw_p:
            return "Hata: `apply_patch` için geçerli bir dosya yolu (`path`) belirtilmedi."
        try:
            file_path = resolve_safe_path(config.workspace_root, raw_p, is_write=True)
        except PermissionError as p_err:
            return f"Güvenlik Hatası: {p_err}"

        if not file_path.is_file():
            return f"Hata: '{raw_p}' dosyası bulunamadı."

        content = file_path.read_text(encoding="utf-8", errors="replace")
        ok, patched_text, err_msg = parse_and_apply_patch(content, patch)
        if not ok or patched_text is None:
            return f"Hata: '{raw_p}' dosyasına yama uygulanamadı: {err_msg}"

        patched_text = sanitize_runaway_repetition(patched_text)
        w_ok, msg = atomic_write(
            target_path=file_path,
            content=patched_text,
            workspace_root=config.workspace_root,
            undo_manager=undo_manager,
            relative_display=raw_p,
            tool_name="apply_patch",
        )
        if not w_ok:
            return f"Hata: '{raw_p}' yaması diske yazılamadı: {msg}"

        if file_tracker is not None:
            file_tracker.invalidate(str(file_path))
            file_tracker.invalidate(raw_p)

        diff_summary = _compact_diff_for_context(generate_unified_diff(content, patched_text, raw_p))
        diag_note = _run_fast_inline_diagnostics(file_path, patched_text)
        if diag_note:
            return (
                f"⚠ [Yama Uygulandı ancak Sözdizimi Hatası]: '{raw_p}' yaması uygulandı "
                f"ANCAK içerikte sözdizimi hataları tespit edildi:\n{diag_note}\n\n{diff_summary}"
            )
        return f"Başarılı: '{raw_p}' dosyasına yama başarıyla uygulandı ve disk doğrulaması yapıldı.\n{diff_summary}"

    def edit_file(
        path: str,
        target: Optional[str] = None,
        replacement: str = "",
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        path_lines = [l.strip().strip("\"'").strip() for l in str(path or "").splitlines() if l.strip()]
        raw_p = path_lines[0] if path_lines else ""
        if not raw_p:
            return "Hata: `edit_file` için geçerli bir dosya yolu (`path`) belirtilmedi."
        path = raw_p

        try:
            file_path = resolve_safe_path(config.workspace_root, path, is_write=True)
        except PermissionError as p_err:
            return f"Güvenlik Hatası: {p_err}"

        if not file_path.is_file():
            return f"Hata: '{path}' dosyası bulunamadı."

        content = file_path.read_text(encoding="utf-8", errors="replace")

        diff_candidate = target if (target and ("@@" in target or target.strip().startswith("---") or target.strip().startswith("diff --git"))) else (
            replacement if (replacement and ("@@" in replacement or replacement.strip().startswith("---") or replacement.strip().startswith("diff --git"))) else None
        )
        if diff_candidate:
            return apply_patch(path=path, patch=diff_candidate)

        s_line: Optional[int] = None
        e_line: Optional[int] = None
        if start_line is not None:
            try:
                s_line = int(start_line)
            except (ValueError, TypeError):
                s_line = None
        if end_line is not None:
            try:
                e_line = int(end_line)
            except (ValueError, TypeError):
                e_line = None

        # Line range mode (start_line, end_line)
        if s_line is not None and e_line is not None:
            c_lines = content.splitlines(keepends=True)
            total_l = len(c_lines)
            if s_line < 1 or e_line < s_line or s_line > total_l:
                return f"Error: Invalid line range {s_line}-{e_line}."
            s_idx = s_line - 1
            e_idx = min(total_l, e_line)
            rep_text = replacement if replacement.endswith("\n") or not replacement else (replacement + "\n")
            updated = "".join(c_lines[:s_idx]) + rep_text + "".join(c_lines[e_idx:])
        else:
            if target is None:
                return "Error: Either `target` block or `start_line`/`end_line` must be specified for `edit_file`."
            if target == replacement or target.strip() == replacement.strip():
                return (
                    f"Error: `target` and `replacement` are identical in '{path}'. "
                    "Please specify the actual change or use 'write_file' to rewrite."
                )
            ambig = _check_ambiguous_target(content, target)
            if ambig:
                return f"Error: Multiple matches found in '{path}'. {ambig}"

            updated = _apply_single_replacement(content, target, replacement)
            if updated is None:
                diag = _diagnose_replacement_failure(content, target)
                return f"Error: Target block not found in '{path}'. {diag}"

        updated = sanitize_runaway_repetition(updated)
        ok, msg = atomic_write(
            target_path=file_path,
            content=updated,
            workspace_root=config.workspace_root,
            undo_manager=undo_manager,
            relative_display=path,
            tool_name="edit_file",
        )
        if not ok:
            return f"Error: Failed to write edit on '{path}': {msg}"

        if file_tracker is not None:
            file_tracker.invalidate(str(file_path))
            file_tracker.invalidate(path)

        diff_summary = _compact_diff_for_context(generate_unified_diff(content, updated, path))
        diag_note = _run_fast_inline_diagnostics(file_path, updated)
        eff_note = _analyze_code_efficiency(file_path, updated) if not diag_note else ""

        ctx_win = _format_context_window(updated, rep_text if target is None else (replacement or target))
        if diag_note:
            return (
                f"⚠ [Edited with Syntax Warning / Düzenlendi ancak Sözdizimi Hatası]: '{path}' updated "
                f"BUT syntax errors were detected:\n{diag_note}\n\n"
                f"{diff_summary}\nPlease fix these issues using `edit_file` in the next step.{ctx_win}"
            )

        return f"Success: '{path}' surgically edited and verified on disk (Başarılı: '{path}' dosyası cerrahi olarak güncellendi ve disk doğrulaması yapıldı).\n{diff_summary}{eff_note}{ctx_win}"

    def _preview_multi_edit(args: Dict[str, Any]) -> Optional[str]:
        raw_path = str(args.get("path", ""))
        edits = args.get("edits") or []
        if not isinstance(edits, list) or not edits:
            return None
        diffs: List[str] = []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in edits:
            if not isinstance(item, dict):
                continue
            item_path = str(item.get("path") or item.get("file_path") or raw_path).strip()
            if not item_path:
                continue
            grouped.setdefault(item_path, []).append(item)

        for p_str, p_edits in grouped.items():
            file_path = resolve_safe_path(config.workspace_root, p_str)
            if not file_path.is_file():
                continue
            try:
                original = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            working = original
            for item in p_edits:
                res = _apply_single_replacement(
                    working,
                    str(item.get("target") or item.get("old_str") or ""),
                    str(item.get("replacement") or item.get("new_str") or ""),
                )
                if res is not None:
                    working = res
            diffs.append(generate_unified_diff(original, working, p_str))
        return "\n".join(diffs) if diffs else None

    def multi_edit_file(
        path: Optional[str] = None,
        edits: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not edits or not isinstance(edits, list):
            return "Hata: `edits` listesi boş olamaz."

        grouped_edits: Dict[str, List[tuple[int, Dict[str, Any]]]] = {}
        for idx, item in enumerate(edits, start=1):
            if not isinstance(item, dict):
                continue
            item_path = str(item.get("path") or item.get("file_path") or path or "").strip()
            if not item_path:
                return f"Hata: {idx}. düzenleme öğesi için hedef dosya yolu (`path`) belirtilmemiş."
            grouped_edits.setdefault(item_path, []).append((idx, item))

        if not grouped_edits:
            return "Hata: `edits` içinde geçerli düzenleme öğesi bulunamadı."

        prepared_writes: List[tuple[str, Path, str, str, int]] = []
        for target_rel_path, indexed_items in grouped_edits.items():
            try:
                file_path = resolve_safe_path(config.workspace_root, target_rel_path, is_write=True)
            except PermissionError as p_err:
                return f"Güvenlik Hatası: {p_err}"

            if not file_path.is_file():
                return f"Hata: '{target_rel_path}' dosyası bulunamadı."

            original = file_path.read_text(encoding="utf-8", errors="replace")
            working = original
            file_changes = 0

            for idx, item in indexed_items:
                t_block = str(item.get("target") or item.get("old_str") or "")
                r_block = str(item.get("replacement") or item.get("new_str") or "")
                if t_block == r_block or t_block.strip() == r_block.strip():
                    continue
                ambig = _check_ambiguous_target(working, t_block)
                if ambig:
                    return f"Hata: '{target_rel_path}' içinde {idx}. düzenleme bloğunda çoklu eşleşme tespit edildi. {ambig}"
                res = _apply_single_replacement(working, t_block, r_block)
                if res is None:
                    diag = _diagnose_replacement_failure(working, t_block)
                    return (
                        f"Hata: '{target_rel_path}' içinde {idx}. düzenleme bloğu (`target`) bulunamadı. {diag}"
                    )
                working = res
                file_changes += 1

            working = sanitize_runaway_repetition(working)
            prepared_writes.append((target_rel_path, file_path, original, working, file_changes))

        total_changes = sum(fc for _, _, _, _, fc in prepared_writes)
        if total_changes == 0:
            files_label = ", ".join(f"'{trp}'" for trp, _, _, _, _ in prepared_writes)
            return (
                f"Hata: {files_label} üzerinde `target` ve `replacement` blokları birebir aynı olduğu için "
                "dosyada hiçbir değişiklik yapılmadı (0 değişiklik). Lütfen geçerli bir değişiklik bloğu belirtin veya "
                "dosyaları baştan oluşturmak için `write_file` / `write_multiple_files` kullanın."
            )

        diff_parts: List[str] = []
        diag_notes: List[str] = []
        touched_files: List[str] = []
        written_count = 0

        try:
            for target_rel_path, file_path, original, working, file_changes in prepared_writes:
                if file_changes == 0:
                    continue
                ok, msg = atomic_write(
                    target_path=file_path,
                    content=working,
                    workspace_root=config.workspace_root,
                    undo_manager=undo_manager,
                    relative_display=target_rel_path,
                    tool_name="multi_edit_file",
                )
                if not ok:
                    raise IOError(f"'{target_rel_path}' atomik yazılamadı: {msg}")
                written_count += 1
                if file_tracker is not None:
                    file_tracker.invalidate(str(file_path))
                    file_tracker.invalidate(target_rel_path)
                touched_files.append(f"'{target_rel_path}'")
                diff_parts.append(_compact_diff_for_context(generate_unified_diff(original, working, target_rel_path)))
                d_note = _run_fast_inline_diagnostics(file_path, working)
                if d_note:
                    diag_notes.append(d_note)
        except Exception as exc:
            if undo_manager is not None:
                for _ in range(written_count):
                    try:
                        undo_manager.undo_last()
                    except Exception:
                        pass
            return f"Hata: Çoklu düzenleme sırasında hata oluştu ve yapılan {written_count} değişiklik geri alındı: {exc}"

        files_label = ", ".join(touched_files)
        combined_diff = "\n".join(diff_parts)
        combined_diag = "".join(diag_notes)
        return (
            f"Success: A total of {total_changes} surgical changes applied across {files_label} and verified on disk (Başarılı: {files_label} üzerinde toplam {total_changes} adet cerrahi değişiklik uygulandı ve disk doğrulaması yapıldı).\n"
            f"{combined_diff}{combined_diag}"
        )

    def _preview_write_multiple_files(args: Dict[str, Any]) -> Optional[str]:
        files = args.get("files") or []
        if not isinstance(files, list) or not files:
            return None
        diffs: List[str] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("path") or "").strip()
            new_content = str(item.get("content", ""))
            if not raw_path:
                continue
            try:
                target = resolve_safe_path(config.workspace_root, raw_path)
            except Exception:
                continue
            old_content = ""
            if target.is_file():
                try:
                    old_content = target.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    old_content = ""
            diffs.append(generate_unified_diff(old_content, new_content, raw_path))
        return "\n".join(diffs) if diffs else None

    def write_multiple_files(files: Any) -> str:
        if isinstance(files, str):
            try:
                import json
                parsed = json.loads(files)
                if isinstance(parsed, list):
                    files = parsed
                elif isinstance(parsed, dict) and "files" in parsed and isinstance(parsed["files"], list):
                    files = parsed["files"]
                elif isinstance(parsed, dict) and "value" in parsed and isinstance(parsed["value"], list):
                    files = parsed["value"]
                elif isinstance(parsed, dict) and "path" in parsed:
                    files = [parsed]
            except Exception:
                try:
                    from evren_code.core.parser import repair_and_parse_json
                    parsed = repair_and_parse_json(files)
                    if isinstance(parsed, dict) and parsed.get("_is_truncated"):
                        return (
                            "Error [write_multiple_files]: Output payload was truncated mid-string due to token limit! "
                            "Batch write aborted to prevent corrupting files. Please write files individually with 'write_file'. "
                            ""
                        )
                    if isinstance(parsed, list):
                        files = parsed
                    elif isinstance(parsed, dict) and "files" in parsed and isinstance(parsed["files"], list):
                        files = parsed["files"]
                    elif isinstance(parsed, dict) and "value" in parsed and isinstance(parsed["value"], list):
                        files = parsed["value"]
                    elif isinstance(parsed, dict) and "path" in parsed:
                        files = [parsed]
                except Exception:
                    pass
        elif isinstance(files, dict) and "files" in files and isinstance(files["files"], list):
            files = files["files"]
        elif isinstance(files, dict) and "value" in files and isinstance(files["value"], list):
            files = files["value"]
        elif isinstance(files, dict) and "path" in files:
            files = [files]

        if not files or not isinstance(files, list):
            return "Hata: `files` parametresi boş olmayan bir liste olmalıdır ([{'path': '...', 'content': '...'}])."

        if len(files) > 20:
            return (
                f"Hata: Tek bir `write_multiple_files` çağrısında en fazla 20 dosya yazılabilir "
                f"(belirtilen: {len(files)} dosya). Model bozulmasını (degeneration) ve aşırı yazmayı "
                "önlemek için işlem durduruldu. Lütfen dosyaları daha küçük gruplar halinde yazın."
            )

        validated_items: List[tuple[str, Path, str]] = []
        for idx, item in enumerate(files, start=1):
            if not isinstance(item, dict):
                return f"Hata: {idx}. öğe geçerli bir sözlük nesnesi değil."
            path_lines = [l.strip().strip("\"'").strip() for l in str(item.get("path") or "").splitlines() if l.strip()]
            raw_p = path_lines[0] if path_lines else ""
            if not raw_p:
                return f"Hata: {idx}. öğede geçerli bir 'path' belirtilmemiş."
            clean_p = raw_p
            c = sanitize_runaway_repetition(str(item.get("content", "")))
            try:
                target_path = resolve_safe_path(config.workspace_root, clean_p, is_write=True)
            except PermissionError as p_err:
                return f"Güvenlik Hatası: {p_err}"
            except Exception as exc:
                return f"Hata: {idx}. dosya yolu '{clean_p}' doğrulanamadı: {exc}"
            validated_items.append((clean_p, target_path, c))

        written_count = 0
        written_targets: List[tuple[str, Path]] = []
        created_display: List[str] = []

        try:
            for raw_p, target_path, c in validated_items:
                ok, msg = atomic_write(
                    target_path=target_path,
                    content=c,
                    workspace_root=config.workspace_root,
                    undo_manager=undo_manager,
                    relative_display=raw_p,
                    tool_name="write_multiple_files",
                )
                if not ok:
                    raise IOError(f"'{raw_p}' atomik yazılamadı: {msg}")
                if file_tracker is not None:
                    file_tracker.record_write(str(target_path))
                    file_tracker.record_write(raw_p)
                written_targets.append((raw_p, target_path))
                created_display.append(f"'{raw_p}'")
                written_count += 1
        except Exception as exc:
            rollback_errors: List[str] = []
            if undo_manager is not None:
                for _ in range(written_count):
                    undo_msg = undo_manager.undo_last()
                    if undo_msg.startswith("Hata:"):
                        rollback_errors.append(undo_msg)

            if rollback_errors:
                err_details = "\n".join(f"  • {e}" for e in rollback_errors)
                return (
                    f"Kritik Hata: Toplu yazma sırasında hata oluştu ('{exc}') ve yazılan dosyalar geri alınırken KISMİ BAŞARISIZLIK yaşandı:\n"
                    f"{err_details}\n"
                    "Lütfen dosya sistemini ve izinleri kontrol edin."
                )
            return f"Hata: Toplu yazma sırasında beklenmeyen bir hata oluştu ve yazılan {written_count} dosya başarıyla geri alındı: {exc}"

        syntax_warnings: List[str] = []
        for raw_p, target_path, c in validated_items:
            diag_note = _run_fast_inline_diagnostics(target_path, c)
            if diag_note:
                syntax_warnings.append(f"• `{raw_p}`:{diag_note}")

        warn_str = ""
        if syntax_warnings:
            warn_str = (
                "\n\n⚠ [Canlı Sözdizimi/AST Uyarısı]: Bazı dosyalarda kritik sözdizimi/bozulma hataları tespit edildi:\n"
                + "\n".join(syntax_warnings)
                + "\nLütfen bir sonraki adımda `edit_file` kullanarak bu hataları düzeltin!"
            )

        return f"Success: {len(created_display)} files created/updated in batch (Başarılı: {len(created_display)} dosya topluca oluşturuldu/güncellendi: {', '.join(created_display)}).{warn_str}"

    registry.register(
        name="read_file",
        description=(
            "Reads the contents of a file within the workspace. "
            "Supports line windowing via start_line and end_line for large files. "
            "To inspect multiple files, prefer 'read_multiple_files' in a single step."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
                "start_line": {"type": "integer", "description": "Optional starting line number (1-based, inclusive)."},
                "end_line": {"type": "integer", "description": "Optional ending line number (1-based, inclusive)."},
            },
            "required": ["path"],
        },
        handler=read_file,
        requires_approval=False,
    )

    registry.register(
        name="read_multiple_files",
        description=(
            "Reads multiple files simultaneously in a single step (e.g. package.json, types.ts, config.py). "
            "Always prefer this over sequential single 'read_file' calls to conserve turn steps and context."
        ),
        parameters={
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to read.",
                }
            },
            "required": ["paths"],
        },
        handler=read_multiple_files,
        requires_approval=False,
    )

    registry.register(
        name="write_file",
        description=(
            "Creates a new file or overwrites an existing file with the provided content. "
            "Specify 'path' first, followed by 'content'. Use 'edit_file' for surgical edits on existing files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write."},
                "content": {"type": "string", "description": "Full text content to write to the file."},
                "mode": {"type": "string", "enum": ["write", "append"], "description": "Write mode ('write': overwrite, 'append': append to end). Default: 'write'."},
            },
            "required": ["path", "content"],
        },
        handler=write_file,
        preview_builder=_preview_write,
    )

    registry.register(
        name="append_file",
        description=(
            "Appends content to the end of an existing file (creates it if missing). "
            "Useful for appending styles, exports, or log entries without rewriting the entire file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the target file."},
                "content": {"type": "string", "description": "Content to append at the end of the file."},
            },
            "required": ["path", "content"],
        },
        handler=append_file,
        preview_builder=_preview_write,
    )

    registry.register(
        name="edit_file",
        description=(
            "Surgically modifies a file by replacing an existing code block ('target') with new text ('replacement'), "
            "or by line windowing ('start_line', 'end_line'). Always prefer this over 'write_file' for existing files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit."},
                "target": {"type": "string", "description": "The exact existing text block to replace."},
                "replacement": {"type": "string", "description": "The replacement text block."},
                "start_line": {"type": "integer", "description": "Optional 1-based start line of replacement window."},
                "end_line": {"type": "integer", "description": "Optional 1-based end line of replacement window."},
            },
            "required": ["path", "replacement"],
        },
        handler=edit_file,
        preview_builder=_preview_edit,
    )

    registry.register(
        name="apply_patch",
        description=(
            "Applies a Unified Diff / patch to a file. "
            "Supports standard '@@ -old,count +new,count @@' hunks to surgically patch code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to patch."},
                "patch": {"type": "string", "description": "Unified Diff patch content."},
            },
            "required": ["path", "patch"],
        },
        handler=apply_patch,
        preview_builder=_preview_patch,
    )

    registry.register(
        name="multi_edit_file",
        description=(
            "Atomically applies multiple distinct search-and-replace edits to a single file in one step."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit."},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "replacement": {"type": "string"},
                        },
                        "required": ["target", "replacement"],
                    },
                    "description": "List of {target, replacement} edit pairs applied sequentially.",
                },
            },
            "required": ["path", "edits"],
        },
        handler=multi_edit_file,
        preview_builder=_preview_multi_edit,
    )

    registry.register(
        name="write_multiple_files",
        description=(
            "Creates or updates multiple files in a single turn step. "
            "Accelerates scaffolding schemas, components, and tests together."
        ),
        parameters={
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path."},
                            "content": {"type": "string", "description": "File content."},
                        },
                        "required": ["path", "content"],
                    },
                    "description": "List of [{'path': '...', 'content': '...'}] file objects.",
                }
            },
            "required": ["files"],
        },
        handler=write_multiple_files,
        requires_approval=True,
        preview_builder=_preview_write_multiple_files,
    )
