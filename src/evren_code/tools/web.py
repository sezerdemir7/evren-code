"""Web / HTTP Content Retrieval and Verification Tool (`fetch_url`, `http_request`, `validate_web_contract`).

Executes HTTP GET/POST/etc. requests to external URLs or local development servers
(`http://localhost:...`), stripping HTML tags to return clean readable text.
Statically validates web contracts across HTML, CSS, and JS in 0ms.
"""

from __future__ import annotations

import difflib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from evren_code.config import AppConfig
from evren_code.tools.filesystem import resolve_safe_path
from evren_code.tools.registry import ToolRegistry


_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WHITESPACE_RE = re.compile(r"\n{3,}")


def strip_html_tags(raw_html: str) -> str:
    """Strips script/style blocks and HTML markup to produce readable text."""
    text = _SCRIPT_STYLE_RE.sub("", raw_html)
    text = _HTML_TAG_RE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)
    return _MULTI_WHITESPACE_RE.sub("\n\n", cleaned)


def register_web_tools(registry: ToolRegistry, config: AppConfig) -> None:
    """Registers web inspection tools (`fetch_url`, `http_request`, `validate_web_contract`)."""

    def fetch_url(url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with 'http://' or 'https://'."

        try:
            with httpx.Client(
                timeout=config.tools.url_fetch_timeout_seconds,
                follow_redirects=True,
            ) as client:
                resp = client.get(url)
        except Exception as exc:
            return f"Error: Failed to fetch '{url}': {exc}"

        content_type = resp.headers.get("content-type", "")
        raw_text = resp.text
        if "text/html" in content_type.lower():
            body = strip_html_tags(raw_text)
        else:
            body = raw_text

        max_chars = config.tools.max_url_chars
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n... [Content truncated at {max_chars} chars / İçerik kırpıldı]"
        return f"URL ({url}) [HTTP {resp.status_code}]:\n{body}"

    registry.register(
        name="fetch_url",
        description="Fetches a web page, API documentation, or local server endpoint (e.g. http://localhost:3000) and returns clean text content.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full HTTP/HTTPS URL to fetch."},
            },
            "required": ["url"],
        },
        handler=fetch_url,
        requires_approval=False,
    )

    def http_request(
        url: str,
        method: str = "GET",
        headers: Optional[dict] = None,
        json_data: Optional[dict] = None,
        data: Optional[str] = None,
        params: Optional[dict] = None,
        timeout: float = 10.0,
    ) -> str:
        """Sends safe HTTP requests to local and external API endpoints."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with 'http://' or 'https://'."

        norm_method = method.strip().upper()
        eff_timeout = min(max(1.0, timeout), config.tools.url_fetch_timeout_seconds * 2)

        try:
            with httpx.Client(timeout=eff_timeout, follow_redirects=True) as client:
                req_kwargs: dict = {}
                if headers:
                    req_kwargs["headers"] = headers
                if params:
                    req_kwargs["params"] = params
                if json_data is not None:
                    req_kwargs["json"] = json_data
                elif data is not None:
                    req_kwargs["content"] = data

                resp = client.request(norm_method, url, **req_kwargs)

            content_type = resp.headers.get("content-type", "")
            raw_text = resp.text
            max_chars = config.tools.max_url_chars
            body_preview = raw_text[:max_chars] if len(raw_text) > max_chars else raw_text
            if len(raw_text) > max_chars:
                body_preview += f"\n... [Response body truncated at {max_chars} chars / Yanıt gövdesi kısaltıldı]"

            return (
                f"HTTP {resp.status_code} ({norm_method} {url})\n"
                f"Content-Type: {content_type}\n"
                f"Response / Yanıt:\n{body_preview}"
            )
        except Exception as exc:
            return f"Error: HTTP request failed ({norm_method} {url}): {exc}"

    registry.register(
        name="http_request",
        description=(
            "Sends an HTTP request (GET, POST, PUT, DELETE, PATCH) to test REST API endpoints, "
            "returning structured JSON or clean text output."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target HTTP/HTTPS URL."},
                "method": {"type": "string", "description": "HTTP method (GET, POST, PUT, DELETE, PATCH). Default GET.", "default": "GET"},
                "headers": {"type": "object", "description": "Optional HTTP request headers."},
                "json_data": {"type": "object", "description": "Optional JSON payload for POST/PUT requests."},
                "params": {"type": "object", "description": "Optional URL query parameters (?key=val)."},
                "timeout": {"type": "number", "description": "Request timeout in seconds (default: 10.0).", "default": 10.0},
            },
            "required": ["url"],
        },
        handler=http_request,
        requires_approval=False,
    )

    def validate_web_contract(
        html_path: Optional[str] = None,
        css_path: Optional[str] = None,
        js_path: Optional[str] = None,
        check_asset_links: bool = True,
    ) -> str:
        """Statically verifies interface contracts between HTML, CSS, and JavaScript (DOM IDs, CSS classes, asset references)
        in 0ms and reports discrepancies.
        """
        workspace_root = config.workspace_root

        # 1. Detect target HTML file
        target_html: Optional[Path] = None
        if html_path:
            p = resolve_safe_path(workspace_root, html_path)
            if not p.is_file():
                return f"Error: HTML file '{html_path}' not found."
            target_html = p
        else:
            cand = workspace_root / "index.html"
            if cand.is_file():
                target_html = cand
            else:
                for f in workspace_root.rglob("*.html"):
                    if "node_modules" not in f.parts and ".git" not in f.parts:
                        target_html = f
                        break

        if not target_html or not target_html.is_file():
            return "Error: No HTML file found to validate. Please pass 'html_path' parameter."

        rel_html_path = target_html.relative_to(workspace_root).as_posix()
        try:
            html_content = target_html.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error: Failed to read '{rel_html_path}': {e}"

        html_lines = html_content.splitlines()

        # Parse HTML Elements
        # a) Classes
        html_classes: Dict[str, List[int]] = {}
        for l_idx, line in enumerate(html_lines, start=1):
            for m in re.finditer(r'(?:class|className)\s*=\s*["\']([^"\']+)["\']', line):
                for cls_token in m.group(1).split():
                    c_clean = cls_token.strip()
                    if c_clean:
                        html_classes.setdefault(c_clean, []).append(l_idx)

        # b) IDs
        html_ids: Dict[str, List[int]] = {}
        for l_idx, line in enumerate(html_lines, start=1):
            for m in re.finditer(r'\bid\s*=\s*["\']([^"\']+)["\']', line):
                id_val = m.group(1).strip()
                if id_val:
                    html_ids.setdefault(id_val, []).append(l_idx)

        # c) Asset Links (Links / Scripts / Images)
        asset_links: List[Tuple[str, str, int]] = []
        for l_idx, line in enumerate(html_lines, start=1):
            for m in re.finditer(r'<(?:link|script|img)\b[^>]*(?:href|src)\s*=\s*["\']([^"\']+)["\']', line, re.IGNORECASE):
                tag_match = re.match(r'<([a-zA-Z0-9]+)', m.group(0))
                tag_name = tag_match.group(1).lower() if tag_match else "tag"
                asset_links.append((tag_name, m.group(1).strip(), l_idx))

        # 2. Locate and Parse CSS Files
        css_files: List[Path] = []
        if css_path:
            cp = resolve_safe_path(workspace_root, css_path)
            if cp.is_file():
                css_files.append(cp)
            else:
                return f"Error: CSS file '{css_path}' not found."
        else:
            for l_idx, line in enumerate(html_lines, start=1):
                for m in re.finditer(r'<link\b[^>]*rel\s*=\s*["\']stylesheet["\'][^>]*href\s*=\s*["\']([^"\']+)["\']', line, re.IGNORECASE):
                    href = m.group(1).split("?")[0].split("#")[0].strip()
                    if not href.startswith(("http://", "https://", "//")):
                        cand = (target_html.parent / href).resolve()
                        if cand.is_file() and cand not in css_files:
                            css_files.append(cand)
            if not css_files:
                for f in workspace_root.rglob("*.css"):
                    if "node_modules" not in f.parts and ".git" not in f.parts:
                        css_files.append(f)

        css_classes: Set[str] = set()
        css_ids: Set[str] = set()
        for cfile in css_files:
            try:
                ctext = cfile.read_text(encoding="utf-8", errors="replace")
                clean_css = re.sub(r"/\*.*?\*/", "", ctext, flags=re.DOTALL)
                for block in re.finditer(r"([^{}]+)\{", clean_css):
                    sel_text = block.group(1).strip()
                    if sel_text.startswith(("@keyframes", "@import", "@font-face")):
                        continue
                    for m in re.finditer(r"\.([a-zA-Z0-9_-]+)", sel_text):
                        css_classes.add(m.group(1))
                    for m in re.finditer(r"#([a-zA-Z0-9_-]+)", sel_text):
                        css_ids.add(m.group(1))
            except OSError:
                pass

        # 3. Locate and Parse JavaScript / TypeScript Files
        js_files: List[Path] = []
        if js_path:
            jp = resolve_safe_path(workspace_root, js_path)
            if jp.is_file():
                js_files.append(jp)
            else:
                return f"Error: JS file '{js_path}' not found."
        else:
            # Check script src in HTML
            for l_idx, line in enumerate(html_lines, start=1):
                for m in re.finditer(r'<script\b[^>]*src\s*=\s*["\']([^"\']+)["\']', line, re.IGNORECASE):
                    src = m.group(1).split("?")[0].split("#")[0].strip()
                    if not src.startswith(("http://", "https://", "//")):
                        cand = (target_html.parent / src).resolve()
                        if cand.is_file() and cand not in js_files:
                            js_files.append(cand)
            if not js_files:
                for f in workspace_root.rglob("*.js"):
                    if "node_modules" not in f.parts and ".git" not in f.parts and "tests" not in f.parts:
                        js_files.append(f)

        js_queried_ids: Dict[str, List[str]] = {}
        js_queried_classes: Dict[str, List[str]] = {}
        for jfile in js_files:
            rel_jf = jfile.relative_to(workspace_root).as_posix()
            try:
                jtext = jfile.read_text(encoding="utf-8", errors="replace")
                clean_js = re.sub(r"/\*.*?\*/", "", jtext, flags=re.DOTALL)
                clean_js = re.sub(r"//.*", "", clean_js)

                # document.getElementById("id")
                for m in re.finditer(r'\bdocument\.getElementById\s*\(\s*["\']([^"\']+)["\']\s*\)', clean_js):
                    js_queried_ids.setdefault(m.group(1), []).append(rel_jf)

                # querySelector("#id")
                for m in re.finditer(r'\bdocument\.querySelector(?:All)?\s*\(\s*["\']#([a-zA-Z0-9_-]+)["\']\s*\)', clean_js):
                    js_queried_ids.setdefault(m.group(1), []).append(rel_jf)

                # querySelector(".class")
                for m in re.finditer(r'\bdocument\.querySelector(?:All)?\s*\(\s*["\']\.([a-zA-Z0-9_-]+)["\']\s*\)', clean_js):
                    js_queried_classes.setdefault(m.group(1), []).append(rel_jf)

                # getElementsByClassName("class")
                for m in re.finditer(r'\bdocument\.getElementsByClassName\s*\(\s*["\']([^"\']+)["\']\s*\)', clean_js):
                    js_queried_classes.setdefault(m.group(1), []).append(rel_jf)
            except OSError:
                pass

        # 4. Verification and Contract Matching
        critical_errors: List[str] = []
        warnings: List[str] = []

        # a) Broken Asset Links
        if check_asset_links:
            for tag_name, link_ref, line_num in asset_links:
                clean_ref = link_ref.split("?")[0].split("#")[0].strip()
                if not clean_ref or clean_ref.startswith(("http://", "https://", "data:", "//", "#", "mailto:", "tel:")):
                    continue
                cand1 = (target_html.parent / clean_ref).resolve()
                cand2 = (workspace_root / clean_ref).resolve()
                if not cand1.exists() and not cand2.exists():
                    critical_errors.append(
                        f"❌ [Broken Asset Link / Kırık Varlık Bağlantısı]: {rel_html_path} (Line {line_num}) `<{tag_name} ...='{link_ref}'>` "
                        f"referenced, but '{clean_ref}' does not exist on disk!"
                    )

        # b) Duplicate DOM IDs
        for id_name, line_nums in html_ids.items():
            if len(line_nums) > 1:
                warnings.append(
                    f"⚠ [Duplicate DOM ID / Mükerrer DOM ID]: 'id=\"{id_name}\"' used {len(line_nums)} times in {rel_html_path} "
                    f"(Lines: {', '.join(map(str, line_nums))}). IDs must be unique across the document."
                )

        # c) JS DOM ID Queries vs HTML IDs
        for qid, caller_files in js_queried_ids.items():
            if qid not in html_ids:
                closest = difflib.get_close_matches(qid, list(html_ids.keys()), n=2, cutoff=0.6)
                caller_str = ", ".join(set(caller_files))
                if closest:
                    sugg = ", ".join(f"'#{m}'" for m in closest)
                    critical_errors.append(
                        f"❌ [Missing/Mismatched DOM ID / Eksik DOM ID]: {caller_str} invoked `document.getElementById(\"{qid}\")`, "
                        f"but no element with id=\"{qid}\" exists in {rel_html_path}! "
                        f"(Similar in HTML: {sugg}). Synchronize ID names between JS and HTML."
                    )
                else:
                    critical_errors.append(
                        f"❌ [Missing DOM ID / Eksik DOM ID]: {caller_str} invoked `document.getElementById(\"{qid}\")`, "
                        f"but no element with id=\"{qid}\" was found in {rel_html_path}."
                    )

        # d) HTML Classes vs CSS Selectors
        if css_files:
            for cls_name, line_nums in html_classes.items():
                if cls_name not in css_classes:
                    closest = difflib.get_close_matches(cls_name, list(css_classes), n=2, cutoff=0.65)
                    if closest:
                        sugg = ", ".join(f"'.{m}'" for m in closest)
                        warnings.append(
                            f"⚠ [Contract Mismatch / Sözleşme Uyuşmazlığı]: Class '.{cls_name}' used in {rel_html_path} (Line {line_nums[0]}) "
                            f"not found in CSS! (Similar in CSS: {sugg})."
                        )

        # e) Unclosed HTML Container Tags
        container_tags = ("div", "section", "main", "header", "footer", "nav", "ul", "ol", "form", "article")
        for tag in container_tags:
            open_count = len(re.findall(rf"<{tag}\b(?![^>]*/>)[^>]*>", html_content, re.IGNORECASE))
            close_count = len(re.findall(rf"</{tag}\s*>", html_content, re.IGNORECASE))
            if open_count != close_count:
                diff = abs(open_count - close_count)
                if open_count > close_count:
                    warnings.append(
                        f"⚠ [Unclosed HTML Tag / Kapatılmamış HTML Etiketi]: {open_count} `<{tag}>` opened in {rel_html_path}, "
                        f"but only {close_count} `</{tag}>` closed ({diff} unclosed)!"
                    )
                else:
                    warnings.append(
                        f"⚠ [Excess Closing Tag / Fazla Kapanış Etiketi]: {close_count} `</{tag}>` closed in {rel_html_path}, "
                        f"but only {open_count} `<{tag}>` opened ({diff} extra)!"
                    )

        # Report Formatting
        css_labels = ", ".join(f.relative_to(workspace_root).as_posix() for f in css_files) or "None"
        js_labels = ", ".join(f.relative_to(workspace_root).as_posix() for f in js_files) or "None"

        summary_lines = [
            "📋 **Web Interface Contract Audit Report / Web Arayüz Sözleşme Denetim Raporu:**",
            f"- **HTML:** `{rel_html_path}` ({len(html_classes)} classes, {len(html_ids)} IDs, {len(asset_links)} assets)",
            f"- **CSS:** `{css_labels}` ({len(css_classes)} class selectors)",
            f"- **JavaScript:** `{js_labels}` ({len(js_queried_ids)} ID queries, {len(js_queried_classes)} class queries)",
            "",
        ]

        if not critical_errors and not warnings:
            return (
                f"✔ **[Web Interface Contract Verified / Web Arayüz Sözleşmesi Doğrulandı / PASS]**\n\n"
                + "\n".join(summary_lines)
                + "Full contract parity verified across HTML, CSS, and JavaScript. "
                "No broken asset links, missing DOM IDs, or mismatched CSS selectors detected (0 errors, 0 warnings)."
            )

        err_section = ""
        if critical_errors:
            err_section = f"### ❌ Critical Contract Errors ({len(critical_errors)}):\n" + "\n".join(f"  • {e}" for e in critical_errors) + "\n\n"

        warn_section = ""
        if warnings:
            warn_section = f"### ⚠ Warnings & Suggestions ({len(warnings)}):\n" + "\n".join(f"  • {w}" for w in warnings) + "\n\n"

        return (
            f"❌ **[Web Interface Contract Mismatch / Web Arayüz Sözleşmesi Uyuşmazlığı / FAILED]**\n\n"
            + "\n".join(summary_lines)
            + err_section
            + warn_section
            + "💡 **Recommendation:** Resolve the missing DOM IDs or class mismatches via `edit_file` to align the web contract."
        )

    registry.register(
        name="validate_web_contract",
        description=(
            "Statically validates interface contracts across HTML, CSS, and JS (DOM IDs, CSS classes, asset references) in 0ms. "
            "Identifies missing DOM IDs, CSS class mismatches, and broken asset links for web verification."
        ),
        parameters={
            "type": "object",
            "properties": {
                "html_path": {"type": "string", "description": "Relative path to target HTML file (e.g. 'index.html'). Auto-detected if omitted."},
                "css_path": {"type": "string", "description": "Relative path to target CSS file. Auto-detected from HTML links if omitted."},
                "js_path": {"type": "string", "description": "Relative path to target JS file. Auto-detected from HTML scripts if omitted."},
                "check_asset_links": {"type": "boolean", "description": "Whether to verify existence of asset link/script/img targets on disk (default: true).", "default": True},
            },
        },
        handler=validate_web_contract,
        requires_approval=False,
    )
