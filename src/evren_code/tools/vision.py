"""Visual Analysis Tool (`analyze_image`).

Leverages EVREN's `qwen3-vl-30b` (`vision_chat`) vision model to inspect
screenshots, UI layouts, bugs, or architectural diagrams.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from evren_code.config import AppConfig
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.tools.filesystem import resolve_safe_path
from evren_code.tools.registry import ToolRegistry


def find_system_browser() -> Optional[str]:
    """Automatically detects a system-installed Chromium-based headless browser (Edge, Chrome, Chromium)."""
    # 1. Binaries on PATH
    for name in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser", "brave"):
        p = shutil.which(name)
        if p and Path(p).is_file():
            return p

    # 2. Windows default paths
    if sys.platform == "win32":
        win_candidates = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge SxS\Application\msedge.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
        for c in win_candidates:
            if c and Path(c).is_file():
                return c

    # 3. macOS default paths
    elif sys.platform == "darwin":
        mac_candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
        for c in mac_candidates:
            if Path(c).is_file():
                return c

    # 4. Linux default paths
    elif sys.platform.startswith("linux"):
        linux_candidates = [
            "/usr/bin/google-chrome",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/usr/bin/microsoft-edge",
        ]
        for c in linux_candidates:
            if Path(c).is_file():
                return c

    return None


def capture_web_page_screenshot(
    workspace_root: Path,
    path_or_url: str,
    output_path: Optional[str] = None,
    viewport: str = "desktop",
    timeout_seconds: float = 20.0,
) -> Tuple[bool, str, Optional[Path]]:
    """Captures a headless browser screenshot of an HTML file or live URL."""
    browser_bin = find_system_browser()
    if not browser_bin:
        return False, "No headless-compatible browser (Edge, Chrome, Chromium) detected on system (Sistemde kurulu headless uyumlu tarayıcı bulunamadı).", None

    vp_map = {
        "desktop": (1280, 800),
        "mobile": (375, 667),
        "tablet": (768, 1024),
    }
    w, h = vp_map.get(viewport.lower(), (1280, 800))

    target_str = path_or_url.strip()
    if target_str.startswith(("http://", "https://")):
        url = target_str
        stem = re.sub(r'[^a-zA-Z0-9_-]', '_', target_str.replace("https://", "").replace("http://", ""))[:20]
    else:
        file_p = resolve_safe_path(workspace_root, target_str)
        if not file_p.is_file():
            return False, f"Error: HTML/web file '{target_str}' not found.", None
        url = file_p.as_uri()
        stem = file_p.stem

    if output_path:
        out_file = resolve_safe_path(workspace_root, output_path, is_write=True)
    else:
        shots_dir = workspace_root / ".evren" / "screenshots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        out_file = shots_dir / f"{stem}_{viewport}_{timestamp}.png"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    profile_dir = tempfile.mkdtemp(prefix="evren_browser_")

    cmd = [
        browser_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-dev-shm-usage",
        "--hide-scrollbars",
        f"--user-data-dir={profile_dir}",
        f"--window-size={w},{h}",
        f"--screenshot={out_file}",
        url,
    ]

    try:
        from evren_code.tools.shell import execute_command_with_tree_kill

        rc, _stdout_str, stderr_str = execute_command_with_tree_kill(
            args=cmd,
            cwd=str(workspace_root),
            timeout=timeout_seconds,
            shell=False,
        )
        if rc != 0 and not out_file.is_file():
            return False, f"Headless tarayıcı çalıştırma hatası: {stderr_str[:300]}", None
        if not out_file.is_file() or out_file.stat().st_size == 0:
            return False, "Ekran görüntüsü dosyası oluşturulamadı veya boş.", None
        return True, "Başarılı", out_file
    except subprocess.TimeoutExpired:
        return False, f"Headless tarayıcı {timeout_seconds}s içinde zaman aşımına uğradı.", None
    except Exception as exc:
        return False, f"Ekran görüntüsü alma hatası: {exc}", None
    finally:
        try:
            shutil.rmtree(profile_dir, ignore_errors=True)
        except Exception:
            pass


# In-memory vision cache to eliminate redundant network roundtrips for unchanged images
_VISION_CACHE: Dict[str, str] = {}


def save_clipboard_image(workspace_root: Path) -> Optional[Path]:
    """Saves an image from the system clipboard to `.evren/screenshots/clipboard_<ts>.png` if present."""
    shots_dir = workspace_root / ".evren" / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    out_file = shots_dir / f"clipboard_{int(time.time() * 1000)}.png"

    try:
        if sys.platform == "win32":
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
                "$img = [System.Windows.Forms.Clipboard]::GetImage(); "
                f"if ($img -ne $null) {{ $img.Save('{str(out_file)}', [System.Drawing.Imaging.ImageFormat]::Png); Write-Output 'SAVED' }}"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "SAVED" in (res.stdout or "") and out_file.is_file() and out_file.stat().st_size > 0:
                return out_file
        elif sys.platform == "darwin":
            res = subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'tell application "System Events" to write (the clipboard as «class PNGf») to (make new file at folder "{str(shots_dir)}" with properties {{name:"{out_file.name}"}})',
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and out_file.is_file() and out_file.stat().st_size > 0:
                return out_file
        elif sys.platform.startswith("linux"):
            for cmd in (
                ["wl-paste", "--type", "image/png"],
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
            ):
                if shutil.which(cmd[0]):
                    res_b = subprocess.run(cmd, capture_output=True, timeout=5)
                    if res_b.returncode == 0 and res_b.stdout:
                        out_file.write_bytes(res_b.stdout)
                        return out_file
    except Exception:
        pass
    return None


def get_clipboard_text() -> str:
    """Retrieves plain text content from the system clipboard if available."""
    try:
        if sys.platform == "win32":
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3,
            )
            if res.returncode == 0:
                return (res.stdout or "").strip()
        elif sys.platform == "darwin":
            res = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                return (res.stdout or "").strip()
        elif sys.platform.startswith("linux"):
            for cmd in (["wl-paste"], ["xclip", "-selection", "clipboard", "-o"]):
                if shutil.which(cmd[0]):
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                    if res.returncode == 0:
                        return (res.stdout or "").strip()
    except Exception:
        pass
    return ""


def register_vision_tools(
    registry: ToolRegistry,
    config: AppConfig,
    llm_client: EvrenLLMClient,
) -> None:
    """Register vision tools."""

    def analyze_image(image_path: str, question: str = "Bu görseli detaylıca analiz et.") -> str:
        clean_path = (image_path or "").strip()
        if clean_path.lower() in ("clipboard", "@clipboard", "pano", "@pano"):
            clip_file = save_clipboard_image(config.workspace_root)
            if not clip_file:
                return "Hata: Sistem panosunda (clipboard) analiz edilecek bir görsel bulunamadı."
            target = clip_file
        else:
            target = resolve_safe_path(config.workspace_root, clean_path)
        if not target.is_file():
            return f"Hata: '{image_path}' görsel dosyası bulunamadı."

        raw_bytes = target.read_bytes()
        file_hash = hashlib.sha256(raw_bytes).hexdigest()
        cache_key = f"{file_hash}:{question.strip().lower()}"

        if cache_key in _VISION_CACHE:
            return f"[Vision Cache / Önbellekten Getirildi ⚡]:\n{_VISION_CACHE[cache_key]}"

        cache_dir = config.workspace_root / ".evren" / "cache"
        cache_file = cache_dir / "vision_cache.json"
        if cache_file.exists():
            try:
                disk_cache = json.loads(cache_file.read_text(encoding="utf-8"))
                if cache_key in disk_cache:
                    _VISION_CACHE[cache_key] = disk_cache[cache_key]
                    return f"[Vision Cache / Önbellekten Getirildi ⚡]:\n{disk_cache[cache_key]}"
            except Exception:
                pass

        mime_type, _ = mimetypes.guess_type(target.name)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"

        b64_data = base64.b64encode(raw_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64_data}"

        vision_model = config.models.get_model("vision")
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        try:
            response = llm_client.chat(
                messages=messages,
                tools_schema=None,
                model_override=vision_model.id,
                stream=False,
            )
            result_content = response.content or ""
            _VISION_CACHE[cache_key] = result_content

            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                disk_cache = {}
                if cache_file.exists():
                    try:
                        disk_cache = json.loads(cache_file.read_text(encoding="utf-8"))
                    except Exception:
                        disk_cache = {}
                disk_cache[cache_key] = result_content
                cache_file.write_text(json.dumps(disk_cache, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            return f"[{vision_model.name} Görsel Analiz Sonucu]:\n{result_content}"
        except Exception as exc:
            return f"Hata: Görsel analiz sırasında sorun oluştu: {exc}"

    registry.register(
        name="analyze_image",
        description="Inspects an image or screenshot file (.png, .jpg, .webp) using the EVREN Vision model and answers visual questions.",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to the target image file."},
                "question": {"type": "string", "description": "Specific question or inspection prompt about the image."},
            },
            "required": ["image_path"],
        },
        handler=analyze_image,
        requires_approval=False,
    )

    def capture_web_screenshot(
        path_or_url: str,
        viewport: str = "desktop",
        output_path: Optional[str] = None,
    ) -> str:
        ok, msg, shot_path = capture_web_page_screenshot(
            config.workspace_root,
            path_or_url,
            output_path=output_path,
            viewport=viewport,
        )
        if not ok or not shot_path:
            return f"Error: Failed to capture screenshot: {msg}"
        try:
            rel_shot = shot_path.relative_to(config.workspace_root).as_posix()
        except ValueError:
            rel_shot = str(shot_path)
        return f"Success: Screenshot for '{path_or_url}' ({viewport}) saved: `{rel_shot}` ({shot_path.stat().st_size} bytes) (Başarılı: '{path_or_url}' ({viewport}) ekran görüntüsü kaydedildi: `{rel_shot}`)."

    registry.register(
        name="capture_web_screenshot",
        description="Captures a headless browser screenshot (.png) of a local HTML file or active web server endpoint.",
        parameters={
            "type": "object",
            "properties": {
                "path_or_url": {"type": "string", "description": "Relative HTML file path or web URL."},
                "viewport": {
                    "type": "string",
                    "enum": ["desktop", "mobile", "tablet"],
                    "description": "Viewport dimensions: desktop (1280x800), mobile (375x667), tablet (768x1024).",
                },
                "output_path": {"type": "string", "description": "Optional destination path for the saved screenshot PNG."},
            },
            "required": ["path_or_url"],
        },
        handler=capture_web_screenshot,
        requires_approval=False,
    )

    def inspect_web_page_visually(
        path_or_url: str,
        viewport: str = "desktop",
        focus: str = "General UI layout, overflow, alignment, and visual contrast analysis (Genel UI, taşma, hizalama)",
    ) -> str:
        ok, msg, shot_path = capture_web_page_screenshot(
            config.workspace_root,
            path_or_url,
            viewport=viewport,
        )
        if not ok or not shot_path:
            return f"Error: Failed to capture page screenshot: {msg}"

        try:
            rel_shot = shot_path.relative_to(config.workspace_root).as_posix()
        except ValueError:
            rel_shot = str(shot_path)

        ui_prompt = (
            f"You are a Principal UI/UX Designer and Frontend Engineer. "
            f"The attached screenshot belongs to '{path_or_url}' in '{viewport}' viewport.\n"
            f"Focus Area: {focus}\n\n"
            "Please visually evaluate these 5 critical criteria and report concisely:\n"
            "1. Overflow & Layout Shifts: Any horizontal/vertical clipping or unintended scrollbars?\n"
            "2. Visual Hierarchy & Typography: Balanced headers, body text, and line heights?\n"
            "3. Color & Contrast: Legible text against backgrounds (WCAG compliance)?\n"
            "4. Buttons & Interactive Elements: Appropriately sized and clearly clickable?\n"
            "5. Concrete Recommendations: 1-2 actionable CSS/HTML fixes if issues exist."
        )

        res = analyze_image(str(rel_shot), question=ui_prompt)
        return (
            f"📸 **[Visual Eye - Canlı Görsel UI İnceleme Raporu / Live UI Inspection Report]**\n"
            f"- **Target:** `{path_or_url}` ({viewport})\n"
            f"- **Screenshot:** `{rel_shot}`\n\n"
            f"{res}"
        )

    registry.register(
        name="inspect_web_page_visually",
        description="Visual Eye: Renders a screenshot of a local HTML page or URL and analyzes UI layout, overflows, contrast, and responsiveness.",
        parameters={
            "type": "object",
            "properties": {
                "path_or_url": {"type": "string", "description": "Target HTML file path or web URL (e.g. index.html)."},
                "viewport": {
                    "type": "string",
                    "enum": ["desktop", "mobile", "tablet"],
                    "description": "Target viewport (desktop: 1280x800, mobile: 375x667, tablet: 768x1024).",
                },
                "focus": {"type": "string", "description": "Optional specific UI focus or element to inspect."},
            },
            "required": ["path_or_url"],
        },
        handler=inspect_web_page_visually,
        requires_approval=False,
    )
