"""Automated PATH Environment Setup Module (`path_setup.py`).

Appends the Python `Scripts` / `bin` directory to the user's OS PATH environment variable
on initial setup (`evren --setup`) so `evren` and `evren-code` can be executed from any terminal.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PathSetupResult:
    """Outcome of PATH inspection and registration."""

    scripts_dir: Path
    already_in_path: bool
    added_to_path: bool
    message: str


def get_scripts_directory() -> Path:
    """Locates the Python `Scripts` (Windows) or `bin` (Unix) executable directory."""
    candidates: List[Path] = []

    scripts_cfg = sysconfig.get_path("scripts")
    if scripts_cfg:
        candidates.append(Path(scripts_cfg))

    exe_parent = Path(sys.executable).resolve().parent
    if os.name == "nt":
        candidates.append(exe_parent / "Scripts")
        candidates.append(exe_parent)
    else:
        candidates.append(exe_parent)
        candidates.append(Path.home() / ".local" / "bin")

    for candidate in candidates:
        if (candidate / "evren.exe").is_file() or (candidate / "evren").is_file():
            return candidate.resolve()

    return candidates[0].resolve()


def _normalize_path_str(p: str) -> str:
    cleaned = p.strip().rstrip("\\/")
    return cleaned.lower() if os.name == "nt" else cleaned


def is_dir_in_path(target_dir: Path, path_value: Optional[str] = None) -> bool:
    """Is dir in path."""
    raw_path = path_value if path_value is not None else os.environ.get("PATH", "")
    target_norm = _normalize_path_str(str(target_dir))
    entries = [_normalize_path_str(item) for item in raw_path.split(os.pathsep) if item.strip()]
    return target_norm in entries


def ensure_cli_on_path() -> PathSetupResult:
    """Ensure cli on path."""
    scripts_dir = get_scripts_directory()

    if os.name == "nt":
        return _ensure_windows_user_path(scripts_dir)
    return _ensure_posix_user_path(scripts_dir)


def _ensure_windows_user_path(scripts_dir: Path) -> PathSetupResult:
    """Ensure windows user path."""
    import winreg

    target_str = str(scripts_dir)
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        ) as key:
            try:
                current_user_path, reg_type = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current_user_path = ""
                reg_type = winreg.REG_EXPAND_SZ

            if is_dir_in_path(scripts_dir, current_user_path):
                if not is_dir_in_path(scripts_dir, os.environ.get("PATH", "")):
                    os.environ["PATH"] = f"{os.environ.get('PATH', '')};{target_str}"
                return PathSetupResult(
                    scripts_dir=scripts_dir,
                    already_in_path=True,
                    added_to_path=False,
                    message=f"'{target_str}' zaten Windows PATH içinde kayıtlı.",
                )

            parts = [p for p in current_user_path.split(";") if p.strip()]
            parts.append(target_str)
            new_user_path = ";".join(parts)

            winreg.SetValueEx(key, "Path", 0, reg_type, new_user_path)
            os.environ["PATH"] = f"{os.environ.get('PATH', '')};{target_str}"

            try:
                import ctypes

                HWND_BROADCAST = 0xFFFF
                WM_SETTINGCHANGE = 0x001A
                SMTO_ABORTIFHUNG = 0x0002
                result = ctypes.c_long()
                ctypes.windll.user32.SendMessageTimeoutW(
                    HWND_BROADCAST,
                    WM_SETTINGCHANGE,
                    0,
                    "Environment",
                    SMTO_ABORTIFHUNG,
                    2000,
                    ctypes.byref(result),
                )
            except Exception:
                pass

            return PathSetupResult(
                scripts_dir=scripts_dir,
                already_in_path=False,
                added_to_path=True,
                message=(
                    f"`evren` komutu Windows PATH'e otomatik eklendi ({target_str}). "
                    "Artık herhangi bir klasörde terminal açıp doğrudan `evren` yazabilirsiniz."
                ),
            )
    except Exception as exc:
        return PathSetupResult(
            scripts_dir=scripts_dir,
            already_in_path=False,
            added_to_path=False,
            message=f"Windows PATH otomatik güncellenemedi: {exc}",
        )


def _ensure_posix_user_path(scripts_dir: Path) -> PathSetupResult:
    """Ensure posix user path."""
    target_str = str(scripts_dir)
    if is_dir_in_path(scripts_dir):
        return PathSetupResult(
            scripts_dir=scripts_dir,
            already_in_path=True,
            added_to_path=False,
            message=f"'{target_str}' zaten PATH içinde.",
        )

    export_line = f'\n# Evren-Code CLI\nexport PATH="$PATH:{target_str}"\n'
    shell = os.environ.get("SHELL", "")
    rc_file = Path.home() / (".zshrc" if "zsh" in shell else ".bashrc")

    try:
        existing = rc_file.read_text(encoding="utf-8") if rc_file.is_file() else ""
        if target_str not in existing:
            with open(rc_file, "a", encoding="utf-8") as f:
                f.write(export_line)
        os.environ["PATH"] = f"{os.environ.get('PATH', '')}:{target_str}"
        return PathSetupResult(
            scripts_dir=scripts_dir,
            already_in_path=False,
            added_to_path=True,
            message=f"`evren` komutu {rc_file} dosyasına eklendi ({target_str}).",
        )
    except Exception as exc:
        return PathSetupResult(
            scripts_dir=scripts_dir,
            already_in_path=False,
            added_to_path=False,
            message=f"PATH otomatik güncellenemedi: {exc}",
        )
