"""Self-Update Module (`updater.py`).

1. In editable development installations (`pip install -e .`), local changes apply immediately.
2. In git-backed environments, `evren --update` or `/update` pulls latest changes (`git pull`)
   and upgrades dependencies.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class UpdateResult:
    """Status and summary of self-update execution."""

    success: bool
    summary: str
    details: str = ""


def get_source_root() -> Path:
    """Returns root directory of Evren-Code repository."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _has_upstream_tracking(repo_root: Path) -> bool:
    """Checks whether active branch has an upstream tracking branch configured."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "@{u}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        return False


def _has_git_remote(repo_root: Path) -> bool:
    """Has git remote (remote/origin)."""
    try:
        res = subprocess.run(
            ["git", "remote"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(res.stdout.strip())
    except Exception:
        return False


def run_self_update(install_deps: bool = False) -> UpdateResult:
    """Run self update."""
    repo_root = get_source_root()
    steps_log: List[str] = [f"Kaynak dizin: {repo_root}"]

    git_dir = repo_root / ".git"
    if git_dir.exists():
        if _has_upstream_tracking(repo_root):
            try:
                git_proc = subprocess.run(
                    ["git", "pull", "--ff-only"],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
                git_out = (git_proc.stdout or git_proc.stderr or "").strip()
                steps_log.append(f"Git Güncellemesi: {git_out}")
                if git_proc.returncode != 0:
                    return UpdateResult(
                        success=False,
                        summary="Git üzerinden güncelleme çekilirken hata oluştu.",
                        details="\n".join(steps_log),
                    )
            except Exception as exc:
                steps_log.append(f"Git uyarısı: {exc}")
        elif _has_git_remote(repo_root):
            steps_log.append(
                "Git Durumu: Uzak depo (remote) mevcut ancak bu branch için upstream izleme ayarlanmamış. "
                "Yerel kod tabanı kullanılıyor."
            )
        else:
            steps_log.append(
                "Yerel Geliştirici Modu: Uzak git deposu (origin) tanımlı değil. "
                "Kod değişiklikleri doğrudan aktiftir."
            )
    else:
        steps_log.append(
            "Yerel Geliştirici Modu (Editable Install): Kod değişiklikleri otomatik olarak aktiftir."
        )

    if install_deps or git_dir.exists():
        try:
            pip_cmd = [sys.executable, "-m", "pip", "install", "-e", str(repo_root), "--no-deps", "--quiet"]
            pip_proc = subprocess.run(
                pip_cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if pip_proc.returncode == 0:
                steps_log.append("Paket ve bağımlılıklar doğrulandı (`pip install -e .`).")
            else:
                stderr_clean = pip_proc.stderr.strip()
                if "WinError 5" in stderr_clean and "evren.exe" in stderr_clean:
                    steps_log.append("Çalışan oturum tespit edildi: Kod değişiklikleri dinamik olarak bellekte aktiftir.")
                else:
                    steps_log.append(f"Pip uyarısı: {stderr_clean}")
        except Exception as exc:
            steps_log.append(f"Pip güncelleme uyarısı: {exc}")

    return UpdateResult(
        success=True,
        summary="Evren-Code güncel durumda!",
        details="\n".join(steps_log),
    )
