"""FileStateTracker: Smart caching mechanism tracking file read and write states
to prevent redundant reads, token waste, and context bloating.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
from typing import Dict, Optional, Tuple


@dataclass
class FileState:
    path: str
    mtime: float
    sha256: str
    size: int
    first_read_step: int
    last_read_step: int
    read_count: int = 1
    stub_count: int = 0
    is_full_read: bool = True
    was_written: bool = False


class FileStateTracker:
    """Tracks hash/mtime status of files read and edited during session.
    
    When the model attempts to repeatedly read the same file without intermediate edits,
    returns a compact stub instead of dumping the full text.
    This prevents token waste, context pollution, and repetitive read loops.
    """

    def __init__(self) -> None:
        self._states: Dict[str, FileState] = {}
        self.current_step: int = 1
        self._lock = threading.Lock()

    def _normalize_path(self, path: str) -> str:
        try:
            return str(Path(path).resolve().as_posix().lower())
        except Exception:
            return path.replace("\\", "/").strip().lower()

    def _compute_sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    def check_file(self, raw_path: str, current_step: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """Check file."""
        step = current_step if current_step is not None else self.current_step
        norm_key = self._normalize_path(raw_path)
        path_obj = Path(raw_path)

        if not path_obj.exists() or not path_obj.is_file():
            return False, None

        try:
            stat = path_obj.stat()
            current_mtime = stat.st_mtime
            current_size = stat.st_size
        except OSError:
            return False, None

        with self._lock:
            if norm_key in self._states:
                state = self._states[norm_key]
                if not state.is_full_read:
                    return False, None

                if state.mtime == current_mtime and state.size == current_size:
                    if state.stub_count >= 1:
                        state.stub_count = 0
                        return False, None
                    state.read_count += 1
                    state.stub_count += 1
                    state.last_read_step = step
                    if getattr(state, "was_written", False):
                        stub = (
                            f"[Dosya Değişmedi / Az Önce Yazıldı]: '{raw_path}'\n"
                            f"Bu dosya Adım {state.first_read_step} içerisinde tarafınızca yeni yazıldı/güncellendi "
                            f"ve tam içeriği bağlamınızda (context) mevcuttur ({current_size} bayt).\n"
                            f"Dosya üzerinde harici bir değişiklik olmadı. "
                            f"İçeriği tekrar okumak yerine mevcut bağlamınızı kullanabilir veya gerekiyorsa cerrahi `edit_file` çağırabilirsiniz."
                        )
                    else:
                        stub = (
                            f"[Dosya Değişmedi / File Unchanged]: '{raw_path}'\n"
                            f"Bu dosya Adım {state.first_read_step} içerisinde zaten okundu "
                            f"ve bağlam geçmişinizde (context) tam içeriğiyle mevcuttur ({current_size} bayt).\n"
                            f"Dosya üzerinde harici bir değişiklik olmadı. "
                            f"Değişiklik yapmak istiyorsanız doğrudan `edit_file` aracını kullanın veya önceki çıktıya başvurun."
                        )
                    return True, stub
                
                current_sha = self._compute_sha256(path_obj)
                if current_sha and current_sha == state.sha256:
                    if state.stub_count >= 1:
                        state.stub_count = 0
                        return False, None
                    state.mtime = current_mtime
                    state.read_count += 1
                    state.stub_count += 1
                    state.last_read_step = step
                    if getattr(state, "was_written", False):
                        stub = (
                            f"[Dosya Değişmedi / Az Önce Yazıldı]: '{raw_path}'\n"
                            f"Bu dosya Adım {state.first_read_step} içerisinde tarafınızca yeni yazıldı/güncellendi "
                            f"ve tam içeriği bağlamınızda (context) mevcuttur ({current_size} bayt).\n"
                            f"İçerik hash'i değişmedi. Doğrudan `edit_file` kullanabilirsiniz."
                        )
                    else:
                        stub = (
                            f"[Dosya Değişmedi / File Unchanged]: '{raw_path}'\n"
                            f"Bu dosya Adım {state.first_read_step} içerisinde zaten okundu "
                            f"ve bağlam geçmişinizde (context) tam içeriğiyle mevcuttur ({current_size} bayt).\n"
                            f"İçerik hash'i değişmedi. Doğrudan `edit_file` kullanabilirsiniz."
                        )
                    return True, stub

            return False, None

    def record_write(
        self,
        raw_path: str,
        current_step: Optional[int] = None,
    ) -> None:
        """Record write."""
        step = current_step if current_step is not None else self.current_step
        norm_key = self._normalize_path(raw_path)
        path_obj = Path(raw_path)
        if not path_obj.exists() or not path_obj.is_file():
            return

        try:
            stat = path_obj.stat()
            sha = self._compute_sha256(path_obj)
            with self._lock:
                self._states[norm_key] = FileState(
                    path=raw_path,
                    mtime=stat.st_mtime,
                    sha256=sha,
                    size=stat.st_size,
                    first_read_step=step,
                    last_read_step=step,
                    read_count=1,
                    is_full_read=True,
                    was_written=True,
                )
        except OSError:
            pass

    def record_read(
        self,
        raw_path: str,
        current_step: Optional[int] = None,
        is_full_read: bool = True,
    ) -> None:
        """Record read."""
        step = current_step if current_step is not None else self.current_step
        norm_key = self._normalize_path(raw_path)
        path_obj = Path(raw_path)
        if not path_obj.exists() or not path_obj.is_file():
            return

        try:
            stat = path_obj.stat()
            sha = self._compute_sha256(path_obj)
            with self._lock:
                prev = self._states.get(norm_key)
                effective_full = is_full_read or (prev is not None and prev.is_full_read and prev.sha256 == sha)
                self._states[norm_key] = FileState(
                    path=raw_path,
                    mtime=stat.st_mtime,
                    sha256=sha,
                    size=stat.st_size,
                    first_read_step=prev.first_read_step if prev else step,
                    last_read_step=step,
                    read_count=(prev.read_count + 1) if prev else 1,
                    is_full_read=effective_full,
                )
        except OSError:
            pass

    def invalidate(self, raw_path: str) -> None:
        """Invalidate."""
        norm_key = self._normalize_path(raw_path)
        clean_raw = raw_path.replace("\\", "/").strip().lower().lstrip("./")
        with self._lock:
            self._states.pop(norm_key, None)
            to_del = [
                k for k in self._states
                if k.lower() == clean_raw
                or k.lower().endswith("/" + clean_raw)
                or clean_raw.endswith("/" + k.lower())
            ]
            for k in to_del:
                self._states.pop(k, None)

    def reset(self) -> None:
        """Reset."""
        with self._lock:
            self._states.clear()
