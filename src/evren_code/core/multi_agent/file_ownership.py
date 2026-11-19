"""File Ownership Manager module."""

from __future__ import annotations

import fnmatch
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from evren_code.core.multi_agent.models import FileLock
from evren_code.core.security import PathContainment

FILE_OWNERSHIP_VIOLATION = "FILE_OWNERSHIP_VIOLATION"


class FileOwnershipManager:
    """File ownership manager implementation."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self._lock = threading.RLock()
        # path -> FileLock (exclusive write)
        self._write_locks: Dict[str, FileLock] = {}
        # path -> Set[owner_task_id] (shared read)
        self._read_locks: Dict[str, Set[str]] = {}
        # task_id -> Set[path] (for fast cleanup)
        self._task_write_paths: Dict[str, Set[str]] = {}
        self._task_read_paths: Dict[str, Set[str]] = {}

    def _normalize_path(self, path_str: str) -> str:
        """Normalize path."""
        if not path_str:
            return ""
        try:
            p = Path(path_str)
            if not p.is_absolute():
                p = (self.workspace_root / p).resolve()
            else:
                p = p.resolve()

            if not PathContainment.is_contained(self.workspace_root, p):
                raise PermissionError(
                    f"[{FILE_OWNERSHIP_VIOLATION}]: '{path_str}' yolu çalışma alanı ({self.workspace_root}) sınırları dışındadır."
                )

            rel = p.relative_to(self.workspace_root.resolve())
            return str(rel).replace("\\", "/").strip().lower()
        except PermissionError:
            raise
        except Exception:
            return path_str.replace("\\", "/").strip().lower()

    def acquire_locks(
        self,
        task_id: str,
        agent_id: str,
        write_files: List[str],
        read_files: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Acquire locks."""
        read_files = read_files or []
        try:
            norm_write = [self._normalize_path(f) if ("*" not in f and "?" not in f) else f.strip().replace("\\", "/").lower() for f in write_files if f.strip()]
            norm_read = [self._normalize_path(f) if ("*" not in f and "?" not in f) else f.strip().replace("\\", "/").lower() for f in read_files if f.strip()]
        except PermissionError as exc:
            return False, str(exc)

        with self._lock:
            for wpath in norm_write:
                if "*" in wpath or "?" in wpath:
                    if wpath in ("*", "**"):
                        other_active_writes = any(tid != task_id for tid in self._task_write_paths if self._task_write_paths[tid])
                        other_active_reads = any(tid != task_id for tid in self._task_read_paths if self._task_read_paths[tid])
                        if other_active_writes or other_active_reads:
                            return False, (
                                f"[{FILE_OWNERSHIP_VIOLATION}] Çalışma alanında başka aktif görevler "
                                f"kilit tutarken genel ('*') yazma kilidi talep edilemez."
                            )
                    continue

                for other_tid, other_paths in self._task_write_paths.items():
                    if other_tid != task_id:
                        if "*" in other_paths or "**" in other_paths:
                            return False, (
                                f"[{FILE_OWNERSHIP_VIOLATION}] '{wpath}' yolu, '{other_tid}' görevinin "
                                f"genel yazma kapsamı ('*') altındadır."
                            )

                existing_lock = self._write_locks.get(wpath)
                if existing_lock and existing_lock.owner_task_id != task_id:
                    return False, (
                        f"[{FILE_OWNERSHIP_VIOLATION}] '{wpath}' dosyası '{existing_lock.owner_task_id}' "
                        f"tarafından kilitlenmiş durumda. Görev '{task_id}' yazma kilidi edinemedi."
                    )
                readers = self._read_locks.get(wpath, set())
                active_readers = [r for r in readers if r != task_id]
                if active_readers:
                    return False, (
                        f"[{FILE_OWNERSHIP_VIOLATION}] '{wpath}' dosyası şu anda başka görevler "
                        f"({active_readers}) tarafından okunuyor. Yazma kilidi bekletilmeli."
                    )

            for rpath in norm_read:
                if "*" in rpath or "?" in rpath:
                    continue
                existing_write = self._write_locks.get(rpath)
                if existing_write and existing_write.owner_task_id != task_id:
                    return False, (
                        f"[{FILE_OWNERSHIP_VIOLATION}] '{rpath}' dosyası '{existing_write.owner_task_id}' "
                        f"tarafından aktif olarak yazılıyor. Okuma kilidi alınamadı."
                    )

            if task_id not in self._task_write_paths:
                self._task_write_paths[task_id] = set()
            if task_id not in self._task_read_paths:
                self._task_read_paths[task_id] = set()

            now = time.time()
            for wpath in norm_write:
                if "*" in wpath or "?" in wpath:
                    self._task_write_paths[task_id].add(wpath)
                    continue
                lock = FileLock(
                    path=wpath,
                    owner_task_id=task_id,
                    owner_agent_id=agent_id,
                    lock_type="exclusive_write",
                    acquired_at=now,
                )
                self._write_locks[wpath] = lock
                self._task_write_paths[task_id].add(wpath)

            for rpath in norm_read:
                if "*" in rpath or "?" in rpath:
                    self._task_read_paths[task_id].add(rpath)
                    continue
                if rpath not in self._read_locks:
                    self._read_locks[rpath] = set()
                self._read_locks[rpath].add(task_id)
                self._task_read_paths[task_id].add(rpath)

            return True, None

    def release_locks(self, task_id: str) -> None:
        """Release locks."""
        with self._lock:
            # Yazma kilitlerini temizle
            write_paths = self._task_write_paths.pop(task_id, set())
            for wpath in write_paths:
                curr = self._write_locks.get(wpath)
                if curr and curr.owner_task_id == task_id:
                    del self._write_locks[wpath]

            # Okuma kilitlerini temizle
            read_paths = self._task_read_paths.pop(task_id, set())
            for rpath in read_paths:
                readers = self._read_locks.get(rpath)
                if readers:
                    readers.discard(task_id)
                    if not readers:
                        del self._read_locks[rpath]

    def verify_write_permission(self, task_id: Optional[str], file_path: str) -> Tuple[bool, Optional[str]]:
        """Verify write permission."""
        if not task_id:
            return True, None

        try:
            norm = self._normalize_path(file_path)
        except PermissionError as exc:
            return False, str(exc)

        with self._lock:
            curr = self._write_locks.get(norm)
            if curr:
                if curr.owner_task_id != task_id:
                    return False, (
                        f"[{FILE_OWNERSHIP_VIOLATION}] '{file_path}' dosyası '{curr.owner_task_id}' "
                        f"görevi tarafından kilitlenmiştir. '{task_id}' yazamaz."
                    )
                return True, None

            assigned = self._task_write_paths.get(task_id, set())
            matches_pattern = False
            for pat in assigned:
                if pat == "*" or pat == "**":
                    matches_pattern = True
                    break
                clean_pat = pat.rstrip("/*")
                if fnmatch.fnmatch(norm, pat) or norm.startswith(clean_pat + "/"):
                    matches_pattern = True
                    break

            if matches_pattern:
                readers = self._read_locks.get(norm, set())
                active_readers = [r for r in readers if r != task_id]
                if active_readers:
                    return False, (
                        f"[{FILE_OWNERSHIP_VIOLATION}] '{file_path}' dosyası şu anda başka görevler "
                        f"({active_readers}) tarafından okunuyor. Yazma kilidi bekletilmeli."
                    )
                now = time.time()
                lock = FileLock(
                    path=norm,
                    owner_task_id=task_id,
                    owner_agent_id="",
                    lock_type="exclusive_write",
                    acquired_at=now,
                )
                self._write_locks[norm] = lock
                self._task_write_paths[task_id].add(norm)
                return True, None

            if norm not in assigned:
                return False, (
                    f"[{FILE_OWNERSHIP_VIOLATION}] '{file_path}' dosyası görev '{task_id}' için "
                    "yazma kapsamına atanmamış veya kilitlenmemiştir."
                )

            return True, None

    def get_active_locks(self) -> List[FileLock]:
        """Get active locks."""
        with self._lock:
            return list(self._write_locks.values())

    def reset(self) -> None:
        """Reset."""
        with self._lock:
            self._write_locks.clear()
            self._read_locks.clear()
            self._task_write_paths.clear()
            self._task_read_paths.clear()
