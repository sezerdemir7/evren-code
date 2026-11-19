"""File Modification Undo and Snapshot Manager (`undo.py`).

Records snapshots of files in a LIFO stack before write/edit operations
(`write_file`, `write_multiple_files`, `edit_file`, `multi_edit_file`).
Allows instant rollback when the user executes `/undo`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import List, Optional


@dataclass
class FileSnapshot:
    """Snapshot of file state before modification."""

    file_path: Path
    relative_display: str
    existed_before: bool
    previous_content: Optional[str]
    tool_name: str


class UndoManager:
    """Manages file modification history in a thread-safe LIFO stack."""

    def __init__(
        self,
        max_history: int = 30,
    ) -> None:
        self.max_history = max_history
        self._stack: List[FileSnapshot] = []
        self._lock = threading.Lock()

    def record_before_write(
        self,
        file_path: Path,
        relative_display: str,
        tool_name: str,
    ) -> None:
        """Record before write."""
        existed = file_path.is_file()
        prev_content: Optional[str] = None
        if existed:
            try:
                prev_content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                prev_content = None

        snapshot = FileSnapshot(
            file_path=file_path,
            relative_display=relative_display,
            existed_before=existed,
            previous_content=prev_content,
            tool_name=tool_name,
        )
        with self._lock:
            self._stack.append(snapshot)
            if len(self._stack) > self.max_history:
                self._stack.pop(0)

    @property
    def can_undo(self) -> bool:
        with self._lock:
            return len(self._stack) > 0

    @property
    def history_count(self) -> int:
        with self._lock:
            return len(self._stack)

    def undo_last(self) -> str:
        """Undo last."""
        with self._lock:
            if not self._stack:
                return "Geri alınacak herhangi bir dosya değişikliği bulunmuyor."
            snap = self._stack.pop()
        try:
            if not snap.existed_before:
                if snap.file_path.is_file():
                    snap.file_path.unlink()
                msg = (
                    f"✔ Geri alındı ({snap.tool_name}): Yeni oluşturulan "
                    f"'{snap.relative_display}' dosyası silindi."
                )
            else:
                snap.file_path.parent.mkdir(parents=True, exist_ok=True)
                snap.file_path.write_text(snap.previous_content or "", encoding="utf-8")
                msg = (
                    f"✔ Geri alındı ({snap.tool_name}): "
                    f"'{snap.relative_display}' önceki haline döndürüldü."
                )
            return msg
        except OSError as exc:
            return f"Hata: '{snap.relative_display}' geri alınırken sorun oluştu: {exc}"

    def revert_turn(self, shadow_git: Optional[Any] = None) -> str:
        """Revert turn (Shadow Git Snapshot)."""
        if shadow_git and hasattr(shadow_git, "revert_to_checkpoint"):
            ok = shadow_git.revert_to_checkpoint()
            if ok:
                with self._lock:
                    self._stack.clear()
                return "✔ [Gölge Git Kontrol Noktası]: Tüm tur değişiklikleri atomik olarak geri sarıldı."
        messages = []
        while self.can_undo:
            messages.append(self.undo_last())
        return "\n".join(messages) if messages else "Geri alınacak değişiklik yok."

