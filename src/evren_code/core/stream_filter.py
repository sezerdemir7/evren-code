"""Live SSE Stream Tag Filter (`StreamTagFilter`).

Open-weight models on EVREN (`auto`, `qwen`, `glm`, `deepseek`) may embed `<tool_call>...</tool_call>`
or `<think>...</think>` tags directly into streamed content. This state machine:
1. Strips `<tool_call>...</tool_call>` blocks from raw text streams to prevent raw JSON leakage.
2. Directs `<think>...</think>` tokens in real time to the dedicated `on_thinking` channel.
3. Buffers boundary chunks (`<tool_` + `call>`) to cleanly resolve split tokens.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable


class _StreamState(str, Enum):
    NORMAL = "normal"
    IN_THINK = "in_think"
    IN_TOOL = "in_tool"


_OPEN_THINK = "<think>"
_CLOSE_THINK = "</think>"
_OPEN_TOOL = "<tool_call>"
_CLOSE_TOOL = "</tool_call>"


class StreamTagFilter:
    """Filters SSE content chunks and routes them to appropriate display callbacks.
    
    Includes a repetition guard mechanism to halt degenerately looping models.
    """

    def __init__(
        self,
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
        on_tool_preparing: Optional[Callable[[str, int], None]] = None,
        repetition_guard: bool = True,
        repetition_threshold: int = 3,
    ) -> None:
        self._on_text = on_text
        self._on_thinking = on_thinking
        self._on_tool_preparing = on_tool_preparing
        self._repetition_guard = repetition_guard
        self._repetition_threshold = max(2, repetition_threshold)
        self._state: _StreamState = _StreamState.NORMAL
        self._buffer: str = ""
        self.repetition_detected: bool = False
        self.repeated_phrase: str = ""
        self._emitted_history: str = ""
        self._thinking_history: str = ""
        self._tool_bytes_count: int = 0
        self._detected_tool_name: str = ""

    def _emit_text(self, text: str) -> None:
        """Emit text."""
        if not text or self.repetition_detected:
            return
        self._emitted_history += text
        if self._repetition_guard:
            rep_found, rep_block = self._check_repetition(self._emitted_history)
            if rep_found:
                self.repetition_detected = True
                self.repeated_phrase = rep_block
                return
        self._on_text(text)

    def feed_thinking(self, text: str) -> None:
        """Feed thinking."""
        if not text or self.repetition_detected:
            return
        self._thinking_history += text
        if self._repetition_guard:
            rep_found, rep_block = self._check_repetition(self._thinking_history)
            if rep_found:
                self.repetition_detected = True
                self.repeated_phrase = rep_block
                return
        self._on_thinking(text)

    def _check_repetition(
        self, text: str, min_len: int = 20, max_len: int = 300
    ) -> tuple[bool, str]:
        """Check repetition."""
        n = len(text)
        eff_threshold = max(3, self._repetition_threshold)
        limit = min(max_len, n // eff_threshold)
        if limit < min_len:
            return False, ""
        for length in range(min_len, limit + 1):
            block = text[-length:]
            if not block.strip():
                continue
            if text.endswith(block * eff_threshold):
                return True, block
        return False, ""

    def feed(self, chunk: str) -> None:
        """Feed (delta)."""
        if not chunk or self.repetition_detected:
            return
        self._buffer += chunk
        self._process_buffer(final=False)

    def flush(self) -> None:
        """Flush."""
        if not self.repetition_detected:
            self._process_buffer(final=True)

    def _process_buffer(self, final: bool) -> None:
        while self._buffer and not self.repetition_detected:
            if self._state == _StreamState.NORMAL:
                idx_think = self._buffer.find(_OPEN_THINK)
                idx_tool = self._buffer.find(_OPEN_TOOL)

                candidates = [
                    (idx_think, _OPEN_THINK, _StreamState.IN_THINK),
                    (idx_tool, _OPEN_TOOL, _StreamState.IN_TOOL),
                ]
                valid = [(pos, tag, st) for pos, tag, st in candidates if pos != -1]
                if valid:
                    pos, tag, next_state = min(valid, key=lambda item: item[0])
                    if pos > 0:
                        self._emit_text(self._buffer[:pos])
                    self._buffer = self._buffer[pos + len(tag) :]
                    self._state = next_state
                    if next_state == _StreamState.IN_TOOL:
                        self._tool_bytes_count = 0
                        self._detected_tool_name = ""
                    continue

                if not final:
                    safe_len = self._find_safe_prefix_len(
                        self._buffer, (_OPEN_THINK, _OPEN_TOOL)
                    )
                    if safe_len > 0:
                        self._emit_text(self._buffer[:safe_len])
                        self._buffer = self._buffer[safe_len:]
                    return
                else:
                    self._emit_text(self._buffer)
                    self._buffer = ""
                    return

            elif self._state == _StreamState.IN_THINK:
                idx_close = self._buffer.find(_CLOSE_THINK)
                if idx_close != -1:
                    if idx_close > 0:
                        self.feed_thinking(self._buffer[:idx_close])
                    self._buffer = self._buffer[idx_close + len(_CLOSE_THINK) :]
                    self._state = _StreamState.NORMAL
                    continue

                if not final:
                    safe_len = self._find_safe_prefix_len(self._buffer, (_CLOSE_THINK,))
                    if safe_len > 0:
                        self.feed_thinking(self._buffer[:safe_len])
                        self._buffer = self._buffer[safe_len:]
                    return
                else:
                    self.feed_thinking(self._buffer)
                    self._buffer = ""
                    return

            elif self._state == _StreamState.IN_TOOL:
                idx_close = self._buffer.find(_CLOSE_TOOL)
                if not self._detected_tool_name:
                    import re
                    m = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', self._buffer)
                    if m:
                        self._detected_tool_name = m.group(1)

                content_len = idx_close if idx_close != -1 else len(self._buffer)
                self._tool_bytes_count = max(self._tool_bytes_count, content_len)
                if self._on_tool_preparing:
                    self._on_tool_preparing(self._detected_tool_name, self._tool_bytes_count)
                if idx_close != -1:
                    self._buffer = self._buffer[idx_close + len(_CLOSE_TOOL) :]
                    self._state = _StreamState.NORMAL
                    self._tool_bytes_count = 0
                    self._detected_tool_name = ""
                    continue

                if not final:
                    keep_tail = len(_CLOSE_TOOL)
                    if len(self._buffer) > keep_tail:
                        self._buffer = self._buffer[-keep_tail:]
                    return
                else:
                    self._buffer = ""
                    return

    @staticmethod
    def _find_safe_prefix_len(text: str, tags: tuple[str, ...]) -> int:
        """Find safe prefix len."""
        max_tag_len = max(len(t) for t in tags)
        search_start = max(0, len(text) - max_tag_len + 1)
        for i in range(search_start, len(text)):
            suffix = text[i:]
            if any(tag.startswith(suffix) for tag in tags):
                return i
        return len(text)
