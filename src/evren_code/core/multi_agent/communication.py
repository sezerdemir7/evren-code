"""Inter-Agent Message Bus module."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from evren_code.core.multi_agent.models import AgentMessage, MessageType


class AgentMessageBus:
    """Manages message exchange queues between agents and coordinator."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._messages: List[AgentMessage] = []
        self._inbox: Dict[str, List[AgentMessage]] = {}

    def send_message(self, message: AgentMessage) -> None:
        """Send message."""
        with self._lock:
            self._messages.append(message)
            recip = message.recipient_id

            if recip == "broadcast":
                for inbox_list in self._inbox.values():
                    inbox_list.append(message)
            else:
                if recip not in self._inbox:
                    self._inbox[recip] = []
                self._inbox[recip].append(message)

    def receive_messages(self, recipient_id: str, clear: bool = True) -> List[AgentMessage]:
        """Receive messages."""
        with self._lock:
            if recipient_id not in self._inbox:
                return []
            msgs = list(self._inbox[recipient_id])
            if clear:
                self._inbox[recipient_id].clear()
            return msgs

    def get_messages_for_task(self, task_id: str) -> List[AgentMessage]:
        """Get messages for task."""
        with self._lock:
            return [m for m in self._messages if m.task_id == task_id]

    def get_messages_by_correlation(self, correlation_id: str) -> List[AgentMessage]:
        """Get messages by correlation (correlation_id)."""
        with self._lock:
            return [m for m in self._messages if m.correlation_id == correlation_id]

    def get_all_messages(self) -> List[AgentMessage]:
        with self._lock:
            return list(self._messages)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
            self._inbox.clear()
