from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from .config import load_settings


@dataclass
class Turn:
    tool: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass
class Thread:
    id: str
    tool: str
    created_at: float
    updated_at: float
    turns: list[Turn] = field(default_factory=list)


class ConversationStore:
    def __init__(self) -> None:
        self._threads: dict[str, Thread] = {}
        self._lock = Lock()

    def create_or_get(self, tool: str, continuation_id: str | None = None) -> Thread:
        with self._lock:
            self._purge_expired_locked()
            if continuation_id and continuation_id in self._threads:
                return self._threads[continuation_id]

            thread_id = str(uuid.uuid4())
            now = time.time()
            thread = Thread(id=thread_id, tool=tool, created_at=now, updated_at=now)
            self._threads[thread_id] = thread
            return thread

    def add_turn(self, thread_id: str, tool: str, payload: dict[str, Any]) -> None:
        with self._lock:
            thread = self._threads.get(thread_id)
            if not thread:
                return
            thread.turns.append(Turn(tool=tool, payload=payload))
            thread.updated_at = time.time()

    def context(self, thread_id: str | None) -> str:
        if not thread_id:
            return ""
        with self._lock:
            self._purge_expired_locked()
            thread = self._threads.get(thread_id)
            if not thread:
                return ""
            max_chars = load_settings().max_context_chars
            chunks: list[str] = []
            total = 0
            for turn in reversed(thread.turns):
                text = f"[{turn.tool}] {turn.payload}"
                if total + len(text) > max_chars:
                    break
                chunks.append(text)
                total += len(text)
            return "\n\n".join(reversed(chunks))

    def _purge_expired_locked(self) -> None:
        ttl = load_settings().conversation_ttl_seconds
        cutoff = time.time() - ttl
        expired = [
            thread_id for thread_id, thread in self._threads.items() if thread.updated_at < cutoff
        ]
        for thread_id in expired:
            del self._threads[thread_id]


store = ConversationStore()
