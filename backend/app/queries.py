"""Recent identify results kept server-side, so "Add to the Archive" can save a walker by
queryId without uploading the video again."""

import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

MAX_QUERIES = 200


@dataclass
class CachedQuery:
    vector: np.ndarray
    display: list[np.ndarray]
    clip_name: str
    expires: float = field(default=0.0)


class QueryCache:
    def __init__(self, ttl_seconds: float, clock=time.monotonic):
        self.ttl = ttl_seconds
        self.clock = clock
        self._items: dict[str, CachedQuery] = {}
        self._lock = threading.Lock()

    def put(self, query: CachedQuery) -> str:
        with self._lock:
            self._prune()
            while len(self._items) >= MAX_QUERIES:
                self._items.pop(next(iter(self._items)))  # oldest first (dicts keep insertion order)
            query_id = uuid.uuid4().hex
            query.expires = self.clock() + self.ttl
            self._items[query_id] = query
            return query_id

    def get(self, query_id: str) -> CachedQuery | None:
        with self._lock:
            self._prune()
            return self._items.get(query_id)

    def pop(self, query_id: str) -> CachedQuery | None:
        with self._lock:
            self._prune()
            return self._items.pop(query_id, None)

    def _prune(self) -> None:
        now = self.clock()
        for key in [k for k, q in self._items.items() if q.expires <= now]:
            del self._items[key]
