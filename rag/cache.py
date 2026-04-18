from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    result: dict[str, Any]


class SemanticLRUCache:
    def __init__(self, capacity: int = 50, similarity_threshold: float = 0.92):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def get(self, query_embedding: np.ndarray) -> dict[str, Any] | None:
        best_key: str | None = None
        best_score = -1.0

        for key, entry in self._store.items():
            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is None or best_score < self.similarity_threshold:
            return None

        entry = self._store.pop(best_key)
        self._store[best_key] = entry
        result = dict(entry.result)
        result["cache_similarity"] = best_score
        result["cache_hit"] = True
        return result

    def put(self, query: str, query_embedding: np.ndarray, result: dict[str, Any]) -> None:
        if query in self._store:
            self._store.pop(query)

        entry = CacheEntry(query=query, embedding=query_embedding.copy(), result=dict(result))
        self._store[query] = entry

        while len(self._store) > self.capacity:
            self._store.popitem(last=False)
