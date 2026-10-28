"""Query vector vs. the Archive: Euclidean distance, smaller = more alike."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Suspect:
    index: int
    distance: float
    # Relative strength among the returned suspects only (they sum to 100), not a probability:
    # a stranger still gets a high number here. isMatch must use the distance (HANDOFF §6.4).
    confidence: float


def rank(query: np.ndarray, features: np.ndarray, top_k: int) -> list[Suspect]:
    if features.shape[0] == 0:
        return []
    distances = np.linalg.norm(features - query[None, :], axis=1)
    order = np.argsort(distances, kind="stable")[:top_k]
    top = distances[order]
    inverse = 1.0 / (top + 1e-6)
    confidence = inverse / inverse.sum() * 100.0
    return [Suspect(int(i), float(d), float(c)) for i, d, c in zip(order, top, confidence, strict=True)]


def is_match(suspects: list[Suspect], threshold: float) -> bool:
    return bool(suspects) and suspects[0].distance <= threshold
