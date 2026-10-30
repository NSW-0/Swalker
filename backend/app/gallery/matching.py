"""Query vector vs. the Archive: Euclidean distance, smaller = more alike."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Suspect:
    index: int  # the person (the group), or the row when rows aren't grouped
    distance: float
    # Relative strength among the returned suspects only (they sum to 100), not a probability:
    # a stranger still gets a high number here. isMatch must use the distance (HANDOFF §6.4).
    confidence: float
    row: int  # the row (clip) that was closest


def rank(query: np.ndarray, features: np.ndarray, top_k: int, groups: list[int] | None = None) -> list[Suspect]:
    """The top_k closest groups. groups[i] says whose row i is (one person can have several clips);
    a person's distance is their closest clip's. Without groups every row is its own."""
    if features.shape[0] == 0:
        return []
    distances = np.linalg.norm(features - query[None, :], axis=1)
    owner = np.arange(len(distances)) if groups is None else np.asarray(groups)
    best: dict[int, int] = {}  # group -> its closest row, in order of distance
    for row in np.argsort(distances, kind="stable"):
        best.setdefault(int(owner[row]), int(row))
        if len(best) == top_k:
            break
    rows = list(best.values())
    top = distances[rows]
    inverse = 1.0 / (top + 1e-6)
    confidence = inverse / inverse.sum() * 100.0
    return [
        Suspect(group, float(d), float(c), row)
        for (group, row), d, c in zip(best.items(), top, confidence, strict=True)
    ]


def is_match(suspects: list[Suspect], threshold: float) -> bool:
    return bool(suspects) and suspects[0].distance <= threshold
