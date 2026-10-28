import numpy as np
import pytest

from app.gallery.matching import is_match, rank


def test_rank_orders_by_distance_and_keeps_top_k():
    features = np.array([[10.0, 0], [1.0, 0], [3.0, 0], [0.5, 0]], np.float32)
    suspects = rank(np.zeros(2, np.float32), features, top_k=3)
    assert [s.index for s in suspects] == [3, 1, 2]
    assert [s.distance for s in suspects] == pytest.approx([0.5, 1.0, 3.0])


def test_confidence_is_relative_and_sums_to_100():
    features = np.array([[1.0, 0], [2.0, 0], [4.0, 0]], np.float32)
    suspects = rank(np.zeros(2, np.float32), features, top_k=3)
    assert sum(s.confidence for s in suspects) == pytest.approx(100)
    assert suspects[0].confidence > suspects[1].confidence > suspects[2].confidence


def test_single_entry_and_empty_gallery():
    one = rank(np.zeros(2, np.float32), np.ones((1, 2), np.float32), top_k=3)
    assert len(one) == 1 and one[0].confidence == pytest.approx(100)
    assert rank(np.zeros(2, np.float32), np.zeros((0, 2), np.float32), top_k=3) == []


def test_is_match_uses_distance_not_confidence():
    far = rank(np.zeros(2, np.float32), np.array([[50.0, 0]], np.float32), top_k=3)
    assert far[0].confidence == pytest.approx(100)  # a stranger still gets "100%"
    assert not is_match(far, threshold=30)
    assert is_match(far, threshold=50)  # the boundary counts as a match
    assert not is_match([], threshold=30)
