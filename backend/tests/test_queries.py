import numpy as np

from app.queries import MAX_QUERIES, CachedQuery, QueryCache


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def q():
    return CachedQuery(vector=np.zeros(4), display=[], clip_name="c.mp4")


def test_expiry():
    clock = Clock()
    cache = QueryCache(ttl_seconds=60, clock=clock)
    qid = cache.put(q())
    clock.now = 59
    assert cache.get(qid) is not None
    clock.now = 61
    assert cache.get(qid) is None


def test_bounded_size_drops_oldest():
    cache = QueryCache(ttl_seconds=60, clock=Clock())
    first = cache.put(q())
    for _ in range(MAX_QUERIES):
        cache.put(q())
    assert cache.get(first) is None
