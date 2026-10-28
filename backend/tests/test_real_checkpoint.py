"""Runs only where the real weights are present (backend/models/), not in CI."""

import numpy as np
import pytest

from app.config import get_settings

CHECKPOINT = get_settings().checkpoint_path
pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason=f"{CHECKPOINT.name} not present")


@pytest.fixture(scope="module")
def embedder():
    import torch

    from app.ml.embed import GaitEmbedder

    return GaitEmbedder(CHECKPOINT, torch.device("cpu"))


def test_checkpoint_loads_strictly_in_eval_mode(embedder):
    assert not embedder.model.training
    assert not any(m.training for m in embedder.model.modules())


def test_signature_is_4096d_and_deterministic(embedder):
    seq = np.random.default_rng(0).random((60, 64, 44)).astype(np.float32)
    a, b = embedder.embed(seq), embedder.embed(seq)
    assert a.shape == (4096,) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)


def test_variable_length_sequences(embedder):
    rng = np.random.default_rng(1)
    for t in (15, 30, 120):
        assert embedder.embed(rng.random((t, 64, 44)).astype(np.float32)).shape == (4096,)
