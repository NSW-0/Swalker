import numpy as np
import pytest

from scripts.casia_eval import VIEWS, fmr_table, genuine_impostor, rank1


def fake_casia(noise, n_subjects=4, seed=0):
    """One base vector per subject; every sequence is that vector plus noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_subjects, 16))
    feats, meta = [], []
    for sid in range(n_subjects):
        for cond, seqs in (("nm", range(1, 7)), ("bg", (1, 2)), ("cl", (1, 2))):
            for seq in seqs:
                for view in VIEWS[:3]:
                    feats.append(base[sid] + rng.normal(scale=noise, size=16))
                    meta.append({"sid": sid, "cond": cond, "seq": seq, "view": view})
    return np.stack(feats), meta


def test_rank1_is_perfect_for_separable_and_chance_for_noise():
    feats, meta = fake_casia(noise=0.01)
    mean, per_view = rank1(feats, meta, ("nm", (5, 6)))
    assert mean == pytest.approx(1.0) and set(per_view) == set(VIEWS[:3])

    feats, meta = fake_casia(noise=100.0, n_subjects=20)
    mean, _ = rank1(feats, meta, ("cl", (1, 2)))
    assert mean < 0.3


def test_genuine_pairs_are_same_subject_same_view_only():
    feats, meta = fake_casia(noise=0.01, n_subjects=3)
    genuine, impostor = genuine_impostor(feats, meta, VIEWS[:3])
    # per view: 3 subjects x 2 probes x 4 gallery seqs genuine; 3x2 x 8 impostor
    assert genuine.size == 3 * 24 and impostor.size == 3 * 48
    assert genuine.max() < impostor.min()


def test_fmr_table_never_exceeds_the_target_rate():
    rng = np.random.default_rng(1)
    genuine, impostor = rng.normal(10, 2, 500), rng.normal(30, 4, 5000)
    for row in fmr_table(genuine, impostor):
        assert row["fmr"] <= row["target_fmr"] + 1e-9
    assert fmr_table(genuine, impostor)[-1]["genuine_accepted"] > 0.99
