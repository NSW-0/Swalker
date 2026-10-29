import numpy as np
import pytest

from scripts.casia_eval import (
    MODES,
    VIEWS,
    find_sequences,
    fmr_table,
    genuine_impostor,
    matches_baseline,
    native_geometry,
    parse_expected,
    rank1,
    rank1_per_probe,
)


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


def test_per_probe_counts_and_same_view_exclusion():
    feats, meta = fake_casia(noise=0.01, n_subjects=3)
    correct, total = rank1_per_probe(feats, meta, ("cl", (1, 2)), exclude_same_view=True)
    assert (correct, total) == (3 * 2 * 3, 3 * 2 * 3)  # 3 subjects x 2 seqs x 3 views, all right


def test_baseline_check():
    expected = parse_expected("NM=99.27, bg=93.26,CL=72.36")
    assert expected == {"NM": 99.27, "BG": 93.26, "CL": 72.36}
    assert matches_baseline({"NM": 99.27, "BG": 93.18, "CL": 72.36}, expected, tolerance=0.1)
    assert not matches_baseline({"NM": 99.27, "BG": 92.44, "CL": 72.36}, expected, tolerance=0.1)


def _make(root, nested):
    for sid in (75, 76):
        base = root / f"{sid:03d}" / (f"{sid:03d}" if nested else "")
        (base / "nm-01" / "090").mkdir(parents=True)


def test_layout_detection_flat_nested_and_wrapped(tmp_path):
    _make(tmp_path / "flat", nested=False)
    _make(tmp_path / "nested", nested=True)
    _make(tmp_path / "wrapped" / "GaitDatasetB-silh", nested=False)
    seqs, layout = find_sequences(tmp_path / "flat", range(75, 77))
    assert len(seqs) == 2 and layout.startswith("flat")
    seqs, layout = find_sequences(tmp_path / "nested", range(75, 77))
    assert len(seqs) == 2 and layout.startswith("nested")
    seqs, layout = find_sequences(tmp_path / "wrapped", range(75, 77))
    assert len(seqs) == 2 and "GaitDatasetB-silh" in layout
    with pytest.raises(SystemExit):
        find_sequences(tmp_path / "flat", range(90, 92))


def _walk(n=8):
    from tests.conftest import person_mask

    return [
        person_mask(h=240, w=320, top=50 + i, height=118 + 3 * (i % 3), width=30, left=60 + 25 * i) for i in range(n)
    ]


def test_native_geometry_with_nothing_kept_is_casia_seq_x():
    frames = _walk()
    for a, b in zip(MODES["casia_seq_x"](frames), native_geometry(False, False)(frames), strict=True):
        assert np.abs(a - b).mean() < 0.005  # same transform; only edge rounding differs


def test_native_mode_is_close_to_train_style():
    frames = _walk()
    for a, b in zip(MODES["train"](frames), MODES["native"](frames), strict=True):
        assert np.abs(a - b).mean() < 0.01  # whole frame at its own scale: only the drawing differs
