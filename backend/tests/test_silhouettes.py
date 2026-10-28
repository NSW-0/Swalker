import itertools

import cv2
import numpy as np
import pytest

from app.errors import NoPersonError, TooFewFramesError
from app.ml.silhouettes import (
    CASIA_BBOX_TOP,
    CASIA_PERSON_HEIGHT,
    FrameRateSampler,
    WalkerTracker,
    casia_silhouette,
    central_window,
    check_enough,
    crop_silhouette,
    pick_display,
)
from tests.conftest import person_mask


def _rows(sil, level=0.05):
    return np.flatnonzero(sil.max(axis=1) > level)


@pytest.mark.parametrize("frame", [(240, 320), (1080, 1920), (720, 1280)])
@pytest.mark.parametrize("person_h", [0.3, 0.9])
def test_casia_framing_matches_training_geometry(frame, person_h):
    h, w = frame
    height = int(h * person_h)
    mask = person_mask(h=h, w=w, top=(h - height) // 3, height=height, width=height // 4, left=w // 3)
    sil = casia_silhouette(mask)

    assert sil.shape == (64, 44) and sil.dtype == np.float32
    assert sil.min() >= 0.0 and sil.max() <= 1.0
    rows = _rows(sil)
    # Same place and size in the frame however big the person was in the video.
    assert rows[0] / 64 == pytest.approx(CASIA_BBOX_TOP, abs=2 / 64)
    assert (rows[-1] + 1 - rows[0]) / 64 == pytest.approx(CASIA_PERSON_HEIGHT, abs=2 / 64)


def test_casia_keeps_soft_greyscale_edges_and_is_narrow():
    # A body-like outline (ellipse), since real masks don't have pixel-aligned edges.
    mask = np.zeros((480, 640), np.uint8)
    cv2.ellipse(mask, (320, 240), (35, 170), 0, 0, 360, 255, -1)
    sil = casia_silhouette(mask)
    assert ((sil > 0.05) & (sil < 0.95)).any()  # not thresholded
    cols = np.flatnonzero(sil.max(axis=0) > 0.05)
    assert cols.size < 15  # squashed 4:3 frame: CASIA-B people are only a few pixels wide


def test_crop_mode_fills_frame_with_hard_values():
    sil = crop_silhouette(person_mask())
    assert sil.shape == (64, 44)
    assert set(np.unique(sil)) <= {0.0, 1.0}
    assert _rows(sil).size > 55


def test_empty_mask():
    empty = np.zeros((240, 320), np.uint8)
    assert casia_silhouette(empty) is None
    assert crop_silhouette(empty) is None


def test_central_window_is_consecutive_and_in_order():
    frames = list(range(200))
    assert central_window(frames, 120) == list(range(40, 160))
    assert central_window(frames[:50], 120) == frames[:50]


def test_pick_display_spreads_across_clip():
    picked = pick_display(list(range(100)), 10)
    assert len(picked) == 10 and picked[0] == 0 and picked[-1] == 99
    assert pick_display([1, 2], 10) == [1, 2]


def test_check_enough():
    with pytest.raises(NoPersonError):
        check_enough(0, 15)
    with pytest.raises(TooFewFramesError):
        check_enough(14, 15)
    check_enough(15, 15)


def test_tracker_follows_the_walker_not_the_most_confident_person():
    walker = np.array([100, 50, 140, 250], float)
    bystander = np.array([300, 80, 330, 200], float)
    tracker = WalkerTracker()
    # First frame: the bigger person is the walker.
    assert tracker.pick(np.stack([bystander, walker]), np.array([0.9, 0.8])) == 1
    # Later the bystander is detected more confidently, but the walker (moved a little) is kept.
    moved = walker + np.array([4, 0, 4, 0])
    assert tracker.pick(np.stack([bystander, moved]), np.array([0.99, 0.6])) == 1


def test_tracker_skips_frames_without_the_walker_then_reacquires():
    tracker = WalkerTracker()
    tracker.pick(np.array([[100, 50, 140, 250]], float), np.array([0.9]))
    elsewhere = np.array([[400, 50, 440, 250]], float)
    for _ in range(WalkerTracker.MAX_MISSES):
        assert tracker.pick(elsewhere, np.array([0.9])) is None
    assert tracker.pick(elsewhere, np.array([0.9])) == 0


def _kept(source_fps, target_fps, n):
    sampler = FrameRateSampler(source_fps, target_fps)
    return [i for i in range(n) if sampler.keep(i)]


def test_high_frame_rate_clips_are_resampled_to_about_25_fps():
    assert len(_kept(60, 25, 600)) == 250  # 10 s at 60 fps -> 10 s at 25 fps
    assert len(_kept(30, 25, 300)) == 250
    kept = _kept(60, 25, 60)
    assert kept[0] == 0 and all(b > a for a, b in itertools.pairwise(kept))  # in order


def test_clips_at_or_below_target_and_unknown_rates_keep_every_frame():
    assert _kept(25, 25, 100) == list(range(100))
    assert _kept(15, 25, 100) == list(range(100))
    assert _kept(0, 25, 100) == list(range(100))  # OpenCV couldn't read the frame rate


def test_casia_applies_the_0_52_horizontal_squash():
    # A person as wide as they are tall: height -> 50.1% of 64 = 32 px, width -> 32 * 0.52 = 16.6 px.
    sil = casia_silhouette(person_mask(h=480, w=640, top=100, height=300, width=300, left=150))
    mass_width = sil.sum(axis=1).max()  # widest row, counting partial edge pixels
    assert mass_width == pytest.approx(0.501 * 64 * 0.52, abs=1.0)
