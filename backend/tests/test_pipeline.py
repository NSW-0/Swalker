import numpy as np

from tests.conftest import FakeEmbedder, FakePipeline, FakeSegmenter


def test_long_videos_stop_at_max_read_frames(settings, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"walk")
    settings.max_read_frames = 20
    settings.min_frames = 10
    walk = FakePipeline(FakeSegmenter(), FakeEmbedder(), settings).analyse(clip)
    assert walk.frames_total == 20 and walk.frames_with_person == 20


def test_frames_used_are_capped_and_display_is_small(settings, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"walk")
    walk = FakePipeline(FakeSegmenter(), FakeEmbedder(), settings).analyse(clip)
    assert walk.frames_total == 40 and walk.frames_used == settings.max_frames == 30
    assert len(walk.display) == settings.display_frames


def test_progress_is_reported_in_stage_order(settings, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"walk")
    seen = []
    FakePipeline(FakeSegmenter(), FakeEmbedder(), settings).analyse(clip, report=lambda s, f: seen.append((s, f)))
    stages = [s for s, _ in seen]
    assert stages[0] == "detecting" and stages[-2:] == ["silhouettes", "embedding"]
    detecting = [f for s, f in seen if s == "detecting"]
    assert detecting == sorted(detecting)


def test_crop_to_person_gives_the_same_silhouette_as_the_full_frame():
    from app.ml.silhouettes import PersonCrop, casia_sequence, casia_silhouette, crop_silhouette, crop_to_person
    from tests.conftest import person_mask

    full = [person_mask(h=480, w=640, top=100 + 5 * i, height=250, width=60, left=150 + 40 * i) for i in range(3)]
    crops = [crop_to_person(m) for m in full]
    assert crops[0].mask.shape[0] < 480 and crops[0].mask.shape[1] < 640
    np.testing.assert_array_equal(casia_silhouette(crops[0].mask), casia_silhouette(full[0]))
    np.testing.assert_array_equal(crop_silhouette(crops[0].mask), crop_silhouette(full[0]))
    # The clip-level framing needs where each crop was: same result as from the whole frames.
    whole = [PersonCrop(m, 0, 0) for m in full]
    for follow_x in (False, True):
        for a, b in zip(casia_sequence(crops, follow_x), casia_sequence(whole, follow_x), strict=True):
            np.testing.assert_array_equal(a, b)
