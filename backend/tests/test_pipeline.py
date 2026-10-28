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
    from app.ml.silhouettes import casia_silhouette, crop_silhouette, crop_to_person
    from tests.conftest import person_mask

    full = person_mask(h=480, w=640, top=100, height=250, width=60, left=300)
    crop = crop_to_person(full)
    assert crop.shape[0] < full.shape[0] and crop.shape[1] < full.shape[1]
    np.testing.assert_array_equal(casia_silhouette(crop), casia_silhouette(full))
    np.testing.assert_array_equal(crop_silhouette(crop), crop_silhouette(full))
