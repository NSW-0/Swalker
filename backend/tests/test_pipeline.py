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
