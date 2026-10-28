import hashlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.ml.embed import FEATURE_DIM
from app.ml.pipeline import GaitPipeline


def person_mask(h=240, w=320, top=60, height=120, width=30, left=140) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    mask[top : top + height, left : left + width] = 255
    return mask


class FakeSegmenter:
    """Every frame shows a person, unless the upload's content contains b"empty" (no person)
    or b"short" (person in only 5 frames)."""

    frames = 40

    def masks(self, video_path: Path):
        content = video_path.read_bytes()
        for i in range(self.frames):
            if b"empty" in content or (b"short" in content and i >= 5):
                yield None
            else:
                yield person_mask(left=100 + i)


def vector_for(content: bytes) -> np.ndarray:
    """Same upload -> same vector; different uploads -> vectors far apart."""
    seed = int.from_bytes(hashlib.sha256(content).digest()[:4], "little")
    return np.random.default_rng(seed).normal(size=FEATURE_DIM).astype(np.float32)


class FakeEmbedder:
    checkpoint_name = "fake.pth"
    next_vector = np.zeros(FEATURE_DIM, np.float32)

    def embed(self, sequence: np.ndarray) -> np.ndarray:
        assert sequence.ndim == 3 and sequence.shape[1:] == (64, 44)
        return self.next_vector


class FakePipeline(GaitPipeline):
    def analyse(self, video_path: Path):
        self.embedder.next_vector = vector_for(video_path.read_bytes())
        return super().analyse(video_path)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        storage_dir=tmp_path / "storage",
        checkpoint_path=Path("fake.pth"),
        match_threshold=5.0,
        min_frames=15,
        max_frames=30,
        max_upload_mb=1,
    )


@pytest.fixture
def client(settings):
    pipeline = FakePipeline(FakeSegmenter(), FakeEmbedder(), settings)
    with TestClient(create_app(settings, pipeline=pipeline)) as c:
        yield c


def video(content: bytes = b"walk-a", name: str = "clip.mp4"):
    return {"video": (name, content, "video/mp4")}
