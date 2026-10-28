"""The whole walk -> signature path, shared by the API and the scripts."""

import threading
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.config import Settings
from app.ml.silhouettes import PREPROCESSORS, central_window, check_enough, crop_to_person, pick_display

# report(stage, fraction of that stage done); stages: detecting, silhouettes, embedding.
Progress = Callable[[str, float], None]
REPORT_EVERY = 5  # frames


def _ignore(stage: str, fraction: float) -> None:
    pass


class Segmenter(Protocol):
    def masks(self, video_path: Path): ...


class Embedder(Protocol):
    checkpoint_name: str

    def embed(self, sequence: np.ndarray) -> np.ndarray: ...


@dataclass
class WalkAnalysis:
    vector: np.ndarray  # (4096,) float32
    frames_total: int
    frames_with_person: int
    frames_used: int
    display: list[np.ndarray]  # a few (64, 44) silhouettes for the UI


class GaitPipeline:
    def __init__(self, segmenter: Segmenter, embedder: Embedder, settings: Settings):
        self.segmenter = segmenter
        self.embedder = embedder
        self.settings = settings
        self.to_silhouette = PREPROCESSORS[settings.preprocess_mode]
        # One video at a time: YOLO and GLTNet share the GPU, and CPU runs are already slow.
        self._lock = threading.Lock()

    @property
    def checkpoint_name(self) -> str:
        return self.embedder.checkpoint_name

    def analyse(self, video_path: Path, report: Progress = _ignore) -> WalkAnalysis:
        max_read = self.settings.max_read_frames
        estimate = getattr(self.segmenter, "estimate_frames", lambda _: None)(video_path)
        expected = min(max_read, estimate) if estimate else max_read

        with self._lock:
            # 1. Find the walker in every frame. Only a small crop around them is kept.
            report("detecting", 0.0)
            crops: list[np.ndarray] = []
            frames_total = 0
            with closing(iter(self.segmenter.masks(video_path))) as masks:
                for mask in masks:
                    frames_total += 1
                    crop = crop_to_person(mask) if mask is not None else None
                    if crop is not None:
                        crops.append(crop)
                    if frames_total % REPORT_EVERY == 0:
                        report("detecting", frames_total / expected)
                    # Long uploads: stop reading; a walk only needs a few seconds.
                    if frames_total >= max_read:
                        break

            # 2. Turn each crop into a 64x44 silhouette.
            report("silhouettes", 0.0)
            silhouettes = [s for s in (self.to_silhouette(c) for c in crops) if s is not None]
            check_enough(len(silhouettes), self.settings.min_frames)
            used = central_window(silhouettes, self.settings.max_frames)

            # 3. Gait signature.
            report("embedding", 0.0)
            vector = self.embedder.embed(np.stack(used))

        return WalkAnalysis(
            vector=vector,
            frames_total=frames_total,
            frames_with_person=len(silhouettes),
            frames_used=len(used),
            display=pick_display(used, self.settings.display_frames),
        )
