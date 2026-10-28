"""The whole walk -> signature path, shared by the API and the scripts."""

import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.config import Settings
from app.ml.silhouettes import PREPROCESSORS, central_window, check_enough, pick_display


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

    def analyse(self, video_path: Path) -> WalkAnalysis:
        with self._lock:
            silhouettes: list[np.ndarray] = []
            frames_total = 0
            with closing(iter(self.segmenter.masks(video_path))) as masks:
                for mask in masks:
                    frames_total += 1
                    sil = self.to_silhouette(mask) if mask is not None else None
                    if sil is not None:
                        silhouettes.append(sil)
                    # Long uploads: stop reading; a walk only needs a few seconds.
                    if frames_total >= self.settings.max_read_frames:
                        break

            check_enough(len(silhouettes), self.settings.min_frames)
            used = central_window(silhouettes, self.settings.max_frames)
            vector = self.embedder.embed(np.stack(used))

        return WalkAnalysis(
            vector=vector,
            frames_total=frames_total,
            frames_with_person=len(silhouettes),
            frames_used=len(used),
            display=pick_display(used, self.settings.display_frames),
        )
