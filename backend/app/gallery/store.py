"""The Archive on disk. What's stored per person: their 4096-d signature (a row of features.npy),
a few small silhouette PNGs for display, and metadata in index.json. Videos are never stored.

    storage/gallery/
      index.json        {"meta": {...}, "entries": [{personId, slug, videoPath, addedAt, silhouetteFiles}]}
      features.npy      (N, 4096) float32, row i belongs to entries[i]
      silhouettes/<slug>/00.png ...
"""

import json
import os
import re
import shutil
import threading
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from app.errors import ApiError
from app.ml.embed import FEATURE_DIM

NAME_MAX = 40
NAME_PATTERN = re.compile(r"^[\w .'\-]+$")  # \w is Unicode-aware, so Arabic names are fine


@dataclass
class GalleryMeta:
    preprocess_mode: str
    checkpoint: str
    feature_dim: int = FEATURE_DIM
    # Frame rate clips were resampled to; 0 = made before resampling existed (all frames kept).
    target_fps: float = 0.0
    # Version of the preprocess_mode's framing (silhouettes.PREPROCESS_VERSIONS); 1 = the first one.
    preprocess_version: int = 1


@dataclass
class GalleryEntry:
    personId: str
    slug: str
    videoPath: str
    addedAt: str
    silhouetteFiles: list[str] = field(default_factory=list)


def validate_name(raw: str) -> str:
    name = " ".join(raw.split())
    if not name:
        raise ApiError("invalid_name", "The case file needs a name.", 422)
    if len(name) > NAME_MAX:
        raise ApiError("invalid_name", f"Names can be at most {NAME_MAX} characters.", 422)
    if not NAME_PATTERN.match(name):
        raise ApiError("invalid_name", "Use letters, numbers, spaces, and . _ - ' only.", 422)
    return name


def _slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")[:30] or "person"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def encode_png(silhouette: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", (np.clip(silhouette, 0, 1) * 255).astype(np.uint8))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


class GalleryStore:
    def __init__(self, root: Path, expected: GalleryMeta):
        self.root = root
        self.sil_dir = root / "silhouettes"
        self.index_path = root / "index.json"
        self.features_path = root / "features.npy"
        self.expected = expected
        self._lock = threading.Lock()

        self.root.mkdir(parents=True, exist_ok=True)
        self.sil_dir.mkdir(exist_ok=True)
        self._load()

    # ── loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.index_path.exists():
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.meta = GalleryMeta(**raw["meta"])
            self.entries = [GalleryEntry(**e) for e in raw["entries"]]
            self.features = (
                np.load(self.features_path).astype(np.float32)
                if self.entries
                else np.zeros((0, FEATURE_DIM), np.float32)
            )
            if self.features.shape != (len(self.entries), self.meta.feature_dim):
                raise RuntimeError(
                    f"{self.features_path} has shape {self.features.shape}, "
                    f"index.json lists {len(self.entries)} entries"
                )
        else:
            self.meta = self.expected
            self.entries = []
            self.features = np.zeros((0, FEATURE_DIM), np.float32)

        # An empty gallery simply adopts the current settings.
        if not self.entries:
            self.meta = self.expected

    @property
    def compatible(self) -> bool:
        """False when the stored vectors were made with other preprocessing or another checkpoint.
        Comparing those against new queries would give meaningless distances."""
        return self.meta == self.expected

    def require_compatible(self) -> None:
        if not self.compatible:
            raise ApiError(
                "gallery_incompatible",
                f"The Archive was built with preprocessing '{self.meta.preprocess_mode}' and "
                f"checkpoint '{self.meta.checkpoint}', but the server now uses "
                f"'{self.expected.preprocess_mode}' / '{self.expected.checkpoint}'. Rebuild the Archive.",
                503,
            )

    # ── reading ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def snapshot(self) -> tuple[list[GalleryEntry], np.ndarray]:
        with self._lock:
            return list(self.entries), self.features

    def find(self, person_id: str) -> tuple[GalleryEntry, np.ndarray] | None:
        with self._lock:
            for i, e in enumerate(self.entries):
                if e.personId == person_id:
                    return e, self.features[i]
        return None

    def name_taken(self, name: str, ignore: str | None = None) -> bool:
        key = name.casefold()
        return any(e.personId.casefold() == key and e.personId != ignore for e in self.entries)

    def _index_of(self, person_id: str) -> int:
        for i, e in enumerate(self.entries):
            if e.personId == person_id:
                return i
        raise ApiError("not_found", f"No case file named '{person_id}'.", 404)

    # ── writing ──────────────────────────────────────────────────────────

    def add(self, name: str, video_name: str, vector: np.ndarray, silhouettes: list[np.ndarray]) -> GalleryEntry:
        name = validate_name(name)
        vector = np.asarray(vector, np.float32).reshape(-1)
        if vector.shape != (self.meta.feature_dim,):
            raise ValueError(f"vector must be ({self.meta.feature_dim},), got {vector.shape}")

        with self._lock:
            self.require_compatible()
            if self.name_taken(name):
                raise ApiError("name_taken", f"A case file named '{name}' already exists.", 409)

            slug = _slugify(name)
            used = {e.slug for e in self.entries}
            base, n = slug, 2
            while slug in used or (self.sil_dir / slug).exists():
                slug, n = f"{base}_{n}", n + 1

            person_dir = self.sil_dir / slug
            person_dir.mkdir()
            files = []
            for i, sil in enumerate(silhouettes):
                fname = f"{i:02d}.png"
                (person_dir / fname).write_bytes(encode_png(sil))
                files.append(f"{slug}/{fname}")

            entry = GalleryEntry(
                personId=name,
                slug=slug,
                videoPath=Path(video_name).name,
                addedAt=datetime.now(UTC).isoformat(timespec="seconds"),
                silhouetteFiles=files,
            )
            features = np.vstack([self.features, vector[None, :]])
            entries = [*self.entries, entry]
            self._persist(entries, features)
            self.entries, self.features = entries, features
            return entry

    def rename(self, person_id: str, new_name: str) -> GalleryEntry:
        """Only the name changes; the vector and the silhouette folder stay as they are."""
        new_name = validate_name(new_name)
        with self._lock:
            i = self._index_of(person_id)
            if self.name_taken(new_name, ignore=person_id):
                raise ApiError("name_taken", f"A case file named '{new_name}' already exists.", 409)
            entries = list(self.entries)
            entries[i] = replace(entries[i], personId=new_name)
            self._persist(entries, self.features)
            self.entries = entries
            return entries[i]

    def remove(self, person_id: str) -> None:
        with self._lock:
            i = self._index_of(person_id)
            gone = self.entries[i]
            entries = self.entries[:i] + self.entries[i + 1 :]
            features = np.delete(self.features, i, axis=0)
            self._persist(entries, features)
            self.entries, self.features = entries, features
        # After the index no longer mentions it, so a failure here only leaves unused PNGs.
        shutil.rmtree(self.sil_dir / gone.slug, ignore_errors=True)

    def _persist(self, entries: list[GalleryEntry], features: np.ndarray) -> None:
        # features first, index second: the index is what says how many rows are valid.
        tmp = self.features_path.with_name("features.tmp.npy")
        np.save(tmp, features.astype(np.float32))
        os.replace(tmp, self.features_path)
        doc = {"meta": asdict(self.meta), "entries": [asdict(e) for e in entries]}
        _atomic_write_bytes(self.index_path, json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"))
