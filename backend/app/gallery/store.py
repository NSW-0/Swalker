"""The Archive on disk. Each person has one or more clips; per clip it stores the 4096-d signature
(a row of features.npy), a few small silhouette PNGs for display, and metadata in index.json.
Videos are never stored.

    storage/gallery/
      index.json        {"version": 2, "meta": {...},
                         "people": [{personId, slug, addedAt,
                                     clips: [{clipId, videoPath, addedAt, silhouetteFiles}]}]}
      features.npy      (clips, 4096) float32, one row per clip, in the order the people and their
                        clips are listed
      silhouettes/<slug>/<clipId>/00.png ...

Version 1 had one clip per person ("entries": [{personId, slug, videoPath, addedAt,
silhouetteFiles}], PNGs in silhouettes/<slug>/); it is read as people with one clip each and
written back as version 2 on the next change.
"""

import json
import os
import re
import shutil
import threading
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from app.errors import ApiError
from app.ml.embed import FEATURE_DIM

NAME_MAX = 40
NAME_PATTERN = re.compile(r"^[\w .'\-]+$")  # \w is Unicode-aware, so Arabic names are fine
INDEX_VERSION = 2


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
class Clip:
    clipId: str
    videoPath: str
    addedAt: str
    silhouetteFiles: list[str] = field(default_factory=list)


@dataclass
class Person:
    personId: str
    slug: str
    addedAt: str
    clips: list[Clip] = field(default_factory=list)


def validate_name(raw: str) -> str:
    name = " ".join(raw.split())
    if not name:
        raise ApiError("invalid_name", "The case file needs a name.", 422)
    if len(name) > NAME_MAX:
        raise ApiError("invalid_name", f"Names can be at most {NAME_MAX} characters.", 422)
    if not NAME_PATTERN.match(name):
        raise ApiError("invalid_name", "Use letters, numbers, spaces, and . _ - ' only.", 422)
    return name


def name_from_filename(filename: str) -> str:
    """A case file name from an upload's file name, for when none was typed: "hamed_bag (1).mp4"
    -> "hamed_bag 1". Characters names can't have become spaces; may be empty."""
    stem = Path(filename).stem
    return " ".join(re.sub(r"[^\w .'\-]+", " ", stem).split())[:NAME_MAX].strip()


def _slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")[:30] or "person"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def encode_png(silhouette: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", (np.clip(silhouette, 0, 1) * 255).astype(np.uint8))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def _read_people(raw: dict) -> list[Person]:
    if "people" in raw:
        return [Person(**{**p, "clips": [Clip(**c) for c in p["clips"]]}) for p in raw["people"]]
    # Version 1: one clip per person.
    return [
        Person(
            personId=e["personId"],
            slug=e["slug"],
            addedAt=e["addedAt"],
            clips=[Clip(uuid.uuid4().hex[:8], e["videoPath"], e["addedAt"], e.get("silhouetteFiles", []))],
        )
        for e in raw["entries"]
    ]


def row_owners(people: list[Person]) -> list[tuple[int, int]]:
    """(person index, clip index) of each row of features, in order."""
    return [(p, c) for p, person in enumerate(people) for c in range(len(person.clips))]


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
            self.people = _read_people(raw)
            rows = sum(len(p.clips) for p in self.people)
            self.features = (
                np.load(self.features_path).astype(np.float32) if rows else np.zeros((0, FEATURE_DIM), np.float32)
            )
            if self.features.shape != (rows, self.meta.feature_dim):
                raise RuntimeError(
                    f"{self.features_path} has shape {self.features.shape}, index.json lists {rows} clips"
                )
        else:
            self.meta = self.expected
            self.people = []
            self.features = np.zeros((0, FEATURE_DIM), np.float32)

        # An empty gallery simply adopts the current settings.
        if not self.people:
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
        """Number of people (not clips)."""
        return len(self.people)

    def snapshot(self) -> tuple[list[Person], np.ndarray]:
        with self._lock:
            return list(self.people), self.features

    def find(self, person_id: str) -> tuple[Person, np.ndarray] | None:
        """The person and their clips' vectors, (clips, 4096) in clip order."""
        with self._lock:
            start = 0
            for p in self.people:
                if p.personId == person_id:
                    return p, self.features[start : start + len(p.clips)]
                start += len(p.clips)
        return None

    def name_taken(self, name: str, ignore: str | None = None) -> bool:
        key = name.casefold()
        return any(p.personId.casefold() == key and p.personId != ignore for p in self.people)

    def _index_of(self, person_id: str) -> int:
        for i, p in enumerate(self.people):
            if p.personId == person_id:
                return i
        raise ApiError("not_found", f"No case file named '{person_id}'.", 404)

    def _first_row(self, person_index: int) -> int:
        return sum(len(p.clips) for p in self.people[:person_index])

    # ── writing ──────────────────────────────────────────────────────────

    def _check_vector(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, np.float32).reshape(-1)
        if vector.shape != (self.meta.feature_dim,):
            raise ValueError(f"vector must be ({self.meta.feature_dim},), got {vector.shape}")
        return vector

    def _write_clip(self, slug: str, video_name: str, silhouettes: list[np.ndarray]) -> Clip:
        clip_id = uuid.uuid4().hex[:8]
        clip_dir = self.sil_dir / slug / clip_id
        clip_dir.mkdir(parents=True)
        files = []
        for i, sil in enumerate(silhouettes):
            fname = f"{i:02d}.png"
            (clip_dir / fname).write_bytes(encode_png(sil))
            files.append(f"{slug}/{clip_id}/{fname}")
        return Clip(clip_id, Path(video_name).name, _now(), files)

    def add(self, name: str, video_name: str, vector: np.ndarray, silhouettes: list[np.ndarray]) -> Person:
        """A new person with their first clip."""
        name = validate_name(name)
        vector = self._check_vector(vector)

        with self._lock:
            self.require_compatible()
            if self.name_taken(name):
                raise ApiError("name_taken", f"A case file named '{name}' already exists.", 409)

            slug = _slugify(name)
            used = {p.slug for p in self.people}
            base, n = slug, 2
            while slug in used or (self.sil_dir / slug).exists():
                slug, n = f"{base}_{n}", n + 1

            clip = self._write_clip(slug, video_name, silhouettes)
            person = Person(personId=name, slug=slug, addedAt=clip.addedAt, clips=[clip])
            people = [*self.people, person]
            features = np.vstack([self.features, vector[None, :]])
            self._persist(people, features)
            self.people, self.features = people, features
            return person

    def add_clip(self, person_id: str, video_name: str, vector: np.ndarray, silhouettes: list[np.ndarray]) -> Person:
        """Another clip of someone already in the Archive (e.g. carrying a bag, another day)."""
        vector = self._check_vector(vector)
        with self._lock:
            self.require_compatible()
            i = self._index_of(person_id)
            person = self.people[i]
            clip = self._write_clip(person.slug, video_name, silhouettes)
            row = self._first_row(i) + len(person.clips)
            features = np.insert(self.features, row, vector, axis=0)
            people = list(self.people)
            people[i] = replace(person, clips=[*person.clips, clip])
            self._persist(people, features)
            self.people, self.features = people, features
            return people[i]

    def remove_clip(self, person_id: str, clip_id: str) -> Person:
        """Drops one clip. The last clip can't go this way: that is deleting the person."""
        with self._lock:
            i = self._index_of(person_id)
            person = self.people[i]
            ids = [c.clipId for c in person.clips]
            if clip_id not in ids:
                raise ApiError("not_found", f"'{person_id}' has no clip '{clip_id}'.", 404)
            if len(ids) == 1:
                raise ApiError("last_clip", "This is their only clip. Delete the case file instead.", 409)
            k = ids.index(clip_id)
            gone = person.clips[k]
            features = np.delete(self.features, self._first_row(i) + k, axis=0)
            people = list(self.people)
            people[i] = replace(person, clips=person.clips[:k] + person.clips[k + 1 :])
            self._persist(people, features)
            self.people, self.features = people, features
        # After the index no longer mentions it, so a failure here only leaves unused PNGs.
        for f in gone.silhouetteFiles:
            (self.sil_dir / f).unlink(missing_ok=True)
        shutil.rmtree(self.sil_dir / person.slug / clip_id, ignore_errors=True)
        return people[i]

    def rename(self, person_id: str, new_name: str) -> Person:
        """Only the name changes; the vectors and the silhouette folder stay as they are."""
        new_name = validate_name(new_name)
        with self._lock:
            i = self._index_of(person_id)
            if self.name_taken(new_name, ignore=person_id):
                raise ApiError("name_taken", f"A case file named '{new_name}' already exists.", 409)
            people = list(self.people)
            people[i] = replace(people[i], personId=new_name)
            self._persist(people, self.features)
            self.people = people
            return people[i]

    def remove(self, person_id: str) -> None:
        with self._lock:
            i = self._index_of(person_id)
            gone = self.people[i]
            first = self._first_row(i)
            features = np.delete(self.features, range(first, first + len(gone.clips)), axis=0)
            people = self.people[:i] + self.people[i + 1 :]
            self._persist(people, features)
            self.people, self.features = people, features
        # After the index no longer mentions it, so a failure here only leaves unused PNGs.
        shutil.rmtree(self.sil_dir / gone.slug, ignore_errors=True)

    def _persist(self, people: list[Person], features: np.ndarray) -> None:
        # features first, index second: the index is what says how many rows are valid.
        tmp = self.features_path.with_name("features.tmp.npy")
        np.save(tmp, features.astype(np.float32))
        os.replace(tmp, self.features_path)
        doc = {"version": INDEX_VERSION, "meta": asdict(self.meta), "people": [asdict(p) for p in people]}
        _atomic_write_bytes(self.index_path, json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"))
