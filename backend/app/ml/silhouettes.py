"""Video -> per-frame person masks (YOLOv8-seg) -> 64x44 silhouettes.

Two ways to turn a mask into a silhouette (Settings.preprocess_mode):

- "casia" (default): frame the person the way CASIA-B silhouettes look, because that is what
  GLTNet was trained on (the training code's CCPG_MATCH_CASIA_GEOMETRY): about half the frame
  height, squashed narrow, soft greyscale edges (HANDOFF §6.1).
- "crop": the original silhouette_extractor.py behaviour (tight crop, hard 0/1). Kept only so
  the two can be compared on CASIA-B test data (scripts/casia_eval.py). Not measured yet.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from app.errors import ApiError, NoPersonError, TooFewFramesError
from app.ml.embed import IMG_H, IMG_W

# CASIA-B framing: the CCPG_MATCH_CASIA_GEOMETRY transform from the training code (casia_b_loader.py
# on Kaggle), as specified by the training side on 2026-10-28.
CASIA_PERSON_HEIGHT = 0.501  # person bbox height / frame height
CASIA_SQUASH = 0.52  # width scale relative to an aspect-preserving resize (CASIA-B's 320x240 -> 44x64)
CASIA_BBOX_TOP = 0.233  # bbox top / frame height
# Bumped whenever the framing changes, so an Archive built the old way is flagged incompatible.
# 1 = 320x240-canvas version (implicit squash 0.516), 2 = CCPG_MATCH_CASIA_GEOMETRY port.
CASIA_GEOMETRY_VERSION = 2
SUPERSAMPLE = 8  # place the person at 8x resolution, then area-average down: sub-pixel accurate, soft edges

PERSON_CLASS = 0  # COCO
CASIA_FPS = 25.0  # CASIA-B videos; the temporal branches learned walking speed at this rate


class FrameRateSampler:
    """Keeps frames so the clip plays at about target_fps, e.g. 60 fps -> every 2nd or 3rd frame.

    The model reads motion between neighbouring frames, so a 60 fps clip fed whole would look like
    a walk at less than half speed. Clips at or below the target are kept whole (they can't be
    sped up without inventing frames).
    """

    def __init__(self, source_fps: float, target_fps: float = CASIA_FPS):
        # OpenCV reports 0 when a file doesn't say its frame rate: keep everything then.
        self.step = source_fps / target_fps if source_fps > target_fps * 1.05 else 1.0
        self._next = 0.0

    def keep(self, index: int) -> bool:
        if index + 1e-9 < self._next:
            return False
        self._next += self.step
        return True


class YoloSegmenter:
    """YOLOv8-seg, loaded once. Yields one binary person mask (or None) per kept frame."""

    def __init__(self, weights: Path, device: str, target_fps: float = CASIA_FPS):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        self.target_fps = target_fps

    def estimate_frames(self, video_path: Path) -> int | None:
        """Frames masks() will yield, from the container's frame count (for progress only)."""
        cap = cv2.VideoCapture(str(video_path))
        try:
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        if count <= 0:
            return None
        return max(1, round(count / FrameRateSampler(fps, self.target_fps).step))

    def masks(self, video_path: Path) -> Iterator[np.ndarray | None]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ApiError("unreadable_video", "The video could not be opened.", 422)
        tracker = WalkerTracker()
        sampler = FrameRateSampler(cap.get(cv2.CAP_PROP_FPS), self.target_fps)
        try:
            index = -1
            while True:
                # grab() without decoding is cheap; only kept frames are decoded and segmented.
                if not cap.grab():
                    break
                index += 1
                if not sampler.keep(index):
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                # retina_masks: masks at the frame's own resolution, not the letterboxed
                # inference size (which would be misaligned when stretched back).
                result = self.model(
                    frame, classes=[PERSON_CLASS], retina_masks=True, device=self.device, verbose=False
                )[0]
                yield walker_mask(result, frame.shape[:2], tracker)
        finally:
            cap.release()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Boxes as (x1, y1, x2, y2)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class WalkerTracker:
    """Follows one person through the clip, so a bystander never gets mixed into the walk.

    The walker is the biggest confident person when first seen; after that, the detection that
    overlaps the walker's last box the most. If the walker is lost for a few frames, it starts over.
    """

    MIN_IOU = 0.2
    MAX_MISSES = 5

    def __init__(self):
        self.box: np.ndarray | None = None
        self.misses = 0

    def pick(self, boxes: np.ndarray, confs: np.ndarray) -> int | None:
        if len(boxes) == 0:
            return self._miss()
        if self.box is None:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas * confs))
        else:
            overlaps = np.array([iou(self.box, b) for b in boxes])
            best = int(np.argmax(overlaps))
            if overlaps[best] < self.MIN_IOU:
                return self._miss()
        self.box, self.misses = boxes[best], 0
        return best

    def _miss(self) -> None:
        self.misses += 1
        if self.misses >= self.MAX_MISSES:
            self.box = None
        return None


def walker_mask(result, frame_hw: tuple[int, int], tracker: WalkerTracker) -> np.ndarray | None:
    """The tracked walker's mask, at frame size, as uint8 0/255."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        tracker.pick(np.zeros((0, 4)), np.zeros(0))
        return None
    cls = result.boxes.cls.cpu().numpy().astype(int)
    people = np.flatnonzero(cls == PERSON_CLASS)
    boxes = result.boxes.xyxy.cpu().numpy()[people]
    confs = result.boxes.conf.cpu().numpy()[people]
    chosen = tracker.pick(boxes, confs)
    if chosen is None:
        return None
    best = people[chosen]
    mask = result.masks.data[best].cpu().numpy().astype(np.float32)
    h, w = frame_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return ((mask > 0.5) * 255).astype(np.uint8)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def crop_to_person(mask: np.ndarray) -> np.ndarray | None:
    """The mask cut down to the person's bounding box plus a zero margin, so a clip's frames can be
    kept in memory between detecting and building silhouettes. Outside the bounding box the mask is
    zero anyway, so casia_silhouette gives exactly the same result on this as on the full frame;
    crop_silhouette does too unless the person touches the frame edge (there it used to clip its
    5% padding at the edge, here the padding is always there)."""
    box = _bbox(mask)
    if box is None:
        return None
    top, bottom, left, right = box
    margin = max(2, (bottom - top) // 10)
    return np.pad(mask[top:bottom, left:right], margin)


def casia_silhouette(mask: np.ndarray) -> np.ndarray | None:
    """CCPG_MATCH_CASIA_GEOMETRY, in the 64x44 model frame:

    1. crop to the silhouette's bounding box
    2. scale so its height is 50.1% of the frame height
    3. squash its width by 0.52 (relative to that height-matched scale)
    4. paste with the bbox top at 23.3% of the frame height

    The spec doesn't fix the horizontal position; the person is centred. Done at SUPERSAMPLE x
    resolution and area-averaged down, so fractional sizes/offsets are kept and edges stay soft
    greyscale (no threshold). Returns float32 (64, 44) in [0, 1], or None for an empty mask.
    """
    box = _bbox(mask)
    if box is None:
        return None
    top, bottom, left, right = box
    person = mask[top:bottom, left:right].astype(np.float32) / 255.0

    frame_h, frame_w = IMG_H * SUPERSAMPLE, IMG_W * SUPERSAMPLE
    target_h = max(1, round(frame_h * CASIA_PERSON_HEIGHT))
    scale = target_h / person.shape[0]
    target_w = min(frame_w, max(1, round(person.shape[1] * scale * CASIA_SQUASH)))
    person = cv2.resize(person, (target_w, target_h), interpolation=cv2.INTER_AREA)

    frame = np.zeros((frame_h, frame_w), np.float32)
    y0 = round(frame_h * CASIA_BBOX_TOP)
    x0 = (frame_w - target_w) // 2
    frame[y0 : y0 + target_h, x0 : x0 + target_w] = person

    out = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return np.clip(out, 0.0, 1.0)


def crop_silhouette(mask: np.ndarray) -> np.ndarray | None:
    """Original silhouette_extractor.py framing: tight crop + 5% padding, hard 0/1."""
    box = _bbox(mask)
    if box is None:
        return None
    top, bottom, left, right = box
    pad_h = max(1, int((bottom - top) * 0.05))
    pad_w = max(1, int((right - left) * 0.05))
    h, w = mask.shape
    crop = mask[max(0, top - pad_h) : min(h, bottom + pad_h), max(0, left - pad_w) : min(w, right + pad_w)]
    out = cv2.resize(crop, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return (out > 127).astype(np.float32)


PREPROCESSORS = {"casia": casia_silhouette, "crop": crop_silhouette}
PREPROCESS_VERSIONS = {"casia": CASIA_GEOMETRY_VERSION, "crop": 1}


def central_window(frames: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
    """Consecutive frames, in order. Long clips keep the middle max_frames (no subsampling:
    the temporal branches read adjacent-frame motion, HANDOFF §6.2)."""
    if len(frames) <= max_frames:
        return frames
    start = (len(frames) - max_frames) // 2
    return frames[start : start + max_frames]


def pick_display(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    if len(frames) <= n:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def check_enough(found: int, min_frames: int) -> None:
    if found == 0:
        raise NoPersonError()
    if found < min_frames:
        raise TooFewFramesError(found, min_frames)
