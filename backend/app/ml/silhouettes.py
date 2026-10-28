"""Video -> per-frame person masks (YOLOv8-seg) -> 64x44 silhouettes.

Two ways to turn a mask into a silhouette (Settings.preprocess_mode):

- "casia" (default): frame the person the way CASIA-B silhouettes look, because that is what
  GLTNet was trained on. CASIA-B silhouettes are whole 320x240 frames squashed to 44x64, so the
  person fills about half the height, is narrow, and the edges are soft greyscale (HANDOFF §6.1).
- "crop": the original silhouette_extractor.py behaviour (tight crop, hard 0/1). Kept only so
  the two can be compared on CASIA-B test data. Not measured yet, so no accuracy claim for either.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from app.errors import ApiError, NoPersonError, TooFewFramesError
from app.ml.embed import IMG_H, IMG_W

# CASIA-B framing. TODO: replace with the exact CCPG_MATCH_CASIA_GEOMETRY numbers from the newest
# casia_b_loader.py on Kaggle; these are the values quoted in HANDOFF §6.1.
CASIA_PERSON_HEIGHT = 0.501  # person bbox height / frame height
CASIA_BBOX_TOP = 0.233  # bbox top / frame height
CASIA_FRAME_H, CASIA_FRAME_W = 240, 320  # CASIA-B's own frame size, so the final resize matches

PERSON_CLASS = 0  # COCO


class YoloSegmenter:
    """YOLOv8-seg, loaded once. Yields one binary person mask (or None) per frame."""

    def __init__(self, weights: Path, device: str):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device

    def masks(self, video_path: Path) -> Iterator[np.ndarray | None]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ApiError("unreadable_video", "The video could not be opened.", 422)
        tracker = WalkerTracker()
        try:
            while True:
                ok, frame = cap.read()
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


def casia_silhouette(mask: np.ndarray) -> np.ndarray | None:
    """Place the person on a 320x240 canvas the way CASIA-B frames them, then squash to 64x44.

    Returns float32 (64, 44) in [0, 1], soft greyscale edges, or None for an empty mask.
    """
    box = _bbox(mask)
    if box is None:
        return None
    top, bottom, left, right = box
    person = mask[top:bottom, left:right].astype(np.float32) / 255.0

    canvas_h, canvas_w = CASIA_FRAME_H, CASIA_FRAME_W
    target_h = max(1, round(canvas_h * CASIA_PERSON_HEIGHT))
    scale = target_h / person.shape[0]
    target_w = min(canvas_w, max(1, round(person.shape[1] * scale)))
    person = cv2.resize(person, (target_w, target_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_h, canvas_w), np.float32)
    y0 = round(canvas_h * CASIA_BBOX_TOP)
    x0 = (canvas_w - target_w) // 2
    canvas[y0 : y0 + target_h, x0 : x0 + target_w] = person

    # Whole 4:3 frame -> 44x64, bilinear, greyscale kept (no threshold), like the training data.
    out = cv2.resize(canvas, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
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
