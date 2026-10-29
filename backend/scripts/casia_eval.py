"""CASIA-B check for the website pipeline: preprocessing Rank-1 (HANDOFF §6.1) + threshold calibration.

Made to run on Kaggle, where CASIA-B and the GPU are. It needs only torch, numpy and OpenCV
(no FastAPI, no ultralytics). In a Kaggle notebook (clones main, so merge the eval-progress PR first):

    !git clone https://github.com/NSW-0/Swalker.git
    %cd Swalker/backend
    !python -m scripts.casia_eval \\
        --root /kaggle/input/<casia-b dataset>/<folder with 001..124> \\
        --checkpoint /kaggle/input/datasets/bbiiggnn/gait-checkpoints/stage2_iter70000.pth \\
        --out /kaggle/working/casia_eval.json

Reading CASIA-B's PNGs from /kaggle/input is the slow part (a network drive), so frames are read
in parallel and the next sequences are read while the current one is embedded. Progress is saved
to --cache every 250 sequences: if the session dies, run the same command again and it carries on.
Run it with "Save Version -> Save & Run All" so it keeps going with the browser tab closed.

(If the repo is private, upload the backend/ folder as a Kaggle dataset and cd into it instead.)

Layout is detected, not assumed (like casia_b_loader.py's 'kaggle' / 'kaggle_nested' / 'local'):
<root>/075/nm-05/090/*.png or <root>/075/075/nm-05/090/*.png, with <root> allowed to be up to two
wrapper folders above the subject folders. Frames are the standard whole 320x240 silhouettes.

What it reports, for the 50 LT test subjects (075-124 by default):

0. A baseline check first. 'train' mode must reproduce this checkpoint's known Rank-1 under the
   corrected evaluation (--expect, default NM 99.27 / BG 93.26 / CL 72.36, within 0.1 points)
   in the 'pooled' protocol. If it doesn't, the script says so and marks the threshold untrusted:
   the harness differs from the training-side evaluation and must be fixed first.

1. Rank-1 for each preprocessing mode: gallery NM 01-04, probes NM 05-06 / BG 01-02 / CL 01-02,
   in three protocols:
     pooled                 train.py's default (rank1_from_features, protocol='pooled'): each
                            probe vs every gallery sequence at another view, mean over probe views
     per-gallery-view       train.py's cross-check / OpenGait style: mean over probe x gallery
                            view pairs, identical view excluded
     per-probe, all views   gallery including the probe's own view (closest to the website,
                            where both clips usually come from the same camera)
     train  - exactly casia_b_loader.load_silhouette(): PIL convert('L'), whole frame resized to
              44x64 with PIL BILINEAR (which, unlike OpenCV INTER_LINEAR, averages when shrinking),
              /255, every frame kept including blank ones
     casia  - each frame's mask re-framed the CASIA-B way (app/ml/silhouettes.py casia_silhouette)
     casia_seq   - the same framing with one scale and vertical offset per clip, so per-frame
                   size and height changes survive (casia_sequence), each frame centred
     casia_seq_x - casia_seq that also keeps the walker's movement across the frame; the
                   website's default (preprocess mode casia_clip)
     crop   - the old silhouette_extractor.py tight crop
     native, native_scale, native_pos - casia_seq_x with its per-clip scale and/or position
              normalisation undone (native_geometry), to see which part costs the accuracy
     native_bin - train's drawing of a binary mask (cost of binarising alone)
     clip_pil - casia_seq_x's geometry drawn like train (silhouettes.casia_clip_frames)
   Modes already measured are off by default, see DEFAULT_MODES.
   The closer casia_seq_x is to train, the less the website's framing costs.
   The CASIA-B mask stands in for the YOLO mask here; YOLO's own errors are not measured.

2. Genuine vs impostor distances (NM probes vs NM gallery, same view, like the website where both
   clips come from the same camera), and the threshold at several false-match rates. This is a
   starting value for SWALKER_MATCH_THRESHOLD; recheck it on the team's own recordings.
"""

import argparse
import io
import json
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.ml.embed import IMG_H, IMG_W
from app.ml.silhouettes import (
    CASIA_BBOX_TOP,
    CASIA_CENTRE_X,
    CASIA_PERSON_HEIGHT,
    CASIA_SQUASH,
    SUPERSAMPLE,
    casia_clip_frames,
    casia_sequence,
    casia_silhouette,
    crop_silhouette,
    crop_to_person,
)

VIEWS = [f"{v:03d}" for v in range(0, 181, 18)]
GALLERY = ("nm", (1, 2, 3, 4))
PROBES = {"NM": ("nm", (5, 6)), "BG": ("bg", (1, 2)), "CL": ("cl", (1, 2))}
SEQ_DIR = re.compile(r"^(nm|bg|cl)-(\d{2})$")
FALSE_MATCH_RATES = (0.001, 0.01, 0.05, 0.10)


def train_style(frame: np.ndarray) -> np.ndarray:
    """casia_b_loader.load_silhouette(), byte for byte: PIL BILINEAR to 44x64, /255, blanks kept."""
    img = Image.fromarray(frame).resize((44, 64), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def as_mask(frame: np.ndarray) -> np.ndarray:
    return ((frame > 127) * 255).astype(np.uint8)


def per_frame(fn):
    return lambda frames: [fn(f) for f in frames]


def native_geometry(keep_scale: bool, keep_position: bool):
    """casia_seq_x with some of its per-clip normalisation undone, to find where its gap to 'train'
    comes from. casia_seq_x maps every frame of a clip with one affine transform: the clip's
    median person height -> 50.1% of the frame, median top -> 23.3%, median centre -> 51.4%.
    keep_scale uses the CASIA frame's own scale instead (240 px high -> 64); keep_position puts
    the clip's median top and centre where they really were in the frame. With both, only the
    website's way of drawing the silhouette is left (binary mask, area-averaged) vs train's
    PIL bilinear on the original PNG."""
    canvas_h, canvas_w = IMG_H * SUPERSAMPLE, IMG_W * SUPERSAMPLE

    def run(frames):
        masks = [as_mask(f) for f in frames]
        boxes = []
        for m in masks:
            rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
            boxes.append((rows[0], rows[-1] + 1, cols[0], cols[-1] + 1) if rows.size else None)
        found = np.array([b for b in boxes if b is not None], dtype=np.float64)
        if found.size == 0:
            return [None] * len(masks)
        tops, bottoms, lefts, rights = found.T
        top_ref, cx_ref = np.median(tops), np.median((lefts + rights) / 2)
        frame_h, frame_w = masks[0].shape
        if keep_scale:
            sy, sx = canvas_h / frame_h, canvas_w / frame_w
        else:
            sy = canvas_h * CASIA_PERSON_HEIGHT / np.median(bottoms - tops)
            sx = sy * CASIA_SQUASH
        if keep_position:
            ty, tx = canvas_h * top_ref / frame_h, canvas_w * cx_ref / frame_w
        else:
            ty, tx = canvas_h * CASIA_BBOX_TOP, canvas_w * CASIA_CENTRE_X
        affine = np.float32([[sx, 0, tx - cx_ref * sx], [0, sy, ty - top_ref * sy]])
        out = []
        for m, box in zip(masks, boxes, strict=True):
            if box is None:
                out.append(None)
                continue
            big = cv2.warpAffine(m.astype(np.float32) / 255.0, affine, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
            out.append(np.clip(cv2.resize(big, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA), 0.0, 1.0))
        return out

    return run


# Each mode turns a sequence's frames into its silhouettes (None = frame dropped).
MODES = {
    "train": per_frame(train_style),
    "casia": per_frame(lambda f: casia_silhouette(as_mask(f))),
    "casia_seq": lambda frames: casia_sequence([crop_to_person(as_mask(f)) for f in frames]),
    # The website's default, preprocess mode casia_clip (named before it was chosen; kept so
    # progress files from that run still resume).
    "casia_seq_x": lambda frames: casia_sequence([crop_to_person(as_mask(f)) for f in frames], follow_x=True),
    "crop": per_frame(lambda f: crop_silhouette(as_mask(f))),
    # Where casia_seq_x's gap to train comes from (native_geometry):
    "native": native_geometry(keep_scale=True, keep_position=True),  # drawing only
    "native_scale": native_geometry(keep_scale=True, keep_position=False),  # + position normalised
    "native_pos": native_geometry(keep_scale=False, keep_position=True),  # + scale normalised
    # Train's drawing from a binary mask: what turning the PNG into a 0/255 mask alone costs.
    "native_bin": per_frame(lambda f: train_style(as_mask(f))),
    # casia_seq_x's geometry drawn like train (binary 320x240 frame, PIL BILINEAR).
    "clip_pil": lambda frames: casia_clip_frames([crop_to_person(as_mask(f)) for f in frames]),
}
# Measured on 2026-10-28/29, pooled NM / BG / CL: crop 69.45 / 41.17 / 12.09, casia 96.09 /
# 78.05 / 35.73, casia_seq 92.55 / 76.40 / 40.18, native 97.27 / 91.79 / 64.27, native_scale
# 97.73 / 91.25 / 61.55, native_pos 98.00 / 89.97 / 63.73 (casia_seq_x 97.82 / 90.06 / 56.64,
# kept as the control).
DEFAULT_MODES = ["train", "casia_seq_x", "native_bin", "clip_pil"]


def detect_layout(root: Path, probe_sid: int) -> tuple[Path, str]:
    """Where the subject folders are, and whether each is nested once (<sid>/<sid>/nm-01/...).

    Mirrors the layouts casia_b_loader.py handles: 'local'/'kaggle' = <base>/<sid>/<cond-seq>/<view>,
    'kaggle_nested' = <base>/<sid>/<sid>/<cond-seq>/<view>. The base may be the given root or a
    folder up to two levels below it (Kaggle datasets often add a wrapper folder).
    """
    name = f"{probe_sid:03d}"
    candidates = [root, *(p for p in root.iterdir() if p.is_dir())]
    candidates += [q for p in candidates[1:] for q in p.iterdir() if q.is_dir()]
    for base in candidates:
        sdir = base / name
        if not sdir.is_dir():
            continue
        if any(SEQ_DIR.match(c.name) for c in sdir.iterdir() if c.is_dir()):
            return base, "flat"
        if (sdir / name).is_dir() and any(SEQ_DIR.match(c.name) for c in (sdir / name).iterdir()):
            return base, "nested"
    raise SystemExit(
        f"couldn't find subject {name} under {root} (looked for {name}/nm-01/000/*.png and "
        f"{name}/{name}/nm-01/000/*.png, up to two folders deep)"
    )


def find_sequences(root: Path, subjects: range) -> tuple[list[dict], str]:
    base, layout = detect_layout(root, subjects.start)
    seqs = []
    for sid in subjects:
        sdir = base / f"{sid:03d}"
        if layout == "nested":
            sdir = sdir / f"{sid:03d}"
        if not sdir.is_dir():
            continue
        for cdir in sorted(sdir.iterdir()):
            m = SEQ_DIR.match(cdir.name)
            if not m:
                continue
            for view in VIEWS:
                vdir = cdir / view
                if vdir.is_dir():
                    seqs.append({"sid": sid, "cond": m[1], "seq": int(m[2]), "view": view, "dir": vdir})
    return seqs, f"{layout} at {base}"


def decode(path: Path) -> np.ndarray:
    """Like load_silhouette(): transparent pixels onto black, then PIL greyscale.

    The file is read in one call and decoded from memory: PIL reading the file itself does several
    small reads, each a round trip on /kaggle/input (likely why disk wait went from ~20 s to ~1000 s per 250)."""
    img = Image.open(io.BytesIO(path.read_bytes()))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, 255)), img.convert("RGBA"))
    return np.asarray(img.convert("L"), dtype=np.uint8)


def read_frames(seq_dir: Path, pool: ThreadPoolExecutor, max_frames: int) -> list[np.ndarray]:
    files = sorted(seq_dir.glob("*.png"))
    if max_frames:
        files = files[:max_frames]
    return list(pool.map(decode, files))


# Bumped whenever a mode's output changes, so a resume never mixes old and new features.
# 2 = train mode via PIL with blank frames kept, empty sequences embedded instead of skipped.
CACHE_VERSION = 2


def load_cache(path: Path, modes: list[str]) -> tuple[dict, list, set]:
    """Features embedded by an earlier, interrupted run: (feats, meta, finished sequence dirs)."""
    feats, meta, done = {m: [] for m in modes}, [], set()
    if not path.exists():
        return feats, meta, done
    data = np.load(path, allow_pickle=False)
    saved = json.loads(str(data["info"]))
    if saved.get("version") != CACHE_VERSION:
        print(f"ignoring {path}: made by an older version of this script, starting over")
        return feats, meta, done
    if saved["modes"] != modes:
        print(f"ignoring {path}: made with modes {saved['modes']}, not {modes}; starting over")
        return feats, meta, done
    for m in modes:
        feats[m] = list(data[m])
    return feats, saved["meta"], set(saved["done"])


def save_cache(path: Path, feats: dict, meta: list, done: set, modes: list[str]) -> None:
    info = json.dumps({"version": CACHE_VERSION, "modes": modes, "meta": meta, "done": sorted(done)})
    arrays = {m: np.stack(v) if v else np.zeros((0, 4096), np.float32) for m, v in feats.items()}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, info=np.array(info), **arrays)
    tmp.replace(path)


def _columns(meta):
    return (np.array([m[k] for m in meta]) for k in ("sid", "cond", "seq", "view"))


def rank1_pooled(feats, meta, probe) -> tuple[float, dict]:
    """train.py rank1_from_features(protocol='pooled'): per probe view, every gallery sequence at
    another view is a candidate; Rank-1 per probe view, then the mean over probe views."""
    sid, cond, seq, view = _columns(meta)
    in_gallery = (cond == GALLERY[0]) & np.isin(seq, GALLERY[1])
    in_probe = (cond == probe[0]) & np.isin(seq, probe[1])
    per_view = {}
    for pv in VIEWS:
        p, g = in_probe & (view == pv), in_gallery & (view != pv)
        if p.any() and g.any():
            d = distances(feats[p], feats[g])
            per_view[pv] = float((sid[g][d.argmin(axis=1)] == sid[p]).mean())
    return (float(np.mean(list(per_view.values()))) if per_view else float("nan")), per_view


def rank1(feats, meta, probe) -> tuple[float, dict]:
    """train.py protocol='per_gallery_view': mean over probe views of the mean over gallery views
    != probe view."""
    sid, cond, seq, view = _columns(meta)
    in_gallery = (cond == GALLERY[0]) & np.isin(seq, GALLERY[1])
    in_probe = (cond == probe[0]) & np.isin(seq, probe[1])

    per_view = {}
    for pv in VIEWS:
        p = in_probe & (view == pv)
        if not p.any():
            continue
        accs = []
        for gv in VIEWS:
            g = in_gallery & (view == gv)
            if gv == pv or not g.any():
                continue
            d = distances(feats[p], feats[g])
            accs.append(float((sid[g][d.argmin(axis=1)] == sid[p]).mean()))
        if accs:
            per_view[pv] = float(np.mean(accs))
    return (float(np.mean(list(per_view.values()))) if per_view else float("nan")), per_view


def rank1_per_probe(feats, meta, probe, exclude_same_view: bool) -> tuple[int, int]:
    """Each probe sequence counted once against the whole NM 01-04 gallery (correct, total),
    e.g. 796/1100. With exclude_same_view, gallery sequences at the probe's own view are left out."""
    sid, cond, seq, view = _columns(meta)
    gal = (cond == GALLERY[0]) & np.isin(seq, GALLERY[1])
    prb = np.flatnonzero((cond == probe[0]) & np.isin(seq, probe[1]))
    if prb.size == 0:
        return 0, 0
    d = distances(feats[prb], feats[gal])
    if exclude_same_view:
        d[view[prb][:, None] == view[gal][None, :]] = np.inf
    return int((sid[gal][d.argmin(axis=1)] == sid[prb]).sum()), int(prb.size)


PROTOCOLS = {
    "per-probe, all views": lambda f, m, p: rank1_per_probe(f, m, p, exclude_same_view=False),
}


def parse_expected(text: str) -> dict[str, float]:
    return {k.strip().upper(): float(v) for k, v in (part.split("=") for part in text.split(","))}


def matches_baseline(row: dict[str, float], expected: dict[str, float], tolerance: float) -> bool:
    return all(abs(row[c] - expected[c]) <= tolerance for c in expected)


def distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    sq = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(sq, 0))


def genuine_impostor(feats, meta, views=VIEWS) -> tuple[np.ndarray, np.ndarray]:
    """NM probe vs NM gallery distances at the same view."""
    sid, cond, seq, view = _columns(meta)
    gal = (cond == "nm") & np.isin(seq, GALLERY[1])
    prb = (cond == "nm") & np.isin(seq, PROBES["NM"][1])
    genuine, impostor = [], []
    for v in views:
        p, g = prb & (view == v), gal & (view == v)
        if not p.any() or not g.any():
            continue
        d = distances(feats[p], feats[g])
        same = sid[p][:, None] == sid[g][None, :]
        genuine.append(d[same])
        impostor.append(d[~same])
    return np.concatenate(genuine or [np.zeros(0)]), np.concatenate(impostor or [np.zeros(0)])


def fmr_table(genuine: np.ndarray, impostor: np.ndarray, rates=FALSE_MATCH_RATES) -> list[dict]:
    """For each target false-match rate: the largest threshold that keeps impostors at or under it."""
    imp = np.sort(impostor)
    rows = []
    for rate in rates:
        k = int(np.floor(rate * imp.size))
        threshold = float(imp[k] - 1e-6) if k < imp.size else float(imp[-1])
        rows.append(
            {
                "target_fmr": rate,
                "threshold": threshold,
                "fmr": float((impostor <= threshold).mean()),
                "genuine_accepted": float((genuine <= threshold).mean()),
            }
        )
    return rows


def describe(d: np.ndarray) -> dict:
    if d.size == 0:
        return {"n": 0}
    p5, p50, p95 = (float(x) for x in np.percentile(d, [5, 50, 95]))
    return {"n": int(d.size), "min": float(d.min()), "p5": p5, "median": p50, "p95": p95, "max": float(d.max())}


def show(stats: dict) -> str:
    return "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=list(MODES))
    parser.add_argument(
        "--calibrate-mode",
        default="clip_pil",
        choices=list(MODES),
        help="mode used for the threshold (casia_seq_x = the website's casia_clip)",
    )
    parser.add_argument("--subjects", default="75-124", help="test subject range, e.g. 75-124")
    parser.add_argument("--max-frames", type=int, default=0, help="cap frames per sequence (0 = whole sequence)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N sequences (quick smoke test)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--expect",
        default="NM=99.27,BG=93.26,CL=72.36",
        help="known Rank-1 (%%) of this checkpoint under the corrected eval; 'train' mode must reproduce it",
    )
    parser.add_argument("--tolerance", type=float, default=0.1, help="allowed difference in points (1/1100 = 0.09)")
    parser.add_argument("--out", type=Path, default=Path("casia_eval.json"))
    parser.add_argument(
        "--cache", type=Path, default=None, help="resume file (default: next to --out); delete it to start over"
    )
    parser.add_argument("--workers", type=int, default=32, help="parallel PNG reads")
    args = parser.parse_args()
    expected = parse_expected(args.expect) if args.expect else {}

    import torch

    from app.ml.embed import GaitEmbedder

    lo, hi = (int(x) for x in args.subjects.split("-"))
    seqs, layout = find_sequences(args.root, range(lo, hi + 1))
    if not seqs:
        raise SystemExit(f"no sequences under {args.root} (expected <root>/075/nm-01/000/*.png)")
    if args.limit:
        seqs = seqs[: args.limit]
    print(f"{len(seqs)} sequences, subjects {lo}-{hi}, layout {layout}, modes {args.modes}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    embedder = GaitEmbedder(args.checkpoint, device)
    print(f"model: {embedder.checkpoint_name} on {device}")

    cache = args.cache or args.out.with_name(args.out.stem + "_cache.npz")
    feats, meta, done = load_cache(cache, args.modes)
    todo = [s for s in seqs if str(s["dir"]) not in done]
    if done:
        print(f"resuming from {cache}: {len(done)} sequences already done, {len(todo)} to go")

    blank, wait, work = 0, 0.0, 0.0
    start = time.time()
    with ThreadPoolExecutor(args.workers) as files, ThreadPoolExecutor(2) as ahead:
        # Up to 4 sequences are read in the background while the current one is embedded.
        queue = deque()
        upcoming = iter(todo)
        for s in upcoming:
            queue.append((s, ahead.submit(read_frames, s["dir"], files, args.max_frames)))
            if len(queue) == 4:
                break
        i = 0
        while queue:
            s, pending = queue.popleft()
            nxt = next(upcoming, None)
            if nxt is not None:
                queue.append((nxt, ahead.submit(read_frames, nxt["dir"], files, args.max_frames)))
            t0 = time.time()
            frames = pending.result()
            t1 = time.time()
            blank += not frames  # CASIA-B has a few empty sequence folders
            for mode in args.modes:
                sils = [x for x in MODES[mode](frames) if x is not None]
                # Like CASIABEvalDataset: a sequence with nothing in it is one blank frame, not dropped.
                feats[mode].append(embedder.embed(np.stack(sils) if sils else np.zeros((1, 64, 44), np.float32)))
            meta.append({k: s[k] for k in ("sid", "cond", "seq", "view")})
            wait, work = wait + t1 - t0, work + time.time() - t1
            done.add(str(s["dir"]))
            i += 1
            if i % 250 == 0 or not queue:
                save_cache(cache, feats, meta, done, args.modes)
                print(
                    f"  {len(done)}/{len(seqs)}  {time.time() - start:.0f}s  "
                    f"(waiting for disk {wait:.0f}s, preprocessing + model {work:.0f}s)",
                    flush=True,
                )
    print(f"embedded {len(meta)} sequences ({blank} empty this run) in {time.time() - start:.0f}s")

    feats = {m: np.stack(v) for m, v in feats.items()}
    results = {"checkpoint": embedder.checkpoint_name, "subjects": args.subjects, "sequences": len(meta), "rank1": {}}

    results["layout"] = layout
    means = {"pooled": rank1_pooled, "per-gallery-view": rank1}
    notes = {"pooled": " (train.py default, the one 99.27 was measured with)"}
    protocols = {**dict.fromkeys(means), **PROTOCOLS}
    for name, per_probe in protocols.items():
        print(f"\nRank-1 (%), {name}{notes.get(name, '')}")
        print(f"{'mode':12s}" + "".join(f"{c:>16s}" for c in PROBES))
        for mode in args.modes:
            row, cells = {}, []
            for cond, probe in PROBES.items():
                if per_probe is None:
                    mean, per_view = means[name](feats[mode], meta, probe)
                    row[cond] = {"pct": mean * 100, "per_view": per_view}
                    cells.append(f"{mean * 100:16.2f}")
                else:
                    correct, total = per_probe(feats[mode], meta, probe)
                    pct = 100 * correct / total if total else float("nan")
                    row[cond] = {"pct": pct, "correct": correct, "total": total}
                    cells.append(f"{pct:7.2f} {correct:>4d}/{total:<4d}")
            results["rank1"].setdefault(name, {})[mode] = row
            print(f"{mode:12s}" + "".join(cells))

    # The harness is trusted only if 'train' mode reproduces the known numbers under some protocol.
    harness_ok = None
    if expected and "train" in args.modes:
        hits = [
            name
            for name, rows in results["rank1"].items()
            if name == "pooled"
            and matches_baseline({c: v["pct"] for c, v in rows["train"].items()}, expected, args.tolerance)
        ]
        harness_ok = bool(hits)
        results["baseline"] = {"expected": expected, "tolerance": args.tolerance, "reproduced_by": hits}
        want = ", ".join(f"{k} {v:.2f}" for k, v in expected.items())
        if hits:
            print(f"\nBASELINE OK: 'train' mode reproduces {want} under: {', '.join(hits)}")
            print("  Compare the other modes with 'train' in that same protocol.")
        else:
            print(f"\nBASELINE NOT REPRODUCED: 'train' mode doesn't give {want} in the pooled protocol.")
            print("  The harness doesn't match the training-side evaluation (preprocessing, protocol or")
            print("  data). Don't use the threshold numbers below; send this output back to fix the script.")

    if args.calibrate_mode in feats:
        f = feats[args.calibrate_mode]
        calib = {}
        for label, views in (("all views", VIEWS), ("090 only", ["090"])):
            genuine, impostor = genuine_impostor(f, meta, views)
            if genuine.size == 0 or impostor.size == 0:
                continue
            calib[label] = {
                "genuine": describe(genuine),
                "impostor": describe(impostor),
                "table": fmr_table(genuine, impostor),
            }
            print(f"\nThreshold, mode '{args.calibrate_mode}', NM vs NM same view, {label}")
            print(f"  genuine   {show(calib[label]['genuine'])}")
            print(f"  impostor  {show(calib[label]['impostor'])}")
            print("  threshold   false-match   genuine accepted")
            for r in calib[label]["table"]:
                print(f"  {r['threshold']:9.2f}   {r['fmr']:9.2%}   {r['genuine_accepted']:9.2%}")
        results["threshold"] = {"mode": args.calibrate_mode, "trusted": harness_ok is True, **calib}
        if harness_ok is not True:
            print("\n(threshold numbers above are UNTRUSTED: the baseline check did not pass or did not run)")

    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
