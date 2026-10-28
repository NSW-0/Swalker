"""CASIA-B check for the website pipeline: preprocessing Rank-1 (HANDOFF §6.1) + threshold calibration.

Made to run on Kaggle, where CASIA-B and the GPU are. It needs only torch, numpy and OpenCV
(no FastAPI, no ultralytics). In a Kaggle notebook:

    !git clone -b nSawafta/backend https://github.com/NSW-0/Swalker.git
    %cd Swalker/backend
    !python -m scripts.casia_eval \\
        --root /kaggle/input/<casia-b dataset>/<folder with 001..124> \\
        --checkpoint /kaggle/input/datasets/bbiiggnn/gait-checkpoints/stage2_iter70000.pth \\
        --out /kaggle/working/casia_eval.json

(If the repo is private, upload the backend/ folder as a Kaggle dataset and cd into it instead.)

Expected layout: <root>/<subject>/<cond-seq>/<view>/*.png, e.g. 075/nm-05/090/075-nm-05-090-031.png,
the standard CASIA-B silhouette release (whole 320x240 frames).

What it reports, for the 50 LT test subjects (075-124 by default):

1. Rank-1 for each preprocessing mode, standard protocol: gallery NM 01-04, probes NM 05-06 /
   BG 01-02 / CL 01-02, all probe views x all other gallery views, identical view excluded.
     train  - whole frame resized to 64x44, bilinear, greyscale (how HANDOFF §6.1 describes
              training; check it against casia_b_loader.py)
     casia  - the website's default: mask re-framed the CASIA-B way (app/ml/silhouettes.py)
     crop   - the old silhouette_extractor.py tight crop
   If casia is close to train, the website's framing is fine. If it's clearly lower, fix it.
   The CASIA-B mask stands in for the YOLO mask here; YOLO's own errors are not measured.

2. Genuine vs impostor distances (NM probes vs NM gallery, same view, like the website where both
   clips come from the same camera), and the threshold at several false-match rates. This is a
   starting value for SWALKER_MATCH_THRESHOLD; recheck it on the team's own recordings.
"""

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from app.ml.silhouettes import casia_silhouette, crop_silhouette

VIEWS = [f"{v:03d}" for v in range(0, 181, 18)]
GALLERY = ("nm", (1, 2, 3, 4))
PROBES = {"NM": ("nm", (5, 6)), "BG": ("bg", (1, 2)), "CL": ("cl", (1, 2))}
SEQ_DIR = re.compile(r"^(nm|bg|cl)-(\d{2})$")
FALSE_MATCH_RATES = (0.001, 0.01, 0.05, 0.10)


def train_style(frame: np.ndarray) -> np.ndarray | None:
    """Whole frame squashed to 64x44, bilinear, greyscale."""
    if frame.max() == 0:
        return None
    return cv2.resize(frame, (44, 64), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def as_mask(frame: np.ndarray) -> np.ndarray:
    return ((frame > 127) * 255).astype(np.uint8)


MODES = {
    "train": train_style,
    "casia": lambda f: casia_silhouette(as_mask(f)),
    "crop": lambda f: crop_silhouette(as_mask(f)),
}


def find_sequences(root: Path, subjects: range) -> list[dict]:
    seqs = []
    for sid in subjects:
        sdir = root / f"{sid:03d}"
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
    return seqs


def rank1(feats, meta, probe, max_gallery_views=None) -> tuple[float, dict]:
    """Mean Rank-1 over probe views, each averaged over gallery views != probe view."""
    sid = np.array([m["sid"] for m in meta])
    cond = np.array([m["cond"] for m in meta])
    seq = np.array([m["seq"] for m in meta])
    view = np.array([m["view"] for m in meta])
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


def distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    sq = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(sq, 0))


def genuine_impostor(feats, meta, views=VIEWS) -> tuple[np.ndarray, np.ndarray]:
    """NM probe vs NM gallery distances at the same view."""
    sid = np.array([m["sid"] for m in meta])
    cond = np.array([m["cond"] for m in meta])
    seq = np.array([m["seq"] for m in meta])
    view = np.array([m["view"] for m in meta])
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
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--calibrate-mode", default="casia", choices=list(MODES), help="mode used for the threshold")
    parser.add_argument("--subjects", default="75-124", help="test subject range, e.g. 75-124")
    parser.add_argument("--max-frames", type=int, default=0, help="cap frames per sequence (0 = whole sequence)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N sequences (quick smoke test)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("casia_eval.json"))
    args = parser.parse_args()

    import torch

    from app.ml.embed import GaitEmbedder

    lo, hi = (int(x) for x in args.subjects.split("-"))
    seqs = find_sequences(args.root, range(lo, hi + 1))
    if not seqs:
        raise SystemExit(f"no sequences under {args.root} (expected <root>/075/nm-01/000/*.png)")
    if args.limit:
        seqs = seqs[: args.limit]
    print(f"{len(seqs)} sequences, subjects {lo}-{hi}, modes {args.modes}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    embedder = GaitEmbedder(args.checkpoint, device)
    print(f"model: {embedder.checkpoint_name} on {device}")

    feats = {m: [] for m in args.modes}
    meta, skipped = [], 0
    start = time.time()
    for i, s in enumerate(seqs):
        frames = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in sorted(s["dir"].glob("*.png"))]
        frames = [f for f in frames if f is not None]
        if args.max_frames:
            frames = frames[: args.max_frames]
        per_mode = {}
        for mode in args.modes:
            sils = [x for x in (MODES[mode](f) for f in frames) if x is not None]
            if not sils:
                break
            per_mode[mode] = embedder.embed(np.stack(sils))
        if len(per_mode) != len(args.modes):
            skipped += 1  # CASIA-B has a few empty or missing sequences
            continue
        for mode, vec in per_mode.items():
            feats[mode].append(vec)
        meta.append({k: s[k] for k in ("sid", "cond", "seq", "view")})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(seqs)}  {time.time() - start:.0f}s")
    print(f"embedded {len(meta)} sequences ({skipped} skipped) in {time.time() - start:.0f}s")

    feats = {m: np.stack(v) for m, v in feats.items()}
    results = {"checkpoint": embedder.checkpoint_name, "subjects": args.subjects, "sequences": len(meta), "rank1": {}}

    print("\nRank-1 (%), identical view excluded")
    print(f"{'mode':8s}" + "".join(f"{c:>8s}" for c in PROBES))
    for mode in args.modes:
        row = {}
        for cond, probe in PROBES.items():
            mean, per_view = rank1(feats[mode], meta, probe)
            row[cond] = {"mean": mean, "per_view": per_view}
        results["rank1"][mode] = row
        print(f"{mode:8s}" + "".join(f"{row[c]['mean'] * 100:8.2f}" for c in PROBES))

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
        results["threshold"] = {"mode": args.calibrate_mode, **calib}

    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
