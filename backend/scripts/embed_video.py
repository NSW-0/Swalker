"""Embed one clip twice and check the signature is 4096-d and identical both times.

python -m scripts.embed_video path/to/walk.mp4 [--save-silhouettes out_dir]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from app.gallery.store import encode_png
from scripts._common import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--save-silhouettes", type=Path, help="write the display silhouettes as PNGs here")
    args = parser.parse_args()

    pipeline = build_pipeline()
    runs = []
    for i in (1, 2):
        start = time.perf_counter()
        walk = pipeline.analyse(args.video)
        runs.append(walk)
        print(
            f"run {i}: {time.perf_counter() - start:.1f}s  frames={walk.frames_total} "
            f"with_person={walk.frames_with_person} used={walk.frames_used}"
        )

    a, b = runs[0].vector, runs[1].vector
    print(f"shape={a.shape} dtype={a.dtype} l2_norm={np.linalg.norm(a):.4f}")
    print(f"identical={np.array_equal(a, b)} max_abs_diff={np.abs(a - b).max():.3g}")

    if args.save_silhouettes:
        args.save_silhouettes.mkdir(parents=True, exist_ok=True)
        for i, sil in enumerate(runs[0].display):
            (args.save_silhouettes / f"{i:02d}.png").write_bytes(encode_png(sil))
        print(f"saved {len(runs[0].display)} silhouettes to {args.save_silhouettes}")


if __name__ == "__main__":
    main()
