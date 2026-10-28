"""Measure genuine vs impostor distances to choose MATCH_THRESHOLD (HANDOFF §6.3).

Folder layout: one sub-folder per person, two or more clips each.

    videos/
      nabeel/walk1.mp4, walk2.mp4
      ahmad/walk1.mp4, walk2.mp4, walk3.mp4

    python -m scripts.calibrate_threshold videos/

Genuine = two different clips of the same person. Impostor = clips of two different people.
Only prints what it measured; the threshold to use is the team's call.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np

from app.errors import ApiError
from scripts._common import VIDEO_EXTENSIONS, build_pipeline

FALSE_MATCH_RATES = (0.0, 0.01, 0.05, 0.10)


def describe(name: str, d: np.ndarray) -> None:
    if d.size == 0:
        print(f"{name:9s} n=0")
        return
    p5, p50, p95 = np.percentile(d, [5, 50, 95])
    print(f"{name:9s} n={d.size:<5d} min={d.min():.2f} p5={p5:.2f} median={p50:.2f} p95={p95:.2f} max={d.max():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", type=Path)
    args = parser.parse_args()

    clips = [(p.parent.name, p) for p in sorted(args.videos.glob("*/*")) if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not clips:
        raise SystemExit(f"no clips found under {args.videos}/<person>/")

    pipeline = build_pipeline()
    people, vectors = [], []
    for person, path in clips:
        try:
            walk = pipeline.analyse(path)
        except ApiError as exc:
            print(f"skip {person}/{path.name}: {exc.message}")
            continue
        print(f"ok   {person}/{path.name}: {walk.frames_used} frames")
        people.append(person)
        vectors.append(walk.vector)

    genuine, impostor = [], []
    for i, j in itertools.combinations(range(len(vectors)), 2):
        d = float(np.linalg.norm(vectors[i] - vectors[j]))
        (genuine if people[i] == people[j] else impostor).append(d)
    genuine, impostor = np.array(genuine), np.array(impostor)

    print(f"\n{len(set(people))} people, {len(vectors)} clips")
    describe("genuine", genuine)
    describe("impostor", impostor)
    if genuine.size == 0 or impostor.size == 0:
        raise SystemExit("need at least two people with two clips each")

    print("\nthreshold    false-match rate   genuine accepted")
    imp_sorted = np.sort(impostor)
    for fmr in FALSE_MATCH_RATES:
        # Largest threshold that lets through at most `fmr` of the impostor pairs.
        k = int(np.floor(fmr * imp_sorted.size))
        threshold = imp_sorted[k] - 1e-6 if k < imp_sorted.size else imp_sorted[-1]
        actual_fmr = float((impostor <= threshold).mean())
        gar = float((genuine <= threshold).mean())
        print(f"{threshold:9.2f}    {actual_fmr:6.1%} (target {fmr:.0%})   {gar:6.1%}")


if __name__ == "__main__":
    main()
