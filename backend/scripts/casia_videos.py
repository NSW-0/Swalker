"""Raw CASIA-B videos -> the Archive, with recordings held out for testing.

Every walk (nm-01..06, bg-01..02, cl-01..02, 11 camera angles, 110 per person) goes through the
same pipeline as an upload: YOLO masks, not CASIA-B's ready-made silhouettes. The bkgrd videos
have nobody in them and are skipped. Three steps, run from backend/:

    python -m scripts.casia_videos embed --root C:\\Users\\nabee\\Desktop\\datasets
    python -m scripts.casia_videos evaluate
    python -m scripts.casia_videos build --root C:\\Users\\nabee\\Desktop\\datasets

embed     computes every clip's vector and display silhouettes into --cache, one file per person:
          ~1.25 s a clip on a GPU, ~4.7 hours for everyone. Run it again and it carries on where
          it stopped. Nothing else is touched.
evaluate  measures from the cache, and only reads it. Same person: Rank-1 for the held-out
          recordings, and how many are accepted at the match line. Strangers: each test clip with
          its own person taken out of the Archive, which is what someone unknown looks like.
          People 001-074 were in the model's training set and 075-124 were not, so they are
          reported apart; only the second group's numbers say how well it recognises new people.
build     makes the Archive from the cache: "Person 001".. with every clip except the held-out
          recordings (bg-02 and cl-02, all angles), and copies those videos to --test-dir to try
          on the website. The previous Archive is moved aside, not deleted. The backend must be
          stopped, because it keeps the Archive in memory and would overwrite the new one.
"""

import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.errors import ApiError
from app.gallery.store import GalleryMeta, GalleryStore
from app.ml.silhouettes import PREPROCESS_VERSIONS

VIDEO_NAME = re.compile(r"^(\d{3})-(nm|bg|cl)-(\d{2})-(\d{3})\.avi$", re.IGNORECASE)
HELD_OUT = {("bg", 2), ("cl", 2)}  # not in the Archive: kept for testing
NORMAL_TEST_SEQ = 6  # nm-06 stands in for a new normal walk (left out of the comparison only)
LAST_TRAINED_SUBJECT = 74  # the LT protocol: the model was trained on 001-074
SIDE_VIEW = 90
TARGET_FALSE_MATCH = (0.001, 0.01, 0.05, 0.10)


@dataclass(frozen=True)
class Video:
    path: Path
    subject: int
    cond: str
    seq: int
    view: int

    @property
    def held_out(self) -> bool:
        return (self.cond, self.seq) in HELD_OUT


def parse_video(path: Path) -> Video | None:
    """None for anything that isn't a walk, like 001-bkgrd-090.avi."""
    m = VIDEO_NAME.match(path.name)
    if m is None:
        return None
    return Video(path, int(m[1]), m[2].lower(), int(m[3]), int(m[4]))


def find_videos(root: Path) -> dict[int, list[Video]]:
    """Per subject, sorted by file name. Only folders named video: --test-dir may be under root."""
    people: dict[int, list[Video]] = {}
    for path in sorted(root.rglob("*.avi"), key=lambda p: p.name.lower()):
        video = parse_video(path) if path.parent.name.lower() == "video" else None
        if video is not None:
            people.setdefault(video.subject, []).append(video)
    return people


def person_name(subject: int) -> str:
    return f"Person {subject:03d}"


def current_meta() -> dict:
    s = get_settings()
    return {
        "preprocess_mode": s.preprocess_mode,
        "preprocess_version": PREPROCESS_VERSIONS[s.preprocess_mode],
        "checkpoint": s.checkpoint_path.name,
        "target_fps": s.target_fps,
    }


def cache_file(cache: Path, subject: int) -> Path:
    return cache / f"{subject:03d}.npz"


def read_info(path: Path) -> dict:
    with np.load(path) as data:
        return json.loads(str(data["info"]))


def write_person(path: Path, meta: dict, names, vectors, display, failed: dict[str, str]) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")  # a run stopped mid-write leaves no half file
    np.savez_compressed(
        tmp,
        info=json.dumps({"meta": meta, "failed": failed}),
        names=np.array(names),
        vectors=np.stack(vectors).astype(np.float32),
        display=(np.stack(display) * 255).astype(np.uint8),
    )
    tmp.replace(path)


# ── embed ─────────────────────────────────────────────────────────────


def embed(args) -> None:
    from scripts._common import build_pipeline  # only this step needs torch and YOLO

    people = find_videos(args.root)
    if args.subjects:
        people = {s: people[s] for s in args.subjects if s in people}
    if not people:
        raise SystemExit(f"no CASIA-B videos found under {args.root}")
    args.cache.mkdir(parents=True, exist_ok=True)
    meta = current_meta()

    todo = []
    for subject in sorted(people):
        path = cache_file(args.cache, subject)
        if path.exists() and read_info(path)["meta"] == meta:
            continue
        todo.append(subject)
    clips = sum(len(people[s]) for s in todo)
    print(f"{len(people)} people, {len(people) - len(todo)} already done, {clips} clips to embed", flush=True)
    if not todo:
        return

    pipeline = build_pipeline()
    start, done = time.time(), 0
    for subject in todo:
        names, vectors, display, failed = [], [], [], {}
        for video in people[subject]:
            try:
                walk = pipeline.analyse(video.path)
            except ApiError as exc:  # e.g. the walker is barely in view
                failed[video.path.name] = exc.code
            else:
                names.append(video.path.name)
                vectors.append(walk.vector)
                display.append(np.stack(walk.display))
            done += 1
        if not names:
            print(f"{subject:03d}: no usable clips, {failed}", flush=True)
            continue
        write_person(cache_file(args.cache, subject), meta, names, vectors, display, failed)
        rate = (time.time() - start) / done
        print(
            f"{subject:03d}: {len(names)} clips, {len(failed)} failed {sorted(failed) or ''}"
            f"  [{done}/{clips}, {rate:.2f} s/clip, {(clips - done) * rate / 3600:.1f} h left]",
            flush=True,
        )


# ── reading the cache ─────────────────────────────────────────────────


@dataclass
class Clips:
    """Every cached clip, sorted by subject so one subject's rows are contiguous."""

    names: list[str]
    subject: np.ndarray
    cond: np.ndarray
    seq: np.ndarray
    view: np.ndarray
    vectors: np.ndarray


def make_clips(names: list[str], vectors: np.ndarray) -> Clips:
    videos = [parse_video(Path(n)) for n in names]
    return Clips(
        names,
        np.array([v.subject for v in videos]),
        np.array([v.cond for v in videos]),
        np.array([v.seq for v in videos]),
        np.array([v.view for v in videos]),
        np.asarray(vectors, np.float32),
    )


def load_clips(cache: Path) -> Clips:
    names, vectors = [], []
    for path in sorted(cache.glob("[0-9][0-9][0-9].npz")):
        with np.load(path) as data:
            names += [str(n) for n in data["names"]]
            vectors.append(data["vectors"])
    if not names:
        raise SystemExit(f"nothing in {cache}: run the embed step first")
    return make_clips(names, np.concatenate(vectors))


# ── evaluate ──────────────────────────────────────────────────────────


def nearest_per_person(test: np.ndarray, gallery: np.ndarray, owners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(tests, people) distance to each person's closest gallery clip, and the sorted people.
    owners must be sorted, as Clips rows are."""
    people, first_row = np.unique(owners, return_index=True)
    out = np.empty((len(test), len(people)), np.float32)
    g2 = (gallery**2).sum(axis=1)
    for start in range(0, len(test), 256):  # (tests, gallery) in one go would need gigabytes
        chunk = test[start : start + 256]
        squared = (chunk**2).sum(axis=1, keepdims=True) + g2 - 2 * chunk @ gallery.T
        out[start : start + 256] = np.minimum.reduceat(squared, first_row, axis=1)
    return np.sqrt(np.maximum(out, 0)), people


def score(clips: Clips, test: np.ndarray, gallery: np.ndarray, threshold: float) -> dict:
    """test and gallery are row masks. Same person: did the closest person turn out right, and was
    it under the line? Strangers: the same clips with their own person taken out."""
    d, people = nearest_per_person(clips.vectors[test], clips.vectors[gallery], clips.subject[gallery])
    n = len(d)
    if n == 0:
        return {"clips": 0}
    truth = np.searchsorted(people, clips.subject[test])
    row = np.arange(n)
    closest = d.argmin(axis=1)
    right = closest == truth
    genuine = d[row, truth]
    others = d.copy()
    others[row, truth] = np.inf
    stranger = others.min(axis=1)
    return {
        "clips": n,
        "rank1": 100 * float(right.mean()),
        "accepted": 100 * float((right & (genuine <= threshold)).mean()),
        "wrong_person_accepted": 100 * float((~right & (d[row, closest] <= threshold)).mean()),
        "stranger_matched": 100 * float((stranger <= threshold).mean()),
        "genuine": genuine,
        "stranger": stranger,
    }


def public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}


def threshold_table(genuine: np.ndarray, stranger: np.ndarray) -> list[dict]:
    """The line that lets a given share of strangers through, and what it accepts of the right people."""
    rows = []
    for rate in TARGET_FALSE_MATCH:
        line = float(np.quantile(stranger, rate))
        rows.append({"threshold": line, "strangers": 100 * rate, "same_person": 100 * float((genuine <= line).mean())})
    return rows


def evaluate_clips(clips: Clips, threshold: float) -> dict:
    held_out = np.array([(c, s) in HELD_OUT for c, s in zip(clips.cond, clips.seq, strict=True)])
    normal = clips.cond == "nm"
    normal_test = normal & (clips.seq == NORMAL_TEST_SEQ)
    unseen = clips.subject > LAST_TRAINED_SUBJECT
    side = clips.view == SIDE_VIEW
    bag, coat = (clips.cond == "bg") & held_out, (clips.cond == "cl") & held_out
    archive = ~held_out  # what build puts in the Archive
    tests = {"normal (nm-06)": normal_test, "bag (bg-02)": bag, "coat (cl-02)": coat}
    # Normal walks are all in the Archive, so nm-06 is left out here to act as a new walk.
    galleries = {
        "Archive with normal clips only": {name: normal & ~normal_test for name in tests},
        "Archive with normal + bg-01 + cl-01": {
            name: archive & ~normal_test if name == "normal (nm-06)" else archive for name in tests
        },
    }
    groups = {
        "new people (075-124, not in training)": unseen,
        "people from training (001-074)": ~unseen,
    }
    out: dict = {"threshold": threshold, "results": {}}
    for group, in_group in groups.items():
        for gallery_name, per_test in galleries.items():
            for test_name, test in tests.items():
                for views, in_view in {"all angles": np.ones_like(side), "side view": side}.items():
                    key = f"{group} | {gallery_name} | {test_name} | {views}"
                    out["results"][key] = public(
                        score(clips, test & in_group & in_view, per_test[test_name], threshold)
                    )

    # The line for the Archive as built, from the new people's clips only.
    for views, in_view in {"all angles": np.ones_like(side), "side view": side}.items():
        genuine, stranger = [], []
        for name, test in tests.items():
            r = score(clips, test & unseen & in_view, galleries["Archive with normal + bg-01 + cl-01"][name], threshold)
            if r["clips"]:
                genuine.append(r["genuine"])
                stranger.append(r["stranger"])
        if genuine:
            out[f"threshold table, {views}"] = threshold_table(np.concatenate(genuine), np.concatenate(stranger))
    return out


def print_report(report: dict) -> None:
    line = report["threshold"]
    print(f"match line {line}. accepted = right person and under the line.")
    print("stranger matched = someone not in the Archive wrongly gets a name.\n")
    print(f"{'':58s}{'clips':>7s}{'Rank-1':>9s}{'accepted':>10s}{'wrong':>8s}{'stranger':>10s}")
    last = None
    for key, r in report["results"].items():
        group, gallery, test, views = key.split(" | ")
        if (group, gallery) != last:
            print(f"\n{group}\n  {gallery}")
            last = (group, gallery)
        if not r["clips"]:
            continue
        print(
            f"    {test:18s}{views:36s}{r['clips']:7d}{r['rank1']:9.2f}{r['accepted']:10.2f}"
            f"{r['wrong_person_accepted']:8.2f}{r['stranger_matched']:10.2f}"
        )
    for key, rows in report.items():
        if key.startswith("threshold table"):
            print(f"\n{key}, new people, Archive with normal + bg-01 + cl-01")
            print(f"{'line':>8s}{'strangers matched':>20s}{'right person under line':>26s}")
            for row in rows:
                print(f"{row['threshold']:8.2f}{row['strangers']:19.1f}%{row['same_person']:25.2f}%")


def evaluate(args) -> None:
    clips = load_clips(args.cache)
    print(f"{len(clips.names)} clips, {len(set(clips.subject.tolist()))} people")
    threshold = args.threshold if args.threshold is not None else get_settings().match_threshold
    report = evaluate_clips(clips, threshold)
    print_report(report)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


# ── build ─────────────────────────────────────────────────────────────


def backend_is_running(port: int = 8000) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/api/health", timeout=2).close()
    except (urllib.error.URLError, OSError):
        return False
    return True


def build(args) -> None:
    if backend_is_running():
        raise SystemExit("The backend is running: close it first, it would overwrite the new Archive.")
    paths = sorted(args.cache.glob("[0-9][0-9][0-9].npz"))
    if not paths:
        raise SystemExit(f"nothing in {args.cache}: run the embed step first")
    settings = get_settings()
    meta = current_meta()
    expected = GalleryMeta(
        preprocess_mode=meta["preprocess_mode"],
        checkpoint=meta["checkpoint"],
        target_fps=meta["target_fps"],
        preprocess_version=meta["preprocess_version"],
    )

    building = settings.storage_dir / "gallery_building"
    shutil.rmtree(building, ignore_errors=True)  # left by a build that stopped; only this script writes it
    store = GalleryStore(building, expected)
    clips_added = 0
    for path in paths:
        with np.load(path) as data:
            info = json.loads(str(data["info"]))
            if info["meta"] != meta:
                raise SystemExit(f"{path} was embedded with {info['meta']}, the server uses {meta}: embed again")
            names = [str(n) for n in data["names"]]
            vectors, display = data["vectors"], data["display"].astype(np.float32) / 255.0
        keep = [i for i, n in enumerate(names) if not parse_video(Path(n)).held_out]
        clips = [(names[i], vectors[i], list(display[i])) for i in keep]
        store.add_many(person_name(int(path.stem)), clips, persist=False)
        clips_added += len(clips)
    store.save()

    gallery = settings.gallery_dir
    if gallery.exists():
        aside = gallery.with_name(f"gallery_before_casia_{datetime.now():%Y%m%d_%H%M%S}")
        gallery.rename(aside)
        print(f"previous Archive moved to {aside}")
    building.rename(gallery)
    print(f"Archive: {len(paths)} people, {clips_added} clips, {len(store.features)} vectors")

    if args.root:
        copied = 0
        for subject, videos in find_videos(args.root).items():
            if not cache_file(args.cache, subject).exists():
                continue
            folder = args.test_dir / person_name(subject)
            for video in videos:
                if video.held_out and not (folder / video.path.name).exists():
                    folder.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(video.path, folder / video.path.name)
                    copied += 1
        print(f"test clips: {copied} copied to {args.test_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=get_settings().storage_dir / "casia_cache")
    steps = parser.add_subparsers(dest="step", required=True)

    step = steps.add_parser("embed", help="compute every clip's vector into the cache")
    step.add_argument("--root", type=Path, required=True, help="folder holding the DatasetB-* folders")
    step.add_argument("--subjects", type=int, nargs="*", help="only these people, e.g. 1 75")
    step.set_defaults(run=embed)

    step = steps.add_parser("evaluate", help="measure from the cache")
    step.add_argument("--threshold", type=float, help="default: the server's match line")
    step.add_argument("--out", type=Path, default=get_settings().storage_dir / "casia_videos_results.json")
    step.set_defaults(run=evaluate)

    step = steps.add_parser("build", help="make the Archive from the cache")
    step.add_argument("--root", type=Path, help="to also copy the held-out videos to --test-dir")
    step.add_argument("--test-dir", type=Path, help="default: <root>/test_clips")
    step.set_defaults(run=build)

    args = parser.parse_args()
    if args.step == "build" and args.root and not args.test_dir:
        args.test_dir = args.root / "test_clips"
    args.run(args)


if __name__ == "__main__":
    main()
