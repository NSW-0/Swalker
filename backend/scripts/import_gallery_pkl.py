"""Import a gallery.pkl made by the old gallery_builder.py into the Archive.

    python -m scripts.import_gallery_pkl gallery.pkl --preprocess-mode crop

The pickle holds no record of how its vectors were made, so --preprocess-mode is required.
The old scripts used "crop" and kept every frame (no frame-rate resampling, --target-fps 0).
If that doesn't match the server's settings, the server will report the Archive as
incompatible and refuse to match against it, by design.
Only import pickles you made yourself: unpickling runs code from the file.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.errors import ApiError
from app.gallery.store import GalleryMeta, GalleryStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pkl", type=Path)
    parser.add_argument("--preprocess-mode", required=True, choices=["casia_clip", "casia", "crop"])
    parser.add_argument("--checkpoint", default=None, help="checkpoint file name (default: the configured one)")
    parser.add_argument("--target-fps", type=float, default=0.0, help="frame rate the vectors were made at (0 = all)")
    args = parser.parse_args()

    settings = get_settings()
    with args.pkl.open("rb") as f:
        data = pickle.load(f)  # trusted: a local file made with gallery_builder.py

    checkpoint = args.checkpoint or settings.checkpoint_path.name
    meta = GalleryMeta(preprocess_mode=args.preprocess_mode, checkpoint=checkpoint, target_fps=args.target_fps)
    store = GalleryStore(settings.gallery_dir, meta)
    if not store.compatible:
        raise SystemExit(f"existing Archive was built with {store.meta}, not {meta}; import into an empty one")

    features = np.asarray(data["features"], np.float32)
    for person_id, vector, video in zip(data["person_ids"], features, data["video_paths"], strict=True):
        try:
            store.add(str(person_id), str(video), vector, silhouettes=[])
            print(f"added {person_id}")
        except ApiError as exc:
            print(f"skip {person_id}: {exc.message}")
    print(f"Archive now has {len(store)} case files ({meta})")


if __name__ == "__main__":
    main()
