# Swalker backend

FastAPI service around GLTNet. Takes a walking clip, resamples it to 25 fps, finds the walker with
YOLOv8-seg, frames each mask as a 64×44 CASIA-B style silhouette, encodes up to 120 consecutive
silhouettes into a 4096-number gait signature, and compares it with every clip in the Archive. Why
each step is done that way: `../PROJECT_SUMMARY.md` §4.

## Setup (Windows, cmd)

```
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dev.txt
copy .env.example .env
```

Use `--index-url https://download.pytorch.org/whl/cpu` instead if there's no NVIDIA GPU.

Weights (git-ignored, never commit them):
- `models/stage2_iter70000.pth`, from the Kaggle dataset `bbiiggnn/gait-checkpoints`.
- `models/yolov8n-seg.pt` is downloaded automatically on first start.

## Run

```
python -m uvicorn app.main:create_app --factory --port 8000
```

Use `python -m uvicorn`, not plain `uvicorn`: a `uvicorn` installed globally would run your global
Python, which may not have the backend's packages (`No module named 'cv2'`).

Then `cd frontend` and run `npm run dev`. Vite proxies `/api` to port 8000. Check
http://localhost:8000/api/health: `modelLoaded` should be `true`. Or run `start.bat` in the repo
folder, which starts both.

## Layout

```
app/
  main.py          app factory; loads YOLO and GLTNet once at startup
  config.py        settings (SWALKER_* in .env)
  api/             routes: health, gallery (the Archive), identify (as a job), uploads
  ml/              combined_glfe_mstfe.py (frozen copy), silhouettes.py (YOLO, tracking, framing),
                   embed.py (GLTNet), pipeline.py (video to signature)
  gallery/         store.py (the Archive on disk), matching.py (distances, closest clip per person)
  jobs.py          identify jobs and their progress
  queries.py       recent walks kept for 30 minutes, so one can be saved without uploading again
scripts/           evaluation, import and measurement scripts (below)
tests/             pytest; fake models, so no weights needed
```

## API

| Method | Path | Does |
|---|---|---|
| GET | `/api/health` | models loaded, device, Archive size, match line, upload limits |
| POST | `/api/identify` | upload a clip; returns a `jobId` straight away |
| GET | `/api/identify/{jobId}/status` | stage, progress, and the result when done |
| GET | `/api/gallery`, `/api/gallery/{id}` | the Archive, one case file with its clips |
| POST | `/api/gallery`, `/api/gallery/from-video` | new person from a walk just identified (`queryId`), or from a video |
| POST | `/api/gallery/{id}/clips`, `/api/gallery/{id}/clips/from-video` | another clip of someone |
| DELETE | `/api/gallery/{id}/clips/{clipId}` | remove a clip (not the last) |
| PATCH / DELETE | `/api/gallery/{id}` | rename, delete |

## Check

```
ruff check .
ruff format --check .
pytest
```

88 tests. They use fake models, so they run without weights. `tests/test_real_checkpoint.py` also
runs when `models/stage2_iter70000.pth` is present. CI runs all three on pull requests to `main`.

## What the Archive stores

`storage/gallery/` (git-ignored) holds, for every person, one or more clips (a normal walk, with a
bag, in a coat…):
- `features.npy`: one 4096-d vector per clip
- `index.json` (version 2): people and their clips (clip file name, date), plus the framing, its
  version, the checkpoint and the frame rate the Archive was built with
- `silhouettes/<person>/<clip>/*.png`: about 10 small silhouettes per clip, for display

Right now it holds the 124 CASIA-B people with 88 clips each, built by `scripts/casia_videos.py`
(`../DATASET.md`).

Uploaded videos are written to a temp file, processed, and deleted. They are never kept.

If you change `SWALKER_PREPROCESS_MODE`, `SWALKER_TARGET_FPS` or the checkpoint, the stored vectors no longer compare
with new ones. The server then reports `galleryCompatible: false` and refuses to match. Move
`storage/gallery` aside and re-add everyone.

## Settings

Every setting is optional; see `.env.example` and `app/config.py`. The important ones:

| Key | Default | Notes |
|---|---|---|
| `SWALKER_CHECKPOINT_PATH` | `models/stage2_iter70000.pth` | swap in a newer checkpoint here; `n_classes` is read from it |
| `SWALKER_DEVICE` | `auto` | `cpu` / `cuda` |
| `SWALKER_PREPROCESS_MODE` | `casia_clip` | `casia_clip` = CASIA-B framing with one scale and position per clip; `casia` = the same per frame; `crop` = the old tight crop. Rank-1 for each in `app/ml/silhouettes.py` |
| `SWALKER_TARGET_FPS` | `25` | clips are resampled to CASIA-B's frame rate first, so 60 fps phone video isn't slow motion to the model |
| `SWALKER_MAX_FRAMES` | `120` | consecutive frames fed to the model (no subsampling, §6.2) |
| `SWALKER_MATCH_THRESHOLD` | `8.0` | measured on raw CASIA-B videos, not yet checked on phone video, see below |

## Scripts

Run them from `backend/` with the venv active.

| Command | What it does |
|---|---|
| `python -m scripts.embed_video clip.mp4 --save-silhouettes out\` | embeds a clip twice: shape, norm, identical? |
| `python -m scripts.casia_eval --root <CASIA-B> --checkpoint <pth>` | on Kaggle: Rank-1 for each preprocessing against the training-side framing (HANDOFF §6.1) plus genuine/impostor distances and a threshold table. Instructions at the top of the script |
| `python -m scripts.casia_videos embed --root <datasets>`, then `evaluate`, then `build --root <datasets>` | raw CASIA-B videos into the Archive with bg-02 and cl-02 held out for testing, and the measurements. Instructions at the top of the script |
| `python -m scripts.calibrate_threshold videos\` | `videos\<person>\*.mp4`, at least 2 people × 2 clips. Prints genuine/impostor distances and the threshold at several false-match rates |
| `python -m scripts.import_gallery_pkl gallery.pkl --preprocess-mode crop` | imports an old `gallery.pkl` |

## Not done yet / not measured

- **Threshold:** 8.0, chosen by the team on 2026-11-05 to accept more of the right people, and
  measured with `casia_videos evaluate`: raw CASIA-B videos through the real YOLO pipeline, the Archive
  as built (124 people, about 88 clips each), test clips from people the model never trained on
  (075-124). At 8.0, right person accepted / stranger matched: side view normal 98 / 38, bag 88 / 8,
  coat 96 / 8 (50 clips each, so one clip is 2 points); all angles normal 96 / 16.55, bag 80 / 7.45, coat
  86 / 4.18 (550 clips each). A stranger is a test clip with its own person taken out of the comparison.
  A stricter line of 7.2 gives 1% of strangers overall. It is still provisional: phone video has not been
  measured against phone video at this Archive size. Check it with `calibrate_threshold` on the team's
  recordings, put the result in `.env`, and set `SWALKER_THRESHOLD_CALIBRATED=true`. Until then the
  Verdict page says the match line is provisional. The dataset is described in `../DATASET.md`.
- **Preprocessing:** measured on CASIA-B with `casia_eval` (2026-10-28/29, pooled Rank-1,
  NM / BG / CL): training-side framing 99.27 / 93.26 / 72.36, `casia_clip` 97.82 / 90.07 / 63.91,
  `casia` 96.09 / 78.05 / 35.73, `crop` 69.45 / 41.17 / 12.09. That uses CASIA-B's own masks;
  YOLO's mask errors on real video are not in these numbers.
- `app/ml/combined_glfe_mstfe.py` is a frozen copy pinned to the checkpoint. Don't edit it here.
