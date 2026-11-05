# Swalker backend

FastAPI service around GLTNet. Takes a walking clip, finds the person with YOLOv8-seg, turns each
frame into a 64×44 silhouette, encodes the sequence into a 4096-number gait signature, and compares
it with the Archive.

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
http://localhost:8000/api/health: `modelLoaded` should be `true`.

## Check

```
ruff check .
ruff format --check .
pytest
```

The tests use fake models, so they run without weights. `tests/test_real_checkpoint.py` also runs
when `models/stage2_iter70000.pth` is present.

## What the Archive stores

`storage/gallery/` (git-ignored) holds:
- `features.npy`: one 4096-d vector per person
- `index.json`: name, clip file name, date, preprocessing + checkpoint used
- `silhouettes/<person>/*.png`: about 10 small silhouettes for display

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
| `SWALKER_MATCH_THRESHOLD` | `7.2` | from raw CASIA-B videos, not yet checked on phone video, see below |

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

- **Threshold:** 7.2 comes from `casia_videos evaluate` (2026-11-05): raw CASIA-B videos through
  the real YOLO pipeline, the Archive as built (124 people, about 88 clips each), test clips from
  people the model never trained on (075-124). 1% of strangers get a name at 7.17 over all angles
  (1650 clips) and 7.16 at the side view (150 clips). At 7.2, right person accepted / stranger
  matched: side view normal 94 / 6, bag 84 / 0, coat 90 / 0 (50 clips each, so one clip is 2 points);
  all angles normal 90.55 / 2.73, bag 63.27 / 0.91, coat 70.18 / 0.18 (550 clips each). A stranger is a
  test clip with its own person taken out of the comparison. It is still provisional: phone video has
  not been measured against phone video in an Archive of this size. Check it with `calibrate_threshold` on
  the team's recordings, put the result in `.env`, and set `SWALKER_THRESHOLD_CALIBRATED=true`. Until
  then the Verdict page says the match line is provisional.
- **Preprocessing:** measured on CASIA-B with `casia_eval` (2026-10-28/29, pooled Rank-1,
  NM / BG / CL): training-side framing 99.27 / 93.26 / 72.36, `casia_clip` 97.82 / 90.07 / 63.91,
  `casia` 96.09 / 78.05 / 35.73, `crop` 69.45 / 41.17 / 12.09. That uses CASIA-B's own masks;
  YOLO's mask errors on real video are not in these numbers.
- `app/ml/combined_glfe_mstfe.py` is a frozen copy pinned to the checkpoint. Don't edit it here.
