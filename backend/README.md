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
uvicorn app.main:create_app --factory --port 8000
```

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
| `SWALKER_PREPROCESS_MODE` | `casia` | `casia` = CASIA-B framing, greyscale (HANDOFF §6.1); `crop` = the old tight crop |
| `SWALKER_TARGET_FPS` | `25` | clips are resampled to CASIA-B's frame rate first, so 60 fps phone video isn't slow motion to the model |
| `SWALKER_MAX_FRAMES` | `120` | consecutive frames fed to the model (no subsampling, §6.2) |
| `SWALKER_MATCH_THRESHOLD` | `30` | **provisional**, see below |

## Scripts

Run them from `backend/` with the venv active.

| Command | What it does |
|---|---|
| `python -m scripts.embed_video clip.mp4 --save-silhouettes out\` | embeds a clip twice: shape, norm, identical? |
| `python -m scripts.casia_eval --root <CASIA-B> --checkpoint <pth>` | on Kaggle: Rank-1 for `train` / `casia` / `crop` silhouettes (HANDOFF §6.1) plus genuine/impostor distances and a threshold table. Instructions at the top of the script |
| `python -m scripts.calibrate_threshold videos\` | `videos\<person>\*.mp4`, at least 2 people × 2 clips. Prints genuine/impostor distances and the threshold at several false-match rates |
| `python -m scripts.import_gallery_pkl gallery.pkl --preprocess-mode crop` | imports an old `gallery.pkl` |

## Not done yet / not measured

- **Threshold:** 30 is a placeholder. Get a first value from `casia_eval` on Kaggle, check it with
  `calibrate_threshold` on the team's recordings, put it in `.env`, and set
  `SWALKER_THRESHOLD_CALIBRATED=true`. Until then the Verdict
  page says the match line is provisional.
- **Preprocessing:** `casia` follows HANDOFF §6.1, but its Rank-1 against `crop` on CASIA-B has
  **not been measured** yet (`casia_eval` does it). The framing numbers in `app/ml/silhouettes.py` come from HANDOFF; replace
  them with `CCPG_MATCH_CASIA_GEOMETRY` from the Kaggle `casia_b_loader.py`.
- **Progress:** identify is a single synchronous request. Job polling (HANDOFF §7.3) is still to do.
- `app/ml/combined_glfe_mstfe.py` is a frozen copy pinned to the checkpoint. Don't edit it here.
