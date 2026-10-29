# Session notes: where the last Claude Code session stopped

Last updated 2026-10-29. Read this after `HANDOFF.md`. It covers work in progress, the next
steps, and things learned the hard way. For what is built overall, see `PROJECT_STATUS.md`.
Update this file at the end of each session.

## Current state

- **Branch:** `nSawafta/eval-read-bytes`, pushed. The PR is **not opened yet**, on purpose: it
  waited for the threshold, which is now set, so it can be opened.
  Commits on it, not yet on `main`:
  - `fbd982b` reads each CASIA-B PNG in one call (it was slow on Kaggle's network mount)
  - `c56f3be` adds clip-level framing modes (`casia_seq`, `casia_seq_x`) to `casia_eval`
  - `f7b1f11` makes the website frame silhouettes per clip (`casia_clip`, now the default)
- **`main`** has `1bd0395` ("Match casia_eval's train mode to the training loader"). It was
  committed and pushed straight to `main` by mistake instead of through a PR. The user was told
  and asked whether to revert it and redo it as a PR. **No answer yet**, so ask again.
- **Backend checks:** 70 pytest tests pass and ruff is clean. Run them from `backend/` with
  `.venv\Scripts\python -m pytest` and `.venv\Scripts\python -m ruff check .`. The system Python
  has no ruff.
- **Local Archive** (`backend/storage/gallery/`) is empty, so changing the framing broke nothing.

## What was done in this session (the framing question, HANDOFF §6.1)

`backend/scripts/casia_eval.py` was run on Kaggle on CASIA-B test subjects 075–124. Its `train`
mode reproduces the training numbers exactly (pooled protocol: NM 99.27 / BG 93.26 / CL 72.36,
`BASELINE OK`), so the other rows can be trusted. Pooled Rank-1, NM / BG / CL:

| mode | NM | BG | CL | note |
|---|---|---|---|---|
| train | 99.27 | 93.26 | 72.36 | the training loader; whole CASIA frame, cannot be used on real video |
| casia_seq_x | 97.82 | 90.06 | 56.64 | **now the website's default, called `casia_clip`** |
| casia | 96.09 | 78.05 | 35.73 | per-frame framing, the old website default |
| casia_seq | 92.55 | 76.40 | 40.18 | one scale per clip, but each frame centred |
| crop | 69.45 | 41.17 | 12.09 | old `silhouette_extractor.py` crop |

In "per-probe, all views" (closest to how the website is used), `casia_seq_x` scores
99.36 / 96.81 / 66.91 and `train` 99.82 / 98.18 / 82.91. These numbers use CASIA-B's own masks;
YOLO's mask quality on real video is **not** measured by them.

## Next steps, in order

1. ~~Threshold run on Kaggle~~ done 2026-10-29. Same Rank-1 as before, `BASELINE OK`. Threshold
   table for `casia_seq_x` (= `casia_clip`), NM vs NM same view:

   | views | 0.1% false match | 1% | 5% | 10% |
   |---|---|---|---|---|
   | all 11 | 7.92 (83.48% genuine accepted) | 9.07 (93.65%) | 10.17 (97.05%) | 10.79 (98.17%) |
   | 090 (side) | 7.31 (93.00%) | 8.23 (95.75%) | 9.42 (99.00%) | 10.05 (99.00%) |

   Genuine median 6.50, impostor median 13.21 (all views).
2. ~~Set the threshold~~ done: `match_threshold = 8.2` in `backend/app/config.py` (1% at the
   side view, the view the recording guide asks for). `threshold_calibrated` stays False until
   it's checked on real recordings with `scripts/calibrate_threshold`.
3. **Open the PR** for `nSawafta/eval-read-bytes` into `main` (the user opens or merges it).
4. **Optional: close more of the CL gap** (56.64 vs 72.36). Ideas not tested yet:
   - Clean CASIA masks to the largest connected component before measuring the box, since
     background noise can throw off the size.
   - Scaling every clip to the same body height removes size differences between people. That
     can't be fixed for real video, where camera distance varies, so maybe accept the gap.
   - Each idea costs a full Kaggle run of about 2 hours. Add it as a new mode in `MODES`; keep
     the existing mode names, so the cache still resumes.
5. **Real recordings** (needs the team): record clips, run `scripts/calibrate_threshold`, build
   the Archive (`PROJECT_STATUS.md` → "Left to do").

## Things learned (don't rediscover them)

- **Training preprocessing** (`casia_b_loader.load_silhouette`, pasted by the user; not in the
  repo): PIL `convert('L')`, then `resize((44, 64), Image.BILINEAR)`, then `/255`. Blank frames
  are kept, and an empty sequence becomes `zeros((1, 64, 44))`. PIL BILINEAR averages when
  shrinking, while cv2 `INTER_LINEAR` does not; that difference alone broke the baseline.
  `CCPG_MATCH_CASIA_GEOMETRY` is False for CASIA-B; it exists only to make CCPG look like CASIA.
- **Training evaluation** (`train.py rank1_from_features`, `protocol='pooled'`) is what the
  99.27 / 93.26 / 72.36 come from. `casia_eval` implements it as `rank1_pooled`.
- **Kaggle `/kaggle/input` is a network mount.** Read each file whole (`path.read_bytes()`)
  before decoding; PIL's many small reads were 250× slower. A fresh session can still be slow
  reading (4244 s of the last 6587 s run), probably a cold cache.
- **The `casia_eval` cache** (`<out>_cache.npz`) is ignored when `CACHE_VERSION` or the list of
  modes changes. Bump `CACHE_VERSION` whenever an existing mode's output changes.
- **`casia_seq_x` in the script and `casia_clip` on the website are the same function**
  (`casia_sequence(..., follow_x=True)` in `backend/app/ml/silhouettes.py`). The script keeps the
  old name so its cache resumes.
- **The Archive records preprocess mode + version + checkpoint + fps.** Changing any of them
  flags existing Archives incompatible (rebuild needed). Bump `PREPROCESS_VERSIONS` in
  `silhouettes.py` when a framing's output changes.
- **The user works in cmd on Windows.** Give cmd-compatible git commands, stage files by name,
  and never commit `backups/`, `*.pth` / `*.pt` or storage. `frontend/src/assets/duckNoGem.png`
  and `wing.png` are the user's untracked files; leave them alone unless asked.
