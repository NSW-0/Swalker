# Session notes: where the last Claude Code session stopped

Last updated 2026-10-29. Read this after `HANDOFF.md`. It covers work in progress, the next
steps, and things learned the hard way. For what is built overall, see `PROJECT_STATUS.md`.
Update this file at the end of each session.

## Current state

- **`main`** has everything up to PR #13: clip-level framing `casia_clip` (website default),
  threshold 8.2, `start.bat`, the `native*` diagnostic modes. `1bd0395` went to `main` without a
  PR; it was decided to leave it.
- **`nSawafta/housekeeping`** still has one unmerged commit, `ea4e6c3` (duck icon: `swalker.ico`
  + `start.bat` makes a git-ignored `Swalker.lnk` shortcut). It needs a second PR.
- **Branch `nSawafta/clip-drawing`** (this session's last work, waiting for a Kaggle run):
  `silhouettes.casia_clip_frames` and the eval modes `native_bin` and `clip_pil`. **Not** the
  website default yet: switch only if the run shows it helps.
- **Backend checks:** 74 pytest tests pass and ruff is clean. Run them from `backend/` with
  `.venv\Scripts\python -m pytest` and `.venv\Scripts\python -m ruff check .`. The system Python
  has no ruff.
- **Local Archive** (`backend/storage/gallery/`) is empty, so changing the framing breaks nothing.

## Where the CL gap comes from (Kaggle run 2026-10-29)

Pooled Rank-1 NM / BG / CL; the `native*` modes undo parts of `casia_seq_x`
(`native_geometry` in `casia_eval.py`):

| mode | NM | BG | CL | differs from train by |
|---|---|---|---|---|
| train | 99.27 | 93.26 | 72.36 | |
| native | 97.27 | 91.79 | 64.27 | drawing only (binary mask, linear warp + area average) |
| native_pos | 98.00 | 89.97 | 63.73 | drawing + per-clip scale |
| native_scale | 97.73 | 91.25 | 61.55 | drawing + per-clip position |
| casia_seq_x | 97.82 | 90.06 | 56.64 | drawing + scale + position (= website) |

So the **drawing costs ~8 CL points** and is fixable. Scale alone costs ~0.5, position ~2.7,
both together ~7.6 (they interact); those can't be kept for real video.

## Next steps, in order

1. **Kaggle run on `nSawafta/clip-drawing`** (`git clone -b nSawafta/clip-drawing ...`, usual
   command; check the modes line says `['train', 'casia_seq_x', 'native_bin', 'clip_pil']`,
   ~40 min). The threshold table now defaults to `clip_pil`.
   - `native_bin` vs `train`: what binarising the mask alone costs (the website only has binary
     YOLO masks, so this part can't be fixed).
   - `clip_pil` vs `casia_seq_x`: what training's drawing (binary 320x240 frame, PIL BILINEAR)
     gains with the website's geometry.
   - If `clip_pil` wins: add it to `PREPROCESSORS` as the new default (e.g. `casia_clip` version 2
     in `PREPROCESS_VERSIONS`, which flags old Archives), set the threshold from its table,
     update README / PROJECT_STATUS / HANDOFF §2 numbers.
2. ~~Threshold run on Kaggle (casia_seq_x)~~ done 2026-10-29. Same Rank-1 as before, `BASELINE OK`. Threshold
   table for `casia_seq_x` (= `casia_clip`), NM vs NM same view:

   | views | 0.1% false match | 1% | 5% | 10% |
   |---|---|---|---|---|
   | all 11 | 7.92 (83.48% genuine accepted) | 9.07 (93.65%) | 10.17 (97.05%) | 10.79 (98.17%) |
   | 090 (side) | 7.31 (93.00%) | 8.23 (95.75%) | 9.42 (99.00%) | 10.05 (99.00%) |

   Genuine median 6.50, impostor median 13.21 (all views).
3. ~~Set the threshold for casia_seq_x~~ done: `match_threshold = 8.2` in `backend/app/config.py` (1% at the
   side view, the view the recording guide asks for). `threshold_calibrated` stays False until
   it's checked on real recordings with `scripts/calibrate_threshold`.
4. **Not started:** averaging several clips per person (HANDOFF §7.5), and the phone-width
   layout (every page is 454 px wide on a 390 px phone; the shared shell, not one page).
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
- **A full `casia_eval` run** takes ~10 min of compute per mode plus reading (seconds when
  Kaggle's disk is warm, over an hour when cold).
- **The user works in cmd on Windows.** Give cmd-compatible git commands, stage files by name,
  and never commit `backups/`, `*.pth` / `*.pt` or storage. `frontend/src/assets/duckNoGem.png`
  and `wing.png` are the user's untracked files; leave them alone unless asked.
