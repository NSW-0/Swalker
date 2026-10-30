# Session notes: where the last Claude Code session stopped

Last updated 2026-10-30. Read this after `HANDOFF.md`. It covers work in progress, the next
steps, and things learned the hard way. For what is built overall, see `PROJECT_STATUS.md`.
Update this file at the end of each session.

## Current state

- **`main`** has everything up to PR #16: clip-level framing, `start.bat` with the duck icon
  (`frontend/src/assets/swalker.ico`; `start.bat` makes a git-ignored `Swalker.lnk`), and the
  diagnostic eval modes. `1bd0395` went to `main` without a PR; it was decided to leave it.
- **Branch `nSawafta/clip-pil-default`** (PR not merged yet as of 2026-10-29): makes
  `casia_clip` draw like training (`casia_clip_frames`, `PREPROCESS_VERSIONS` 2, so Archives
  built with version 1 are flagged) and sets the threshold to 8.5 from its table.
- **Branch `nSawafta/distance-scale`** (PR not opened as of 2026-10-30): the Verdict distance
  scale stretches to the farthest suspect instead of clamping at twice the match line. Not looked
  at in a browser yet; the user is checking it.
- **Backend checks:** 74 pytest tests pass and ruff is clean. Run them from `backend/` with
  `.venv\Scripts\python -m pytest` and `.venv\Scripts\python -m ruff check .`. The system Python
  has no ruff.
- **Local Archive** (`backend/storage/gallery/`) is empty, so changing the framing breaks nothing.

## First real recordings (2026-10-30)

Archive: 10 people, one normal side-view walk each (`C:\Users\nabee\Pictures\gait\`). Test clips
in `C:\Users\nabee\Pictures\gait_test\`: `hamed_bag`, `hoda_bag`, `Nabeel_CL`. Measured with the
real pipeline (`casia_clip` v2):

- Archive vs Archive (all different people): min 9.80 (Hodayfah-Ibrahim), median 12.66, max 18.60.
- hamed_bag: Hamed 12.66 (#1, next Raid 14.52). Nabeel_CL: Nabeel 11.24 (#1, next Mostafa
  12.30). hoda_bag: Mohamad 10.53 (#1, **wrong**), Hodayfah 10.97 (#2).
- So bag/coat distances to the right person (11-12.7) overlap with distances between different
  people (from 9.8): no threshold separates them. **Not measured yet:** normal-vs-normal of the
  same person on real video (asked the user for a second normal clip of 4-5 people).
- The user set `SWALKER_MATCH_THRESHOLD=12` in the git-ignored `backend/.env` to try something;
  the committed default is still 8.5. Delete that `.env` line to go back.
- Tried and rejected (2026-10-30): judging each distance against that person's usual distance
  to the others (z-norm, or ratio to their nearest other). hoda_bag stays wrong (Mohamad -1.00
  vs Hodayfah -0.99 z-norm; rank 4 by ratio). The Archive by then also held `Nabeel_Cl` (the coat
  clip added as its own case file), so Nabeel_CL can't be used as a test clip any more.
- Options given to the user (2026-10-30), in order: (1) second normal clip + second bag clip
  for 4-5 people, to measure normal-vs-normal and to test (2); (2) several clips per person in
  the Archive, matched by the closest clip and shown under one name; (3) a "possible match"
  band between the line and ~12 when #1 leads #2 clearly (margin needs data: right answers led
  by 1.86 and 1.06, the wrong one by 0.44); (4) consistent recording setup. Retraining is out of
  scope (CLAUDE.md).

## How the website's framing was chosen (Kaggle runs 2026-10-28/29)

`casia_eval` on CASIA-B test subjects 075-124; `train` reproduces the training numbers exactly
(`BASELINE OK`). Pooled Rank-1 NM / BG / CL:

| mode | NM | BG | CL | what it is |
|---|---|---|---|---|
| train | 99.27 | 93.26 | 72.36 | the training loader; whole CASIA frame, can't be used on real video |
| native_bin | 99.27 | 93.26 | 72.36 | train from a binarised mask: identical, CASIA-B PNGs are binary |
| native | 97.27 | 91.79 | 64.27 | real geometry, website's old drawing (linear warp + area average) |
| native_pos | 98.00 | 89.97 | 63.73 | old drawing + per-clip scale |
| native_scale | 97.73 | 91.25 | 61.55 | old drawing + per-clip position |
| **clip_pil** | **97.82** | **90.07** | **63.91** | **website now (`casia_clip` v2)**: per-clip scale + position, training's drawing |
| casia_seq_x | 97.82 | 90.06 | 56.64 | `casia_clip` v1: per-clip scale + position, old drawing |
| casia | 96.09 | 78.05 | 35.73 | per-frame framing |
| casia_seq | 92.55 | 76.40 | 40.18 | per-clip scale, each frame centred |
| crop | 69.45 | 41.17 | 12.09 | old `silhouette_extractor.py` crop |

Per-probe, all views (closest to the website): train 99.82 / 98.18 / 82.91, clip_pil
99.64 / 95.99 / 74.45, casia_seq_x 99.36 / 96.81 / 66.91.

The remaining ~8.5 CL points come from normalising each clip's scale and position, which real
video needs (size and place in the frame depend on the camera). These numbers use CASIA-B's own
masks; YOLO's mask quality on real video is **not** measured by them.

## Next steps, in order

1. **Merge `nSawafta/clip-pil-default`** (the user opens/merges the PR).
2. Threshold for `clip_pil` (= `casia_clip` v2), NM vs NM same view, from the 2026-10-29 run:

   | views | 0.1% false match | 1% | 5% | 10% |
   |---|---|---|---|---|
   | all 11 | 8.27 (86.63% genuine accepted) | 9.41 (95.00%) | 10.55 (97.71%) | 11.18 (98.40%) |
   | 090 (side) | 7.47 (93.00%) | 8.52 (96.50%) | 9.65 (98.75%) | 10.29 (99.75%) |

   Set to 8.5 (1% at the side view, the view the recording guide asks for).
   `threshold_calibrated` stays False until it's checked on real recordings with
   `scripts/calibrate_threshold`. (The old 8.2 was for `casia_seq_x`.)
3. The framing question (HANDOFF §6.1) is closed unless real recordings show a problem. Any
   further `casia_eval` experiment: add a mode to `MODES` and put it in `DEFAULT_MODES` with
   `train` (needed for the baseline check); a run is ~10 min per mode.
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
- **Eval mode names vs website modes:** `clip_pil` in the script = `casia_clip` version 2 on the
  website (`casia_clip_frames`); `casia_seq_x` = version 1 (`casia_sequence(..., follow_x=True)`).
- **The silhouette drawing matters as much as the geometry.** The model was trained on PIL
  BILINEAR shrinks of binary 320x240 frames; drawing the same shape with area averaging cost
  ~7 CL points. Keep any new framing drawn through `casia_clip_frames`'s PIL path.
- **The Archive records preprocess mode + version + checkpoint + fps.** Changing any of them
  flags existing Archives incompatible (rebuild needed). Bump `PREPROCESS_VERSIONS` in
  `silhouettes.py` when a framing's output changes.
- **A full `casia_eval` run** takes ~10 min of compute per mode plus reading (seconds when
  Kaggle's disk is warm, over an hour when cold).
- **The user works in cmd on Windows.** Give cmd-compatible git commands, stage files by name,
  and never commit `backups/`, `*.pth` / `*.pt` or storage. `frontend/src/assets/duckNoGem.png`
  and `wing.png` are the user's untracked files; leave them alone unless asked.
