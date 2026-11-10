# Changelog

What changed in Swalker, newest first, one line per change with the PR that brought it to `main`.
Measurements are given with what they were measured on. Add an entry with every PR (see CLAUDE.md).

## 2026-11-10

- **Project summary and READMEs** (#26): `PROJECT_SUMMARY.md`, how the website was built: the method, every pipeline
  step and why, how CASIA-B was used, the measurements and the limits. All three READMEs rewritten
  (the frontend one still said there was no backend; the backend one described one vector per
  person). `PROJECT_STATUS.md` and a note at the top of `HANDOFF.md` brought up to date.

## 2026-11-05

- **Phone layout** (#25): navigation in a bar along the bottom, every page fits the width and
  scrolls inside, dialogs use more of the screen. Checked in headless Edge at 320, 360, 390 and 430 px
  wide, a phone on its side and a small tablet: nothing wider than the screen in 78 captures. All in one
  screen-size block in `index.css`. PC at 7 window sizes, 70 captures compared before and after: only the
  random case number's tag moved, 40 of 35,812 element lines.
- **Match line 8, was 7.2** (#24), the team's choice to accept more of the right people. New people,
  right person accepted / stranger matched: side view normal 98 / 38, bag 88 / 8, coat 96 / 8 (50 clips
  each); all angles normal 96 / 16.55, bag 80 / 7.45, coat 86 / 4.18 (550 clips each). Not checked on
  phone video, still marked provisional.
- **`DATASET.md`** (#24): what CASIA-B holds, how it is split and what was measured on it. Docs
  brought up to date.
- **Raw CASIA-B videos into the Archive** (#23, `scripts/casia_videos.py`): vectors for all 110 walks
  of each of the 124 people through the real YOLO pipeline (resumable cache), an Archive with a first
  name per person (001 = Ahmad, 002 = Omar) and half of the bag and coat recordings, the other half
  (bg-02, cl-02) copied to `datasets/tests`, a report of what the dataset holds per person
  (`datasets/dataset_report.md`), and measurements that keep people from training (001-074) apart
  from new people (075-124). The old Archive goes to the Recycle Bin. Case file silhouettes load
  lazily.
- **Archive built from all 124 people** (#23): 10,909 clips, 2,728 test videos in `datasets/tests`.
  3 of the 13,640 walks could not be read (109-nm-01 at 126, 144 and 162 degrees). New people
  (075-124), all angles, 550 clips each, Rank-1 normal / bag / coat: 99.64 / 92.91 / 62.18 with normal
  clips only, 99.82 / 96.55 / 96.91 with bg-01 and cl-01 added.
- **Match line 7.2, was 8.5** (#23): at 8.5 this larger Archive gave 21.82% of strangers a name; 1% is
  at 7.17 (all angles, 1650 clips). At 7.2, new people, right person accepted / stranger matched: side
  view normal 94 / 6, bag 84 / 0, coat 90 / 0 (50 clips each); all angles normal 90.55 / 2.73, bag
  63.27 / 0.91, coat 70.18 / 0.18 (550 clips each). Not checked on phone video, still marked
  provisional. `.env.example` also named the wrong default framing (`casia`, now `casia_clip`).

## 2026-10-30

- **Changelog** (this file) and the rule in CLAUDE.md to keep it updated (#22).
- **Add clip from a case file goes straight to that person** (#21): no New / Existing choice or
  person picker when "Add clip" is pressed inside a case file.
- **Several clips per person in the Archive** (#20). A person can have a normal clip, a bag clip,
  a coat clip…; every clip is compared and each person counts once, by their closest clip. Add
  Case File chooses New / Existing person; a case file lists its clips (remove, "Add clip"); the
  Verdict page can save a walk as another clip of someone; the lineup says which clip matched.
  `index.json` version 2; older Archives are read as one clip per person. Not yet measured on real
  video whether it helps.
- **Case file name optional** (#19): left empty, the clip's file name is used
  (`hamed_bag (1).mp4` → `hamed_bag 1`). Verdict distance marker is a thick line instead of a dot.
- **Verdict distance scale fits real distances** (#18): it used to end at twice the match line, so
  real distances (10–19) piled up at the end. Now it ends just past the farthest suspect.
- **First real recordings measured** (notes only, in #18/#19): 10 people, bag/coat test clips.
  Bag/coat distances to the right person (11–12.7) overlap with distances between different
  people (from 9.8), so no threshold separates them. Score normalisation tried and rejected.

## 2026-10-29

- **Silhouettes drawn like the training loader** (#17): `casia_clip` version 2, a binary 320×240
  frame shrunk with PIL bilinear. CASIA-B pooled Rank-1 NM / BG / CL 97.82 / 90.07 / 63.91 (was
  97.82 / 90.06 / 56.64; training framing 99.27 / 93.26 / 72.36). Match threshold 8.5 (1% false
  matches at the side view on CASIA-B, 96.50% of same-person pairs accepted).
- **Eval modes that locate the gap to training** (#16): `native*`, `native_bin`, `clip_pil` in
  `casia_eval`. Found the drawing cost ~8 CL points; binary masks cost nothing.
- **Demo launcher** (#13, #14, #15): `start.bat` starts backend + frontend and opens the site; a
  Swalker shortcut with the detective-duck icon (`frontend/src/assets/swalker.ico`). CLAUDE.md
  brought up to date; `SESSION_NOTES.md` for handing over between sessions.
- **Clip-level framing** (#12): one scale and position per clip (`casia_clip` v1), keeping the
  walker's movement across the frame. CASIA-B pooled Rank-1 97.82 / 90.06 / 56.64 (per-frame
  framing was 96.09 / 78.05 / 35.73). First threshold 8.2.

## 2026-10-28

- **CASIA-B eval reads files faster** (#11): each PNG read in one call (Kaggle's disk is a
  network mount).
- **CASIA-B eval resumes and runs in parallel** (#10).
- **Eval reproduces training exactly** (`1bd0395`, committed straight to `main`): `train` mode
  uses PIL like `casia_b_loader`, and the `pooled` protocol; gives 99.27 / 93.26 / 72.36.
- **Geometry port, self-checking eval, identify as a job** (#9): CASIA-B framing from the training
  code, `casia_eval` checks the baseline first, the Verdict loader follows real job progress,
  About page shows the corrected accuracy.
- **Backend** (#7, #8): FastAPI + PyTorch backend (GLTNet loaded once, YOLOv8-seg, walker
  tracking, 25 fps resampling, Archive storing vectors and silhouettes but never videos,
  threshold matching), frontend wired to it and all mock data removed, Verdict page redesigned,
  rename/delete in the Archive, CI for the backend.

## Before

- 2026-10-19: handoff document for the backend work (`HANDOFF.md`), output page (#6).
- 2026-09-17 to 09-19: the frontend: nav bar, demo pages, details, Archive storage and editing,
  About page (#1–#5).
