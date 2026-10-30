# Changelog

What changed in Swalker, newest first, one line per change with the PR that brought it to `main`.
Measurements are given with what they were measured on. Add an entry with every PR (see CLAUDE.md).

## Not merged yet

- **Add clip from a case file goes straight to that person**: no New / Existing choice or person
  picker when "Add clip" is pressed inside a case file. Branch `nSawafta/add-clip-for-person`.

## 2026-10-30

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
