# Swalker: project status

Last updated 2026-10-28. What has been built, and what is still left. For how the model works and
the background behind each item, see `HANDOFF.md`. For running the backend, see `backend/README.md`.

## Done

### Backend (`backend/`): new, FastAPI + PyTorch
- **Models load once at startup.** GLTNet comes from a frozen copy of `combined_glfe_mstfe.py`
  pinned to `stage2_iter70000.pth`, loaded with `strict=True`, always in `eval()` and `no_grad()`.
  Only `feat` is used. YOLOv8n-seg also loads once. If loading fails, the API still starts and
  `/api/health` says why.
- **Video to silhouettes**, rebuilt from the old `silhouette_extractor.py`:
  - CASIA-B style framing by default (`casia_clip`), using the training code's
    `CCPG_MATCH_CASIA_GEOMETRY` numbers: height 50.1% of the frame, width squashed by 0.52, bbox
    top at 23.3%, centre at 51.4%, greyscale kept (HANDOFF §6.1). One scale and position is used
    for the whole clip, so each frame keeps the size changes, bobbing and movement across the
    frame that the camera saw. Per-frame framing (`casia`) and the old tight crop (`crop`) are
    still selectable. The Archive records the framing and its version.
  - Measured on CASIA-B (2026-10-28, `casia_eval`, pooled Rank-1 NM / BG / CL, CASIA-B's own
    masks): training-side framing 99.27 / 93.26 / 72.36, `casia_clip` 97.82 / 90.06 / 56.64,
    `casia` 96.09 / 78.05 / 35.73, `crop` 69.45 / 41.17 / 12.09.
  - Frames are fed in order and consecutive, up to 120. No more spreading 30 frames across the
    clip (§6.2).
  - **Walker tracking:** the same person is followed across frames by box overlap. Before, each
    frame took "the most confident person", so bystanders got mixed into the walk.
  - YOLO masks are taken at frame resolution (`retina_masks`), so they are no longer misaligned.
  - **Frame rate:** clips are resampled to 25 fps (CASIA-B's rate) before segmenting, so a 60 fps
    phone clip isn't fed as slow motion. The Archive records the frame rate, so Archives built
    before this change are flagged incompatible.
  - Reading stops after 450 frames, so long uploads can't take minutes.
- **Matching:** Euclidean distance, top 3, and a match threshold, so strangers come out as
  "Unknown" (§6.3). Raw distances are always returned. "Strength" is labelled as relative only (§6.4).
- **Archive storage** (`backend/storage/gallery/`, git-ignored): per person, a 4096-d vector, ~10
  small silhouette PNGs and metadata. **Videos are never stored**: uploads are deleted after
  processing. The Archive records which preprocessing and checkpoint built it, and refuses to
  match if those change.
- **API:**
  - `GET /api/health`
  - `GET /api/gallery`
  - `GET /api/gallery/{id}`
  - `POST /api/identify` starts a job and returns a `jobId` straight away;
    `GET /api/identify/{jobId}/status` reports `queued | detecting | silhouettes | embedding |
    searching | done | error` with overall progress, and the result when done (HANDOFF §7.3)
  - `POST /api/gallery` (save a walker by `queryId`, no re-upload)
  - `POST /api/gallery/from-video`

  Errors the UI can show: no person, too few frames, wrong type, too large, model not loaded,
  name taken, expired query.
- **Archive editing:** `PATCH /api/gallery/{id}` renames (the vector and silhouettes stay),
  `DELETE /api/gallery/{id}` removes the vector, the entry and its silhouettes.
- **Scripts:**
  - `casia_eval`, for Kaggle: first checks that `train` mode reproduces the known baseline
    (NM 99.27 / BG 93.26 / CL 72.36) and marks the threshold untrusted if not; then Rank-1 for each
    preprocessing mode in three protocols (with counts like 796/1100), genuine/impostor distances
    and a threshold table. Detects flat, nested and wrapped dataset layouts. Run on real CASIA-B on
    Kaggle on 2026-10-28: baseline reproduced exactly
  - `embed_video` (determinism check)
  - `calibrate_threshold` (genuine vs impostor distances)
  - `import_gallery_pkl` (import an old `gallery.pkl`)
- **Quality:** ruff (lint + format) and 70 pytest tests. A new GitHub Actions job,
  `.github/workflows/backend-ci.yml`, runs them on PRs.

### Frontend (`frontend/`): wired to the real API, mock data removed
- **Mock files deleted:** `mockGallery.ts`, `mockVerdict.ts`, `archiveStore.ts`,
  `MockSilhouette.tsx`. The `?mock=` switch is gone.
- **Crime Scene:** a real video picker that shows the file name and size and checks type and size
  before uploading.
- **Verdict:**
  - The loader follows the real job: its four labels track the server's stages (each shown at
    least ~0.45 s so they're readable) and the ring follows the reported progress; at least 3.5 s.
  - Error screen for failed investigations.
  - Real silhouettes.
  - "Open a case file" saves an unknown walker.
- **Verdict page redesign:**
  - verdict card with a distance scale (identical → match line → strangers) and a pin for this walk
  - a plain-language sentence, and three key numbers (distance, match line, runner-up / how far
    past the line)
  - the clue and the on-file walk playing side by side as silhouette animations
  - a compact lineup with per-suspect distance bars
  - variants for match, unknown, unknown-then-saved, empty Archive and errors
  - fits 1440×900 and 1280×720 without scrolling and stacks on phones
- **Archive:** loads from the server, with loading, empty and error states. Case file dialog shows
  silhouettes, the vector preview, L2 norm and date. **"Add Case File" works** (name + video upload).
  Case files can be **renamed** and **deleted** (delete asks to confirm inside the dialog).
- **About page** shows the corrected accuracy: NM 99.27%, BG 93.26%, CL 72.36%.
- `vite.config.ts` proxies `/api` to `localhost:8000`.

### Checked so far
- The real checkpoint gives a 4096-d vector, and the same clip gives an identical vector twice
  (CUDA, ~1.3 s warm).
- End-to-end through the browser (headless Edge) with real models, all without console errors:
  match, unknown, save, empty Archive and the no-person error.
- **Limit of these checks:** there were no real walking videos available. The test clips are panned
  photos, which prove the pipeline runs but say **nothing about accuracy**.

## Left to do (in order)

1. **Record real walking clips** of the team and others: side view, whole body in frame, 3–5 s,
   2–3 clips per person (HANDOFF §7.5).
2. **Calibrate the match threshold.** The first value comes from `scripts/casia_eval.py` on
   Kaggle, with the `casia_seq_x` threshold table (= the website's `casia_clip`); see
   `SESSION_NOTES.md` for the exact command. Then run `python -m scripts.calibrate_threshold videos\`
   on the team's clips to check it, set `SWALKER_MATCH_THRESHOLD` in `backend/.env`, and set
   `SWALKER_THRESHOLD_CALIBRATED=true`. Until then the 30 is a placeholder, and the Verdict page
   says the match line is provisional.
3. **Build the real Archive** from the recorded clips (Archive → "Add Case File").

## Known issues / notes
- **The site is wider than a phone screen** (every page measures 454 px wide on a 390 px phone).
  This comes from the shared layout (96 px nav rail plus page widths), not one page, so it needs
  a mobile pass on the shell.
- Several clips of one person aren't averaged yet: each case file holds one clip's vector.
  Averaging may help (HANDOFF §7.5).
- The Archive is a single process's files. Run one backend instance, not several.
- `backend/app/ml/combined_glfe_mstfe.py` must not be edited. A new architecture comes with its
  own checkpoint: replace both together.
