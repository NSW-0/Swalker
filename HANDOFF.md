# Swalker — Project Handoff (frontend done, backend next)

Written 2026-10-19 for whoever picks this up next, most likely Claude Code working in this repo.
Read this whole file before writing any backend code. `CLAUDE.md` has the short version of the rules.

---

## 1. What this project is

**Swalker** is the website for **GLTNet**, a gait recognition model: it identifies a person from
the way they walk, without needing their face. It's a graduation project at Birzeit University
(ECE department).

- Students: Nabeel Sawafta, Ahmad Karmi, Kareem Hamza
- Supervisor: Dr. Rawan Soboh
- Goal of the website: show the model working, as a website rather than as a Python script.
  A visitor uploads a walking clip, the site "investigates" it and says who it is, or that the
  person is unknown and can be added to the Archive.

The site has a detective theme throughout: the upload page is "The Crime Scene", the gallery of
known people is "The Archive", the result is "The Verdict", and a detective duck mascot appears
on every page.

Names you'll see for the same model: **GLTNet** (paper / README), **Combined GLFE + MSTFE**
(About page, Research Summary), `GLFEMSTFEModel` (the PyTorch class).

---

## 2. Status at a glance

| Part | Status |
|---|---|
| Model architecture | Final. Do not change. |
| Trained weights | Done: `stage2_iter70000.pth` (on Kaggle, see §5) |
| Research training runs | Still going on Kaggle, separate from the website (see §5.3) |
| Python inference scripts | Rewritten into `backend/app/ml/` (2026-10-28). §6.2, §6.3 (threshold mechanism), §6.5 and §6.6 fixed; §6.1 implemented as `casia_clip` mode and measured on CASIA-B (pooled Rank-1 97.82 / 90.07 / 63.91 vs 99.27 / 93.26 / 72.36 for the training framing, see `backend/README.md`) |
| Frontend | Done, wired to the real API; mock data removed (2026-10-28) |
| Backend | Steps 1–5 of §7.6 done (`backend/README.md`). Left: calibrate the threshold on real recordings, measure §6.1, progress reporting (§7.3) |

---

## 3. Repository layout

```
gait-website/
  README.md
  HANDOFF.md            <- this file
  CLAUDE.md             <- short rules for Claude Code
  .gitignore
  .github/workflows/ci.yml   <- frontend lint + build on PRs to main
  frontend/             <- React site (done)
  backend/              <- does not exist yet, to be created
  mockups/              <- design mockups, git-ignored, not part of the site
```

### 3.1 Frontend

React 19 + Vite + TypeScript, react-router-dom (`createBrowserRouter`), MUI 9 (dark theme,
primary `#F58F7C`), i18next (set up, barely used). ESLint flat config; Husky + lint-staged run
`eslint --fix` on staged `.ts/.tsx` before every commit.

Run it: `cd frontend && npm install && npm run dev` → http://localhost:5173
Check it: `npm run lint` and `npm run build` (this is exactly what CI runs).

| Route | File | What it does now |
|---|---|---|
| `/` | `routes/Home.tsx` | Hero text + "Let's Investigate" button |
| `/crime-scene` | `routes/Demo.tsx` | "Add Clue" (**fake**: just flips a flag and shows `clip_001.mp4`), then "Investigate the Clue" → Verdict |
| `/crime-scene/verdict` | `routes/Verdict.tsx` | Loading screen (fixed 3.5 s), then the result (mock) |
| `/archive` | `routes/Archive.tsx` | Grid of case files from mock gallery; dialog shows the 4096-d vector preview. `?person=<id>` opens that card. "Add Case File" card **does nothing yet** |
| `/about` | `routes/About.tsx` | How it works, what gait recognition is, the model and its numbers |

Supporting files:

| File | Role |
|---|---|
| `components/layout/Navbar.tsx` | Left icon rail with MUI tooltips |
| `routes/RootLayout.tsx` | Picks the background image per route (`pageBackgroundClass` map) |
| `components/verdict/InvestigationLoader.tsx` | Duck sprite in a progress ring, 4 step labels |
| `components/verdict/MockSilhouette.tsx` | **Placeholder** SVG walking figures. Replace with real silhouette images from the backend |
| `data/mockGallery.ts` | 20 fake people with seeded random 4096-d vectors. Same shape as `gallery.pkl` |
| `data/mockVerdict.ts` | Fake identification using the same math as `identify.py` |
| `data/archiveStore.ts` | Mock gallery + walkers saved from the Verdict page (memory only, lost on reload) |

### 3.2 The Verdict page (the page the backend mostly feeds)

1. **Loading**: duck sprite (`assets/sprite_frames.png`, 8 frames) inside a progress ring, with
   step labels *Spotting the suspect → Lifting the silhouettes → Reading the walk → Searching the
   Archive*. These four labels deliberately match the real pipeline stages, so a backend that
   reports progress can drive them for real.
2. **Match**: prime suspect, a closeness bar with a wing image marking the match line, the clue's
   silhouette strip next to the matched person's strip, a top-3 lineup, "How we got here" stats,
   and the gem duck.
3. **No match**: "Unknown walker", nearest 3 walks, and **Add to the Archive** (asks for a case
   file name, then saves the walker). After saving, the same walk is recognised next time.

Dev switch: no-match is the default mock right now; `/crime-scene/verdict?mock=match` shows a match.

The frontend's result type (in `data/mockVerdict.ts`) is the contract the backend should return:

```ts
interface Suspect { personId: string; distance: number; confidence: number }
interface VerdictResult {
  caseId: string
  clipName: string
  framesWithPerson: number
  gallerySize: number
  suspects: Suspect[]        // top 3, sorted by distance ascending
  isMatch: boolean           // suspects[0].distance <= MATCH_THRESHOLD
  queryVector: number[]      // only used by the mock to save; the real API should use an id instead
}
```

---

## 4. How the model works (the logic of the project)

```
walking video
  → YOLOv8-seg finds the person in each frame and gives a pixel mask
  → each mask becomes a 64×44 silhouette (white person, black background)
  → a sequence of silhouettes goes into GLTNet
  → GLTNet outputs one 4096-number vector = the person's "gait signature"
  → compare that vector with every person stored in the gallery (Euclidean distance)
  → closest person = suspect; if even the closest is too far away = unknown walker
```

**Model facts**

- Class: `GLFEMSTFEModel(n_parts=8, branch_c=128, head_out_c=256, local_stride=3, n_classes=<from checkpoint>)`
  in `combined_glfe_mstfe.py`. About 6.5M parameters.
- Input: tensor `(B, 1, T, 64, 44)`, values in [0, 1]. T can vary at inference (the head takes
  a max over time).
- `forward` returns `(feat, logits)`. Use only `feat` (shape `(B, 4096)`). `logits` is the training
  classifier and is meaningless for new people.
- GLFE spatial frontend (GLConv + LTA) → two temporal branches (small scale = adjacent frames,
  large scale = 3-frame intervals) → GeM pooling per horizontal strip → 16 strips × 256 = 4096.
- Trained with triplet loss + cross-entropy on CASIA-B (74 training people, LT protocol).
- Gallery matching: Euclidean distance between 4096-d vectors, smaller = more alike.

**Two rules that already caused real bugs in this project:**

1. Always run inference under `model.eval()` **and** `torch.no_grad()`. The network is full of
   BatchNorm. A gallery once encoded in train mode made a healthy model look broken
   (NM 84.91% read as 22.55%).
2. `torch.load` on torch ≥ 2.6 defaults to `weights_only=True`. Current checkpoints load fine that
   way; keep it like that, don't store non-primitive objects in anything you save.

**Accuracy (CASIA-B, LT protocol, 50 test subjects, 11 views)**

| | NM (normal) | BG (bag) | CL (coat) |
|---|---|---|---|
| Originally published | 99.18% | 92.44% | 72.36% |
| Re-measured with corrected evaluation (2026-09-22) | **99.27%** | **93.26%** | **72.36%** |

The About page shows the corrected numbers (since 2026-10-28). The originals were measured with the
gallery encoded in a different BatchNorm mode than the probes; same checkpoint, CL unchanged (796/1100).
CL (clothing change) is the known weak spot.

---

## 5. Model files and where they live

### 5.1 Code

| File | What it is | Where |
|---|---|---|
| `combined_glfe_mstfe.py` | Model architecture (final) | Claude project "Gait website" + Kaggle `gait-code` dataset |
| `silhouette_extractor.py` | Video → YOLOv8 masks → 64×44 silhouettes | Claude project |
| `gallery_builder.py` | Folder of videos → `gallery.pkl` | Claude project |
| `identify.py` | Video → top-K matches against `gallery.pkl` | Claude project |
| `casia_b_loader.py`, `train.py` | Training code. Newest versions are on Kaggle (`gait-code`); the Claude project has an older copy | Kaggle |
| `README_pipeline.md` | How to use the scripts + recording tips | Claude project |

**None of these are in this repo yet.** Before starting the backend, copy the first four into the
backend (e.g. `backend/app/ml/`). Use them as the starting point, but read §6 first: they need
changes before they're fit for a web service.

### 5.2 Weights

- `stage2_iter70000.pth`, Kaggle dataset `bbiiggnn/gait-checkpoints`
  (on Kaggle: `/kaggle/input/datasets/bbiiggnn/gait-checkpoints/stage2_iter70000.pth`).
  The 2026-09-22 re-measurement used a copy in a `gait-v2-checkpoints` dataset.
- The checkpoint is a dict; weights are under `ckpt['model']`. `n_classes` may be missing in this
  older file; `identify.py` already falls back to 74, which is correct for it.
- YOLO: `yolov8n-seg.pt`, downloaded automatically by `ultralytics` on first use.
- **Never commit `.pth` / `.pt` files.** Put them in `backend/models/` (git-ignored) and document the
  download step.

### 5.3 Training is still going on, separately

Training runs on Kaggle (GPU T4) and has nothing to do with the website code. In the latest log
(2026-10-07), run 6 (fixed recipe, CASIA-B only) and run 7 (pretrain on CCPG, then CASIA-B) were
in progress, both trying to improve CL. Neither had beaten the published checkpoint yet. If a newer checkpoint ever beats `stage2_iter70000.pth`, the backend should
be able to swap it in by changing one config value. `n_classes` is read from the checkpoint, so
different checkpoints load without code changes. Do not touch training code from this repo.

---

## 6. Problems in the existing inference scripts (read before building)

These are not style issues. Each one can make the website's answers worse than the model's real
accuracy, and none of them would show up as an error.

### 6.1 Preprocessing does not match what the model was trained on — **most important**

The model was trained and evaluated on CASIA-B silhouettes that are **whole 320×240 video frames**
resized to 64×44. In those, the person fills about **50% of the frame height** and is only about
**6 pixels wide** (the 4:3 frame is squashed into a tall 44×64 image), and pixel values are **soft
greyscale** (bilinear resize).

`silhouette_extractor.crop_and_resize_silhouette()` does something different: it **crops tightly
around the person** so they fill the whole 64×44 image, then **thresholds to hard 0/1**. A
code review on the training side measured exactly this kind of framing gap between CASIA-B and the
CCPG dataset and found it large (person ~6 px wide vs ~31 px wide). The website would be feeding
the model a different kind of image than it learned from.

What to do:
1. **Measure first.** Using CASIA-B test sequences (they're on Kaggle), encode the same sequences
   two ways: (a) the training/eval way (full frame, bilinear, greyscale) and (b) the deployment
   way (`crop_and_resize_silhouette`). Compare Rank-1. If (b) is clearly lower, the preprocessing
   must change.
2. **Likely fix:** re-frame each YOLO mask the CASIA-B way before resizing: crop to the person,
   scale so they are ~50.1% of the frame height, place them at CASIA-B's vertical position
   (bbox top ≈ 0.233 of frame height) on a 4:3 canvas, resize the canvas to 44×64 with bilinear
   interpolation, and **keep the greyscale values (don't threshold)**. The training code already
   has this exact re-framing for CCPG: `CCPG_MATCH_CASIA_GEOMETRY` in the newest
   `casia_b_loader.py` on Kaggle. Reuse its numbers.
3. Gallery and query must go through the **same** preprocessing, and if it changes, every stored
   gallery vector must be rebuilt.

### 6.2 Frame sampling does not match either

- Training uses **30 consecutive frames**; evaluation feeds the **whole sequence** in order.
- `sample_frames()` spreads 30 frames evenly across the whole clip (`np.linspace`). On a 5-second
  30 fps clip that keeps every 5th frame, so the walk looks 5× faster than anything the model saw,
  and the temporal branches are built to read adjacent-frame motion.
- Recommended: feed all extracted frames in order (cap the length, e.g. ~90–120 frames, for speed),
  the same way evaluation does. Measure with the same Rank-1 check as §6.1.

### 6.3 No "unknown person" decision

`identify.py` always returns its closest match, so a stranger always gets "identified" as someone.
The Verdict page needs a **distance threshold** (`MATCH_THRESHOLD`): if the best distance is above
it, the result is "Unknown walker".

How to choose it: collect genuine distances (same person, different clip) and impostor distances
(different people), e.g. from the CASIA-B test set and from the team's own recorded videos, and
pick a threshold with an acceptable false-match rate. The mock uses 30, which only fits the mock
vectors. Make it a config value and return it in the API response so the closeness bar can draw
the match line.

### 6.4 "Confidence" is relative, not a probability

`identify.py` turns the top-3 distances into percentages that always add up to 100%. A stranger
can still show "61%". The frontend labels it "strength" with a note saying so. Keep returning the
raw distances; that's what the match decision should use.

### 6.5 Performance

- `extract_silhouettes_from_video()` loads YOLO **on every call**. In a service, load YOLO and
  GLTNet **once at startup**.
- YOLOv8n-seg is about 10 fps on CPU, so a 5 s clip at 30 fps is ~15 s on CPU, plus the model.
  The UI must handle a real wait (see §7.3). A GPU makes this a few seconds.

### 6.6 What the gallery stores

`gallery.pkl` = `{'person_ids': [...], 'features': (N, 4096), 'video_paths': [...]}`. It does
**not** store silhouettes, but the Verdict page shows an "On file" silhouette strip for the matched
person, so the backend should also keep a few silhouette frames per person when it adds them.

---

## 7. What's left to do: the backend

### 7.1 Suggested stack

- **Python + FastAPI + uvicorn**, because the model is PyTorch. (The original project checklist
  also named FastAPI.)
- Linting/tests: `ruff` + `pytest`. Add a backend job to CI next to the frontend one.
- Storage: keep it simple. A gallery folder with a `features.npy` + `index.json` (or SQLite), plus
  silhouette PNGs per person. Add a one-off importer for an existing `gallery.pkl`.
- Config through env vars / `.env` (checkpoint path, device `cpu`/`cuda`, match threshold, max
  frames, upload size limit).

Suggested layout (adjust as needed):

```
backend/
  app/
    main.py            FastAPI app, startup loads YOLO + GLTNet once
    config.py
    api/               routes
    ml/                combined_glfe_mstfe.py, silhouettes.py (fixed preprocessing), embed.py
    gallery/           storage + matching
  models/              .pth / .pt weights (git-ignored)
  storage/             uploads, gallery data (git-ignored)
  tests/
  pyproject.toml or requirements.txt
```

### 7.2 API the frontend needs

| Method | Path | Used by | Does |
|---|---|---|---|
| GET | `/api/health` | dev | Model loaded? Device? Gallery size? |
| GET | `/api/gallery` | Archive | List of case files: `personId`, `videoPath`, `addedAt`, a short vector preview (first 8 values), L2 norm, silhouette thumbnail URL |
| GET | `/api/gallery/{personId}` | Archive dialog | One case file, with its silhouette frames |
| POST | `/api/identify` | Crime Scene → Verdict | Upload a video (multipart). Returns the `VerdictResult` shape above **plus** `queryId`, `threshold`, `querySilhouettes` (≈10 frames as PNG URLs or data URLs) and, for each suspect, a thumbnail URL. Keeps the query's 4096-d vector server-side under `queryId` for a while |
| POST | `/api/gallery` | Verdict "Add to the Archive" | `{ name, queryId }` → saves that walker as a new case file without re-uploading. 409 if the name exists |
| POST | `/api/gallery/from-video` | Archive "Add Case File" | Upload a video + name → new case file |

Errors the UI must be able to show: no person found in the video, too few usable frames, file too
large / wrong type, model not loaded.

### 7.3 Long requests and the loading screen

`/api/identify` can take 10–30 s on CPU. Two options:
- **Simple first version**: one synchronous request with a long timeout; the loader keeps animating
  until the response arrives (keep the 3.5 s as a minimum display time).
- **Better**: `POST /api/identify` returns a `jobId`; the frontend polls `GET /api/jobs/{jobId}`
  (or uses Server-Sent Events) for `stage` = `detecting | silhouettes | embedding | searching |
  done | error` and a percentage. These map one-to-one to the loader's four labels.

### 7.4 Frontend changes that go with the backend

- `vite.config.ts`: proxy `/api` → `http://localhost:8000` for development.
- **Crime Scene**: replace the fake "Add Clue" with a real video file picker (show file name and
  size, validate type and size), then pass the file to the Verdict page (router state or a small
  store) and call `/api/identify`.
- **Verdict**: replace `buildMockVerdict()` with the API call; drive `InvestigationLoader` from real
  progress (it currently takes `duration` and animates by time); replace `MockSilhouette` with
  `<img>` strips from the API; save via `POST /api/gallery` with `queryId`; handle errors.
  The `?mock=` switch can stay for development.
- **Archive**: load from `/api/gallery` instead of `mockGallery` / `archiveStore`; wire up
  "Add Case File".
- **About**: decide whether to show the original or the corrected accuracy numbers (§4).
- Then delete or clearly fence off the mock files.

### 7.5 Building the real gallery

The demo needs real people in the Archive. Record the team (and anyone willing) using the
guidelines from `README_pipeline.md`:
side view (90°) works best, camera 4–6 m from the walking path, camera height 1.0–1.2 m, whole
body visible, 3–5 s of walking (at least one full gait cycle), steady camera, good light.
Several clips per person and averaging their vectors helps.

### 7.6 Suggested order

1. Copy the model code into `backend/`, load the checkpoint, embed one video from a script. Confirm
   the output is a 4096-d vector and that the same clip gives the same vector twice.
2. Do the §6.1 / §6.2 measurement and fix the preprocessing.
3. Gallery storage + matching + threshold (§6.3).
4. FastAPI endpoints (§7.2) with tests.
5. Frontend wiring (§7.4).
6. Record real gallery videos (§7.5) and calibrate the threshold on them.
7. Progress reporting for the loader (§7.3), CI job for the backend.

---

## 8. Conventions and gotchas

**Git**
- Feature branches named `nSawafta/<topic>`, merged into `main` by PR (or locally). CI runs
  frontend lint + build on PRs to `main`.
- The machine is **Windows** and the shell is **cmd**. In cmd, `&` separates commands, so a branch
  name like `output&load` breaks. Use `-`.
- Add files by name rather than `git add .`. Large original art files (`duckNoGem.png` 6.5 MB,
  `wing.png`) are deliberately not committed; the site uses the small `_web` copies.
- "LF will be replaced by CRLF" warnings on commit are harmless.

**Frontend styling**
- Colours live as CSS variables in `frontend/src/index.css` (`--bg-deep #2C2B30`,
  `--accent #F58F7C`, `--accent-shadow #C9787C`, `--muted`, `--text`…).
- Generic rules `.page h1` and `.page p` exist. A plain page class like `.foo-text` **loses** to
  `.page p` (specificity 0,1,0 vs 0,1,1). Always scope page styles under the page wrapper, e.g.
  `.page-verdict-content .foo-text`. This bug happened four times.
- Pages use a fixed centred header and a fixed-height content area with internal scrolling, so
  `.app-shell` never grows taller than the screen. When it grew, the `background-size: cover`
  image looked zoomed in.
- Each page has its own background (`RootLayout.tsx` map + `.app-shell.page-*` rules) and a duck
  in a corner.
- Wrapping a react-router `NavLink` in an MUI `Tooltip` breaks NavLink's function-style
  `className`; compute `isActive` with `useLocation()` instead (see `Navbar.tsx`).

**Assets created from the originals**
- `sprite_frames.png`: `sprite.png` cut into 8 equal cells (the original's frames aren't on a grid).
- `duckNoGem_web.webp`: 600×800 copy of the 6.5 MB `duckNoGem.png`.
- `wing_web.png`: `wing.png` cropped to the wing.
If the originals are edited, regenerate these.

---

## 9. Where more detail lives

In the claude.ai project "Gait website":
- `Combined_GLFE_MSTFE_Research_Summary_Final.md`: architecture, training, results, recording guide
- `README_pipeline.md`: how the inference scripts are used
- `NEW_CLAUDE_HANDOFF.md`: the training-side handoff (CCPG plan, hyperparameters)
- `claude/CODE_REVIEW_FINDINGS_2026-09-21.md`: training/eval bugs found and fixed, incl. the
  CASIA-B vs CCPG geometry measurement behind §6.1
- `claude/BASELINE_REEVAL_2026-09-22.md`: the corrected accuracy numbers
- `claude/RUN_LOG_2026-09-23.md`: every training run and its results
- `claude/VERDICT_PAGE_NOTES.md`: Verdict page design and backend needs
- `claude/WEBSITE_SETUP.md`: how the frontend was scaffolded
