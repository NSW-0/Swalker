# Swalker: gait recognition website

The website for **GLTNet**, a gait recognition model that identifies a person from the way they
walk, without their face. Birzeit University ECE graduation project by Nabeel Sawafta, Ahmad Karmi
and Kareem Hamza, supervised by Dr. Rawan Soboh.

Upload a walking clip on **The Crime Scene**. The site finds the walker, turns the walk into a
4096-number gait signature and compares it with everyone in **The Archive**. **The Verdict** then
names the closest person, or says the walker is unknown and offers to add them.

```
gait-website/
  frontend/   React 19 + Vite + TypeScript + MUI site, see frontend/README.md
  backend/    FastAPI + PyTorch service around the model, see backend/README.md
  start.bat   starts both and opens the site
```

## Running the demo

Double-click `start.bat`, or run it from cmd in this folder. It checks the setup, starts the backend
and the frontend in their own windows, waits for the models to load and opens
http://localhost:5173. Close the two windows to stop.

First-time setup:
1. **Backend:** a Python environment and the model checkpoint, see [`backend/README.md`](backend/README.md).
2. **Frontend:** `npm install` in `frontend/`. `start.bat` does this on its first run.

The Archive is not in the repo. It currently holds the 124 people of the CASIA-B dataset, built with
`backend/scripts/casia_videos.py` (see [`DATASET.md`](DATASET.md)).

## Results

Details and every table: [`RESULTS.md`](RESULTS.md).

- **The model** (CASIA-B benchmark, people it never trained on): Rank-1 99.27% normal walking, 93.26%
  with a bag, 72.36% in a coat.
- **The website's pipeline** on the same silhouettes: 97.82 / 90.07 / 63.91. The original script's
  tight crop gave 69.45 / 41.17 / 12.09.
- **Raw CASIA-B video through the website**, with a bag and a coat clip of each person in the Archive:
  Rank-1 99.82 / 96.55 / 96.91. Over all camera angles at the match line of 8, the right person is
  accepted 96 / 80 / 86% of the time, and 16.55 / 7.45 / 4.18% of strangers get a name.
- **Phone video:** the match line for phone clips is not set yet. It needs two normal walks of the
  same people.

## Documents

| File | What it is |
|---|---|
| [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) | How the website was built: method, pipeline, dataset use, measurements, limits |
| [`RESULTS.md`](RESULTS.md) | Every measurement, with what it was measured on |
| [`DATASET.md`](DATASET.md) | CASIA-B: what it holds, how it was split, results on it |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | What is built, what is left, notes for whoever continues |
| [`CHANGELOG.md`](CHANGELOG.md) | Every change, with its date and pull request |
| [`HANDOFF.md`](HANDOFF.md) | The model, the original scripts' problems and the plan the backend followed (2026-10-19) |
| [`CLAUDE.md`](CLAUDE.md) | Short rules for Claude Code in this repo |

## Checks

Pull requests to `main` run them automatically:
- **Frontend:** `npm run lint` and `npm run build`.
- **Backend:** `ruff check .`, `ruff format --check .` and `pytest`.

Model weights, the Archive and uploaded videos are never committed.
