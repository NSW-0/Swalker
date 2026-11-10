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

## Documents

| File | What it is |
|---|---|
| [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) | How the website was built: method, pipeline, dataset use, measurements, limits |
| [`DATASET.md`](DATASET.md) | CASIA-B: what it holds, how it was split, results on it |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | What is built and what is left |
| [`CHANGELOG.md`](CHANGELOG.md) | Every change, with its date and pull request |
| [`HANDOFF.md`](HANDOFF.md) | The model, the original scripts' problems and the plan the backend followed (2026-10-19) |
| [`SESSION_NOTES.md`](SESSION_NOTES.md) | Working notes between development sessions |
| [`CLAUDE.md`](CLAUDE.md) | Short rules for Claude Code in this repo |

## Checks

Pull requests to `main` run them automatically:
- **Frontend:** `npm run lint` and `npm run build`.
- **Backend:** `ruff check .`, `ruff format --check .` and `pytest`.

Model weights, the Archive and uploaded videos are never committed.
