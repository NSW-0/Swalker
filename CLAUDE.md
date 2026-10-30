# Swalker / GLTNet website

Gait recognition website for a Birzeit University ECE graduation project. React frontend
(`frontend/`) and FastAPI + PyTorch backend (`backend/`), wired together; no mock data left. What
remains is mostly real recordings of people and checking the threshold on them.

**Read `HANDOFF.md` before starting any task.** It explains the model, the frontend, the API the
frontend expects, and known problems in the inference scripts. Then read `SESSION_NOTES.md`:
where the last session stopped, the next steps, and lessons learned. Update it before a session
ends.

**Keep `CHANGELOG.md` up to date:** every change gets a dated line there, in the same branch as
the change (under "Not merged yet" until its PR is merged, then under the merge date with the PR
number). The user wants all changes tracked.

## Commands

Frontend (`frontend/`):
- `npm install`, `npm run dev` (http://localhost:5173)
- `npm run lint` and `npm run build`: both must pass; CI runs them on PRs to `main`

Backend (`backend/`, see `backend/README.md`), with the venv in `backend/.venv`:
- `.venv\Scripts\python -m uvicorn app.main:create_app --factory --port 8000`
- `.venv\Scripts\python -m ruff check .`, `ruff format .` and `-m pytest`: all must pass; CI runs
  them on PRs to `main`. The system Python has no ruff.
- `scripts/casia_eval.py` runs on Kaggle (CASIA-B + GPU), not locally.

## Rules

- Do not change the model architecture (`combined_glfe_mstfe.py`) or anything about training.
  Training happens on Kaggle, outside this repo.
- Inference always under `model.eval()` and `torch.no_grad()`. Use only `feat` from
  `model(x)`, never the classifier logits.
- Load YOLO and the model once at startup, not per request.
- Before trusting identification results, check that preprocessing matches training
  (HANDOFF.md §6.1–6.2). Gallery and query must use identical preprocessing.
- Never commit model weights (`*.pth`, `*.pt`), uploads, or large original images.
- Show the user any accuracy claim only with the measurement behind it.

## Environment

- Developer machine is Windows, shell is cmd. Give git commands that work in cmd (no `&` in
  branch names, no bash-only syntax).
- Branches: `nSawafta/<topic>`, merged into `main` via PR. Stage files by name.

## Frontend conventions

- Scope page CSS under the page wrapper (`.page-verdict-content .x`), otherwise `.page p`
  overrides it.
- Keep pages from scrolling as a whole: fixed header + fixed-height content with internal scroll.
- Colours come from the CSS variables in `frontend/src/index.css`; MUI theme is in `main.tsx`.
