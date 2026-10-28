# GLTNet Gait Recognition — Website

Monorepo for the GLTNet gait recognition website (Birzeit University ECE graduation project).

```
gait-website/
  frontend/   # React 19 + Vite + TypeScript site — see frontend/README.md
  backend/    # FastAPI + PyTorch service around the model — see backend/README.md
```

## Frontend

See [`frontend/README.md`](./frontend/README.md) for setup and run instructions.

## Backend

See [`backend/README.md`](./backend/README.md). In short: put the weights in `backend/models/`,
run `python -m uvicorn app.main:create_app --factory --port 8000` from `backend/`, then `npm run dev` in
`frontend/`.
