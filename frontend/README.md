# GLTNet Gait Recognition — Web Frontend

Frontend for the GLTNet gait recognition project (Birzeit University ECE graduation project).
Built with React 19, Vite, TypeScript, and React Router.

There is no backend wired in yet — this is frontend scaffolding only. A backend/API for the
actual model will be added later once the site's UI is further along.

## Requirements

- Node.js 20+ and npm

## Setup

```bash
npm install
```

This also installs the git pre-commit hook (via the `prepare` script → Husky), so ESLint
runs automatically on staged `.ts`/`.tsx` files before every commit.

## Run locally

```bash
npm run dev
```

Opens the dev server, by default at http://localhost:5173.

## Other scripts

```bash
npm run build     # type-check and build for production (outputs to dist/)
npm run preview   # preview the production build locally
npm run lint      # run ESLint across the project
```

## Project structure

```
src/
  main.tsx              # entry point, mounts RouterProvider
  router.tsx             # route definitions (createBrowserRouter)
  routes/                 # one file per route/page
    RootLayout.tsx        # shared layout (Navbar + Footer + <Outlet/>)
    Home.tsx
    Demo.tsx               # placeholder for the interactive gait demo
    About.tsx
    NotFound.tsx
  components/
    layout/                # Navbar, Footer, and other layout pieces
  index.css                # global styles
```

As the site grows, add new pages under `src/routes/` and register them in `src/router.tsx`.
Reusable UI pieces go under `src/components/`.

## Linting

ESLint is configured via the flat config in `eslint.config.js`, with:

- `@eslint/js` recommended rules
- `typescript-eslint` recommended rules
- `eslint-plugin-react-hooks` recommended rules
- `eslint-plugin-react-refresh` (Vite HMR safety)

Run `npm run lint` any time, or rely on the pre-commit hook to catch issues automatically.

## Git hooks (Husky + lint-staged)

A pre-commit hook (`.husky/pre-commit`) runs `lint-staged`, which runs `eslint --fix` on any
staged `.ts`/`.tsx` files. This is set up automatically when you run `npm install` (via the
`prepare` script). If hooks ever stop firing, re-run `npx husky`.
