# Swalker frontend

The website: React 19, Vite, TypeScript, React Router and MUI, with a detective theme. It needs the
backend (`../backend`) running for anything past the Home and About pages.

## Requirements

- Node.js 20+ and npm (CI uses Node 22)
- The backend on port 8000, see [`../backend/README.md`](../backend/README.md). Or start both with
  `start.bat` in the repo folder.

## Setup and run

```
npm install
npm run dev
```

The site opens at http://localhost:5173. The dev server passes every `/api` request on to
http://localhost:8000 (`vite.config.ts`).

`npm install` also installs a pre-commit hook (Husky + lint-staged) that runs `eslint --fix` on
staged `.ts`/`.tsx` files. If it stops firing, run `npx husky`.

## Scripts

```
npm run dev       # development server
npm run build     # type-check and build for production (dist/)
npm run preview   # serve the production build
npm run lint      # ESLint
```

`npm run lint` and `npm run build` must pass; CI runs them on pull requests to `main`.

## Pages

| Route | File | What it does |
|---|---|---|
| `/` | `routes/Home.tsx` | Landing page |
| `/crime-scene` | `routes/Demo.tsx` | Pick a walking clip; checks type and size before upload |
| `/crime-scene/verdict` | `routes/Verdict.tsx` | Runs the identification, shows the loading screen, then the result: match or unknown, distance scale, silhouettes, the closest 3 walks. An unknown walker can be saved to the Archive |
| `/archive` | `routes/Archive.tsx` | Everyone in the Archive. A case file shows each clip; add a person or a clip, remove a clip, rename, delete. `?person=<name>` opens that case file |
| `/about` | `routes/About.tsx` | How it works, gait recognition, the model and its accuracy |

## Structure

```
src/
  main.tsx                   entry point and MUI theme
  router.tsx                 routes
  routes/                    one file per page, plus RootLayout.tsx (navigation + per-page background)
  api/client.ts              calls to the backend, its types, friendly error messages
  api/useUploadLimits.ts     upload limits from /api/health
  data/clueStore.ts          the clip picked on the Crime Scene, passed to the Verdict page
  components/layout/         Navbar (Footer.tsx is not used at the moment)
  components/verdict/        loading screen, distance scale, silhouette player and thumbnails
  i18n/                      i18next, set up but barely used
  assets/                    backgrounds, ducks, sprite, swalker.ico
  index.css                  all styles; colours are CSS variables at the top
```

## Styling rules

- Colours come from the CSS variables at the top of `src/index.css`; the MUI theme is in `main.tsx`.
- Scope page styles under the page wrapper (`.page-verdict-content .x`), otherwise the generic
  `.page p` rule wins.
- Pages don't scroll as a whole: a header plus a fixed-height area that scrolls inside. A taller page
  would also zoom the background image.
- **Phones:** every phone style is in the "Phones" block at the end of `index.css`. It applies
  below 760 px wide, or below 1000 px on a touch screen. The navigation moves to a bar along the
  bottom. Keeping phone styles there means the PC layout can't change.
