"""Swalker API. Run from backend/: uvicorn app.main:create_app --factory --port 8000"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import gallery, health, identify
from app.config import Settings, get_settings
from app.errors import register_error_handlers
from app.gallery.store import GalleryMeta, GalleryStore
from app.ml.pipeline import GaitPipeline
from app.queries import QueryCache
from app.state import AppState

log = logging.getLogger("swalker")


def load_pipeline(state: AppState) -> None:
    """YOLO + GLTNet, once per process. A failure leaves the API up so /api/health can say why."""
    from app.ml.embed import GaitEmbedder, resolve_device
    from app.ml.silhouettes import YoloSegmenter

    s = state.settings
    try:
        device = resolve_device(s.device)
        embedder = GaitEmbedder(s.checkpoint_path, device)
        segmenter = YoloSegmenter(s.yolo_weights, str(device), s.target_fps)
        state.pipeline = GaitPipeline(segmenter, embedder, s)
        state.device = str(device)
        log.info("models loaded on %s (%s, preprocess=%s)", device, embedder.checkpoint_name, s.preprocess_mode)
    except Exception as exc:  # reported through /api/health
        state.load_error = f"{type(exc).__name__}: {exc}"
        log.exception("model loading failed")


def create_app(settings: Settings | None = None, pipeline: GaitPipeline | None = None) -> FastAPI:
    """pipeline: pass one in (tests) to skip loading the real models."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = settings or get_settings()
    store = GalleryStore(
        settings.gallery_dir,
        GalleryMeta(
            preprocess_mode=settings.preprocess_mode,
            checkpoint=settings.checkpoint_path.name,
            target_fps=settings.target_fps,
        ),
    )
    state = AppState(
        settings=settings,
        store=store,
        queries=QueryCache(settings.query_ttl_minutes * 60),
        pipeline=pipeline,
        device="injected" if pipeline else "unknown",
    )
    if not store.compatible:
        log.error(
            "gallery at %s was built with %s; server uses %s. Rebuild it before identifying.",
            settings.gallery_dir,
            store.meta,
            store.expected,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state.pipeline is None:
            load_pipeline(state)
        yield

    app = FastAPI(title="Swalker API", lifespan=lifespan)
    app.state.swalker = state
    register_error_handlers(app)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
    for module in (health, gallery, identify):
        app.include_router(module.router, prefix="/api")
    app.mount("/api/media", StaticFiles(directory=store.sil_dir), name="media")
    return app
