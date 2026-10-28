"""Everything loaded once per process, reachable from routes via request.app.state.swalker."""

from dataclasses import dataclass

from fastapi import Request

from app.config import Settings
from app.errors import ApiError
from app.gallery.store import GalleryStore
from app.ml.pipeline import GaitPipeline
from app.queries import QueryCache


@dataclass
class AppState:
    settings: Settings
    store: GalleryStore
    queries: QueryCache
    pipeline: GaitPipeline | None = None
    device: str = "unknown"
    load_error: str | None = None

    def require_pipeline(self) -> GaitPipeline:
        if self.pipeline is None:
            raise ApiError("model_not_loaded", f"The model isn't loaded: {self.load_error or 'still starting'}.", 503)
        return self.pipeline


def get_state(request: Request) -> AppState:
    return request.app.state.swalker
