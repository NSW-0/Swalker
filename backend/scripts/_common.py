"""Shared setup for the command-line scripts: same models and settings as the API."""

from app.config import get_settings
from app.ml.embed import GaitEmbedder, resolve_device
from app.ml.pipeline import GaitPipeline
from app.ml.silhouettes import YoloSegmenter

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv"}


def build_pipeline() -> GaitPipeline:
    settings = get_settings()
    device = resolve_device(settings.device)
    print(f"device={device} checkpoint={settings.checkpoint_path.name} preprocess={settings.preprocess_mode}")
    embedder = GaitEmbedder(settings.checkpoint_path, device)
    segmenter = YoloSegmenter(settings.yolo_weights, str(device), settings.target_fps)
    return GaitPipeline(segmenter, embedder, settings)
