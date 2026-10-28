"""Settings, read from environment variables or backend/.env (prefix SWALKER_)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWALKER_", env_file=BACKEND_DIR / ".env", extra="ignore")

    checkpoint_path: Path = BACKEND_DIR / "models" / "stage2_iter70000.pth"
    # Downloaded by ultralytics on first use if missing.
    yolo_weights: Path = BACKEND_DIR / "models" / "yolov8n-seg.pt"
    device: Literal["auto", "cpu", "cuda"] = "auto"

    storage_dir: Path = BACKEND_DIR / "storage"

    # How each frame's mask becomes a 64x44 silhouette. See ml/silhouettes.py.
    # Gallery and query must use the same mode; the gallery records which one it was built with.
    preprocess_mode: Literal["casia", "crop"] = "casia"

    # Frames fed to the model: consecutive, in order, capped for speed (HANDOFF §6.2).
    max_frames: int = 120
    min_frames: int = 15
    # Video frames read at most (~15 s at 30 fps). Recordings should be 3-5 s of walking.
    max_read_frames: int = 450
    # Silhouettes kept for display (Verdict strip, Archive case file). Not fed to the model.
    display_frames: int = 10

    # Euclidean distance at or below which the closest walker counts as a match (HANDOFF §6.3).
    # Provisional until scripts/calibrate_threshold.py has been run on real recordings;
    # set threshold_calibrated=true only after that.
    match_threshold: float = 30.0
    threshold_calibrated: bool = False
    top_k: int = 3

    max_upload_mb: int = 100
    query_ttl_minutes: int = 30

    cors_origins: list[str] = ["http://localhost:5173"]

    @field_validator("checkpoint_path", "yolo_weights", "storage_dir")
    @classmethod
    def _relative_to_backend(cls, value: Path) -> Path:
        return value if value.is_absolute() else BACKEND_DIR / value

    @property
    def gallery_dir(self) -> Path:
        return self.storage_dir / "gallery"


@lru_cache
def get_settings() -> Settings:
    return Settings()
