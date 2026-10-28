from fastapi import APIRouter, Depends

from app.api.uploads import VIDEO_EXTENSIONS
from app.state import AppState, get_state

router = APIRouter()


@router.get("/health")
def health(state: AppState = Depends(get_state)):
    s = state.settings
    return {
        "modelLoaded": state.pipeline is not None,
        "loadError": state.load_error,
        "device": state.device,
        "checkpoint": s.checkpoint_path.name,
        "preprocessMode": s.preprocess_mode,
        "gallerySize": len(state.store),
        "galleryCompatible": state.store.compatible,
        "threshold": s.match_threshold,
        "thresholdCalibrated": s.threshold_calibrated,
        "maxUploadMb": s.max_upload_mb,
        "videoExtensions": list(VIDEO_EXTENSIONS),
    }
