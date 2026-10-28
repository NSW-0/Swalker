from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.serialize import png_data_url, silhouette_urls, thumbnail_url
from app.api.uploads import saved_upload
from app.gallery.matching import is_match, rank
from app.queries import CachedQuery
from app.state import AppState, get_state

router = APIRouter()


@router.post("/identify")
def identify(video: UploadFile = File(...), state: AppState = Depends(get_state)):
    """Synchronous: on CPU this takes 10-30 s; the Verdict loader keeps animating meanwhile."""
    pipeline = state.require_pipeline()
    state.store.require_compatible()
    settings = state.settings

    with saved_upload(video, settings.max_upload_mb) as path:
        walk = pipeline.analyse(path)

    entries, features = state.store.snapshot()
    suspects = rank(walk.vector, features, settings.top_k)
    clip_name = Path(video.filename or "clip").name
    query_id = state.queries.put(CachedQuery(vector=walk.vector, display=walk.display, clip_name=clip_name))

    return {
        "caseId": f"{int(query_id[:8], 16) % 10000:04d}",
        "queryId": query_id,
        "clipName": clip_name,
        "framesTotal": walk.frames_total,
        "framesWithPerson": walk.frames_with_person,
        "framesUsed": walk.frames_used,
        "gallerySize": len(entries),
        "threshold": settings.match_threshold,
        "thresholdCalibrated": settings.threshold_calibrated,
        "isMatch": is_match(suspects, settings.match_threshold),
        "suspects": [
            {
                "personId": entries[s.index].personId,
                "distance": s.distance,
                "confidence": s.confidence,
                "thumbnailUrl": thumbnail_url(entries[s.index]),
                "silhouetteUrls": silhouette_urls(entries[s.index]),
            }
            for s in suspects
        ],
        "querySilhouettes": [png_data_url(s) for s in walk.display],
    }
