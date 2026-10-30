"""Identify as a job (HANDOFF §7.3): POST starts it and answers at once with a jobId; the Verdict
loader polls GET /api/identify/{jobId}/status, which reports the real stage until the result is in."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.serialize import png_data_url, silhouette_urls, thumbnail_url
from app.api.uploads import save_upload
from app.errors import ApiError
from app.gallery.matching import is_match, rank
from app.gallery.store import row_owners
from app.ml.pipeline import WalkAnalysis
from app.queries import CachedQuery
from app.state import AppState, get_state

router = APIRouter()
log = logging.getLogger("swalker")


@router.post("/identify", status_code=202)
def start_identify(video: UploadFile = File(...), state: AppState = Depends(get_state)):
    # Everything that can be checked straight away fails here, before a job exists.
    state.require_pipeline()
    state.store.require_compatible()
    path = save_upload(video, state.settings.max_upload_mb)
    clip_name = Path(video.filename or "clip").name

    job = state.jobs.create()
    state.executor.submit(run_identify, state, job.id, path, clip_name)
    return {"jobId": job.id, "status": job.status}


@router.get("/identify/{job_id}/status")
def identify_status(job_id: str, state: AppState = Depends(get_state)):
    job = state.jobs.get(job_id)
    if job is None:
        raise ApiError("job_not_found", "That investigation isn't running any more. Start it again.", 404)
    return job.public()


def run_identify(state: AppState, job_id: str, path: Path, clip_name: str) -> None:
    jobs = state.jobs
    try:
        walk = state.require_pipeline().analyse(path, report=lambda stage, f: jobs.stage(job_id, stage, f))
        jobs.stage(job_id, "searching")
        jobs.finish(job_id, verdict(state, walk, clip_name))
    except ApiError as exc:
        jobs.fail(job_id, exc.code, exc.message)
    except Exception:  # reported to the page as a generic failure; details in the server log
        log.exception("identify job %s failed", job_id)
        jobs.fail(job_id, "server_error", "Something went wrong while investigating this clip.")
    finally:
        path.unlink(missing_ok=True)  # uploads are never kept


def verdict(state: AppState, walk: WalkAnalysis, clip_name: str) -> dict:
    settings = state.settings
    people, features = state.store.snapshot()
    owners = row_owners(people)
    # Every clip is compared; each person counts once, with their closest clip.
    suspects = rank(walk.vector, features, settings.top_k, groups=[p for p, _ in owners])
    query_id = state.queries.put(CachedQuery(vector=walk.vector, display=walk.display, clip_name=clip_name))
    return {
        "caseId": f"{int(query_id[:8], 16) % 10000:04d}",
        "queryId": query_id,
        "clipName": clip_name,
        "framesTotal": walk.frames_total,
        "framesWithPerson": walk.frames_with_person,
        "framesUsed": walk.frames_used,
        "gallerySize": len(people),
        "threshold": settings.match_threshold,
        "thresholdCalibrated": settings.threshold_calibrated,
        "isMatch": is_match(suspects, settings.match_threshold),
        "suspects": [suspect(people, owners, s) for s in suspects],
        "querySilhouettes": [png_data_url(s) for s in walk.display],
    }


def suspect(people, owners, s) -> dict:
    person = people[s.index]
    clip = person.clips[owners[s.row][1]]
    return {
        "personId": person.personId,
        "distance": s.distance,
        "confidence": s.confidence,
        "clipCount": len(person.clips),
        "matchedClip": clip.videoPath,  # which of their clips was closest
        "thumbnailUrl": thumbnail_url(clip),
        "silhouetteUrls": silhouette_urls(clip),
    }
