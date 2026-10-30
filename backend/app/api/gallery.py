from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from pydantic import BaseModel, Field

from app.api.serialize import entry_detail, entry_summary
from app.api.uploads import saved_upload
from app.errors import ApiError
from app.gallery.store import name_from_filename, validate_name
from app.state import AppState, get_state

router = APIRouter()


@router.get("/gallery")
def list_gallery(state: AppState = Depends(get_state)):
    entries, features = state.store.snapshot()
    return {
        "entries": [entry_summary(e, features[i]) for i, e in enumerate(entries)],
        "compatible": state.store.compatible,
    }


@router.get("/gallery/{person_id}")
def get_case_file(person_id: str, state: AppState = Depends(get_state)):
    found = state.store.find(person_id)
    if found is None:
        raise ApiError("not_found", f"No case file named '{person_id}'.", 404)
    return entry_detail(*found)


class Rename(BaseModel):
    name: str = Field(max_length=200)


@router.patch("/gallery/{person_id}")
def rename_case_file(person_id: str, body: Rename, state: AppState = Depends(get_state)):
    entry = state.store.rename(person_id, body.name)
    return entry_detail(*state.store.find(entry.personId))


@router.delete("/gallery/{person_id}", status_code=204)
def delete_case_file(person_id: str, state: AppState = Depends(get_state)):
    state.store.remove(person_id)
    return Response(status_code=204)


class SaveQuery(BaseModel):
    name: str = Field(max_length=200)
    queryId: str = Field(max_length=64)


@router.post("/gallery", status_code=201)
def save_query(body: SaveQuery, state: AppState = Depends(get_state)):
    query = state.queries.get(body.queryId)
    if query is None:
        raise ApiError("query_expired", "That investigation has expired. Investigate the clip again.", 404)
    entry = state.store.add(body.name, query.clip_name, query.vector, query.display)
    state.queries.pop(body.queryId)
    return entry_detail(entry, query.vector)


@router.post("/gallery/from-video", status_code=201)
def add_from_video(
    name: str = Form("", max_length=200),
    video: UploadFile = File(...),
    state: AppState = Depends(get_state),
):
    # Cheap checks before the slow part. No name typed: the clip's file name is used.
    name = validate_name(name if name.strip() else name_from_filename(video.filename or ""))
    state.store.require_compatible()
    if state.store.name_taken(name):
        raise ApiError("name_taken", f"A case file named '{name}' already exists.", 409)
    pipeline = state.require_pipeline()

    with saved_upload(video, state.settings.max_upload_mb) as path:
        walk = pipeline.analyse(path)
    entry = state.store.add(name, video.filename or path.name, walk.vector, walk.display)
    return entry_detail(entry, walk.vector)
