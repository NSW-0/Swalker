from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from pydantic import BaseModel, Field

from app.api.serialize import person_detail, person_summary
from app.api.uploads import saved_upload
from app.errors import ApiError
from app.gallery.store import name_from_filename, validate_name
from app.queries import CachedQuery
from app.state import AppState, get_state

router = APIRouter()


def detail(state: AppState, person_id: str) -> dict:
    found = state.store.find(person_id)
    if found is None:
        raise ApiError("not_found", f"No case file named '{person_id}'.", 404)
    return person_detail(*found)


def take_query(state: AppState, query_id: str) -> CachedQuery:
    query = state.queries.get(query_id)
    if query is None:
        raise ApiError("query_expired", "That investigation has expired. Investigate the clip again.", 404)
    return query


@router.get("/gallery")
def list_gallery(state: AppState = Depends(get_state)):
    people, features = state.store.snapshot()
    entries, row = [], 0
    for p in people:
        entries.append(person_summary(p, features[row : row + len(p.clips)]))
        row += len(p.clips)
    return {"entries": entries, "compatible": state.store.compatible}


@router.get("/gallery/{person_id}")
def get_case_file(person_id: str, state: AppState = Depends(get_state)):
    return detail(state, person_id)


class Rename(BaseModel):
    name: str = Field(max_length=200)


@router.patch("/gallery/{person_id}")
def rename_case_file(person_id: str, body: Rename, state: AppState = Depends(get_state)):
    person = state.store.rename(person_id, body.name)
    return detail(state, person.personId)


@router.delete("/gallery/{person_id}", status_code=204)
def delete_case_file(person_id: str, state: AppState = Depends(get_state)):
    state.store.remove(person_id)
    return Response(status_code=204)


class SaveQuery(BaseModel):
    name: str = Field(max_length=200)
    queryId: str = Field(max_length=64)


@router.post("/gallery", status_code=201)
def save_query(body: SaveQuery, state: AppState = Depends(get_state)):
    """An investigated walk becomes a new person, without uploading it again."""
    query = take_query(state, body.queryId)
    person = state.store.add(body.name, query.clip_name, query.vector, query.display)
    state.queries.pop(body.queryId)
    return detail(state, person.personId)


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
    person = state.store.add(name, video.filename or path.name, walk.vector, walk.display)
    return detail(state, person.personId)


class AddClipFromQuery(BaseModel):
    queryId: str = Field(max_length=64)


@router.post("/gallery/{person_id}/clips", status_code=201)
def add_clip_from_query(person_id: str, body: AddClipFromQuery, state: AppState = Depends(get_state)):
    """An investigated walk becomes another clip of someone already in the Archive."""
    query = take_query(state, body.queryId)
    state.store.add_clip(person_id, query.clip_name, query.vector, query.display)
    state.queries.pop(body.queryId)
    return detail(state, person_id)


@router.post("/gallery/{person_id}/clips/from-video", status_code=201)
def add_clip_from_video(person_id: str, video: UploadFile = File(...), state: AppState = Depends(get_state)):
    # Cheap checks before the slow part.
    detail(state, person_id)
    state.store.require_compatible()
    pipeline = state.require_pipeline()

    with saved_upload(video, state.settings.max_upload_mb) as path:
        walk = pipeline.analyse(path)
    state.store.add_clip(person_id, video.filename or path.name, walk.vector, walk.display)
    return detail(state, person_id)


@router.delete("/gallery/{person_id}/clips/{clip_id}")
def remove_clip(person_id: str, clip_id: str, state: AppState = Depends(get_state)):
    state.store.remove_clip(person_id, clip_id)
    return detail(state, person_id)
