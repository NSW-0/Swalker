from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import video


def test_health(client):
    body = client.get("/api/health").json()
    assert body["modelLoaded"] is True
    assert body["gallerySize"] == 0
    assert body["galleryCompatible"] is True
    assert body["threshold"] == 5.0 and body["thresholdCalibrated"] is False
    assert ".mp4" in body["videoExtensions"]


def test_identify_on_empty_archive_is_unknown(client):
    res = client.post("/api/identify", files=video())
    assert res.status_code == 200
    body = res.json()
    assert body["isMatch"] is False and body["suspects"] == []
    assert body["framesWithPerson"] == 40 and body["framesUsed"] == 30
    assert len(body["querySilhouettes"]) == 10
    assert body["querySilhouettes"][0].startswith("data:image/png;base64,")
    assert "queryVector" not in body  # the vector stays on the server


def test_unknown_walker_saved_then_recognised(client):
    first = client.post("/api/identify", files=video(b"walk-nabeel", "a.mp4")).json()
    saved = client.post("/api/gallery", json={"name": "Nabeel", "queryId": first["queryId"]})
    assert saved.status_code == 201
    assert saved.json()["videoPath"] == "a.mp4"
    assert len(saved.json()["silhouetteUrls"]) == 10

    again = client.post("/api/identify", files=video(b"walk-nabeel", "b.mp4")).json()
    assert again["isMatch"] is True
    assert again["suspects"][0]["personId"] == "Nabeel"
    assert again["suspects"][0]["distance"] == 0.0
    assert again["suspects"][0]["thumbnailUrl"].startswith("/api/media/")
    assert client.get(again["suspects"][0]["thumbnailUrl"]).headers["content-type"] == "image/png"

    stranger = client.post("/api/identify", files=video(b"walk-someone-else")).json()
    assert stranger["isMatch"] is False
    assert stranger["suspects"][0]["personId"] == "Nabeel"  # nearest, but past the match line


def test_save_errors(client):
    qid = client.post("/api/identify", files=video()).json()["queryId"]
    assert client.post("/api/gallery", json={"name": "Ahmad", "queryId": qid}).status_code == 201

    expired = client.post("/api/gallery", json={"name": "Other", "queryId": qid})  # used up
    assert expired.status_code == 404 and expired.json()["error"] == "query_expired"

    qid2 = client.post("/api/identify", files=video(b"x")).json()["queryId"]
    taken = client.post("/api/gallery", json={"name": "ahmad", "queryId": qid2})
    assert taken.status_code == 409 and taken.json()["error"] == "name_taken"
    bad = client.post("/api/gallery", json={"name": "../x", "queryId": qid2})
    assert bad.status_code == 422 and bad.json()["error"] == "invalid_name"
    # A failed save keeps the query, so the user can pick another name.
    assert client.post("/api/gallery", json={"name": "Ahmad 2", "queryId": qid2}).status_code == 201


def test_gallery_list_and_detail(client):
    client.post("/api/gallery/from-video", data={"name": "Kareem"}, files=video(b"k", "kareem.mp4"))
    entries = client.get("/api/gallery").json()["entries"]
    assert [e["personId"] for e in entries] == ["Kareem"]
    assert len(entries[0]["vectorPreview"]) == 8 and entries[0]["featureDim"] == 4096
    assert entries[0]["l2Norm"] > 0

    detail = client.get("/api/gallery/Kareem").json()
    assert len(detail["silhouetteUrls"]) == 10
    missing = client.get("/api/gallery/Nobody")
    assert missing.status_code == 404 and missing.json()["error"] == "not_found"


def test_from_video_rejects_taken_name_before_processing(client):
    client.post("/api/gallery/from-video", data={"name": "Kareem"}, files=video(b"k"))
    res = client.post("/api/gallery/from-video", data={"name": "KAREEM"}, files=video(b"k2"))
    assert res.status_code == 409


def test_upload_errors(client):
    wrong = client.post("/api/identify", files=video(name="notes.txt"))
    assert wrong.status_code == 415 and wrong.json()["error"] == "unsupported_type"

    big = client.post("/api/identify", files=video(b"0" * (1024 * 1024 + 1)))
    assert big.status_code == 413 and big.json()["error"] == "file_too_large"

    nobody = client.post("/api/identify", files=video(b"empty"))
    assert nobody.status_code == 422 and nobody.json()["error"] == "no_person"

    short = client.post("/api/identify", files=video(b"short"))
    assert short.status_code == 422 and short.json()["error"] == "too_few_frames"

    missing = client.post("/api/identify")
    assert missing.status_code == 422 and missing.json()["error"] == "invalid_request"


def test_uploads_are_not_kept(client, settings):
    client.post("/api/gallery/from-video", data={"name": "A"}, files=video())
    client.post("/api/identify", files=video())
    stored = {p.suffix for p in settings.storage_dir.rglob("*") if p.is_file()}
    assert stored <= {".json", ".npy", ".png"}


def test_model_not_loaded(settings, monkeypatch):
    import app.main

    def fail(state):
        state.load_error = "FileNotFoundError: no checkpoint"

    monkeypatch.setattr(app.main, "load_pipeline", fail)
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/health").json()["modelLoaded"] is False
        res = c.post("/api/identify", files=video())
        assert res.status_code == 503 and res.json()["error"] == "model_not_loaded"
