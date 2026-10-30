from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import identify, video


def result(client, content=b"walk-a", name="clip.mp4") -> dict:
    status = identify(client, content, name)
    assert status["status"] == "done", status
    assert status["progress"] == 1.0
    return status["result"]


def test_health(client):
    body = client.get("/api/health").json()
    assert body["modelLoaded"] is True
    assert body["gallerySize"] == 0
    assert body["galleryCompatible"] is True
    assert body["threshold"] == 5.0 and body["thresholdCalibrated"] is False
    assert ".mp4" in body["videoExtensions"]


def test_identify_starts_a_job_and_reports_status(client):
    res = client.post("/api/identify", files=video())
    assert res.status_code == 202
    assert res.json()["status"] == "queued" and len(res.json()["jobId"]) == 32
    status = client.get(f"/api/identify/{res.json()['jobId']}/status")
    assert status.status_code == 200
    assert status.json()["status"] in ("queued", "detecting", "silhouettes", "embedding", "searching", "done")

    missing = client.get("/api/identify/nope/status")
    assert missing.status_code == 404 and missing.json()["error"] == "job_not_found"


def test_identify_on_empty_archive_is_unknown(client):
    body = result(client)
    assert body["isMatch"] is False and body["suspects"] == []
    assert body["framesWithPerson"] == 40 and body["framesUsed"] == 30
    assert len(body["querySilhouettes"]) == 10
    assert body["querySilhouettes"][0].startswith("data:image/png;base64,")
    assert "queryVector" not in body  # the vector stays on the server


def test_unknown_walker_saved_then_recognised(client):
    first = result(client, b"walk-nabeel", "a.mp4")
    saved = client.post("/api/gallery", json={"name": "Nabeel", "queryId": first["queryId"]})
    assert saved.status_code == 201
    assert saved.json()["videoPath"] == "a.mp4"
    assert len(saved.json()["silhouetteUrls"]) == 10

    again = result(client, b"walk-nabeel", "b.mp4")
    assert again["isMatch"] is True
    assert again["suspects"][0]["personId"] == "Nabeel"
    assert again["suspects"][0]["distance"] == 0.0
    assert again["suspects"][0]["thumbnailUrl"].startswith("/api/media/")
    assert client.get(again["suspects"][0]["thumbnailUrl"]).headers["content-type"] == "image/png"

    stranger = result(client, b"walk-someone-else")
    assert stranger["isMatch"] is False
    assert stranger["suspects"][0]["personId"] == "Nabeel"  # nearest, but past the match line


def test_save_errors(client):
    qid = result(client)["queryId"]
    assert client.post("/api/gallery", json={"name": "Ahmad", "queryId": qid}).status_code == 201

    expired = client.post("/api/gallery", json={"name": "Other", "queryId": qid})  # used up
    assert expired.status_code == 404 and expired.json()["error"] == "query_expired"

    qid2 = result(client, b"x")["queryId"]
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


def test_upload_errors_fail_before_a_job_starts(client):
    wrong = client.post("/api/identify", files=video(name="notes.txt"))
    assert wrong.status_code == 415 and wrong.json()["error"] == "unsupported_type"

    big = client.post("/api/identify", files=video(b"0" * (1024 * 1024 + 1)))
    assert big.status_code == 413 and big.json()["error"] == "file_too_large"

    missing = client.post("/api/identify")
    assert missing.status_code == 422 and missing.json()["error"] == "invalid_request"


def test_video_problems_end_the_job_with_an_error(client):
    nobody = identify(client, b"empty")
    assert nobody["status"] == "error" and nobody["error"]["error"] == "no_person"
    assert "result" not in nobody

    short = identify(client, b"short")
    assert short["status"] == "error" and short["error"]["error"] == "too_few_frames"


def test_uploads_are_not_kept(client, settings, tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    client.post("/api/gallery/from-video", data={"name": "A"}, files=video())
    result(client)
    identify(client, b"empty")  # a failed job cleans up too
    assert list((tmp_path / "tmp").iterdir()) == []
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


def test_rename_and_delete_case_files(client):
    client.post("/api/gallery/from-video", data={"name": "Kareem"}, files=video(b"k"))
    client.post("/api/gallery/from-video", data={"name": "Nabeel"}, files=video(b"n"))

    renamed = client.patch("/api/gallery/Kareem", json={"name": "Kareem Hamza"})
    assert renamed.status_code == 200 and renamed.json()["personId"] == "Kareem Hamza"
    assert len(renamed.json()["silhouetteUrls"]) == 10
    assert client.get(renamed.json()["silhouetteUrls"][0]).status_code == 200
    taken = client.patch("/api/gallery/Kareem%20Hamza", json={"name": "nabeel"})
    assert taken.status_code == 409

    assert client.delete("/api/gallery/Nabeel").status_code == 204
    assert client.delete("/api/gallery/Nabeel").status_code == 404
    names = [e["personId"] for e in client.get("/api/gallery").json()["entries"]]
    assert names == ["Kareem Hamza"]

    # The renamed walker is still recognised under the new name.
    again = result(client, b"k")
    assert again["isMatch"] and again["suspects"][0]["personId"] == "Kareem Hamza"


def test_from_video_without_a_name_uses_the_clip_name(client):
    res = client.post("/api/gallery/from-video", files=video(b"h", "hamed_bag (1).mp4"))
    assert res.status_code == 201 and res.json()["personId"] == "hamed_bag 1"
    res = client.post("/api/gallery/from-video", data={"name": "  "}, files=video(b"n", "Nabeel.mp4"))
    assert res.status_code == 201 and res.json()["personId"] == "Nabeel"
    # Same clip name again: taken, like a typed name.
    res = client.post("/api/gallery/from-video", files=video(b"n2", "nabeel.MP4"))
    assert res.status_code == 409 and res.json()["error"] == "name_taken"


def test_several_clips_per_person(client):
    client.post("/api/gallery/from-video", data={"name": "Hamed"}, files=video(b"hamed-normal", "hamed.mp4"))
    client.post("/api/gallery/from-video", data={"name": "Raid"}, files=video(b"raid", "raid.mp4"))

    # A bag clip of Hamed came out unknown; it's added to his case file from the Verdict page.
    bag = result(client, b"hamed-bag", "hamed_bag.mp4")
    added = client.post("/api/gallery/Hamed/clips", json={"queryId": bag["queryId"]})
    assert added.status_code == 201
    assert added.json()["clipCount"] == 2
    assert [c["videoPath"] for c in added.json()["clips"]] == ["hamed.mp4", "hamed_bag.mp4"]
    assert all(len(c["silhouetteUrls"]) == 10 for c in added.json()["clips"])
    used = client.post("/api/gallery/Hamed/clips", json={"queryId": bag["queryId"]})
    assert used.status_code == 404 and used.json()["error"] == "query_expired"

    # A coat clip, uploaded straight into the Archive.
    coat = client.post("/api/gallery/Hamed/clips/from-video", files=video(b"hamed-coat", "hamed_coat.mp4"))
    assert coat.status_code == 201 and coat.json()["clipCount"] == 3
    listed = {e["personId"]: e["clipCount"] for e in client.get("/api/gallery").json()["entries"]}
    assert listed == {"Hamed": 3, "Raid": 1}
    assert client.get("/api/health").json()["gallerySize"] == 2  # people, not clips

    # Another bag walk matches his bag clip, and Hamed is listed once.
    again = result(client, b"hamed-bag", "hamed_bag2.mp4")
    assert again["isMatch"] and again["suspects"][0]["personId"] == "Hamed"
    assert again["suspects"][0]["distance"] == 0.0
    assert again["suspects"][0]["matchedClip"] == "hamed_bag.mp4"
    assert again["suspects"][0]["clipCount"] == 3
    assert [s["personId"] for s in again["suspects"]] == ["Hamed", "Raid"]

    # Removing a clip; the last one can't go this way.
    clips = coat.json()["clips"]
    after = client.delete(f"/api/gallery/Hamed/clips/{clips[1]['clipId']}")
    assert after.status_code == 200 and after.json()["clipCount"] == 2
    client.delete(f"/api/gallery/Hamed/clips/{clips[2]['clipId']}")
    last = client.delete(f"/api/gallery/Hamed/clips/{clips[0]['clipId']}")
    assert last.status_code == 409 and last.json()["error"] == "last_clip"

    missing = client.post("/api/gallery/Nobody/clips/from-video", files=video(b"x"))
    assert missing.status_code == 404
