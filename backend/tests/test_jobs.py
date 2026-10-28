from app.jobs import JobStore


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_progress_follows_the_stages_and_never_goes_back():
    jobs = JobStore(ttl_seconds=60)
    job = jobs.create()
    assert jobs.get(job.id).public() == {"jobId": job.id, "status": "queued", "progress": 0.0}
    jobs.stage(job.id, "detecting", 0.5)
    assert jobs.get(job.id).progress == 0.375  # half of the 0-0.75 detecting span
    jobs.stage(job.id, "detecting", 0.2)  # a late, smaller report doesn't move it back
    assert jobs.get(job.id).progress == 0.375
    jobs.stage(job.id, "embedding")
    assert jobs.get(job.id).status == "embedding" and jobs.get(job.id).progress == 0.8
    jobs.finish(job.id, {"isMatch": True})
    done = jobs.get(job.id).public()
    assert done["status"] == "done" and done["progress"] == 1.0 and done["result"] == {"isMatch": True}
    jobs.stage(job.id, "detecting", 0.1)  # nothing changes a finished job
    assert jobs.get(job.id).status == "done"


def test_errors_and_expiry():
    clock = Clock()
    jobs = JobStore(ttl_seconds=60, clock=clock)
    job = jobs.create()
    jobs.fail(job.id, "no_person", "No person was found in the video.")
    assert jobs.get(job.id).public()["error"] == {"error": "no_person", "message": "No person was found in the video."}
    clock.now = 61
    assert jobs.get(job.id) is None
