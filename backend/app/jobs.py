"""Identify jobs: POST /api/identify starts one, GET /api/identify/{jobId}/status reports where it is
(HANDOFF §7.3). The stages are the four the Verdict loader shows, plus queued/done/error."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

MAX_JOBS = 200

# Share of the overall progress bar each stage covers. Finding the person (YOLO on every frame)
# is by far the slowest part; the others take a fraction of a second on a GPU.
STAGE_SPAN = {
    "queued": (0.0, 0.0),
    "detecting": (0.0, 0.75),
    "silhouettes": (0.75, 0.8),
    "embedding": (0.8, 0.95),
    "searching": (0.95, 1.0),
    "done": (1.0, 1.0),
}


@dataclass
class Job:
    id: str
    status: str = "queued"
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    expires: float = 0.0

    def public(self) -> dict[str, Any]:
        body: dict[str, Any] = {"jobId": self.id, "status": self.status, "progress": round(self.progress, 3)}
        if self.result is not None:
            body["result"] = self.result
        if self.error is not None:
            body["error"] = self.error
        return body


@dataclass
class JobStore:
    ttl_seconds: float
    clock: Any = time.monotonic
    _jobs: dict[str, Job] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self) -> Job:
        with self._lock:
            self._prune()
            while len(self._jobs) >= MAX_JOBS:
                self._jobs.pop(next(iter(self._jobs)))  # oldest first
            job = Job(id=uuid.uuid4().hex, expires=self.clock() + self.ttl_seconds)
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._prune()
            job = self._jobs.get(job_id)
            return Job(**vars(job)) if job else None  # a copy, so readers never see a half update

    def stage(self, job_id: str, status: str, fraction: float = 0.0) -> None:
        lo, hi = STAGE_SPAN[status]
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status not in ("done", "error"):
                job.status = status
                job.progress = max(job.progress, lo + (hi - lo) * min(1.0, max(0.0, fraction)))

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            if job := self._jobs.get(job_id):
                job.status, job.progress, job.result = "done", 1.0, result

    def fail(self, job_id: str, code: str, message: str) -> None:
        with self._lock:
            if job := self._jobs.get(job_id):
                job.status, job.error = "error", {"error": code, "message": message}

    def _prune(self) -> None:
        now = self.clock()
        for key in [k for k, j in self._jobs.items() if j.expires <= now]:
            del self._jobs[key]
