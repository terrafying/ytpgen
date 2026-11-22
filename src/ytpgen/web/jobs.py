"""In-process background job runner for the web UI.

Heavy operations (ingest / analyze / generate / auto / render) run on
worker threads so a 1h+ video analysis never hangs a browser request.
Jobs are tracked in a simple registry with status, progress lines, and
the resulting artifact path; the UI polls /jobs/<id>/status.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_MAX_FINISHED = 50  # keep last N finished jobs for status polling


@dataclass
class Job:
    id: str
    kind: str
    label: str
    fn: Callable[[], Any]
    status: str = "queued"  # queued | running | done | error
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str = ""
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "log": self.log[-30:],
            "result": self.result
            if isinstance(self.result, (str, int, float, dict, list, type(None)))
            else str(self.result),
        }


class JobRunner:
    """Tiny thread-pool job registry. One process, no external broker."""

    def __init__(self, workers: int = 2) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(workers)
        self._threads: list[threading.Thread] = []

    def submit(self, kind: str, label: str, fn: Callable[[], Any]) -> str:
        """Enqueue *fn*; returns the job id immediately."""
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, label=label, fn=fn)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        t = threading.Thread(target=self._run, args=(job,), daemon=True,
                             name=f"ytpgen-job-{job.id}")
        self._threads.append(t)
        t.start()
        logger.info("job %s queued: %s", job.id, label)
        return job.id

    def _run(self, job: Job) -> None:
        with self._sem:
            job.status = "running"
            job.started_at = time.time()
            try:
                job.result = job.fn()
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 — jobs isolate their own failures
                job.status = "error"
                job.error = str(exc)
                logger.exception("job %s failed", job.id)
            finally:
                job.finished_at = time.time()
                self._gc()

    def _gc(self) -> None:
        with self._lock:
            finished = [j for j in self._order if self._jobs[j].status in ("done", "error")]
            excess = max(0, len(finished) - _MAX_FINISHED)
            for jid in finished[:excess]:
                self._jobs.pop(jid, None)
                self._order.remove(jid)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 25) -> list[Job]:
        with self._lock:
            ids = list(reversed(self._order))
        return [self._jobs[i] for i in ids if i in self._jobs][:limit]

    def status(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None
