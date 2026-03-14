from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.models import (
    JobDetail,
    JobSummary,
    RequestResult,
)


class JobState:
    """Mutable state for a single batch job."""

    def __init__(self, job_id: str, total_requests: int) -> None:
        self.job_id = job_id
        self.status: str = "pending"  # pending | running | completed | failed | cancelled
        self.total_requests = total_requests
        self.completed: int = 0
        self.failed: int = 0
        self.created_at: datetime = datetime.now(timezone.utc)
        self.finished_at: datetime | None = None
        self.cancel_event: asyncio.Event = asyncio.Event()
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def _results_path(self) -> Path:
        return Path(settings.data_dir) / f"{self.job_id}.jsonl"

    async def add_result(self, result: RequestResult) -> None:
        async with self._lock:
            if result.status == "success":
                self.completed += 1
            else:
                self.failed += 1

            # Append to JSONL file
            line = result.model_dump_json() + "\n"
            self._results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._results_path, "a") as f:
                f.write(line)

    def mark_running(self) -> None:
        self.status = "running"

    def mark_finished(self) -> None:
        self.finished_at = datetime.now(timezone.utc)
        if self.cancel_event.is_set():
            self.status = "cancelled"
        elif self.failed == self.total_requests:
            self.status = "failed"
        else:
            self.status = "completed"

    def cancel(self) -> None:
        self.cancel_event.set()

    def to_summary(self) -> JobSummary:
        return JobSummary(
            job_id=self.job_id,
            status=self.status,
            total_requests=self.total_requests,
            completed=self.completed,
            failed=self.failed,
            created_at=self.created_at,
            finished_at=self.finished_at,
        )

    def to_detail(self, limit: int = 50, offset: int = 0) -> JobDetail:
        results: list[RequestResult] = []
        if self._results_path.exists():
            with open(self._results_path) as f:
                for i, line in enumerate(f):
                    if i < offset:
                        continue
                    if len(results) >= limit:
                        break
                    line = line.strip()
                    if line:
                        results.append(RequestResult.model_validate_json(line))
        return JobDetail(
            job_id=self.job_id,
            status=self.status,
            total_requests=self.total_requests,
            completed=self.completed,
            failed=self.failed,
            created_at=self.created_at,
            finished_at=self.finished_at,
            results=results,
            limit=limit,
            offset=offset,
        )


class _QueuedJob:
    """A job waiting to be processed, bundled with its parameters."""

    def __init__(
        self,
        job: JobState,
        requests: list,
        model: str | None,
        concurrency: int | None,
        max_retries: int | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> None:
        self.job = job
        self.requests = requests
        self.model = model
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens


class JobStore:
    """In-memory store for all jobs. Processes jobs sequentially via an async queue."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._queue: asyncio.Queue[_QueuedJob] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def create_job(self, total_requests: int) -> JobState:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = JobState(job_id=job_id, total_requests=total_requests)
        self._jobs[job_id] = job
        return job

    async def enqueue(self, queued_job: _QueuedJob) -> None:
        await self._queue.put(queued_job)

    def get_job(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobSummary]:
        return [job.to_summary() for job in self._jobs.values()]

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def start_worker(self) -> None:
        """Start the background worker that processes jobs one by one."""
        import logging
        from app.processor import process_batch

        logger = logging.getLogger("batchllm.worker")

        async def _worker() -> None:
            logger.info("Job queue worker started")
            while True:
                queued = await self._queue.get()
                job = queued.job
                logger.info(
                    "Starting job %s (%d requests, %d jobs still queued)",
                    job.job_id,
                    job.total_requests,
                    self._queue.qsize(),
                )
                try:
                    await process_batch(
                        requests=queued.requests,
                        job=job,
                        model=queued.model,
                        concurrency=queued.concurrency,
                        max_retries=queued.max_retries,
                        temperature=queued.temperature,
                        max_tokens=queued.max_tokens,
                    )
                except Exception:
                    logger.exception("Job %s crashed", job.job_id)
                    job.mark_finished()
                finally:
                    self._queue.task_done()

        self._worker_task = asyncio.create_task(_worker())

    async def stop_worker(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None


# Singleton
job_store = JobStore()
