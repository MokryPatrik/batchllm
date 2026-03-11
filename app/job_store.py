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

    def to_detail(self) -> JobDetail:
        results: list[RequestResult] = []
        if self._results_path.exists():
            with open(self._results_path) as f:
                for line in f:
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
        )


class JobStore:
    """In-memory store for all jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}

    def create_job(self, total_requests: int) -> JobState:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = JobState(job_id=job_id, total_requests=total_requests)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobSummary]:
        return [job.to_summary() for job in self._jobs.values()]


# Singleton
job_store = JobStore()
