from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.auth import verify_api_key
from app.job_store import _QueuedJob, job_store
from app.models import (
    CancelJobResponse,
    CreateJobRequest,
    CreateJobResponse,
    JobDetail,
    JobListResponse,
)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/jobs", response_model=CreateJobResponse, status_code=202)
async def create_job(body: CreateJobRequest) -> CreateJobResponse:
    if not body.requests:
        raise HTTPException(status_code=400, detail="No requests provided")

    # Assign IDs to requests that don't have one
    for req in body.requests:
        if req.id is None:
            req.id = f"req_{uuid.uuid4().hex[:8]}"

    job = job_store.create_job(total_requests=len(body.requests))

    # Enqueue for sequential processing
    await job_store.enqueue(
        _QueuedJob(
            job=job,
            requests=body.requests,
            model=body.model,
            concurrency=body.concurrency,
            max_retries=body.max_retries,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    )

    return CreateJobResponse(
        job_id=job.job_id,
        status=job.status,
        total_requests=job.total_requests,
        created_at=job.created_at,
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs() -> JobListResponse:
    return JobListResponse(jobs=job_store.list_jobs())


@router.get("/jobs/{job_id}", response_model=JobDetail)
async def get_job(job_id: str) -> JobDetail:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_detail()


@router.delete("/jobs/{job_id}", response_model=CancelJobResponse)
async def cancel_job(job_id: str) -> CancelJobResponse:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("pending", "running"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{job.status}'",
        )
    job.cancel()
    return CancelJobResponse(job_id=job.job_id, status="cancelled")
