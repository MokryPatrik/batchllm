from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# --- Request models ---


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMRequest(BaseModel):
    id: str | None = None
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class CreateJobRequest(BaseModel):
    requests: list[LLMRequest]
    model: str | None = None
    concurrency: int | None = None
    max_retries: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None


# --- Response models ---


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RequestResult(BaseModel):
    id: str
    status: str  # "success" | "error"
    response: str | None = None
    error: str | None = None
    attempts: int = 1
    usage: TokenUsage | None = None


class JobSummary(BaseModel):
    job_id: str
    status: str  # pending | running | completed | failed | cancelled
    total_requests: int
    completed: int = 0
    failed: int = 0
    created_at: datetime
    finished_at: datetime | None = None


class JobDetail(JobSummary):
    results: list[RequestResult] = []


class CreateJobResponse(BaseModel):
    job_id: str
    status: str
    total_requests: int
    created_at: datetime


class JobListResponse(BaseModel):
    jobs: list[JobSummary]


class CancelJobResponse(BaseModel):
    job_id: str
    status: str


class ErrorResponse(BaseModel):
    detail: str
