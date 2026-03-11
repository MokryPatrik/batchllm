from __future__ import annotations

import asyncio
import logging
import uuid

import litellm

from app.config import settings
from app.job_store import JobState
from app.models import LLMRequest, RequestResult, TokenUsage

logger = logging.getLogger("batchllm.processor")


async def _process_single_request(
    request: LLMRequest,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    job: JobState,
) -> RequestResult:
    """Process a single LLM request with retries and concurrency control."""
    req_id = request.id or f"auto_{uuid.uuid4().hex[:8]}"
    req_model = request.model or model
    req_temp = request.temperature if request.temperature is not None else temperature
    req_max_tokens = request.max_tokens if request.max_tokens is not None else max_tokens

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    last_error = ""
    for attempt in range(1, max_retries + 1):
        # Check cancellation before each attempt
        if job.cancel_event.is_set():
            return RequestResult(
                id=req_id,
                status="error",
                error="Job cancelled",
                attempts=attempt,
            )

        try:
            async with semaphore:
                response = await litellm.acompletion(
                    model=req_model,
                    messages=messages,
                    temperature=req_temp,
                    max_tokens=req_max_tokens,
                )

            content = response.choices[0].message.content or ""
            usage = None
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return RequestResult(
                id=req_id,
                status="success",
                response=content,
                usage=usage,
                attempts=attempt,
            )

        except Exception as e:
            last_error = str(e)
            logger.warning(
                "Request %s attempt %d/%d failed: %s",
                req_id,
                attempt,
                max_retries,
                last_error,
            )
            if attempt < max_retries:
                backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s, ...
                await asyncio.sleep(backoff)

    return RequestResult(
        id=req_id,
        status="error",
        error=last_error,
        attempts=max_retries,
    )


async def process_batch(
    requests: list[LLMRequest],
    job: JobState,
    model: str | None = None,
    concurrency: int | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> None:
    """Process a batch of LLM requests with controlled concurrency."""
    _model = model or settings.default_model
    _concurrency = concurrency or settings.default_concurrency
    _max_retries = max_retries or settings.default_max_retries
    _temperature = temperature if temperature is not None else settings.default_temperature
    _max_tokens = max_tokens or settings.default_max_tokens

    semaphore = asyncio.Semaphore(_concurrency)
    job.mark_running()

    total = len(requests)
    completed_count = 0

    async def _handle(req: LLMRequest) -> None:
        nonlocal completed_count
        result = await _process_single_request(
            request=req,
            model=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
            max_retries=_max_retries,
            semaphore=semaphore,
            job=job,
        )
        await job.add_result(result)
        completed_count += 1
        if completed_count % max(1, total // 20) == 0 or completed_count == total:
            logger.info(
                "Job %s progress: %d/%d (%.1f%%)",
                job.job_id,
                completed_count,
                total,
                completed_count / total * 100,
            )

    tasks = [asyncio.create_task(_handle(req)) for req in requests]
    await asyncio.gather(*tasks)
    job.mark_finished()
    logger.info(
        "Job %s finished: %d succeeded, %d failed out of %d",
        job.job_id,
        job.completed,
        job.failed,
        total,
    )
