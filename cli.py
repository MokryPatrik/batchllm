"""CLI entry point for file-based batch processing (no web server)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

from app.config import settings
from app.job_store import JobState
from app.models import LLMRequest, LLMMessage, RequestResult
from app.processor import process_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("batchllm.cli")


def load_requests(path: Path) -> list[LLMRequest]:
    requests: list[LLMRequest] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON on line %d: %s", lineno, e)
                continue

            req_id = data.get("id", f"req_{uuid.uuid4().hex[:8]}")
            messages = [LLMMessage(**m) for m in data.get("messages", [])]
            if not messages:
                logger.warning("Line %d: no messages, skipping", lineno)
                continue

            requests.append(
                LLMRequest(
                    id=req_id,
                    messages=messages,
                    model=data.get("model"),
                    temperature=data.get("temperature"),
                    max_tokens=data.get("max_tokens"),
                )
            )
    return requests


def load_completed_ids(path: Path) -> set[str]:
    """Load IDs of already-completed requests for resume support."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "id" in data:
                    ids.add(data["id"])
            except json.JSONDecodeError:
                continue
    return ids


async def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    all_requests = load_requests(input_path)
    logger.info("Loaded %d requests from %s", len(all_requests), input_path)

    # Resume: skip already completed
    completed_ids = load_completed_ids(output_path)
    if completed_ids:
        before = len(all_requests)
        all_requests = [r for r in all_requests if r.id not in completed_ids]
        logger.info(
            "Resuming: skipping %d already completed, %d remaining",
            before - len(all_requests),
            len(all_requests),
        )

    if not all_requests:
        logger.info("No requests to process")
        return

    # Override data_dir so results go to the output file location
    settings.data_dir = str(output_path.parent)

    job = JobState(job_id=output_path.stem, total_requests=len(all_requests))

    await process_batch(
        requests=all_requests,
        job=job,
        model=args.model,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    logger.info(
        "Done: %d succeeded, %d failed. Results in %s",
        job.completed,
        job.failed,
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BatchLLM CLI")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", default="output.jsonl", help="Output JSONL file")
    parser.add_argument("--model", default=None, help="LiteLLM model identifier")
    parser.add_argument("--concurrency", type=int, default=None, help="Max parallel requests")
    parser.add_argument("--max-retries", type=int, default=None, help="Max retries per request")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
