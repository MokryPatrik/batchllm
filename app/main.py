import logging

from fastapi import FastAPI

from app.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="BatchLLM",
    description="Batch LLM request processor with configurable concurrency and provider support via LiteLLM",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
