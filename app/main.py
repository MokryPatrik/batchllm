import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.job_store import job_store
from app.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch the sequential job worker
    await job_store.start_worker()
    yield
    # Shutdown: stop the worker
    await job_store.stop_worker()


app = FastAPI(
    title="BatchLLM",
    description="Batch LLM request processor with configurable concurrency and provider support via LiteLLM",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
