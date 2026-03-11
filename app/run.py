"""Entrypoint that reads BATCHLLM_PORT from env so the port is configurable at runtime."""
import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
