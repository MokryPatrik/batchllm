from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server auth
    api_key: str = "changeme"

    # LLM defaults
    default_model: str = "moonshot/moonshot-v1-8k"
    default_concurrency: int = 10
    default_max_retries: int = 3
    default_temperature: float = 0.7
    default_max_tokens: int = 1024

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Storage
    data_dir: str = "./data"

    model_config = {"env_prefix": "BATCHLLM_"}


settings = Settings()
