from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://axolotl:axolotl@localhost:5432/axolotl"
    redis_url: str = "redis://localhost:6379/0"

    # LLM
    default_model: str = "openai:gpt-4.1-mini"
    default_temperature: float = 0.0

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # App
    app_env: str = "development"
    api_port: int = 8000
    admin_port: int = 8080

    # LangSmith (optional)
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "axolotl"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
