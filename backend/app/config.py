"""Central place for environment-driven settings.

Everything else in the app imports `settings` from here instead of calling
os.environ directly, so there's exactly one place that knows how config is
sourced.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://aiops:aiops@localhost:5432/aiopsforge"
    redis_url: str = "redis://localhost:6379/0"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    mcp_server_url: str = "http://localhost:9000/mcp"
    workspace_root: str = "/workspace"
    max_qa_attempts: int = 3
    embedding_model: str = "nomic-embed-text"
    memory_similarity_threshold: float = 0.75
    memory_top_k: int = 3
    short_term_ttl_seconds: int = 3600
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
