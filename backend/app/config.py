"""Central place for environment-driven settings.

Everything else in the app imports `settings` from here instead of calling
os.environ directly, so there's exactly one place that knows how config is
sourced.
"""
from typing import Optional

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
    # How many of the newest memory_entries rows retrieve_similar() scores.
    # (Previously it only ever looked at the newest `memory_top_k` rows.)
    memory_scan_limit: int = 500
    # User intervention (app.pipeline.intervention). A pause that nobody
    # answers auto-approves after this many seconds so an abandoned browser
    # tab can't wedge a run forever; 0 = wait indefinitely.
    intervention_timeout_seconds: int = 900
    # Max times one pause point may loop back ("re-plan", "regenerate")
    # before the pipeline proceeds with the latest result.
    intervention_max_revisions: int = 5
    # Batch processing & experiments (app.batch / app.experiments).
    # Concurrent runs per batch (Ollama itself may serialise generation; see README),
    # and a hard cap on the runs one batch/experiment may create.
    batch_max_concurrency: int = 5
    batch_max_runs: int = 100
    # Metrics & observability (app.pipeline.metrics). Rates are USD per 1K
    # tokens. Local Ollama has no API bill, so the defaults are a
    # cloud-equivalent reference price that makes runs/models comparable;
    # set both to 0 if you only care about tokens and latency.
    metrics_enabled: bool = True
    token_rate_in_per_1k: float = 0.0005
    token_rate_out_per_1k: float = 0.0015
    # Failure recovery: raw cosine (0-1) + type/category boosts, see
    # app.memory.find_similar_failures.
    failure_similarity_threshold: float = 0.65
    # "auto": ask the LLM to refine the repair hint only when the
    # deterministic parser is low-confidence. "always" | "never" also valid.
    failure_analyzer_llm_mode: str = "auto"
    # Ollama sampling / context options (Phase 18). None = leave Ollama's own default
    # (which is random sampling with a small context window). Per-run overrides in
    # app.run_config win over these. For reproducible experiments set a seed and
    # temperature 0, and record whatever you use in the paper.
    ollama_temperature: Optional[float] = None
    ollama_seed: Optional[int] = None
    ollama_num_ctx: Optional[int] = None
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
