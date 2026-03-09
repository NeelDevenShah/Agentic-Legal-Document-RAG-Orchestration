from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime configuration loaded from the environment."""

    data_dir: Path
    chunk_size: int = 1200
    chunk_overlap: int = 180
    top_k: int = 5
    provider: str = "gemini"
    model_name: str = "gemini-2.5-flash"
    embedding_model_name: str = "models/embedding-001"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str = ".qdrant"
    qdrant_collection_name: str = "virallens_corpus"
    mysql_url: str | None = None
    mysql_messages_table: str = "messages"
    mysql_sessions_table: str = "sessions"
    mysql_memory_table: str = "memory_entries"
    temperature: float = 0.0
    retry_attempts: int = 4
    embedding_batch_size: int = 64
    embedding_max_concurrency: int = 4
    jina_api_key: str | None = None
    jina_reranker_model: str = "jina-reranker-v3"
    rerank_candidate_multiplier: int = 4
    default_question: str = (
        "Summarize the most important issues, parties, and recurring themes in the provided documents."
    )

    @classmethod
    def from_env(cls) -> "AppConfig":
        data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
        return cls(
            data_dir=data_dir,
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
            top_k=int(os.getenv("TOP_K", "5")),
            provider=os.getenv("MODEL_PROVIDER", "gemini").strip().lower(),
            model_name=os.getenv("MODEL_NAME", "gemini-2.5-flash").strip(),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "models/embedding-001",
            ).strip(),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_path=os.getenv("QDRANT_PATH", ".qdrant").strip(),
            qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "virallens_corpus").strip(),
            mysql_url=os.getenv("MYSQL_URL") or None,
            mysql_messages_table=os.getenv("MYSQL_MESSAGES_TABLE", "messages").strip(),
            mysql_sessions_table=os.getenv("MYSQL_SESSIONS_TABLE", "sessions").strip(),
            mysql_memory_table=os.getenv("MYSQL_MEMORY_TABLE", "memory_entries").strip(),
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0")),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "4")),
            embedding_batch_size=max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))),
            embedding_max_concurrency=max(1, int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "4"))),
            jina_api_key=os.getenv("JINA_API_KEY") or None,
            jina_reranker_model=os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v3").strip(),
            rerank_candidate_multiplier=max(1, int(os.getenv("RERANK_CANDIDATE_MULTIPLIER", "4"))),
            default_question=os.getenv(
                "DEFAULT_QUESTION",
                "Summarize the most important issues, parties, and recurring themes in the provided documents.",
            ).strip(),
        )