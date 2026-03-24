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
    openai_api_key: str | None = None
    model_name: str = "gpt-5.4-mini"
    embedding_model_name: str = "text-embedding-3-small"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str = ".qdrant"
    qdrant_collection_name: str = "virallens_corpus"
    retry_attempts: int = 4
    embedding_batch_size: int = 64
    embedding_max_concurrency: int = 4
    upload_staging_dir: Path = Path("data/uploads")
    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = 7860
    jina_api_key: str | None = None
    jina_reranker_model: str = "jina-reranker-v3"
    rerank_candidate_multiplier: int = 4
    default_question: str = (
        "Summarize the most important issues, parties, and recurring themes in the provided documents."
    )

    @classmethod
    def from_env(cls) -> "AppConfig":
        provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
        if provider != "openai":
            raise ValueError("Only MODEL_PROVIDER=openai is supported.")

        data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
        return cls(
            data_dir=data_dir,
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
            top_k=int(os.getenv("TOP_K", "5")),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            model_name=os.getenv("MODEL_NAME", "gpt-5.4-mini").strip(),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "text-embedding-3-small",
            ).strip(),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_path=os.getenv("QDRANT_PATH", ".qdrant").strip(),
            qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "virallens_corpus").strip(),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "4")),
            embedding_batch_size=max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))),
            embedding_max_concurrency=max(1, int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "4"))),
            upload_staging_dir=Path(os.getenv("UPLOAD_STAGING_DIR", "data/uploads")).resolve(),
            gradio_server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0").strip(),
            gradio_server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
            jina_api_key=os.getenv("JINA_API_KEY") or None,
            jina_reranker_model=os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v3").strip(),
            rerank_candidate_multiplier=max(1, int(os.getenv("RERANK_CANDIDATE_MULTIPLIER", "4"))),
            default_question=os.getenv(
                "DEFAULT_QUESTION",
                "Summarize the most important issues, parties, and recurring themes in the provided documents.",
            ).strip(),
        )
