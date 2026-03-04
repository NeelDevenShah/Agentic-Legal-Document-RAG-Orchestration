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
    provider: str = "groq"
    model_name: str = "llama-3.1-8b-instant"
    embedding_model_name: str = "nvidia/nemotron-3-embed-1b:free"
    temperature: float = 0.0
    retry_attempts: int = 4
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
            provider=os.getenv("MODEL_PROVIDER", "groq").strip().lower(),
            model_name=os.getenv("MODEL_NAME", "llama-3.1-8b-instant").strip(),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "nvidia/nemotron-3-embed-1b:free",
            ).strip(),
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0")),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "4")),
            default_question=os.getenv(
                "DEFAULT_QUESTION",
                "Summarize the most important issues, parties, and recurring themes in the provided documents.",
            ).strip(),
        )