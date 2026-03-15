from __future__ import annotations

from config import AppConfig


def clear_persistence_stores(config: AppConfig, *, qdrant_clear_message: str) -> str:
    return qdrant_clear_message
