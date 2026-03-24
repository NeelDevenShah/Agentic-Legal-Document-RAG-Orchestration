from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_name: str
    api_key: str | None


def build_chat_model(config: ModelConfig):
    if not config.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env to run the demo.")
    return ChatOpenAI(
        model=config.model_name,
        api_key=config.api_key,
    )