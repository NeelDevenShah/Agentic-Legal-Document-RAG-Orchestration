from __future__ import annotations

from dataclasses import dataclass
import os

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


@dataclass(frozen=True, slots=True)
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 0.0


def build_chat_model(config: ModelConfig):
    provider = config.provider.lower().strip()
    if provider == "groq":
        return ChatGroq(model=config.model_name, temperature=config.temperature)
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env to run the OpenAI-backed demo."
            )
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )
    raise ValueError(
        f"Unsupported model provider '{config.provider}'. Use 'groq' or 'openai'."
    )