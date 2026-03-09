from __future__ import annotations

from dataclasses import dataclass
import os

from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass(frozen=True, slots=True)
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 0.0


def build_chat_model(config: ModelConfig):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env to run the demo."
        )
    return ChatGoogleGenerativeAI(
        model=config.model_name,
        temperature=config.temperature,
        api_key=api_key,
    )