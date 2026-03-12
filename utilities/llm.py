from __future__ import annotations

from dataclasses import dataclass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


@dataclass(frozen=True, slots=True)
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 0.0


def build_chat_model(config: ModelConfig):
    provider = config.provider.lower().strip()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env to run the demo."
            )
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    elif provider == "gemini":
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
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'gemini'.")