from __future__ import annotations

from dataclasses import dataclass

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass(frozen=True, slots=True)
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 0.0


def build_chat_model(config: ModelConfig):
    provider = config.provider.lower().strip()
    if provider == "groq":
        return ChatGroq(model=config.model_name, temperature=config.temperature)
    if provider in {"gemini", "google"}:
        return ChatGoogleGenerativeAI(model=config.model_name, temperature=config.temperature)
    raise ValueError(
        f"Unsupported model provider '{config.provider}'. Use 'groq' or 'gemini'."
    )
