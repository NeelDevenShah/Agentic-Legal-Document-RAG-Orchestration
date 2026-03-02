from __future__ import annotations

from dataclasses import dataclass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
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
    if provider in {"gemini", "google"}:
        return ChatGoogleGenerativeAI(model=config.model_name, temperature=config.temperature)
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env to run the OpenRouter-backed demo."
            )
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Multi-Agent RAG",
            },
        )
    raise ValueError(
        f"Unsupported model provider '{config.provider}'. Use 'groq', 'gemini', or 'openrouter'."
    )