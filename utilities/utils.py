from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar
import random
import re
import time

from langchain_core.messages import AIMessage

T = TypeVar("T")

_RATE_LIMIT_MARKERS = (
    "rate limit",
    "too many requests",
    "429",
    "temporarily unavailable",
    "resource exhausted",
    "timeout",
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return normalize_whitespace(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text"} and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
            elif isinstance(item, str):
                parts.append(item)
        return normalize_whitespace(" ".join(parts))
    if content is None:
        return ""
    return normalize_whitespace(str(content))


def extract_final_ai_text(result: Any) -> str:
    messages = []
    if isinstance(result, dict):
        messages = list(result.get("messages", []))
    elif hasattr(result, "get"):
        try:
            messages = list(result.get("messages", []))  # type: ignore[attr-defined]
        except Exception:
            messages = []

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = message_content_to_text(message.content)
            if text:
                return text
        if getattr(message, "type", None) == "ai":
            text = message_content_to_text(getattr(message, "content", ""))
            if text:
                return text

    if isinstance(result, dict):
        for key in ("answer", "final_answer", "output"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return normalize_whitespace(value)

    return ""


def retry_with_backoff(
    action: Callable[[], T],
    *,
    attempts: int,
    label: str,
) -> T:
    delay = 1.0
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:  # pragma: no cover - provider/network dependent
            last_error = exc
            message = str(exc).lower()
            is_retryable = any(marker in message for marker in _RATE_LIMIT_MARKERS)
            if attempt == attempts or not is_retryable:
                raise
            sleep_for = delay + random.uniform(0.0, 0.35)
            print(f"{label} failed on attempt {attempt}/{attempts}; retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
            delay *= 2

    assert last_error is not None
    raise last_error