from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from deepagents import create_deep_agent

from .prompts import (
    DEEPAGENTS_SYSTEM_PROMPT,
    RESEARCH_SUBAGENT_PROMPT,
    SYNTHESIS_SUBAGENT_PROMPT,
)
from .retrieval import CorpusIndex, format_search_hits
from .utils import extract_final_ai_text, retry_with_backoff


def build_search_tool(index: CorpusIndex):
    @tool
    def search_corpus(query: str, top_k: int = 5) -> str:
        """Search the corpus for relevant PDF chunks and return cited evidence."""

        hits = index.search(query, top_k=top_k)
        return format_search_hits(hits)

    return search_corpus


def build_deepagents_graph(
    *,
    index: CorpusIndex,
    model,
):
    search_tool = build_search_tool(index)
    subagents = [
        {
            "name": "research-analyst",
            "description": "Find the best supporting chunks and extract evidence.",
            "system_prompt": RESEARCH_SUBAGENT_PROMPT,
            "model": model,
            "tools": [search_tool],
        },
        {
            "name": "synthesis-writer",
            "description": "Turn evidence into a concise, cited answer.",
            "system_prompt": SYNTHESIS_SUBAGENT_PROMPT,
            "model": model,
            "tools": [],
        },
    ]

    return create_deep_agent(
        model=model,
        tools=[search_tool],
        system_prompt=DEEPAGENTS_SYSTEM_PROMPT,
        subagents=subagents,
    )


def run_deepagents_rag(
    *,
    question: str,
    index: CorpusIndex,
    model,
    retry_attempts: int,
) -> str:
    agent = build_deepagents_graph(index=index, model=model)
    result = retry_with_backoff(
        lambda: agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 25},
        ),
        attempts=retry_attempts,
        label="DeepAgents orchestration",
    )
    answer = extract_final_ai_text(result)
    if answer:
        return answer
    return extract_final_ai_text({"messages": result.get("messages", [])})
