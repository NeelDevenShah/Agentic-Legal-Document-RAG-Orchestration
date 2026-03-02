from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from prompts import LANGGRAPH_SYSTEM_PROMPT
from utilities import CorpusIndex, extract_final_ai_text, format_search_hits, retry_with_backoff


class RagState(TypedDict, total=False):
    question: str
    context: str
    answer: str


def build_langgraph_rag_graph(
    *,
    index: CorpusIndex,
    model,
    top_k: int,
    retry_attempts: int,
):
    def retrieve(state: RagState) -> dict[str, str]:
        hits = index.search(state["question"], top_k=top_k)
        return {"context": format_search_hits(hits)}

    def answer(state: RagState) -> dict[str, str]:
        messages = [
            SystemMessage(content=LANGGRAPH_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n\n"
                    f"Retrieved context:\n{state['context']}\n\n"
                    "Write the best possible answer and cite the corpus metadata inline."
                )
            ),
        ]
        response = retry_with_backoff(
            lambda: model.invoke(messages),
            attempts=retry_attempts,
            label="LangGraph answer generation",
        )
        return {"answer": extract_final_ai_text({"messages": [response]})}

    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def run_langgraph_rag(
    *,
    question: str,
    index: CorpusIndex,
    model,
    top_k: int,
    retry_attempts: int,
) -> str:
    graph = build_langgraph_rag_graph(
        index=index,
        model=model,
        top_k=top_k,
        retry_attempts=retry_attempts,
    )
    result = retry_with_backoff(
        lambda: graph.invoke({"question": question}),
        attempts=retry_attempts,
        label="LangGraph orchestration",
    )
    answer = result.get("answer", "")
    if answer:
        return answer
    return extract_final_ai_text(result)
