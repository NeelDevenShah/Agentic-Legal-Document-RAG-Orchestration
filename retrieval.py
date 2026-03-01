from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(slots=True)
class SearchHit:
    score: float
    document: Document


@dataclass(slots=True)
class CorpusIndex:
    chunks: list[Document]
    vectorizer: TfidfVectorizer
    matrix: Any

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        query_matrix = self.vectorizer.transform([query])
        scores = cosine_similarity(query_matrix, self.matrix).ravel()
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchHit(score=float(scores[index]), document=self.chunks[index])
            for index in ranked_indices
            if scores[index] > 0
        ]


def build_corpus_index(chunks: list[Document]) -> CorpusIndex:
    if not chunks:
        raise ValueError("Cannot build an index from an empty chunk list")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20_000,
    )
    matrix = vectorizer.fit_transform([chunk.page_content for chunk in chunks])
    return CorpusIndex(chunks=chunks, vectorizer=vectorizer, matrix=matrix)


def format_search_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "No high-confidence matches were found in the corpus."

    lines: list[str] = []
    for hit in hits:
        metadata = hit.document.metadata
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        chunk_index = metadata.get("chunk_index", "?")
        excerpt = hit.document.page_content[:550]
        lines.append(
            f"- [{source} | page {page} | chunk {chunk_index} | score {hit.score:.3f}] {excerpt}"
        )
    return "\n".join(lines)
