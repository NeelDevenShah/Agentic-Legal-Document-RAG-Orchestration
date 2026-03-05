from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .utils import message_content_to_text, normalize_whitespace


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_METADATA_PROMPT = (
    "You create document-level metadata for retrieval. "
    "Return valid JSON only with these keys: title, document_type, summary, keywords, entities, topics, important_dates, parties, jurisdiction. "
    "Use concise strings or arrays of strings. If a field is unknown, use null or an empty list. Do not wrap the answer in markdown."
)


@dataclass(slots=True)
class SearchHit:
    score: float
    document: Document


@dataclass(slots=True)
class BM25Index:
    tokenized_documents: list[list[str]]
    document_frequencies: list[Counter[str]]
    document_lengths: list[int]
    average_document_length: float
    idf: dict[str, float]
    k1: float = 1.5
    b: float = 0.75

    def score(self, query: str) -> np.ndarray:
        query_tokens = _tokenize(query)
        scores = np.zeros(len(self.tokenized_documents), dtype=np.float32)
        if not query_tokens:
            return scores

        query_terms = Counter(query_tokens)
        for term, term_frequency_in_query in query_terms.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, doc_frequencies in enumerate(self.document_frequencies):
                term_frequency = doc_frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                document_length = self.document_lengths[index]
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * document_length / self.average_document_length
                )
                scores[index] += (
                    idf
                    * term_frequency
                    * (self.k1 + 1)
                    / denominator
                    * term_frequency_in_query
                )
        return scores

    def ranked_indices(self, query: str, *, limit: int) -> list[int]:
        scores = self.score(query)
        ranked = np.argsort(scores)[::-1]
        return [index for index in ranked[:limit] if scores[index] > 0]


@dataclass(slots=True)
class CorpusIndex:
    chunks: list[Document]
    chunk_ids: list[str]
    chunk_id_to_index: dict[str, int]
    embeddings: Embeddings
    qdrant_client: QdrantClient
    collection_name: str
    bm25_index: BM25Index

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        semantic_ranked = self._semantic_ranked_indices(query, limit=max(top_k * 5, 10))
        bm25_ranked = self.bm25_index.ranked_indices(query, limit=max(top_k * 5, 10))
        fused_scores = _reciprocal_rank_fusion([semantic_ranked, bm25_ranked])

        ranked_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        return [
            SearchHit(
                score=float(fused_scores[index]),
                document=_document_with_retrieval_metadata(self.chunks[index], fused_scores[index]),
            )
            for index in ranked_indices
        ]

    def _semantic_ranked_indices(self, query: str, *, limit: int) -> list[int]:
        query_vector = self.embeddings.embed_query(query)
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=False,
        )

        ranked_indices: list[int] = []
        for result in results:
            index = self.chunk_id_to_index.get(str(result.id))
            if index is not None:
                ranked_indices.append(index)
        return ranked_indices


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _reciprocal_rank_fusion(ranked_lists: list[list[int]], *, k: int = 60) -> dict[int, float]:
    fused_scores: dict[int, float] = defaultdict(float)
    for ranked_indices in ranked_lists:
        for rank, index in enumerate(ranked_indices, start=1):
            fused_scores[index] += 1.0 / (k + rank)
    return fused_scores


def _document_with_retrieval_metadata(document: Document, retrieval_score: float) -> Document:
    metadata = dict(document.metadata)
    metadata["retrieval_score"] = retrieval_score
    return Document(page_content=document.page_content, metadata=metadata)


def _clean_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start : end + 1]
    return stripped


def _parse_metadata_payload(text: str) -> dict[str, Any]:
    cleaned = _clean_json_text(text)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Metadata payload must be a JSON object")
    return payload


def _ensure_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _fallback_document_metadata(source: str, text: str) -> dict[str, Any]:
    tokens = _tokenize(text)
    keyword_counts = Counter(token for token in tokens if len(token) > 3)
    keywords = [token for token, _ in keyword_counts.most_common(8)]
    return {
        "title": Path(source).stem.replace("_", " ").strip() or source,
        "document_type": "pdf",
        "summary": normalize_whitespace(text[:500]) if text else "",
        "keywords": keywords,
        "entities": [],
        "topics": keywords[:5],
        "important_dates": [],
        "parties": [],
        "jurisdiction": None,
    }


def _generate_document_metadata(metadata_llm: Any, source: str, text: str) -> dict[str, Any]:
    if metadata_llm is None:
        return _fallback_document_metadata(source, text)

    prompt_text = normalize_whitespace(text[:12_000])
    messages = [
        SystemMessage(content=_METADATA_PROMPT),
        HumanMessage(
            content=(
                f"Source file: {source}\n\n"
                f"Document text:\n{prompt_text}\n\n"
                "Return a single JSON object with the requested metadata fields."
            )
        ),
    ]

    try:
        response = metadata_llm.invoke(messages)
        payload = _parse_metadata_payload(message_content_to_text(getattr(response, "content", response)))
    except Exception:
        return _fallback_document_metadata(source, text)

    profile = _fallback_document_metadata(source, text)
    profile.update({
        "title": str(payload.get("title") or profile["title"]),
        "document_type": str(payload.get("document_type") or profile["document_type"]),
        "summary": normalize_whitespace(str(payload.get("summary") or profile["summary"])),
        "keywords": _ensure_string_list(payload.get("keywords")) or profile["keywords"],
        "entities": _ensure_string_list(payload.get("entities")),
        "topics": _ensure_string_list(payload.get("topics")) or profile["topics"],
        "important_dates": _ensure_string_list(payload.get("important_dates")),
        "parties": _ensure_string_list(payload.get("parties")),
        "jurisdiction": payload.get("jurisdiction") or profile["jurisdiction"],
    })
    return profile


def _profile_to_text(profile: dict[str, Any]) -> str:
    parts = [
        profile.get("title", ""),
        profile.get("document_type", ""),
        profile.get("summary", ""),
        ", ".join(profile.get("keywords", [])),
        ", ".join(profile.get("entities", [])),
        ", ".join(profile.get("topics", [])),
        ", ".join(profile.get("important_dates", [])),
        ", ".join(profile.get("parties", [])),
        str(profile.get("jurisdiction") or ""),
    ]
    return normalize_whitespace(" ".join(part for part in parts if part))


def _safe_chunk_id(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("path") or metadata.get("source") or "chunk")
    chunk_index = str(metadata.get("chunk_index") or metadata.get("chunk_label") or "0")
    source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_") or "chunk"
    return f"{source}__{chunk_index}"


def _group_chunks_by_source(chunks: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        source = str(chunk.metadata.get("path") or chunk.metadata.get("source") or "unknown")
        grouped[source].append(chunk)
    return grouped


def _build_bm25_index(texts: list[str]) -> BM25Index:
    tokenized_documents = [_tokenize(text) for text in texts]
    document_lengths = [len(tokens) for tokens in tokenized_documents]
    average_document_length = float(sum(document_lengths) / len(document_lengths)) if document_lengths else 0.0

    document_frequencies: list[Counter[str]] = [Counter(tokens) for tokens in tokenized_documents]
    document_occurrences: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_occurrences.update(set(tokens))

    total_documents = len(tokenized_documents)
    idf = {
        term: float(np.log(1 + (total_documents - document_count + 0.5) / (document_count + 0.5)))
        for term, document_count in document_occurrences.items()
    }
    return BM25Index(
        tokenized_documents=tokenized_documents,
        document_frequencies=document_frequencies,
        document_lengths=document_lengths,
        average_document_length=average_document_length or 1.0,
        idf=idf,
    )


def _chunk_sort_key(document: Document) -> tuple[str, int]:
    metadata = document.metadata
    source = str(metadata.get("path") or metadata.get("source") or "")
    chunk_index = metadata.get("chunk_index")
    try:
        chunk_number = int(chunk_index)
    except (TypeError, ValueError):
        chunk_number = 0
    return source, chunk_number


def create_qdrant_client(
    *,
    qdrant_path: str | Path = ".qdrant",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
) -> QdrantClient:
    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return QdrantClient(path=str(Path(qdrant_path).expanduser()))


def _create_embeddings(embedding_model_name: str | None = None) -> Embeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env to build the embedding index."
        )

    return OpenAIEmbeddings(
        model=(embedding_model_name or os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")).strip(),
        api_key=api_key,
    )


def clear_qdrant_collection(
    *,
    qdrant_path: str | Path = ".qdrant",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = "virallens_corpus",
    embedding_model_name: str | None = None,
) -> str:
    qdrant_client = create_qdrant_client(
        qdrant_path=qdrant_path,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
        return f"Qdrant: deleted collection `{collection_name}`."
    return f"Qdrant: no collection named `{collection_name}`."


def load_corpus_index_from_qdrant(
    *,
    embedding_model_name: str | None = None,
    qdrant_path: str | Path = ".qdrant",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = "virallens_corpus",
) -> CorpusIndex:
    qdrant_client = create_qdrant_client(
        qdrant_path=qdrant_path,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )
    if not qdrant_client.collection_exists(collection_name):
        raise RuntimeError("No indexed corpus found. Upload PDFs if needed, then click Index new files.")

    enriched_chunks: list[Document] = []
    chunk_ids: list[str] = []
    next_offset = None
    while True:
        records, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break
        for record in records:
            payload = dict(record.payload or {})
            page_content = str(payload.pop("page_content", ""))
            metadata = dict(payload)
            if "chunk_id" not in metadata:
                metadata["chunk_id"] = str(record.id)
            enriched_chunks.append(Document(page_content=page_content, metadata=metadata))
            chunk_ids.append(str(metadata["chunk_id"]))
        if next_offset is None:
            break

    if not enriched_chunks:
        raise RuntimeError("The Qdrant collection is empty. Click Index new files to build the index.")

    paired = sorted(zip(chunk_ids, enriched_chunks, strict=False), key=lambda item: _chunk_sort_key(item[1]))
    chunk_ids = [chunk_id for chunk_id, _ in paired]
    enriched_chunks = [chunk for _, chunk in paired]

    search_texts = [
        str(chunk.metadata.get("search_text") or chunk.page_content)
        for chunk in enriched_chunks
    ]
    embeddings = _create_embeddings(embedding_model_name)
    bm25_index = _build_bm25_index(search_texts)
    chunk_id_to_index = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}

    return CorpusIndex(
        chunks=enriched_chunks,
        chunk_ids=chunk_ids,
        chunk_id_to_index=chunk_id_to_index,
        embeddings=embeddings,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        bm25_index=bm25_index,
    )


def build_corpus_index(
    chunks: list[Document],
    *,
    embedding_model_name: str | None = None,
    metadata_llm: Any | None = None,
    qdrant_path: str | Path = ".qdrant",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = "virallens_corpus",
) -> CorpusIndex:
    if not chunks:
        raise ValueError("Cannot build an index from an empty chunk list")

    embeddings = _create_embeddings(embedding_model_name)

    grouped_chunks = _group_chunks_by_source(chunks)
    source_profiles = {
        source: _generate_document_metadata(metadata_llm, source, "\n\n".join(chunk.page_content for chunk in source_chunks))
        for source, source_chunks in grouped_chunks.items()
    }

    enriched_chunks: list[Document] = []
    chunk_ids: list[str] = []
    search_texts: list[str] = []

    for chunk in chunks:
        metadata = dict(chunk.metadata)
        source = str(metadata.get("path") or metadata.get("source") or "unknown")
        profile = source_profiles[source]
        metadata.update(profile)
        metadata["document_profile_text"] = _profile_to_text(profile)
        metadata["search_text"] = normalize_whitespace(f"{metadata['document_profile_text']} {chunk.page_content}")
        metadata["chunk_id"] = _safe_chunk_id(metadata)

        enriched_chunk = Document(page_content=chunk.page_content, metadata=metadata)
        enriched_chunks.append(enriched_chunk)
        chunk_ids.append(metadata["chunk_id"])
        search_texts.append(metadata["search_text"])

    qdrant_client = create_qdrant_client(
        qdrant_path=qdrant_path,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    embeddings_matrix = np.asarray(
        embeddings.embed_documents(search_texts),
        dtype=np.float32,
    )
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings_matrix.shape[1], distance=Distance.COSINE),
    )
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=chunk_ids[index],
                vector=embeddings_matrix[index].tolist(),
                payload={
                    **enriched_chunks[index].metadata,
                    "page_content": enriched_chunks[index].page_content,
                },
            )
            for index in range(len(enriched_chunks))
        ],
    )

    bm25_index = _build_bm25_index(search_texts)
    chunk_id_to_index = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}

    return CorpusIndex(
        chunks=enriched_chunks,
        chunk_ids=chunk_ids,
        chunk_id_to_index=chunk_id_to_index,
        embeddings=embeddings,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        bm25_index=bm25_index,
    )


def format_search_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "No high-confidence matches were found in the corpus."

    lines: list[str] = []
    for hit in hits:
        metadata = hit.document.metadata
        title = metadata.get("title") or metadata.get("source", "unknown")
        document_type = metadata.get("document_type", "document")
        page = metadata.get("page", "?")
        chunk_index = metadata.get("chunk_index", "?")
        excerpt = hit.document.page_content[:550]
        lines.append(
            f"- [{title} | {document_type} | page {page} | chunk {chunk_index} | score {hit.score:.3f}] {excerpt}"
        )
    return "\n".join(lines)