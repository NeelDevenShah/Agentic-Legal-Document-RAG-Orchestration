from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .utils import message_content_to_text, normalize_whitespace, retry_with_backoff


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_METADATA_PROMPT = (
    "Extract metadata from a legal document header. Return ONLY valid JSON (no markdown, no explanation).\n"
    "Required fields: title, document_type, summary, keywords, parties, jurisdiction\n"
    "Instructions:\n"
    "- title: Case name or document title (e.g., 'LAST ATLANTIS CAPITAL LLC v. AGS SPECIALIST PARTNERS')\n"
    "- document_type: Type of legal document (e.g., 'court_opinion', 'memorandum_order', 'motion', etc.)\n"
    "- summary: One sentence summary of the document's purpose\n"
    "- keywords: Array of 3-5 key legal concepts (e.g., ['Rule 10b-5', 'motion for summary judgment', 'dismissal'])\n"
    "- parties: Array of party names with their roles (e.g., ['LAST ATLANTIS CAPITAL LLC (Plaintiff)', 'AGS SPECIALIST PARTNERS (Defendant)'])\n"
    "- jurisdiction: Full jurisdiction information (e.g., 'United States District Court, Northern District of Illinois, Eastern Division')\n"
    "If a field cannot be determined, use null or empty array. Return JSON object only."
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
    retry_attempts: int = 4
    numeric_id_to_index: dict[int, int] = field(default_factory=dict)
    jina_api_key: str | None = None
    jina_reranker_model: str = "jina-reranker-v3"
    rerank_candidate_multiplier: int = 4

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        semantic_ranked = self._semantic_ranked_indices(query, limit=max(top_k * 5, 10))
        bm25_ranked = self.bm25_index.ranked_indices(query, limit=max(top_k * 5, 10))
        fused_scores = _reciprocal_rank_fusion([semantic_ranked, bm25_ranked])

        candidate_limit = max(top_k * self.rerank_candidate_multiplier, top_k)
        candidate_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:candidate_limit]

        if self.jina_api_key and candidate_indices:
            reranked = self._rerank(query, candidate_indices, top_k=top_k)
            if reranked is not None:
                return reranked

        ranked_indices = candidate_indices[:top_k]
        return [
            SearchHit(
                score=float(fused_scores[index]),
                document=_document_with_retrieval_metadata(self.chunks[index], fused_scores[index]),
            )
            for index in ranked_indices
        ]

    def _rerank(self, query: str, candidate_indices: list[int], *, top_k: int) -> list[SearchHit] | None:
        documents = [self.chunks[index].page_content for index in candidate_indices]
        try:
            results = retry_with_backoff(
                lambda: _jina_rerank(
                    query,
                    documents,
                    api_key=self.jina_api_key,
                    model=self.jina_reranker_model,
                    top_n=top_k,
                ),
                attempts=self.retry_attempts,
                label="Jina rerank",
            )
        except Exception:
            return None

        return [
            SearchHit(
                score=float(relevance_score),
                document=_document_with_retrieval_metadata(
                    self.chunks[candidate_indices[local_index]], relevance_score
                ),
            )
            for local_index, relevance_score in results
        ]

    def _semantic_ranked_indices(self, query: str, *, limit: int) -> list[int]:
        query_vector = retry_with_backoff(
            lambda: self.embeddings.embed_query(query),
            attempts=self.retry_attempts,
            label="Embeddings query",
        )
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=False,
        ).points

        ranked_indices: list[int] = []
        for result in results:
            index = self.numeric_id_to_index.get(result.id)
            if index is not None:
                ranked_indices.append(index)
        return ranked_indices


def _jina_rerank(
    query: str,
    documents: list[str],
    *,
    api_key: str,
    model: str,
    top_n: int,
) -> list[tuple[int, float]]:
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": model,
            "query": query,
            "top_n": top_n,
            "documents": documents,
            "return_documents": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return [(result["index"], result["relevance_score"]) for result in payload["results"]]


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
    """Fallback metadata when LLM extraction fails. Returns minimal data."""
    return {
        "title": Path(source).stem.replace("_", " ").strip() or source,
        "document_type": "legal_document",
        "summary": "",
        "keywords": [],
        "parties": [],
        "jurisdiction": "",
    }


def _generate_document_metadata(
    metadata_llm: Any,
    source: str,
    text: str,
    *,
    retry_attempts: int = 4,
) -> dict[str, Any]:
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
        response = retry_with_backoff(
            lambda: metadata_llm.invoke(messages),
            attempts=retry_attempts,
            label=f"metadata LLM ({source})",
        )
        payload = _parse_metadata_payload(message_content_to_text(getattr(response, "content", response)))
    except Exception:
        return _fallback_document_metadata(source, text)

    profile = _fallback_document_metadata(source, text)
    profile.update({
        "title": str(payload.get("title") or profile["title"]),
        "document_type": str(payload.get("document_type") or profile["document_type"]),
        "summary": normalize_whitespace(str(payload.get("summary") or profile["summary"])),
        "keywords": _ensure_string_list(payload.get("keywords")) or profile["keywords"],
        "parties": _ensure_string_list(payload.get("parties")) or profile.get("parties", []),
        "jurisdiction": str(payload.get("jurisdiction") or profile.get("jurisdiction") or ""),
    })
    return profile


def _profile_to_text(profile: dict[str, Any]) -> str:
    """Extract legal document metadata keywords for searching."""
    parts = []

    if profile.get("title"):
        parts.append(profile["title"])

    if profile.get("parties"):
        parties = profile["parties"]
        if isinstance(parties, list):
            parts.extend(str(p) for p in parties if p)
        elif parties:
            parts.append(str(parties))

    if profile.get("jurisdiction"):
        jurisdiction = profile["jurisdiction"]
        if jurisdiction:
            parts.append(str(jurisdiction))

    if profile.get("keywords"):
        keywords = profile["keywords"]
        if isinstance(keywords, list):
            parts.extend(str(k) for k in keywords if k)
        elif keywords:
            parts.append(str(keywords))

    return normalize_whitespace(" ".join(str(part) for part in parts if part))


def _safe_chunk_id(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("path") or metadata.get("source") or "chunk")
    chunk_index = str(metadata.get("chunk_index") or metadata.get("chunk_label") or "0")
    source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_") or "chunk"
    return f"{source}__{chunk_index}"


def _chunk_id_to_numeric(chunk_id: str) -> int:
    """Convert string chunk ID to a numeric ID (u64) using MD5 hash."""
    digest = hashlib.md5(chunk_id.encode()).digest()[:8]
    return int.from_bytes(digest, byteorder="big", signed=False)


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


def _create_embeddings(
    *,
    embedding_model_name: str,
    openai_api_key: str | None,
) -> Embeddings:
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env to build the embedding index.")
    return OpenAIEmbeddings(model=embedding_model_name, api_key=openai_api_key)


def _embed_documents_concurrently(
    embeddings: Embeddings,
    texts: list[str],
    *,
    batch_size: int,
    max_concurrency: int,
    retry_attempts: int,
) -> list[list[float]]:
    """Embed ``texts`` in batches sent concurrently, each batch retried with backoff.

    Results are reassembled in the original order regardless of completion order.
    """

    batch_size = max(1, batch_size)
    max_concurrency = max(1, max_concurrency)

    batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
    results: list[list[list[float]] | None] = [None] * len(batches)

    def embed_batch(batch: list[str]) -> list[list[float]]:
        return embeddings.embed_documents(batch)

    with ThreadPoolExecutor(max_workers=min(max_concurrency, len(batches))) as executor:
        future_to_index = {
            executor.submit(
                retry_with_backoff,
                lambda batch=batch: embed_batch(batch),
                attempts=retry_attempts,
                label=f"Embeddings batch {index + 1}/{len(batches)}",
            ): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()

    return [vector for batch_result in results if batch_result is not None for vector in batch_result]


def clear_qdrant_collection(
    *,
    qdrant_path: str | Path = ".qdrant",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = "virallens_corpus",
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
    embedding_model_name: str,
    openai_api_key: str | None,
    qdrant_path: str | Path = ".qdrant",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = "virallens_corpus",
    retry_attempts: int = 4,
    jina_api_key: str | None = None,
    jina_reranker_model: str = "jina-reranker-v3",
    rerank_candidate_multiplier: int = 4,
) -> CorpusIndex:
    qdrant_client = create_qdrant_client(
        qdrant_path=qdrant_path,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )
    if not qdrant_client.collection_exists(collection_name):
        raise RuntimeError("No indexed corpus found. Upload PDFs, then click Index uploaded PDFs.")

    enriched_chunks: list[Document] = []
    chunk_ids: list[str] = []
    numeric_ids: list[int] = []
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
            numeric_ids.append(record.id)
        if next_offset is None:
            break

    if not enriched_chunks:
        raise RuntimeError("The Qdrant collection is empty. Upload PDFs, then click Index uploaded PDFs.")

    paired = sorted(
        zip(chunk_ids, enriched_chunks, numeric_ids, strict=False),
        key=lambda item: _chunk_sort_key(item[1]),
    )
    chunk_ids = [chunk_id for chunk_id, _, _ in paired]
    enriched_chunks = [chunk for _, chunk, _ in paired]
    numeric_ids = [numeric_id for _, _, numeric_id in paired]

    search_texts = [
        normalize_whitespace(
            f"{_profile_to_text(chunk.metadata)} {chunk.page_content}"
        )
        for chunk in enriched_chunks
    ]
    embeddings = _create_embeddings(
        embedding_model_name=embedding_model_name,
        openai_api_key=openai_api_key,
    )
    bm25_index = _build_bm25_index(search_texts)
    chunk_id_to_index = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    numeric_id_to_index = {numeric_id: index for index, numeric_id in enumerate(numeric_ids)}

    return CorpusIndex(
        chunks=enriched_chunks,
        chunk_ids=chunk_ids,
        chunk_id_to_index=chunk_id_to_index,
        embeddings=embeddings,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        bm25_index=bm25_index,
        retry_attempts=retry_attempts,
        numeric_id_to_index=numeric_id_to_index,
        jina_api_key=jina_api_key,
        jina_reranker_model=jina_reranker_model,
        rerank_candidate_multiplier=rerank_candidate_multiplier,
    )


def build_corpus_index(
    chunks: list[Document],
    *,
    embedding_model_name: str,
    openai_api_key: str | None,
    metadata_llm: Any | None = None,
    qdrant_path: str | Path = ".qdrant",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = "virallens_corpus",
    retry_attempts: int = 4,
    embedding_batch_size: int = 64,
    embedding_max_concurrency: int = 4,
    jina_api_key: str | None = None,
    jina_reranker_model: str = "jina-reranker-v3",
    rerank_candidate_multiplier: int = 4,
) -> CorpusIndex:
    if not chunks:
        raise ValueError("Cannot build an index from an empty chunk list")

    embeddings = _create_embeddings(
        embedding_model_name=embedding_model_name,
        openai_api_key=openai_api_key,
    )

    grouped_chunks = _group_chunks_by_source(chunks)
    source_profiles = {}
    for source, source_chunks in grouped_chunks.items():
        first_chunk_text = source_chunks[0].page_content if source_chunks else ""
        source_profiles[source] = _generate_document_metadata(
            metadata_llm,
            source,
            first_chunk_text,
            retry_attempts=retry_attempts,
        )

    enriched_chunks: list[Document] = []
    chunk_ids: list[str] = []
    search_texts: list[str] = []

    for chunk in chunks:
        metadata = dict(chunk.metadata)
        source = str(metadata.get("path") or metadata.get("source") or "unknown")
        profile = source_profiles[source]
        metadata.update(profile)
        metadata["chunk_id"] = _safe_chunk_id(metadata)

        metadata_keywords = _profile_to_text(profile)
        search_text = normalize_whitespace(f"{metadata_keywords} {chunk.page_content}")

        enriched_chunk = Document(page_content=chunk.page_content, metadata=metadata)
        enriched_chunks.append(enriched_chunk)
        chunk_ids.append(metadata["chunk_id"])
        search_texts.append(search_text)

    qdrant_client = create_qdrant_client(
        qdrant_path=qdrant_path,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )

    existing_chunks = []
    existing_chunk_ids = []
    existing_search_texts = []

    if qdrant_client.collection_exists(collection_name):
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
                existing_chunk = Document(page_content=page_content, metadata=metadata)
                existing_chunks.append(existing_chunk)
                existing_chunk_ids.append(str(metadata["chunk_id"]))
                existing_search_texts.append(
                    normalize_whitespace(f"{_profile_to_text(metadata)} {page_content}")
                )
            if next_offset is None:
                break

    merged_by_id: dict[str, tuple[Document, str]] = {
        chunk_id: (chunk, search_text)
        for chunk_id, chunk, search_text in zip(
            existing_chunk_ids,
            existing_chunks,
            existing_search_texts,
            strict=False,
        )
    }
    for chunk_id, chunk, search_text in zip(chunk_ids, enriched_chunks, search_texts, strict=False):
        merged_by_id[chunk_id] = (chunk, search_text)

    all_chunk_ids = list(merged_by_id)
    all_chunks = [chunk for chunk, _ in merged_by_id.values()]
    all_search_texts = [search_text for _, search_text in merged_by_id.values()]

    embeddings_matrix = np.asarray(
        _embed_documents_concurrently(
            embeddings,
            all_search_texts,
            batch_size=embedding_batch_size,
            max_concurrency=embedding_max_concurrency,
            retry_attempts=retry_attempts,
        ),
        dtype=np.float32,
    )

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embeddings_matrix.shape[1], distance=Distance.COSINE),
        )

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=_chunk_id_to_numeric(all_chunk_ids[index]),
                vector=embeddings_matrix[index].tolist(),
                payload={
                    **all_chunks[index].metadata,
                    "page_content": all_chunks[index].page_content,
                    "chunk_id": all_chunk_ids[index],
                },
            )
            for index in range(len(all_chunks))
        ],
    )

    bm25_index = _build_bm25_index(all_search_texts)
    chunk_id_to_index = {chunk_id: index for index, chunk_id in enumerate(all_chunk_ids)}
    numeric_id_to_index = {_chunk_id_to_numeric(chunk_id): index for index, chunk_id in enumerate(all_chunk_ids)}

    return CorpusIndex(
        chunks=all_chunks,
        chunk_ids=all_chunk_ids,
        chunk_id_to_index=chunk_id_to_index,
        embeddings=embeddings,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        bm25_index=bm25_index,
        retry_attempts=retry_attempts,
        numeric_id_to_index=numeric_id_to_index,
        jina_api_key=jina_api_key,
        jina_reranker_model=jina_reranker_model,
        rerank_candidate_multiplier=rerank_candidate_multiplier,
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
