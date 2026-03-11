from __future__ import annotations

import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .utils import normalize_whitespace


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Chunk documents using Level 2: Legal-Aware Smart Chunking.

    Improves on basic chunking by:
    1. Detecting legal section headers (MEMORANDUM, OPINION, etc.)
    2. Preserving clause boundaries (WHEREAS, THEREFORE, PROVIDED)
    3. Respecting sentence structure while maintaining legal context
    4. Adding metadata flags for chunk type (header, clause, standard)
    """

    # Level 2: Enhanced separator hierarchy for legal documents
    legal_aware_separators = [
        "\n\n\n",                                    # Major section breaks
        r"^(MEMORANDUM|OPINION|ORDER|RULING|DECISION|JUDGMENT)\b",  # Legal headers
        r"^(WHEREAS|THEREFORE|PROVIDED|SUBJECT TO|NOW THEREFORE)\b",  # Legal clauses
        r"^(§|SECTION|ARTICLE|CHAPTER|PART)\s+\d+", # Numbered sections
        "\n\n",                                     # Paragraph breaks
        r"(?<=[.!?])\s+(?=[A-Z])",                # Sentence boundaries
        r"(?<=;)\s+",                             # Clause boundaries
        "\n",                                      # Line breaks
        ". ",                                      # Sentence endings
        "; ",                                      # Semicolons
        " ",                                       # Word spaces
        "",                                        # Character split
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=legal_aware_separators,
        length_function=len,
        is_separator_regex=True,
    )

    chunks = splitter.split_documents(documents)

    chunked_documents: list[Document] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = chunk_index
        metadata["chunk_label"] = f"chunk-{chunk_index}"

        chunk_text = chunk.page_content

        # Level 2: Add metadata flags for chunk type detection
        is_header = bool(re.match(
            r"^\s*(MEMORANDUM|OPINION|ORDER|RULING|DECISION|JUDGMENT|§|SECTION|ARTICLE)",
            chunk_text
        ))
        is_legal_clause = bool(re.search(
            r"\b(WHEREAS|THEREFORE|PROVIDED|SUBJECT TO|NOW THEREFORE|Rule\s+\d+|U\.S\.C\.|§)\b",
            chunk_text
        ))

        metadata["is_header"] = is_header
        metadata["is_legal_clause"] = is_legal_clause
        metadata["chunk_size"] = len(chunk_text)

        normalized_text = normalize_whitespace(chunk_text)
        chunked_documents.append(
            Document(
                page_content=normalized_text,
                metadata=metadata,
            )
        )

    return chunked_documents