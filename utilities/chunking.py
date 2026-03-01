from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .utils import normalize_whitespace


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Chunk page documents with a page-aware recursive splitter."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    chunked_documents: list[Document] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = chunk_index
        metadata["chunk_label"] = f"chunk-{chunk_index}"
        chunked_documents.append(
            Document(
                page_content=normalize_whitespace(chunk.page_content),
                metadata=metadata,
            )
        )

    return chunked_documents