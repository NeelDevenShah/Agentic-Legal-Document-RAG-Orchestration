from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from .utils import normalize_whitespace


def load_pdf_documents(data_dir: Path) -> list[Document]:
    """Extract page-level documents from all PDFs in the supplied directory."""

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents: list[Document] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_index, page in enumerate(reader.pages, start=1):
            text = normalize_whitespace(page.extract_text() or "")
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "path": str(pdf_path),
                        "page": page_index,
                    },
                )
            )

    if not documents:
        raise ValueError(f"No extractable text found in any PDF under {data_dir}")

    return documents


def load_pdf_documents_from_paths(pdf_paths: Iterable[Path]) -> list[Document]:
    """Extract page-level documents from a provided list of PDF files."""

    documents: list[Document] = []
    unique_paths = sorted({Path(path).resolve() for path in pdf_paths})

    for pdf_path in unique_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"Uploaded PDF does not exist: {pdf_path}")
        reader = PdfReader(str(pdf_path))
        for page_index, page in enumerate(reader.pages, start=1):
            text = normalize_whitespace(page.extract_text() or "")
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "path": str(pdf_path),
                        "page": page_index,
                    },
                )
            )

    if not documents:
        raise ValueError("No extractable text found in the uploaded PDF files")

    return documents