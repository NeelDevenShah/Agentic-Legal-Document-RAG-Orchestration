from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from .utils import normalize_whitespace

DEFAULT_UPLOAD_STAGING_DIR = Path("data/uploads").resolve()


def coerce_upload_path(uploaded_file) -> Path | None:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, Path):
        return uploaded_file
    if isinstance(uploaded_file, str):
        return Path(uploaded_file)
    if isinstance(uploaded_file, dict):
        raw_path = uploaded_file.get("path") or uploaded_file.get("name")
        return Path(raw_path) if raw_path else None

    raw_path = getattr(uploaded_file, "path", None) or getattr(uploaded_file, "name", None)
    if raw_path:
        return Path(raw_path)
    if isinstance(uploaded_file, (bytes, bytearray)):
        return None
    return Path(uploaded_file)


def stage_uploaded_pdfs(
    uploaded_files,
    *,
    staging_dir: Path = DEFAULT_UPLOAD_STAGING_DIR,
) -> list[Path]:
    if not uploaded_files:
        return []

    items = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_paths: list[Path] = []
    for item in items:
        source_path = coerce_upload_path(item)
        if source_path is None:
            continue
        if not source_path.exists():
            raise FileNotFoundError(
                f"Upload expired or missing: {source_path.name}. "
                "Remove it from the list, upload again, then click Index uploaded PDFs."
            )

        destination = staging_dir / source_path.name
        if destination.exists() and destination.resolve() != source_path.resolve():
            stem = source_path.stem
            suffix = source_path.suffix
            counter = 1
            while destination.exists():
                destination = staging_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        if destination.resolve() != source_path.resolve():
            shutil.copy2(source_path, destination)
        staged_paths.append(destination)

    return staged_paths


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
