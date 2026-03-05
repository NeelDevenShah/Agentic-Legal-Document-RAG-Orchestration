from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from config import AppConfig
from deepagents_flow import run_deepagents_rag
from graph_flow import run_langgraph_rag
from utilities import (
    CorpusIndex,
    ModelConfig,
    build_chat_model,
    build_corpus_index,
    chunk_documents,
    clear_qdrant_collection,
    load_corpus_index_from_qdrant,
    load_pdf_documents,
    load_pdf_documents_from_paths,
)
from utilities.persistence import clear_persistence_stores


@dataclass(slots=True)
class _RuntimeState:
    index: CorpusIndex | None = None
    chunk_count: int = 0
    source_label: str | None = None


_RUNTIME = _RuntimeState()


def _uploaded_pdf_paths(uploaded_files) -> list[Path]:
    paths: list[Path] = []
    for uploaded_file in uploaded_files or []:
        path = getattr(uploaded_file, "path", None) or getattr(uploaded_file, "name", None) or uploaded_file
        paths.append(Path(path))
    return paths


def _qdrant_settings(config: AppConfig) -> dict[str, object]:
    return {
        "qdrant_path": config.qdrant_path,
        "qdrant_url": config.qdrant_url,
        "qdrant_api_key": config.qdrant_api_key,
        "collection_name": config.qdrant_collection_name,
        "embedding_model_name": config.embedding_model_name,
    }


def _build_runtime(config: AppConfig, *, metadata_llm, uploaded_files=None):
    if uploaded_files:
        documents = load_pdf_documents_from_paths(_uploaded_pdf_paths(uploaded_files))
        source_label = "uploaded PDFs"
    else:
        documents = load_pdf_documents(config.data_dir)
        source_label = str(config.data_dir)

    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    index = build_corpus_index(
        chunks,
        metadata_llm=metadata_llm,
        **_qdrant_settings(config),
    )
    return index, chunks, source_label


def _build_model(config: AppConfig):
    return build_chat_model(
        ModelConfig(
            provider=config.provider,
            model_name=config.model_name,
            temperature=config.temperature,
        )
    )


def _require_api_key(provider: str) -> None:
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env to run the Groq-backed demo."
        )
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env to run the OpenAI-backed demo."
        )


def _require_embedding_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env to build or query the embedding index."
        )


def _set_runtime(index, chunks, source_label: str) -> str:
    _RUNTIME.index = index
    _RUNTIME.chunk_count = len(chunks)
    _RUNTIME.source_label = source_label
    return f"Indexed {len(chunks)} chunks from {source_label}."


def _resolve_index(config: AppConfig):
    if _RUNTIME.index is not None:
        return _RUNTIME.index, _RUNTIME.chunk_count, _RUNTIME.source_label or "memory cache"

    index = load_corpus_index_from_qdrant(**_qdrant_settings(config))
    _RUNTIME.index = index
    _RUNTIME.chunk_count = len(index.chunks)
    _RUNTIME.source_label = "stored Qdrant index"
    return index, _RUNTIME.chunk_count, _RUNTIME.source_label


def _index_new_files(config: AppConfig, uploaded_files=None) -> str:
    if not uploaded_files:
        raise gr.Error("Please upload at least one PDF before clicking Index new files.")
    _require_embedding_api_key()
    _require_api_key(config.provider)
    model = _build_model(config)
    index, chunks, source_label = _build_runtime(
        config,
        metadata_llm=model,
        uploaded_files=uploaded_files,
    )
    return _set_runtime(index, chunks, source_label)


def _clear_db(config: AppConfig) -> str:
    _RUNTIME.index = None
    _RUNTIME.chunk_count = 0
    _RUNTIME.source_label = None

    qdrant_message = clear_qdrant_collection(**_qdrant_settings(config))
    return clear_persistence_stores(config, qdrant_clear_message=qdrant_message)


def _build_answers(
    *,
    question: str,
    flow: str,
    config: AppConfig,
):
    _require_api_key(config.provider)
    _require_embedding_api_key()
    model = _build_model(config)
    index, chunk_count, source_label = _resolve_index(config)

    outputs: list[str] = [f"Using {chunk_count} indexed chunks from {source_label}."]

    if flow in {"langgraph", "both"}:
        langgraph_answer = run_langgraph_rag(
            question=question,
            index=index,
            model=model,
            top_k=config.top_k,
            retry_attempts=config.retry_attempts,
        )
        outputs.append("=== LangGraph flow ===")
        outputs.append(langgraph_answer)

    if flow in {"deepagents", "both"}:
        deepagents_answer = run_deepagents_rag(
            question=question,
            index=index,
            model=model,
            retry_attempts=config.retry_attempts,
        )
        outputs.append("=== DeepAgents flow ===")
        outputs.append(deepagents_answer)

    return "\n\n".join(outputs)


def build_demo() -> gr.Blocks:
    load_dotenv()
    base_config = AppConfig.from_env()

    with gr.Blocks(title="Virallens Multi-Agent RAG") as demo:
        gr.Markdown(
            "# Virallens Multi-Agent RAG\n"
            "Upload PDFs (optional), click **Index new files**, then **Run** a question. "
            "Use **Clear DB** to wipe Qdrant and MySQL storage."
        )
        uploaded_files = gr.File(
            label="Upload PDFs (optional)",
            file_count="multiple",
            file_types=[".pdf"],
        )
        with gr.Row():
            index_button = gr.Button("Index new files", variant="primary")
            clear_button = gr.Button("Clear DB", variant="stop")
            run_button = gr.Button("Run")
        index_status = gr.Textbox(
            label="Index status",
            lines=2,
            interactive=False,
            placeholder="Index new files to load PDFs into Qdrant before running a query.",
        )
        with gr.Row():
            question = gr.Textbox(
                label="Question",
                value=base_config.default_question,
                lines=3,
            )
        with gr.Row():
            flow = gr.Radio(
                choices=["langgraph", "deepagents", "both"],
                value="both",
                label="Flow",
            )

        output = gr.Textbox(label="Answer", lines=24)

        index_button.click(
            lambda uploaded: _index_new_files(base_config, uploaded_files=uploaded),
            inputs=uploaded_files,
            outputs=index_status,
        )
        clear_button.click(
            lambda: _clear_db(base_config),
            inputs=None,
            outputs=index_status,
        )
        run_button.click(
            lambda question_text, selected_flow: _build_answers(
                question=question_text,
                flow=selected_flow,
                config=base_config,
            ),
            inputs=[question, flow],
            outputs=output,
        )

    return demo


def main() -> int:
    load_dotenv()
    demo = build_demo()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
