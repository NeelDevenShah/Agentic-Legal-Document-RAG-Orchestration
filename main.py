from __future__ import annotations

from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from config import AppConfig
from deepagents_flow import run_deepagents_rag
from graph_flow import run_langgraph_rag
from utilities import (
    ModelConfig,
    build_chat_model,
    build_corpus_index,
    chunk_documents,
    load_pdf_documents,
    load_pdf_documents_from_paths,
)


def _uploaded_pdf_paths(uploaded_files) -> list[Path]:
    paths: list[Path] = []
    for uploaded_file in uploaded_files or []:
        path = getattr(uploaded_file, "path", None) or getattr(uploaded_file, "name", None) or uploaded_file
        paths.append(Path(path))
    return paths


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
        embedding_model_name=config.embedding_model_name,
        metadata_llm=metadata_llm,
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
    import os

    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env to run the Groq-backed demo."
        )
    if provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env to run the Gemini-backed demo."
        )
    if provider == "openrouter" and not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_KEY")):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env to run the OpenRouter-backed demo."
        )


def _build_answers(
    *,
    question: str,
    flow: str,
    config: AppConfig,
    uploaded_files=None,
):
    _require_api_key(config.provider)
    model = _build_model(config)
    index, chunks, source_label = _build_runtime(
        config,
        metadata_llm=model,
        uploaded_files=uploaded_files,
    )

    outputs: list[str] = [f"Loaded {len(chunks)} chunks from {source_label}"]

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
        gr.Markdown("# Virallens Multi-Agent RAG\nQuery PDFs from `data/` or upload your own documents, then run LangGraph or DeepAgents.")
        uploaded_files = gr.File(
            label="Upload PDFs (optional)",
            file_count="multiple",
            file_types=[".pdf"],
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
        run_button = gr.Button("Run")

        def run(question_text, selected_flow, uploaded_files_value):
            return _build_answers(
                question=question_text,
                flow=selected_flow,
                config=base_config,
                uploaded_files=uploaded_files_value,
            )

        run_button.click(
            run,
            inputs=[question, flow, uploaded_files],
            outputs=output,
        )

    return demo


def main() -> int:
    load_dotenv()
    demo = build_demo()
    demo.launch()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
