from __future__ import annotations

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
)


def _build_runtime(config: AppConfig):
    documents = load_pdf_documents(config.data_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    index = build_corpus_index(chunks)
    return index, chunks


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
):
    _require_api_key(config.provider)
    index, chunks = _build_runtime(config)
    model = _build_model(config)

    outputs: list[str] = [f"Loaded {len(chunks)} chunks from {config.data_dir}"]

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
        gr.Markdown("# Virallens Multi-Agent RAG\nQuery the PDFs in `data/` with LangGraph or DeepAgents.")
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

        def run(question_text, selected_flow):
            return _build_answers(question=question_text, flow=selected_flow, config=base_config)

        run_button.click(
            run,
            inputs=[question, flow],
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
