# Virallens Multi-Agent RAG

This repository implements the Q1 assignment as a small multi-agent RAG demo over the PDFs in `data/`.

## What is included

- A page-aware PDF ingestion pipeline.
- Recursive chunking with overlap for legal-style documents.
- A shared retrieval index built with `TF-IDF` and cosine similarity.
- A LangGraph flow for direct grounded question answering.
- A DeepAgents flow with a research subagent and a synthesis subagent.
- A Gradio web interface.

## Agent roles

- `LangGraph retriever`: pulls the most relevant chunks for a question.
- `LangGraph answerer`: drafts the final answer from the retrieved context.
- `research-analyst`: uses the corpus search tool to collect evidence.
- `synthesis-writer`: turns evidence into a concise cited response.
- `DeepAgents orchestrator`: coordinates the subagents and keeps the workflow grounded.

## Flow overview

```mermaid
flowchart LR
  Q[User question] --> L[Load PDFs]
  L --> C[Chunk pages]
  C --> I[Build TF-IDF index]
  I --> G1[LangGraph retrieve]
  G1 --> A1[LangGraph answer]
  I --> D1[DeepAgents orchestrator]
  D1 --> R[research-analyst]
  R --> S[synthesis-writer]
  S --> A2[Final answer]
```

## Approach

The corpus is loaded page by page, normalized, and split with a recursive splitter that prefers paragraph boundaries before falling back to sentence and character boundaries. Retrieval is handled by a local semantic embedding index using `nvidia/nemotron-3-embed-1b:free` through OpenRouter, so the demo does not need a separate vector database. The LLM layer can be backed by Groq, Gemini, or OpenRouter through LangChain integrations, and the code retries transient rate-limit style failures with exponential backoff.

The Gradio UI also accepts optional PDF uploads. When files are uploaded, the app chunks and indexes those documents instead of the default `data/` folder.

The LangGraph path is the simplest grounded RAG pipeline: retrieve the best chunks, then answer only from that context. The DeepAgents path exposes the same corpus through a tool, then delegates retrieval and synthesis to dedicated subagents.

## Run locally

1. Copy `.env.sample` to `.env` and set the active provider key, including `OPENROUTER_API_KEY` if you want the OpenRouter route.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the demo:

```bash
python -m app.main
```

This opens the Gradio app with the question box and flow selector. All runtime settings come from `.env`.

## Notes

- The UI is intentionally simple and Gradio-based.
- All runtime settings are loaded from `.env` through `app/config.py`.
- The `app/main.py` entrypoint launches the Gradio app and serves as the sample main file requested in the brief.
- Writing code under `app/` is a standard small-project layout; `src/` is also common, but `app/` keeps the launch path obvious here.
