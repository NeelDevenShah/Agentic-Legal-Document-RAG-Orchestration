# Agentic Legal Document RAG Orchestration

This repository implements a small multi-agent RAG workflow using LangChain,
LangGraph, and DeepAgents. The application indexes user-uploaded PDFs, retrieves
grounded evidence with hybrid search, and can answer questions through either a
LangGraph flow, a DeepAgents flow, or both side by side in the Gradio UI.

## Agent Roles

### LangGraph Answer Agent

The LangGraph path is a compact retrieve-and-answer workflow. It retrieves the
most relevant chunks from the shared corpus index and sends the formatted
evidence to one LLM call with strict grounding instructions.

### DeepAgents Orchestrator

The DeepAgents path uses a top-level orchestrator with two specialized subagents:

- `research-analyst`: searches the corpus through the `search_corpus` tool and
  extracts cited evidence notes.
- `synthesis-writer`: turns the research notes into a concise final answer with
  source-aware citations.

## Reasoning Flows

### LangGraph Flow Architecture

`rag_pipeline/graph_flow.py` defines a single-agent retrieve-then-answer state
machine:

```text
START -> retrieve -> answer -> END
```

- **Retrieve Node**: calls `CorpusIndex.search(question, top_k)` and formats
  the best matches with source, page, chunk, and score metadata.
- **Answer Node**: invokes the configured chat model with the retrieved context
  and `LANGGRAPH_SYSTEM_PROMPT`.
- **Reasoning Style**: direct grounded synthesis from retrieved evidence, with
  instructions to avoid unsupported claims and cite corpus metadata inline.

This flow is fast and predictable because it performs one retrieval step and one
answer-generation step.

### DeepAgents Flow Architecture

`rag_pipeline/deepagents_flow.py` creates a multi-agent orchestration flow with
a shared corpus search tool:

```text
User question -> orchestrator -> research-analyst -> synthesis-writer -> answer
```

- **Orchestrator Agent**: manages the top-level execution, delegates research,
  and routes gathered evidence into final synthesis.
- **Research-Analyst Subagent**: uses `RESEARCH_SUBAGENT_PROMPT` and the
  `search_corpus` tool to find relevant vector/BM25 evidence. It returns
  bullet-point research notes with strict chunk/source citations.
- **Synthesis-Writer Subagent**: uses `SYNTHESIS_SUBAGENT_PROMPT` and operates
  on the research notes rather than running retrieval itself. It produces the
  final synthesized answer with citations.

This demonstrates multi-agent decomposition while reusing the same retrieval
backend as LangGraph.

## Approach

### Data Loading

`rag_pipeline/utilities/data.py` reads each uploaded PDF page with `pypdf` and
stores source metadata: file name, staged path, and page number. Uploaded PDFs
are staged into `data/uploads/` before indexing so the app does not depend on
temporary upload paths.

### Chunking

`rag_pipeline/utilities/chunking.py` uses a legal-aware recursive chunking
strategy with:

- major section breaks,
- legal headings such as `MEMORANDUM`, `OPINION`, and `ORDER`,
- legal clause markers such as `WHEREAS`, `THEREFORE`, and `SUBJECT TO`,
- numbered section boundaries,
- paragraph, sentence, clause, line, word, and character fallbacks.

Each chunk keeps page/source metadata and adds chunk-level flags such as
`chunk_index`, `is_header`, `is_legal_clause`, and `chunk_size`.

### Retrieval

`rag_pipeline/utilities/retrieval.py` builds a shared `CorpusIndex`:

- dense embeddings stored in Qdrant,
- a lightweight BM25 lexical index,
- Reciprocal Rank Fusion to combine semantic and lexical candidates,
- optional Jina reranking when `JINA_API_KEY` is configured.

The same index is used by both orchestration flows.

### Rate Limiting

Provider calls are wrapped with exponential backoff in
`rag_pipeline/utilities/utils.py`. Embedding creation is batched and
concurrency-limited through
`EMBEDDING_BATCH_SIZE` and `EMBEDDING_MAX_CONCURRENCY`.

## Dataset and Evaluation

During development, generated evaluation questions were refined to be
self-contained and content-focused. The final test cases avoid meta-questions
such as "according to this chunk" and instead ask about concrete legal issues,
parties, statutes, facts, holdings, and allegations from the provided corpus.

The benchmark compared LangGraph and DeepAgents on the same retrieval backend.
Saved experiment outputs are kept under `experiments/` and should be treated as
historical results; rerunning them may produce slightly different values because
LLM APIs, latency, and rate limits vary over time.

### Benchmark Summary

| Metric                          | Exp. 1 LangGraph | Exp. 1 DeepAgents | Exp. 2 LangGraph | Exp. 2 DeepAgents | Exp. 3 Optimized LangGraph | Exp. 3 DeepAgents |
| ------------------------------- | ---------------: | ----------------: | ---------------: | ----------------: | -------------------------: | ----------------: |
| Accuracy Pass Rate (Score >= 7) |    88.0% (22/25) |     84.0% (21/25) |    84.0% (42/50) |     86.0% (43/50) |             100.0% (50/50) |     86.0% (43/50) |
| Mean Accuracy Score             |             8.56 |              8.40 |             8.38 |              8.28 |                       9.28 |              8.28 |
| Mean Citation Score             |             8.24 |              7.88 |             8.10 |              7.84 |                       8.68 |              7.84 |
| Mean Latency per Query          |            2.39s |            45.95s |            2.54s |            50.26s |                      2.54s |            50.26s |

### Experiment 2 Breakdown

| Classification Band | Score Range | LangGraph Flow | DeepAgents Flow |
| ------------------- | ----------: | -------------: | --------------: |
| Passed              |    7.0-10.0 |  42/50 (84.0%) |   43/50 (86.0%) |
| Partially Passed    |     4.0-6.0 |   7/50 (14.0%) |     4/50 (8.0%) |
| Failed              |     0.0-3.0 |    1/50 (2.0%) |     3/50 (6.0%) |

Experiment 2 showed that DeepAgents slightly improved pass count, but LangGraph
was much faster and had stronger citation quality. DeepAgents latency was driven
by multi-agent delegation, search loops, and occasional rate-limit retries.

### Key Finding

The results suggest that the best orchestration choice depends on task type. For
complex research tasks that require decomposition, cross-checking, and synthesis
across multiple evidence threads, the multi-agent DeepAgents pattern can be more
useful than a single-agent or simple RAG setup because each subagent has a
specialized responsibility. For direct retrieval-heavy questions where the main
need is to find the right passages and answer quickly, the simpler LangGraph RAG
flow shines because it has much lower latency, fewer moving parts, and more
predictable execution.

### Optimization Notes

The optimized LangGraph run in Experiment 3 reached a 100.0% pass rate on the 50-question benchmark. The main improvements were stricter answer-completeness instructions, stronger entity grounding, better handling of context spread across chunk boundaries, and a larger retrieved context window with optional Jina reranking.

Based on these experiments, LangGraph is the recommended default workflow for this project because it provides the best balance of accuracy, citation quality, and latency. The low latency of LangGraph (approximately 2.5 s/query) enabled rapid prompt iteration and retrieval tuning, allowing optimization efforts to be prioritized within the available project timeline and compute budget. While the DeepAgents workflow could likely achieve comparable accuracy through similar prompt engineering and retrieval optimizations, those experiments were not pursued due to its substantially higher latency (approximately 50 s/query, around 20× slower than LangGraph). The significantly longer execution time made iterative optimization considerably more time- and cost-intensive, so the available effort was focused on optimizing the LangGraph pipeline. Consequently, DeepAgents is included to demonstrate a multi-agent decomposition architecture over the same RAG backend, while LangGraph represents the fully optimized, production-recommended workflow.

## Repository Structure

```text
.
├── rag_pipeline/           # Application package
│   ├── prompts/             # System prompts for LangGraph and DeepAgents
│   ├── utilities/           # Loading, chunking, retrieval, LLM, and retry helpers
│   ├── app.py               # Gradio application
│   ├── config.py            # Environment-driven runtime configuration
│   ├── graph_flow.py        # LangGraph RAG flow
│   └── deepagents_flow.py   # DeepAgents multi-agent flow
├── scripts/                 # Optional evaluation and test-case generation scripts
├── main.py                  # Thin local/Docker entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Configuration

Copy the sample environment file and add API keys:

```bash
cp .env.sample .env
```

Required configuration:

```text
OPENAI_API_KEY=...
MODEL_PROVIDER=openai
MODEL_NAME=gpt-5.4-mini
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

Only OpenAI chat and embedding models are supported in the runnable app.

Optional:

- `JINA_API_KEY` enables neural reranking.
- `QDRANT_URL` can point to the Docker Compose Qdrant service or a remote Qdrant
  instance. Without it, the app uses the local `QDRANT_PATH`.

## Run Locally

```bash
pip install -r requirements.txt
cp .env.sample .env
python main.py
```

Open `http://localhost:7860`.
Upload one or more PDFs in the UI, click `Index uploaded PDFs`, then run a
question.

## Run With Docker

```bash
docker compose up --build
```

Open `http://localhost:7860`.
Upload one or more PDFs in the UI, click `Index uploaded PDFs`, then run a
question.

## Evaluation Utilities

The repo also includes optional scripts used during development:

- `scripts/generate_test_cases.py`: creates self-contained RAG evaluation
  questions from corpus chunks.
- `scripts/run_benchmark_eval.py`: runs both flows against a CSV test set and
  writes aggregate metrics.
