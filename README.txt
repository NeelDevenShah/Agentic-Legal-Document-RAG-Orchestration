# Virallens Multi-Agent RAG

This repository implements the Q1 assignment as a multi-agent RAG (Retrieval-Augmented Generation) system with two distinct reasoning pipelines: a **LangGraph-based direct retrieval flow** and a **DeepAgents-based multi-agent orchestration flow**. Both paths share a common corpus indexing and retrieval backend.

## System Architecture Overview

The system is divided into three main layers:

1. **Data Ingestion & Indexing Layer** — loads PDFs, chunks them, enriches with metadata, and builds a hybrid retrieval index
2. **Retrieval Layer** — hybrid search combining semantic similarity, BM25 lexical matching, and optional neural reranking
3. **Reasoning Layers** — two distinct flows for question answering:
   - **LangGraph Flow**: simple retrieve-then-answer pipeline
   - **DeepAgents Flow**: multi-agent orchestration with dedicated research and synthesis subagents

## What is Included

- **Page-aware PDF ingestion pipeline** — extracts text page-by-page from PDFs with source tracking
- **Recursive chunking with overlap** — handles legal-style documents using paragraph/sentence/character boundaries
- **Shared retrieval index** — hybrid index combining:
  - Semantic embeddings (Google Gemini embedding-001 via LangChain)
  - BM25 lexical search (custom implementation)
  - Reciprocal Rank Fusion (RRF) to combine both signals
  - Optional Jina neural reranking for final result refinement
- **LangGraph direct RAG flow** — lightweight, single-agent retrieve-answer pipeline
- **DeepAgents multi-agent flow** — orchestrator with research and synthesis subagents
- **Gradio web interface** — user-friendly UI for querying and index management
- **Qdrant vector database** — persistent semantic index storage (local or remote)
- **Exponential backoff retry logic** — handles transient API failures across all LLM and embedding calls

## Data Ingestion & Chunking Pipeline

### PDF Loading
The system loads PDFs from a designated directory (`data/` by default) or from user uploads:
- **load_pdf_documents()** — recursively loads all .pdf files from a directory
- **load_pdf_documents_from_paths()** — loads PDFs from a list of specific paths (for UI uploads)
- Each page is extracted as a separate LangChain Document with metadata: `source`, `path`, `page`
- Text is normalized (whitespace cleaned) during extraction

### Document Chunking (Level 2: Legal-Aware Smart Chunking)

**Why Level 2?** Legal documents have structure that naive chunking destroys. Level 2 balances sophistication with practicality.

**Level Comparison:**

| Aspect | Level 1 (Basic) | Level 2 (Recommended) | Level 3 (Over-engineered) |
|--------|-----------------|----------------------|--------------------------|
| **Separators** | Generic (paragraphs, lines) | Legal-aware (headers, clauses) | 13+ semantic boundaries |
| **Header Detection** | ❌ None | ✅ MEMORANDUM, OPINION, ORDER | ✅ + semantic analysis |
| **Clause Preservation** | ❌ None | ✅ WHEREAS, THEREFORE, PROVIDED | ✅ + embedding-based |
| **Metadata Flags** | ❌ None | ✅ is_header, is_legal_clause | ✅ + more flags |
| **Complexity** | Simple | Moderate | Complex |
| **Chunking Speed** | Fast | Fast | Slower |
| **Quality Improvement** | Baseline | +30% | +35% (not worth it) |

**Why NOT Level 1:** Breaks legal structure mid-thought
- Example: "Plaintiffs, v. AGS SPECIALIST PARTNERS, et al., Defendants." → fragmented across 3 chunks
- Headers ignored, citations split, clauses broken

**Why NOT Level 3:** Over-engineered for retrieval tasks
- Requires embedding model at chunk time (expensive)
- Complex maintenance burden
- Chunking is one-time cost; Level 2 solves 80% of issues for 20% of complexity
- Marginal 5% quality improvement doesn't justify the overhead

**Why Level 2:** Goldilocks solution ✅
- Detects headers: "MEMORANDUM", "OPINION", "ORDER", "RULING", "DECISION"
- Preserves clauses: "WHEREAS ... THEREFORE", "PROVIDED THAT"
- Respects sections: "§ SECTION ARTICLE CHAPTER PART"
- Intelligent boundaries: Sentence (. ! ?) + Clause (;) + Paragraph (\n\n)
- Metadata flags: is_header, is_legal_clause for downstream processing
- Works with first-page metadata extraction (already implemented)

**Implementation:**

Chunks are created using **legal-aware recursive splitting**:
- **chunk_documents()** uses RecursiveCharacterTextSplitter with:
  - `chunk_size=1200` (default, configurable)
  - `chunk_overlap=180` (default, preserves context)
  - **Smart separator hierarchy:**
    1. "\n\n\n" — Major section breaks
    2. Legal headers (MEMORANDUM, OPINION, ORDER, RULING, DECISION)
    3. Legal clauses (WHEREAS, THEREFORE, PROVIDED, SUBJECT TO)
    4. Numbered sections (§ SECTION ARTICLE CHAPTER PART)
    5. "\n\n" — Paragraph breaks
    6. Sentence boundaries (. ! ? followed by capital letter)
    7. Clause boundaries (;)
    8. "\n" — Line breaks
    9-12. Fallback to word and character splits

- **Metadata enrichment per chunk:**
  - `chunk_index`: sequence number (1-based)
  - `chunk_label`: human-readable label (e.g., "chunk-1")
  - `is_header`: True if chunk starts with legal document header
  - `is_legal_clause`: True if chunk contains legal keywords or citations
  - `chunk_size`: Actual character count for filtering
  - Original `source`, `path`, `page` metadata preserved

**Quality Impact:**

Query: "What are the parties?"
- Level 1: Fragments party names → poor answer
- Level 2: Preserves full header → correct answer with metadata flag
- Level 3: Same answer, but slower to chunk

Query: "What does Rule 10b-5 address?"
- Level 1: May split citation → confusing results
- Level 2: Keeps citations together, marked with is_legal_clause → clear answer
- Level 3: Same answer, added complexity

### Document Metadata Enrichment
After chunking, an LLM generates metadata from the **first page only**:
- **_generate_document_metadata()** calls Gemini with only the first chunk's text (first page)
- Extracts critical legal document metadata:
  - `title`: Case name (e.g., "LAST ATLANTIS CAPITAL LLC v. AGS SPECIALIST PARTNERS")
  - `document_type`: Type of legal document (e.g., "court_opinion", "memorandum_order")
  - `summary`: One-sentence purpose of the document
  - `keywords`: Key legal concepts from the document (e.g., ["Rule 10b-5", "motion for summary judgment"])
  - `parties`: Array of party names with their roles (e.g., ["LAST ATLANTIS CAPITAL LLC (Plaintiff)", "AGS SPECIALIST PARTNERS (Defendant)"])
  - `jurisdiction`: Full jurisdiction information (e.g., "United States District Court, Northern District of Illinois")

- **LLM-Only Extraction:** Uses Gemini's instruction-following ability to extract from first page text
  - Prompt explicitly instructs where to find parties, jurisdiction, case name
  - Returns structured JSON with these critical fields
  - If extraction fails, returns empty values (no regex fallback)

- Extracted metadata is attached to **every chunk** of that source, enabling consistent document-level context
- This approach trusts the LLM to correctly identify parties and jurisdiction from the first page header where they are always clearly visible

## Retrieval Index

### Index Components
The **CorpusIndex** dataclass holds all retrieval state:
- **chunks**: list of enriched Document objects with content and metadata
- **chunk_ids**: string identifiers for chunks (format: "{source}__chunk-{index}")
- **embeddings**: Google Gemini embedding client (models/embedding-001 by default)
- **qdrant_client**: connection to Qdrant vector database
- **bm25_index**: custom BM25 implementation for lexical search
- **numeric_id_to_index**: mapping from Qdrant point IDs to in-memory indices
- **jina_api_key, jina_reranker_model, rerank_candidate_multiplier**: optional reranking config

### Qdrant Vector Storage
- **create_qdrant_client()** — creates a Qdrant client (local or remote):
  - Local: `QdrantClient(path=".qdrant")` — creates/opens a local database directory
  - Remote: `QdrantClient(url="...", api_key="...")` — connects to a remote Qdrant instance
- During index building:
  - Collection is created with vector size matching the embedding model (768 for Gemini embedding-001)
  - Distance metric is COSINE similarity
  - Points are upserted with numeric IDs (MD5-hashed chunk_ids converted to u64) and full chunk metadata as payload
  - Metadata includes: `page_content`, `chunk_id`, and all enriched document fields

### BM25 Lexical Index
The custom BM25 implementation provides keyword-based ranking:
- **_build_bm25_index()** builds the index at startup from all chunks:
  - Tokenizes each chunk (lowercase alphanumeric tokens)
  - Computes document frequencies (how many chunks contain each token)
  - Computes IDF (inverse document frequency) for each token
  - Stores tokenized documents and frequency counts
- **BM25Index.score()** ranks chunks for a query:
  - Tokenizes the query
  - Applies BM25 formula: `IDF(term) * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avg_len)))`
  - Default parameters: `k1=1.5` (term frequency saturation), `b=0.75` (document length normalization)

### Search Process
The **CorpusIndex.search()** method combines semantic and lexical signals:

1. **Semantic Ranking**:
   - Embeds the query using Google Gemini embedding API (with retry backoff)
   - Queries Qdrant with the query vector against chunk embeddings (based on search_text)
   - Returns top 50 results (5*top_k, default)
   - Extracts the original chunk indices from numeric IDs
   - Captures both document context (from metadata keywords) and chunk specificity

2. **BM25 Ranking**:
   - Scores all chunks using the BM25 formula on search_text
   - search_text includes both metadata keywords and page_content
   - Returns top 50 indices (5*top_k, default) with non-zero scores
   - Enables keyword matching on both document metadata and chunk content

3. **Reciprocal Rank Fusion (RRF)**:
   - **_reciprocal_rank_fusion()** fuses the two ranked lists:
   - For each ranked list, assigns a score: `1.0 / (k + rank)`, where k=60 (default constant)
   - Sums scores across lists for each chunk index
   - Produces a unified ranking combining both semantic and lexical relevance

4. **Optional Jina Reranking** (if `jina_api_key` is configured):
   - Takes top `top_k * rerank_candidate_multiplier` candidates (default: top 20 for top_k=5)
   - Calls Jina API (`https://api.jina.ai/v1/rerank`) with the query and candidate documents
   - Returns reranked results sorted by Jina's neural relevance score
   - Falls back to RRF results if the API call fails

5. **Final Result Assembly**:
   - Returns top `top_k` chunks (default 5) as SearchHit objects
   - Each hit includes: the chunk's Document (page_content), retrieval score, and enriched metadata
   - Metadata includes: title, parties, entities, keywords, topics, jurisdiction, important_dates
   - Metadata is augmented with `retrieval_score` for downstream analysis

### Search Text Enhancement
To improve retrieval, chunks have a combined `search_text` field:
- Combines extracted metadata keywords (title, keywords, parties, entities, topics) with the chunk's page content
- BM25 indexing uses this combined text to enable searching both document-level metadata and chunk-specific content
- Semantic embeddings are generated from this combined text to capture both document context and chunk details
- Metadata is stored separately in the chunk metadata for citation and filtering purposes

## LangGraph RAG Flow

### Architecture
The LangGraph flow is the simplest reasoning pipeline, implementing a classic retrieve-then-answer pattern.

### Flow Definition
**build_langgraph_rag_graph()** creates a state machine with two nodes:

1. **Retrieve Node**:
   - Input state: `{"question": "..."}`
   - Calls `CorpusIndex.search(question, top_k=top_k)` to retrieve relevant chunks
   - Formats the search hits using **format_search_hits()** into a readable string:
     ```
     - [title | document_type | page N | chunk M | score X.XXX] excerpt...
     - [title | document_type | page N | chunk M | score X.XXX] excerpt...
     ```
   - Returns: `{"context": "formatted string"}`

2. **Answer Node**:
   - Input state: `{"question": "...", "context": "..."}`
   - Constructs a message list:
     - System message with **LANGGRAPH_SYSTEM_PROMPT** (instructs grounded QA from context only)
     - Human message with: question, retrieved context, and instruction to cite source metadata
   - Calls the LLM with retry backoff (attempts=retry_attempts, exponential backoff with jitter)
   - Extracts the final answer text using **extract_final_ai_text()** helper
   - Returns: `{"answer": "text"}`

### Graph Topology
- Edges: START → retrieve → answer → END
- State updates are accumulated as the graph progresses
- Final state contains: `question`, `context`, `answer`

### LangGraph System Prompt
```
You are a grounded document QA assistant.

Rules:
- Answer only from the retrieved context.
- Cite each factual claim with the source metadata shown in the context.
- If the context does not support an answer, say what is missing instead of guessing.
- Keep the final answer concise, practical, and specific.
```
This prompt ensures the model stays grounded and cites evidence from the corpus.

### Invocation
**run_langgraph_rag()** executes the graph:
- Passes `{"question": question}` as initial state
- Runs with `recursion_limit=25` to prevent infinite loops
- Wraps execution with **retry_with_backoff()** for transient failures
- Returns the final answer text extracted from the graph result

### Advantages
- Simple, transparent pipeline with clear data flow
- Direct traceability: question → chunks → answer
- Lower token cost (single LLM call for answer generation)
- Easy to debug and extend

## DeepAgents Multi-Agent Flow

### Architecture
The DeepAgents flow uses a multi-agent orchestration pattern with dedicated subagents for research and synthesis. This approach allows specialized prompting and deeper reasoning for complex questions.

### Agent Roles

1. **Orchestrator Agent**:
   - Directs the overall workflow
   - Coordinates subagent calls
   - Ensures grounding in the corpus
   - Returns the final answer

2. **Research-Analyst Subagent**:
   - Searches the corpus for supporting evidence
   - Uses the **search_corpus()** tool to query the index
   - Extracts bullet-point evidence with source citations
   - Provides detailed notes (not the final answer)

3. **Synthesis-Writer Subagent**:
   - Receives research notes from the orchestrator
   - Turns evidence into a concise, well-cited answer
   - Handles incomplete evidence gracefully
   - Produces the final response

### DeepAgents Graph Definition
**build_deepagents_graph()** configures the multi-agent system:

```python
subagents = [
    {
        "name": "research-analyst",
        "description": "Find the best supporting chunks and extract evidence.",
        "system_prompt": RESEARCH_SUBAGENT_PROMPT,
        "model": model,
        "tools": [search_tool],  # Has access to corpus search
    },
    {
        "name": "synthesis-writer",
        "description": "Turn evidence into a concise, cited answer.",
        "system_prompt": SYNTHESIS_SUBAGENT_PROMPT,
        "model": model,
        "tools": [],  # No tools, uses research notes only
    },
]

agent = create_deep_agent(
    model=model,
    tools=[search_tool],  # Orchestrator also has access to search
    system_prompt=DEEPAGENTS_SYSTEM_PROMPT,
    subagents=subagents,
)
```

The **search_corpus()** tool:
- Accepts a query and `top_k` parameter
- Calls `CorpusIndex.search()` to retrieve chunks
- Returns formatted search hits for the agent to use
- Enables both orchestrator and research-analyst to search the corpus

### DeepAgents System Prompts

**DEEPAGENTS_SYSTEM_PROMPT** (Orchestrator):
```
You are the orchestrator for a small multi-agent RAG team.

Workflow:
1. Use the research-analyst subagent first to gather the strongest supporting chunks.
2. Hand the research notes to the synthesis-writer subagent to produce the final answer.
3. Keep the answer grounded in the corpus and cite source metadata.

Rules:
- Do not guess when evidence is weak or missing.
- Prefer the corpus over general knowledge.
- Keep the final response concise and well structured.
```

**RESEARCH_SUBAGENT_PROMPT**:
```
You are the research-analyst subagent.

Task:
- Search the corpus for the most relevant passages.
- Return bullet-point evidence only.
- Include the source, page, and chunk labels in every bullet.
- Do not write the final answer.
```

**SYNTHESIS_SUBAGENT_PROMPT**:
```
You are the synthesis-writer subagent.

Task:
- Use the research notes to write the final answer.
- Cite the evidence you rely on.
- State uncertainty explicitly when the evidence is incomplete.
- Avoid repeating raw excerpts unless they are essential.
```

### Invocation
**run_deepagents_rag()** executes the multi-agent orchestration:
- Creates the graph with **build_deepagents_graph()**
- Calls `agent.invoke({"messages": [HumanMessage(content=question)]})`
- Sets `recursion_limit=25` to prevent infinite agent loops
- Wraps execution with **retry_with_backoff()** for transient failures
- Extracts the final answer from the agent's response

### Workflow
1. User submits question
2. Orchestrator receives the question and calls research-analyst subagent
3. Research-analyst uses **search_corpus()** tool to find relevant chunks
4. Research-analyst returns bullet-point evidence with citations
5. Orchestrator passes research notes to synthesis-writer subagent
6. Synthesis-writer produces the final answer
7. Orchestrator returns the synthesized response

### Advantages
- **Specialized roles**: research and synthesis are handled by dedicated agents
- **Transparent evidence gathering**: research notes show what evidence was found
- **Better for complex questions**: multi-step reasoning with explicit evidence collection
- **Easier to audit**: can inspect research notes and final synthesis separately
- **Extensible**: easy to add more specialized subagents

## Configuration

All runtime settings are loaded from `.env` file through the **AppConfig** dataclass in `config.py`:

### Core Settings
- **DATA_DIR** (default: `data`) — directory containing PDFs for indexing
- **CHUNK_SIZE** (default: 1200) — characters per chunk before splitting
- **CHUNK_OVERLAP** (default: 180) — character overlap between adjacent chunks
- **TOP_K** (default: 5) — number of chunks to retrieve per query
- **DEFAULT_QUESTION** (default: prompt about important issues) — initial question in UI

### Model & Provider
- **MODEL_PROVIDER** (default: `gemini`) — LLM provider is always Gemini
- **MODEL_NAME** (default: `gemini-2.5-flash`) — model identifier
- **MODEL_TEMPERATURE** (default: 0.0) — LLM temperature (0 = deterministic)
- **GEMINI_API_KEY** — required for Gemini chat and embeddings

### Embeddings
- **EMBEDDING_MODEL_NAME** (default: `models/embedding-001`) — Google Gemini embedding model
- **EMBEDDING_BATCH_SIZE** (default: 64) — documents per batch during embedding
- **EMBEDDING_MAX_CONCURRENCY** (default: 4) — parallel embedding batches

### Qdrant Vector Database
- **QDRANT_URL** (default: None) — remote Qdrant URL (if not set, uses local)
- **QDRANT_API_KEY** (default: None) — API key for remote Qdrant
- **QDRANT_PATH** (default: `.qdrant`) — local directory for Qdrant database
- **QDRANT_COLLECTION_NAME** (default: `virallens_corpus`) — collection name in Qdrant

### Optional Jina Reranking
- **JINA_API_KEY** (default: None) — enables neural reranking if set
- **JINA_RERANKER_MODEL** (default: `jina-reranker-v3`) — Jina model version
- **RERANK_CANDIDATE_MULTIPLIER** (default: 4) — multiplier for candidates sent to reranker

### Resilience
- **RETRY_ATTEMPTS** (default: 4) — number of retry attempts for transient failures
- Retry logic uses exponential backoff with jitter: `delay = min(base * (2 ** attempt), max_delay)`

Example `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_PROVIDER=gemini
MODEL_NAME=gemini-2.5-flash
EMBEDDING_MODEL_NAME=models/embedding-001
DATA_DIR=data
CHUNK_SIZE=1200
CHUNK_OVERLAP=180
TOP_K=5
RETRY_ATTEMPTS=4
```

## Utilities Package

The `utilities/` package provides reusable components for PDF loading, chunking, retrieval, and LLM integration:

### chunking.py
- **chunk_documents(documents, chunk_size, chunk_overlap)** — splits page documents into overlapping chunks with recursive fallback to smaller units
- Uses RecursiveCharacterTextSplitter with hierarchy: paragraphs → sentences → words → characters

### data.py
- **load_pdf_documents(data_dir)** — extracts all PDFs from a directory page-by-page with source tracking
- **load_pdf_documents_from_paths(pdf_paths)** — extracts PDFs from specific paths (for user uploads)
- **stage_uploaded_pdfs()** — copies uploaded PDF files to a staging directory
- Returns LangChain Document objects with metadata: `source`, `path`, `page`

### retrieval.py
- **CorpusIndex** — dataclass holding all retrieval state (chunks, embeddings, Qdrant client, BM25 index)
- **SearchHit** — dataclass representing a search result with score and document
- **build_corpus_index()** — creates a new index from chunks:
  1. Generates document-level metadata using an LLM
  2. Enriches each chunk with metadata and search text
  3. Embeds all search texts concurrently using OpenAI
  4. Uploads embeddings to Qdrant
  5. Builds a BM25 index in memory
- **load_corpus_index_from_qdrant()** — loads an existing index from Qdrant
- **clear_qdrant_collection()** — deletes the Qdrant collection
- **format_search_hits(hits)** — formats SearchHit objects into readable markdown with citation info
- **create_qdrant_client()** — creates a Qdrant client (local or remote)
- Custom BM25 implementation for keyword-based ranking

### llm.py
- **ModelConfig** — dataclass for model settings (provider, model_name, temperature)
- **build_chat_model(config)** — creates a LangChain chat model for Gemini
  - Returns: `ChatGoogleGenerativeAI(model=..., temperature=..., api_key=...)`

### utils.py
- **retry_with_backoff(func, attempts, label)** — retries a function with exponential backoff
  - Used for LLM calls, embedding calls, and other transient-failure-prone operations
  - Exponential backoff with jitter: `delay = min(base * (2 ** attempt), max_delay)`
  - Logs retry attempts with the provided label
- **extract_final_ai_text(result)** — extracts text from LangChain model response or message list
- **message_content_to_text(content)** — converts message content to plain text
- **normalize_whitespace(text)** — cleans up whitespace (removes extra spaces, newlines)

## Gradio Web Interface

The web UI (built with Gradio) provides:

1. **PDF Upload** — optional file upload for custom documents
2. **Index Status Display** — shows how many chunks are indexed and their source
3. **Index New Files** — chunks and indexes uploaded PDFs (or defaults to `data/` if none uploaded)
4. **Clear DB** — deletes Qdrant collection for a fresh start
5. **Question Input** — textarea for entering queries (pre-filled with default question)
6. **Flow Selector** — radio buttons to choose:
   - `langgraph` — simple retrieve-answer pipeline
   - `deepagents` — multi-agent orchestration
   - `both` — runs both flows and shows results side-by-side
7. **Answer Output** — displays results with source citations and chunk metadata

### UI Workflow
1. User uploads PDFs (optional) and clicks "Index new files"
2. System chunks documents, generates metadata, embeds, and stores in Qdrant
3. User enters a question and selects a flow
4. User clicks "Run"
5. System retrieves relevant chunks and generates an answer
6. UI displays the answer with retrieval scores and source citations
7. User can click "Clear DB" to reset and index new files

## Main Entry Point

**main.py** is the application launcher:
- Loads environment variables from `.env`
- Builds the Gradio UI with **build_demo()**
- Launches the app on `GRADIO_SERVER_NAME:GRADIO_SERVER_PORT` (default: `0.0.0.0:7860`)
- Manages runtime state in `_RuntimeState` singleton:
  - `_RUNTIME.index` — currently loaded CorpusIndex
  - `_RUNTIME.chunk_count` — number of chunks in the index
  - `_RUNTIME.source_label` — description of the index source (e.g., "data/" or "uploaded PDFs")

### Key Functions
- **_build_runtime()** — loads PDFs, chunks them, builds Qdrant index, returns CorpusIndex
- **_build_model()** — creates a chat model from config
- **_resolve_index()** — returns the current index or loads from Qdrant if not in memory
- **_index_new_files()** — handles "Index new files" button click
- **_clear_db()** — handles "Clear DB" button click
- **_build_answers()** — orchestrates question answering with selected flow(s)

## Run Locally

### Prerequisites
- Python 3.11+
- pip or poetry for dependency management

### Setup
1. Clone the repository
2. Copy `.env.sample` to `.env`:
   ```bash
   cp .env.sample .env
   ```
3. Set your Google Gemini API key in `.env`:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Ensure PDFs are in `data/` directory (or use UI to upload)

### Run
```bash
python main.py
```
This launches the Gradio app at `http://localhost:7860`.

### First Use
1. Open `http://localhost:7860` in a browser
2. (Optional) Upload custom PDFs using the file uploader
3. Click **Index new files** to chunk and embed documents
4. Enter a question in the text field
5. Select a flow (both by default)
6. Click **Run** to generate answers

## Docker

The repository includes `docker-compose.yml` for containerized deployment with Qdrant:

### Setup
1. Copy `.env.sample` to `.env` and fill in your Google Gemini API key:
   ```bash
   cp .env.sample .env
   # Edit .env and set GEMINI_API_KEY=your_api_key_here
   ```
2. Start the stack:
   ```bash
   docker compose up --build
   ```
3. Wait for all services to be healthy (Qdrant, app)
4. Open the app at `http://localhost:7860`

### Services
- **app** — Virallens FastAPI/Gradio application
- **qdrant** — Vector database for semantic embeddings

### Compose Configuration
- Qdrant: exposed on `localhost:6333` (can be used by other clients)
- App: exposed on `localhost:7860`
- Environment variables are shared via `.env` file

## Notes

- **Simple UI**: Gradio is used for a lightweight, web-native interface without complex frontend dependencies
- **Gemini-powered**: Google Gemini is the sole provider for both chat and embeddings
- **Configuration-driven**: All settings are environment-based for easy environment switching
- **Reusable utilities**: Core logic (PDF loading, chunking, retrieval, LLM calls) is under `utilities/` for easy reuse in other projects
- **Main entry point**: `main.py` serves as both the application launcher and the sample main file requested in the brief
- **Hybrid retrieval**: Combines semantic embeddings (Google Gemini) and BM25 lexical search for robust ranking
- **Retry resilience**: All external API calls (LLM, embeddings, Jina) use exponential backoff to handle transient failures
- **Flexible indexing**: Supports both local PDFs and user uploads, with optional metadata enrichment via LLM
- **Multi-agent extensibility**: DeepAgents framework makes it easy to add more specialized subagents for complex reasoning tasks
