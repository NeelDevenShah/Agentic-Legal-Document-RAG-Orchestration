from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from config import AppConfig
from deepagents_flow import run_deepagents_rag
from graph_flow import run_langgraph_rag
from utilities.chunking import chunk_documents
from utilities.data import load_pdf_documents
from utilities.llm import ModelConfig, build_chat_model
from utilities.retrieval import build_corpus_index, load_corpus_index_from_qdrant
from utilities.utils import retry_with_backoff


def evaluate_single_sample(
    sample: dict[str, str],
    idx: int,
    total: int,
    index,
    model,
    config: AppConfig,
    judge_prompt_template: str,
) -> dict[str, object]:
    sample_idx = sample.get("sample_index", str(idx))
    chunk_id = sample.get("chunk_id", f"chunk-{idx}")
    source_file = sample.get("source_file", "")
    page = sample.get("page", "1")
    question = sample.get("question", "")
    expected_answer = sample.get("expected_answer", "")
    chunk_content = sample.get("chunk_content", "")

    print(f"[{idx}/{total}] Processing TC-{sample_idx}: {chunk_id} ({source_file} p.{page})")

    # 1. Run LangGraph Flow
    t0 = time.perf_counter()
    def _run_lg():
        return run_langgraph_rag(
            question=question,
            index=index,
            model=model,
            top_k=config.top_k,
            retry_attempts=6,
        )

    try:
        langgraph_ans = retry_with_backoff(_run_lg, attempts=6, label=f"LangGraph TC-{sample_idx}")
    except Exception as e:
        langgraph_ans = f"ERROR: {e}"
    langgraph_time = time.perf_counter() - t0

    time.sleep(1.0)  # Rate limiting spacing

    # 2. Run DeepAgents Flow
    t0 = time.perf_counter()
    def _run_da():
        return run_deepagents_rag(
            question=question,
            index=index,
            model=model,
            retry_attempts=6,
        )

    try:
        deepagents_ans = retry_with_backoff(_run_da, attempts=6, label=f"DeepAgents TC-{sample_idx}")
    except Exception as e:
        deepagents_ans = f"ERROR: {e}"
    deepagents_time = time.perf_counter() - t0

    time.sleep(1.0)  # Rate limiting spacing

    # 3. LLM Judge Evaluation
    judge_prompt = judge_prompt_template.format(
        question=question,
        expected_answer=expected_answer,
        chunk_id=chunk_id,
        source_file=source_file,
        page=page,
        chunk_content=chunk_content,
        langgraph_answer=langgraph_ans,
        deepagents_answer=deepagents_ans,
    )

    def _judge():
        res = model.invoke(judge_prompt)
        raw = getattr(res, "content", str(res)).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        if "langgraph" not in parsed or "deepagents" not in parsed:
            raise ValueError(f"Invalid judge format: {raw}")
        return parsed

    judge_eval = retry_with_backoff(_judge, attempts=6, label=f"Judge TC-{sample_idx}")

    lg_eval = judge_eval.get("langgraph", {})
    da_eval = judge_eval.get("deepagents", {})

    print(
        f"  --> TC-{sample_idx} Complete | "
        f"LangGraph: Score={lg_eval.get('accuracy_score', 0)}/10 ({langgraph_time:.1f}s) | "
        f"DeepAgents: Score={da_eval.get('accuracy_score', 0)}/10 ({deepagents_time:.1f}s)"
    )

    return {
        "sample_index": int(sample_idx) if str(sample_idx).isdigit() else idx,
        "chunk_id": chunk_id,
        "source_file": source_file,
        "page": page,
        "question": question,
        "expected_answer": expected_answer,
        "langgraph_answer": langgraph_ans,
        "langgraph_latency_sec": round(langgraph_time, 3),
        "langgraph_accuracy_score": float(lg_eval.get("accuracy_score", 0)),
        "langgraph_citation_score": float(lg_eval.get("citation_score", 0)),
        "langgraph_passed": bool(lg_eval.get("passed", False)),
        "langgraph_reasoning": str(lg_eval.get("reasoning", "")),
        "deepagents_answer": deepagents_ans,
        "deepagents_latency_sec": round(deepagents_time, 3),
        "deepagents_accuracy_score": float(da_eval.get("accuracy_score", 0)),
        "deepagents_citation_score": float(da_eval.get("citation_score", 0)),
        "deepagents_passed": bool(da_eval.get("passed", False)),
        "deepagents_reasoning": str(da_eval.get("reasoning", "")),
    }


def run_evaluation(
    *,
    input_csv_path: str = "rag_test_cases.csv",
    output_results_csv: str = "evaluation_results.csv",
    max_workers: int = 2,
):
    load_dotenv()
    config = AppConfig.from_env()

    print("Initializing LLM model...")
    model = build_chat_model(
        ModelConfig(
            provider=config.provider,
            model_name=config.model_name,
            temperature=config.temperature,
        )
    )

    print("Loading documents and building Corpus Index...")
    documents = load_pdf_documents(config.data_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    index = build_corpus_index(
        chunks,
        metadata_llm=model,
        retry_attempts=6,
        embedding_batch_size=config.embedding_batch_size,
        embedding_max_concurrency=config.embedding_max_concurrency,
        jina_api_key=None,  # Use local hybrid BM25 + Qdrant RRF for evaluation
        qdrant_path=config.qdrant_path,
        qdrant_url=config.qdrant_url,
        qdrant_api_key=config.qdrant_api_key,
        collection_name=config.qdrant_collection_name,
        embedding_model_name=config.embedding_model_name,
        provider=config.provider,
    )
    print(f"CorpusIndex ready with {len(index.chunks)} chunks.\n")

    # Read test cases from CSV
    test_cases_file = Path(input_csv_path).resolve()
    if not test_cases_file.exists():
        raise FileNotFoundError(f"Test cases file missing: {test_cases_file}")

    with open(test_cases_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_cases = list(reader)

    print(f"Starting rate-limit-protected evaluation of {len(test_cases)} test cases (workers={max_workers})...\n")

    judge_prompt_template = (
        "You are an impartial expert evaluator for Retrieval-Augmented Generation (RAG) systems.\n"
        "Evaluate the following two RAG model outputs against the Question, Expected Ground Truth Answer, and Source Context Chunk.\n\n"
        "Question: {question}\n"
        "Expected Ground Truth Answer: {expected_answer}\n"
        "Source Context Chunk ({chunk_id}, {source_file} p.{page}):\n"
        "{chunk_content}\n\n"
        "--- SYSTEM 1: LangGraph (Single-agent direct retrieve-and-answer) ---\n"
        "{langgraph_answer}\n\n"
        "--- SYSTEM 2: DeepAgents (Multi-agent research and synthesis) ---\n"
        "{deepagents_answer}\n\n"
        "Evaluation Instructions:\n"
        "- Assess accuracy_score (0 to 10): Does the generated answer correctly answer the question based on the expected answer and source text?\n"
        "- Assess citation_score (0 to 10): Does the answer properly attribute evidence to document metadata or source details?\n"
        "- Determine passed (true if accuracy_score >= 7, else false).\n"
        "- Provide brief reasoning for each score.\n\n"
        "Return ONLY a JSON object formatted strictly as follows (no markdown blocks, no extra commentary):\n"
        "{{\n"
        '  "langgraph": {{\n'
        '    "accuracy_score": <number 0-10>,\n'
        '    "citation_score": <number 0-10>,\n'
        '    "passed": <true/false>,\n'
        '    "reasoning": "<string>"\n'
        "  }},\n"
        '  "deepagents": {{\n'
        '    "accuracy_score": <number 0-10>,\n'
        '    "citation_score": <number 0-10>,\n'
        '    "passed": <true/false>,\n'
        '    "reasoning": "<string>"\n'
        "  }}\n"
        "}}\n"
    )

    eval_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_single_sample,
                sample,
                idx,
                len(test_cases),
                index,
                model,
                config,
                judge_prompt_template,
            )
            for idx, sample in enumerate(test_cases, start=1)
        ]
        for future in as_completed(futures):
            try:
                res = future.result()
                eval_results.append(res)
            except Exception as e:
                print(f"Error in evaluation task: {e}")

    # Sort results by sample index
    eval_results.sort(key=lambda r: r["sample_index"])

    # Export results CSV
    results_csv_file = Path(output_results_csv).resolve()
    if eval_results:
        fieldnames = list(eval_results[0].keys())
        with open(results_csv_file, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(eval_results)

    # Calculate aggregate metrics
    total_samples = len(eval_results)
    lg_pass_count = sum(1 for r in eval_results if r["langgraph_passed"])
    da_pass_count = sum(1 for r in eval_results if r["deepagents_passed"])

    lg_avg_acc = sum(r["langgraph_accuracy_score"] for r in eval_results) / total_samples
    da_avg_acc = sum(r["deepagents_accuracy_score"] for r in eval_results) / total_samples

    lg_avg_cite = sum(r["langgraph_citation_score"] for r in eval_results) / total_samples
    da_avg_cite = sum(r["deepagents_citation_score"] for r in eval_results) / total_samples

    lg_avg_lat = sum(r["langgraph_latency_sec"] for r in eval_results) / total_samples
    da_avg_lat = sum(r["deepagents_latency_sec"] for r in eval_results) / total_samples

    lg_pass_rate = (lg_pass_count / total_samples) * 100
    da_pass_rate = (da_pass_count / total_samples) * 100

    summary_md = f"""
# RAG Benchmark Evaluation Results Summary

| Metric | LangGraph Flow | DeepAgents Flow |
|---|---|---|
| **Accuracy Pass Rate (Score >= 7)** | **{lg_pass_rate:.1f}%** ({lg_pass_count}/{total_samples}) | **{da_pass_rate:.1f}%** ({da_pass_count}/{total_samples}) |
| **Mean Accuracy Score (0-10)** | **{lg_avg_acc:.2f}** | **{da_avg_acc:.2f}** |
| **Mean Citation Score (0-10)** | **{lg_avg_cite:.2f}** | **{da_avg_cite:.2f}** |
| **Mean Latency per Query** | **{lg_avg_lat:.2f}s** | **{da_avg_lat:.2f}s** |

Results exported to: `{results_csv_file}`
"""
    print("\n" + "=" * 50)
    print(summary_md)
    print("=" * 50 + "\n")

    return eval_results


if __name__ == "__main__":
    run_evaluation()
