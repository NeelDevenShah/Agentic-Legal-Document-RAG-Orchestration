from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from dotenv import load_dotenv

from config import AppConfig
from utilities.chunking import chunk_documents
from utilities.data import load_pdf_documents
from utilities.llm import ModelConfig, build_chat_model
from utilities.utils import retry_with_backoff


def generate_rag_test_cases(
    *,
    num_samples: int = 25,
    output_csv_path: str = "rag_test_cases.csv",
    seed: int = 42,
):
    load_dotenv()
    config = AppConfig.from_env()

    print(f"Loading documents from {config.data_dir}...")
    documents = load_pdf_documents(config.data_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    total_chunks = len(chunks)
    print(f"Loaded {len(documents)} document pages -> {total_chunks} chunks.")

    # Randomly select num_samples chunks
    random.seed(seed)
    sampled_indices = sorted(random.sample(range(total_chunks), min(num_samples, total_chunks)))
    sampled_chunks = [chunks[i] for i in sampled_indices]

    print(f"Selected {len(sampled_chunks)} random chunks for benchmark test generation.")

    model = build_chat_model(
        ModelConfig(
            provider=config.provider,
            model_name=config.model_name,
            temperature=0.0,
        )
    )

    prompt_template = (
        "You are an expert QA dataset generator for evaluating Retrieval-Augmented Generation (RAG) systems.\n"
        "Given the following legal document chunk, generate:\n"
        "1. A specific, clear, answerable question that can be answered strictly using ONLY the information in this chunk.\n"
        "2. A concise, accurate expected answer derived directly from the text of the chunk.\n\n"
        "Return ONLY a JSON object with keys 'question' and 'expected_answer'. Do not include markdown codeblocks or extra text.\n\n"
        "Document Source: {source}\n"
        "Page: {page}\n"
        "Chunk ID: {chunk_id}\n\n"
        "Chunk Text:\n"
        "{chunk_text}\n"
    )

    test_cases = []

    for idx, chunk in enumerate(sampled_chunks, start=1):
        chunk_id = chunk.metadata.get("chunk_label", f"chunk-{idx}")
        source_file = chunk.metadata.get("source", "unknown.pdf")
        page = chunk.metadata.get("page", 1)
        chunk_text = chunk.page_content

        prompt = prompt_template.format(
            source=source_file,
            page=page,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
        )

        def _invoke_llm():
            res = model.invoke(prompt)
            content = getattr(res, "content", str(res)).strip()
            # Clean JSON formatting if wrapped in markdown blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            return json.loads(content)

        print(f"[{idx}/{len(sampled_chunks)}] Generating Q&A for {chunk_id} ({source_file}, p.{page})...")
        parsed = retry_with_backoff(_invoke_llm, attempts=config.retry_attempts, label=f"Q&A Gen {chunk_id}")

        question = parsed.get("question", "").strip()
        expected_answer = parsed.get("expected_answer", "").strip()

        test_cases.append({
            "sample_index": idx,
            "chunk_id": chunk_id,
            "source_file": source_file,
            "page": page,
            "question": question,
            "expected_answer": expected_answer,
            "chunk_content": chunk_text,
        })

    # Save to CSV
    csv_file = Path(output_csv_path).resolve()
    fieldnames = [
        "sample_index",
        "chunk_id",
        "source_file",
        "page",
        "question",
        "expected_answer",
        "chunk_content",
    ]

    with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_cases)

    print(f"\nSuccessfully generated {len(test_cases)} test cases and saved to {csv_file}")
    return test_cases


if __name__ == "__main__":
    generate_rag_test_cases()
