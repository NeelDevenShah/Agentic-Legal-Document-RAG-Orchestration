from __future__ import annotations

import argparse
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


PROMPT_TEMPLATE = (
    "You are an expert QA dataset generator for evaluating Retrieval-Augmented Generation (RAG) systems.\n"
    "Given the following legal document chunk, generate:\n"
    "1. A specific, clear, answerable question that can be answered strictly using ONLY the information in this chunk.\n"
    "2. A concise, accurate expected answer derived directly from the text of the chunk.\n\n"
    "CRITICAL RULES:\n"
    "- The question must be COMPLETELY self-contained. Include names of specific companies (e.g., Facebook, Macquarie, Costco), parties (e.g., Moab Partners, Amalgamated Bank, Chiueh, Knight, Upright Trust), or specific acts/laws (e.g., Exchange Act Section 10(b), Rule 10b-5(b)) so that a RAG system can retrieve the correct chunk based on the question alone. NEVER use ambiguous terms like 'the company', 'this case', 'the defendants', or 'the fund' without introducing their specific name.\n"
    "- The question must be purely CONTENT-based. Ask about factual findings, allegations, legal holdings, arguments, or rules.\n"
    "- Do NOT ask about document formatting, headings, page numbers, or tables (e.g., do NOT ask 'What heading is on page 51?' or 'What page numbers are listed for X?').\n"
    "- The question must NEVER contain meta-references to the chunk, excerpt, or document structure. Do NOT use phrases like 'according to the chunk', 'in this chunk', 'in this excerpt', 'as relevant to this chunk', 'according to the text', or 'according to the document'.\n\n"
    "Return ONLY a JSON object with keys 'question' and 'expected_answer'. Do not include markdown codeblocks or extra text.\n\n"
    "Document Source: {source}\n"
    "Page: {page}\n"
    "Chunk ID: {chunk_id}\n\n"
    "Chunk Text:\n"
    "{chunk_text}\n"
)


def generate_rag_test_cases(
    *,
    num_samples: int = 25,
    output_csv_path: str = "rag_test_cases.csv",
    existing_csv_path: str | None = None,
    seed: int = 42,
):
    load_dotenv()
    config = AppConfig.from_env()

    existing_test_cases = []
    used_chunk_ids = set()
    used_contents_normalized = set()

    if existing_csv_path:
        existing_file = Path(existing_csv_path).resolve()
        if existing_file.exists():
            with open(existing_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_test_cases.append(row)
                    used_chunk_ids.add(row["chunk_id"])
                    used_contents_normalized.add(" ".join(row["chunk_content"].split()).strip().lower())
            print(f"Loaded {len(existing_test_cases)} existing test cases from {existing_file.name}.")

    print(f"Loading documents from {config.data_dir}...")
    documents = load_pdf_documents(config.data_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    total_chunks = len(chunks)
    print(f"Loaded {len(documents)} document pages -> {total_chunks} chunks.")

    # Filter out already used chunks and boilerplate chunks
    available_chunks = []
    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_label")
        content_text = chunk.page_content.strip()
        content_norm = " ".join(content_text.split()).lower()

        # Skip used chunks, very short text (boilerplate), or empty headers
        if chunk_id in used_chunk_ids or content_norm in used_contents_normalized:
            continue
        if len(content_text) < 150 or content_text.upper() in ["APPENDIX", "TABLE OF AUTHORITIES"]:
            continue

        available_chunks.append(chunk)

    needed_samples = max(0, num_samples - len(existing_test_cases))
    print(f"Available non-duplicate chunks: {len(available_chunks)}. Need {needed_samples} new test cases.")

    random.seed(seed)
    sampled_chunks = random.sample(available_chunks, min(needed_samples, len(available_chunks)))

    model = build_chat_model(
        ModelConfig(
            provider=config.provider,
            model_name=config.model_name,
            temperature=0.0,
        )
    )

    new_test_cases = []

    for idx, chunk in enumerate(sampled_chunks, start=1):
        chunk_id = chunk.metadata.get("chunk_label", f"chunk-{idx}")
        source_file = chunk.metadata.get("source", "unknown.pdf")
        page = chunk.metadata.get("page", 1)
        chunk_text = chunk.page_content

        prompt = PROMPT_TEMPLATE.format(
            source=source_file,
            page=page,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
        )

        def _invoke_llm():
            res = model.invoke(prompt)
            content = getattr(res, "content", str(res)).strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            return json.loads(content)

        overall_idx = len(existing_test_cases) + idx
        print(f"[{idx}/{needed_samples}] Generating Q&A for {chunk_id} ({source_file}, p.{page})...")
        parsed = retry_with_backoff(_invoke_llm, attempts=config.retry_attempts, label=f"Q&A Gen {chunk_id}")

        question = parsed.get("question", "").strip()
        expected_answer = parsed.get("expected_answer", "").strip()

        new_test_cases.append({
            "sample_index": overall_idx,
            "chunk_id": chunk_id,
            "source_file": source_file,
            "page": page,
            "question": question,
            "expected_answer": expected_answer,
            "chunk_content": chunk_text,
        })

    # Combine existing and new test cases
    combined_test_cases = []
    for idx, tc in enumerate(existing_test_cases, start=1):
        tc["sample_index"] = idx
        combined_test_cases.append(tc)
    combined_test_cases.extend(new_test_cases)

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
        writer.writerows(combined_test_cases)

    print(f"\nSuccessfully saved {len(combined_test_cases)} total test cases to {csv_file}")
    return combined_test_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG test cases")
    parser.add_argument("--num-samples", type=int, default=25, help="Total number of test cases desired")
    parser.add_argument("--output", type=str, default="rag_test_cases.csv", help="Output CSV path")
    parser.add_argument("--existing", type=str, default=None, help="Existing CSV to extend")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    generate_rag_test_cases(
        num_samples=args.num_samples,
        output_csv_path=args.output,
        existing_csv_path=args.existing,
        seed=args.seed,
    )
