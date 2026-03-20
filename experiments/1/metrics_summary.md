# RAG Benchmark Evaluation Results Summary

## Aggregate Metrics

| Metric | LangGraph Flow | DeepAgents Flow |
|---|---|---|
| **Accuracy Pass Rate (Score >= 7)** | **88.0%** (22/25) | **84.0%** (21/25) |
| **Mean Accuracy Score (0-10)** | **8.56** | **8.40** |
| **Mean Citation Score (0-10)** | **8.24** | **7.88** |
| **Mean Latency per Query** | **2.39s** | **45.95s** |

## Per-Case Latency & Accuracy Breakdown

| Case # | Question | LangGraph Score | LangGraph Latency | DeepAgents Score | DeepAgents Latency |
|---|---|---|---|---|---|
| 1 | What SEC Staff report is cited as having been published in 2013 a... | 9.0 | 2.23s | 9.0 | 18.38s |
| 2 | In Facebook, Inc. v. Amalgamated Bank, when was the judgment of t... | 2.0 | 1.80s | 7.0 | 17.94s |
| 3 | What theory underlies private securities actions alleging that di... | 10.0 | 1.72s | 10.0 | 25.21s |
| 4 | What did respondents allege about certain risk disclosures in Met... | 9.0 | 2.37s | 7.0 | 23.77s |
| 5 | Why does the petition argue that the Ninth Circuit’s standard for... | 8.0 | 3.11s | 6.0 | 25.92s |
| 6 | Why does the Supreme Court say that Meta's risk disclosures conce... | 9.0 | 2.62s | 8.0 | 27.61s |
| 7 | Under Item 105, how would a reasonable investor interpret a risk ... | 10.0 | 2.56s | 10.0 | 22.32s |
| 8 | What does SEC Item 106 require registrants to disclose about cybe... | 8.0 | 2.54s | 9.0 | 25.29s |
| 9 | What did Costco Wholesale Corp. warn in its October 10, 2023 Form... | 10.0 | 1.94s | 10.0 | 22.82s |
| 10 | If a risk factor discussion is longer than 15 pages, what must be... | 10.0 | 1.83s | 10.0 | 26.98s |
| 11 | Can a failure to disclose information required by Item 303 of SEC... | 10.0 | 2.00s | 10.0 | 148.66s |
| 12 | What does Section 13(a) of the Exchange Act require issuers to fi... | 8.0 | 1.94s | 8.0 | 56.77s |
| 13 | What was the crux of Moab Partners, L. P.'s argument in its lawsu... | 8.0 | 2.46s | 8.0 | 20.76s |
| 14 | Under Exchange Act Rule 10b-5(b), what is the difference between ... | 8.0 | 2.79s | 8.0 | 25.54s |
| 15 | What does the Supreme Court say about when a failure to disclose ... | 10.0 | 2.25s | 10.0 | 201.04s |
| 16 | In the legal briefs, which editions of the Restatement of Contrac... | 10.0 | 1.88s | 4.0 | 66.98s |
| 17 | Why does Knight argue that it is entitled to summary judgment in ... | 7.0 | 3.02s | 10.0 | 29.13s |
| 18 | What do the plaintiffs assert about Knight’s potential liability ... | 9.0 | 2.44s | 9.0 | 29.01s |
| 19 | What means or instrumentalities did the defendants allegedly use ... | 9.0 | 3.26s | 9.0 | 25.78s |
| 20 | What fundamental policy did the 2021 Order find Upright Trust est... | 10.0 | 1.94s | 10.0 | 81.74s |
| 21 | What would have happened to the Upright Fund’s semiconductor hold... | 6.0 | 3.11s | 3.0 | 29.68s |
| 22 | What disclosures must a fund include in its next shareholder repo... | 4.0 | 2.25s | 5.0 | 33.61s |
| 23 | Did Chiueh share the 2019 Deficiency Letter with the Board before... | 10.0 | 2.97s | 10.0 | 35.20s |
| 24 | Under the Ninth Claim for Relief, what does the Commission allege... | 10.0 | 2.25s | 10.0 | 64.54s |
| 25 | Under which Investment Company Act provision are the defendants a... | 10.0 | 2.56s | 10.0 | 64.02s |

Results exported to: `/home/neel/Desktop/AIEngineerAssignment-Virallens/experiments/1/evaluation_results.csv`
