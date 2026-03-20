# RAG Benchmark Evaluation Results Summary

## Aggregate Metrics

| Metric | LangGraph Flow | DeepAgents Flow |
|---|---|---|
| **Accuracy Pass Rate (Score >= 7)** | **84.0%** (42/50) | **86.0%** (43/50) |
| **Mean Accuracy Score (0-10)** | **8.38** | **8.28** |
| **Mean Citation Score (0-10)** | **8.10** | **7.84** |
| **Mean Latency per Query** | **2.54s** | **50.26s** |

## Per-Case Latency & Accuracy Breakdown

| Case # | Question | LangGraph Score | LangGraph Latency | DeepAgents Score | DeepAgents Latency |
|---|---|---|---|---|---|
| 1 | What SEC Staff report is cited as having been published in 2013 a... | 9.0 | 1.71s | 9.0 | 20.77s |
| 2 | In Facebook, Inc. v. Amalgamated Bank, when was the judgment of t... | 4.0 | 1.61s | 7.0 | 24.73s |
| 3 | What theory underlies private securities actions alleging that di... | 9.0 | 2.05s | 9.0 | 31.35s |
| 4 | What did respondents allege about certain risk disclosures in Met... | 8.0 | 2.37s | 7.0 | 29.11s |
| 5 | Why does the petition argue that the Ninth Circuit’s standard for... | 8.0 | 3.28s | 7.0 | 22.86s |
| 6 | Why does the Supreme Court say that Meta's risk disclosures conce... | 10.0 | 3.48s | 8.0 | 30.64s |
| 7 | Under Item 105, how would a reasonable investor interpret a risk ... | 9.0 | 4.31s | 10.0 | 54.08s |
| 8 | What does SEC Item 106 require registrants to disclose about cybe... | 8.0 | 3.16s | 8.0 | 23.06s |
| 9 | What did Costco Wholesale Corp. warn in its October 10, 2023 Form... | 10.0 | 1.95s | 10.0 | 23.78s |
| 10 | If a risk factor discussion is longer than 15 pages, what must be... | 10.0 | 2.14s | 9.0 | 35.66s |
| 11 | Can a failure to disclose information required by Item 303 of SEC... | 10.0 | 3.19s | 10.0 | 29.19s |
| 12 | What does Section 13(a) of the Exchange Act require issuers to fi... | 3.0 | 1.99s | 9.0 | 21.22s |
| 13 | What was the crux of Moab Partners, L. P.'s argument in its lawsu... | 8.0 | 2.46s | 8.0 | 29.21s |
| 14 | Under Exchange Act Rule 10b-5(b), what is the difference between ... | 9.0 | 3.73s | 9.0 | 30.02s |
| 15 | What does the Supreme Court say about when a failure to disclose ... | 10.0 | 2.94s | 10.0 | 82.09s |
| 16 | In the legal briefs, which editions of the Restatement of Contrac... | 6.0 | 2.16s | 5.0 | 32.90s |
| 17 | Why does Knight argue that it is entitled to summary judgment in ... | 4.0 | 2.53s | 9.0 | 24.12s |
| 18 | What do the plaintiffs assert about Knight’s potential liability ... | 9.0 | 2.50s | 8.0 | 69.02s |
| 19 | What means or instrumentalities did the defendants allegedly use ... | 8.0 | 2.55s | 7.0 | 34.24s |
| 20 | What fundamental policy did the 2021 Order find Upright Trust est... | 10.0 | 2.05s | 10.0 | 29.52s |
| 21 | What would have happened to the Upright Fund’s semiconductor hold... | 8.0 | 4.92s | 3.0 | 57.75s |
| 22 | What disclosures must a fund include in its next shareholder repo... | 4.0 | 2.16s | 5.0 | 91.16s |
| 23 | Did Chiueh share the 2019 Deficiency Letter with the Board before... | 10.0 | 1.88s | 10.0 | 20.02s |
| 24 | Under the Ninth Claim for Relief, what does the Commission allege... | 10.0 | 1.75s | 10.0 | 119.35s |
| 25 | Under which Investment Company Act provision are the defendants a... | 10.0 | 4.93s | 10.0 | 36.51s |
| 26 | What conduct does the Commission allege Chiueh engaged in that vi... | 4.0 | 2.66s | 9.0 | 33.23s |
| 27 | What was one of Upright Trust's Fund fundamental policies stated ... | 9.0 | 3.04s | 9.0 | 115.67s |
| 28 | What did the Ninth Circuit ultimately explain was the basis for i... | 10.0 | 2.18s | 10.0 | 54.21s |
| 29 | According to Investment Company Act Section 13(a)(3), what must a... | 10.0 | 1.82s | 10.0 | 57.07s |
| 30 | Did the Ninth Circuit hold that Rule 10b-5(b) liability can be pr... | 10.0 | 3.85s | 10.0 | 40.96s |
| 31 | What is the 'virtual certainty' standard for risk-factor statemen... | 7.0 | 2.87s | 0.0 | 650.69s |
| 32 | What did the Ninth Circuit panel majority explain made Alphabet’s... | 9.0 | 2.35s | 9.0 | 71.48s |
| 33 | What risk did Facebook warn in its Form 10-K could result from a ... | 10.0 | 1.76s | 10.0 | 29.21s |
| 34 | Before 2005, were companies required to disclose material future ... | 10.0 | 2.16s | 9.0 | 25.04s |
| 35 | Under SEC Item 105, why would a reasonable investor not be misled... | 9.0 | 2.54s | 9.0 | 30.09s |
| 36 | Why did Petitioners argue that a typical risk disclosure under It... | 8.0 | 2.68s | 8.0 | 29.71s |
| 37 | In Facebook, Inc. et al. v. Amalgamated Bank et al., who are the ... | 6.0 | 2.10s | 5.0 | 26.05s |
| 38 | What conduct did the Commission allege that Defendants engaged in... | 4.0 | 1.93s | 3.0 | 21.85s |
| 39 | Why were Chiueh’s false statements considered material to a reaso... | 9.0 | 2.18s | 8.0 | 24.17s |
| 40 | What does Rule 10b–5(b) make unlawful, and what two things does i... | 10.0 | 1.79s | 10.0 | 22.85s |
| 41 | What conduct is the court permanently enjoining Chiueh, and those... | 9.0 | 2.54s | 8.0 | 33.10s |
| 42 | What actual loss to the Upright Trust Fund resulted from the sale... | 10.0 | 2.15s | 9.0 | 29.42s |
| 43 | Since when has Upright Trust been registered as an open-end inves... | 10.0 | 1.79s | 10.0 | 24.88s |
| 44 | What did the SEC’s 1964 Securities Act guidance advise offerors o... | 8.0 | 2.36s | 9.0 | 29.41s |
| 45 | What did the Sixth Circuit in Bondali conclude about the plaintif... | 9.0 | 1.97s | 8.0 | 22.45s |
| 46 | How did Defendants classify Company A during the Relevant Period,... | 10.0 | 1.96s | 10.0 | 31.67s |
| 47 | What does Regulation S-K Item 105 require filers to provide, and ... | 7.0 | 2.87s | 6.0 | 31.67s |
| 48 | Under Exchange Act Section 10(b) and Rule 10b-5(b), when can an o... | 9.0 | 2.83s | 10.0 | 29.76s |
| 49 | According to Judge Bumatay’s dissent, what did Meta’s risk statem... | 10.0 | 2.14s | 10.0 | 24.02s |
| 50 | What does Investment Company Act Section 8(a), 15 U.S.C. § 80a-8(... | 8.0 | 2.87s | 8.0 | 21.94s |

Results exported to: `/home/neel/Desktop/AIEngineerAssignment-Virallens/experiments/2/evaluation_results.csv`
