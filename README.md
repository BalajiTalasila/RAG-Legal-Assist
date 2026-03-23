# RAG-Legal-Assist
### NLP-Driven Legal Document Processing System with Adaptive Hybrid Legal Chunking (AHLC)

---

## Overview

RAG-LegalAssist is an AI-powered legal document assistant designed to process large-scale legal texts and generate **accurate, context-grounded, and citation-based answers**.

The system leverages **Retrieval-Augmented Generation (RAG)** combined with a novel **Adaptive Hybrid Legal Chunker (AHLC)** to improve retrieval precision and reduce hallucination in legal question answering.

---

## Problem Statement

Legal documents are:

* Lengthy and structurally complex
* Rich in hierarchical sections (Articles, Clauses, etc.)
* Difficult for traditional keyword-based systems
* Challenging for LLMs due to context limits and hallucination

This project addresses these issues by combining:

* Intelligent chunking
* Vector-based retrieval
* LLM-based grounded generation

---

## Key Features

* High-precision legal retrieval using vector embeddings
* LLM-based answer generation with citations
* Adaptive Hybrid Legal Chunker (AHLC) *(Novel Contribution)*
* Comprehensive evaluation framework
* Multiple evaluation phases (retrieval, grounding, robustness)
* Zero hallucination observed in experiments

---

## System Architecture

1. **Preprocessing**

   * Clean legal documents
   * Remove noise (headers, citations, etc.)

2. **Chunking (Core Innovation)**

   * Sliding Window (baseline)
   * Recursive Splitter (LangChain)
   * Semantic Chunking
   * **AHLC (proposed method)**

3. **Embedding & Indexing**

   * Sentence embeddings (BGE model)
   * FAISS vector database

4. **Retrieval**

   * Top-K relevant chunks selected

5. **Generation**

   * LLM (Qwen 2.5) generates answers using retrieved context

6. **Evaluation**

   * Retrieval metrics
   * Answer quality metrics
   * RAG-specific metrics

---

## Novel Contribution — AHLC

Adaptive Hybrid Legal Chunker (AHLC) dynamically combines:

* Structure-aware splitting (Articles, Sections)
* Semantic similarity segmentation
* Token-length enforcement

### Goal:

Improve **context relevance** and **reduce retrieval noise** in legal documents.

---

## Evaluation Phases

### Phase 1 — Large Scale Evaluation (50 Queries)

| Metric        | Value  |
| ------------- | ------ |
| Precision@5   | 0.40   |
| Recall@5      | 1.00   |
| MRR           | 1.00   |
| nDCG@5        | 1.00   |
| Groundedness  | 0.8749 |
| Faithfulness  | 0.9664 |
| Hallucination | 0.0000 |

High faithfulness and zero hallucination
Strong semantic grounding

---

### Phase 2 — Multi-Query Type Evaluation

Tested across:

* Legal grounds
* Articles applied
* Court decisions
* Reasoning
* Case summaries

| Metric        | Value  |
| ------------- | ------ |
| Groundedness  | 0.8833 |
| Faithfulness  | 0.9578 |
| Hallucination | 0.0000 |

Robust across different query types

---

### Phase 3 — Prompt Engineering

Compared multiple prompt designs.

**Final Selected Prompt:**

```
You are a legal assistant.
Use ONLY the provided legal context to answer.
Always cite chunk references like (Chunk #).
```

✔ Best grounding
✔ Zero hallucination
✔ Stable performance

---

### Phase 4 — Paraphrase Robustness

Evaluated using different phrasings of same query.

| Category      | Groundedness | Faithfulness |
| ------------- | ------------ | ------------ |
| Legal Grounds | 0.877        | 0.933        |
| Articles      | 0.878        | 1.000        |
| Decision      | 0.881        | 1.000        |

Low variance → strong robustness

---

### Phase 5 — Chunker Comparison (SOTA vs AHLC)

| Chunker             | Groundedness | Faithfulness |
| ------------------- | ------------ | ------------ |
| Sliding Window      | **0.8913**   | 0.9186       |
| Recursive           | 0.8883       | 0.9047       |
| Semantic            | 0.8883       | 0.9047       |
| **AHLC (Proposed)** | 0.8803       | **0.9333**   |

### Insight

* Sliding chunking → higher raw similarity
* AHLC → **better structural support (faithfulness)**

✔ AHLC improves **legal answer reliability**

---

## Metrics Used

### Retrieval Metrics

* Precision@K
* Recall@K
* MRR
* nDCG

### Answer Quality

* Exact Match
* F1 Score
* BLEU
* ROUGE

### RAG-Specific Metrics

* Groundedness
* Faithfulness
* Hallucination Rate

---

## Dataset

* **European Court of Human Rights (ECtHR) Dataset**
* Contains:

  * Full judgments
  * Legal reasoning
  * Structured case data

---

## Tech Stack

* Python
* PyTorch
* Transformers (HuggingFace)
* Sentence Transformers
* FAISS
* LangChain
* NLTK

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Preprocess data
python src/process.py

# Step 2: Build vector index
python src/embed_store.py

# Step 3: Run RAG system
python app.py
```

---

## Key Findings

* RAG significantly reduces hallucination in legal QA
* Prompt engineering improves grounding
* AHLC improves **faithfulness and structural alignment**
* System is robust across queries and paraphrases

---

## Future Work

* Automatic ground truth extraction
* Advanced evaluation (human-in-loop)
* Cross-jurisdiction legal datasets
* Integration with real-time legal systems
* Improved adaptive thresholding

---

## Authors

* Your Team Name / Members

---

## License

This project is for academic and research purposes.
