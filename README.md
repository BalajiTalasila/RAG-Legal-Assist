# RAG-Legal-Assist

### NLP-Driven Legal Document Processing System with Adaptive Hybrid Legal Chunking (AHLC)

---

## Overview

RAG-LegalAssist is an AI-powered legal document processing and question-answering system designed to retrieve relevant legal information and generate context-grounded, citation-supported responses.

The system combines Retrieval-Augmented Generation (RAG) with the proposed Adaptive Hybrid Legal Chunking (AHLC) approach to improve the organization and retrieval of legal text and support grounded legal question answering.

---

## Problem Statement

Legal documents are:

* Lengthy and structurally complex
* Rich in hierarchical legal structures
* Difficult to retrieve effectively using keyword-based methods alone
* Challenging for language models because of context limitations and hallucination risks

This project addresses these challenges through:

* Legal document preprocessing
* Structure-aware and semantic chunking
* Vector-based retrieval
* Retrieval-augmented response generation
* Retrieval and generation quality evaluation

---

## Key Features

* Adaptive Hybrid Legal Chunking (AHLC)
* Structure-aware legal document segmentation
* Semantic similarity-based segmentation
* Length-constrained chunk generation
* BGE-based text embeddings
* FAISS vector similarity retrieval
* Retrieval-Augmented Generation (RAG)
* Citation-supported responses
* Retrieval and generation evaluation
* Prompt engineering evaluation
* Paraphrase robustness evaluation
* Chunking-method comparison

---

## System Pipeline

The overall processing pipeline is:

Original ECtHR Dataset
        ↓
Preprocessing and Cleaning
        ↓
Cleaned Case Records
        ↓
Adaptive Hybrid Legal Chunking (AHLC)
        ↓
Text Embeddings
        ↓
FAISS Vector Index
        ↓
Top-K Retrieval
        ↓
Prompt Construction
        ↓
Qwen2.5-1.5B-Instruct
        ↓
Citation-Grounded Legal Response

---

## Data Processing

The original ECtHR legal judgment data were preprocessed to remove redundant symbols, excessive whitespace, and non-informative markers while preserving important legal structural information such as case identifiers and legal references.

The resulting cleaned case records are stored in:

```text
processed/clean_cases.csv
```

## License
This project is licensed under the MIT License.
