# Evaluating of LLM Applications
Code examples for systematically evaluating LLM applications using the RAGAS framework. Includes a case study comparing different RAG configurations with quantifiable metrics for context quality, response accuracy, and error susceptibility.


## TL;DR

This project demonstrates how to systematically evaluate Large Language Model (LLM) applications, especially those using Retrieval-Augmented Generation (RAG) pipelines. It leverages the open-source [RAGAS](https://docs.ragas.io/en/stable/) framework to provide quantitative, reproducible metrics for LLM system performance, moving beyond subjective or ad hoc testing. The repository includes a technical case study, code for document-based Q&A evaluation, and tools for vector search, reranking, and metric-driven analysis.

---

## Folder Overview

- **application/**
  Core logic for the LLM evaluation pipeline:
  - `application.py`: Main application class orchestrating LLM, vector store, and reranker.
  - `docling_parser.py`: Parses and chunks PDF documents for embedding.
  - `vector_store.py`: Handles document embedding and similarity search using ChromaDB and OpenAI.
  - `reranker.py`: Reranks retrieved documents using a neural model for improved relevance.

- **chroma_db/**
  Persistent vector database files and collections for document embeddings.

- **data/**
  - `raw/`: Source PDF documents (e.g., annual reports).
  - `questions_groundtruth.json`: Test questions and reference answers for evaluation.

- **documentation/**
  - `blog-article.md`: In-depth articles on LLM evaluation and the RAGAS methodology.
  - `evaluation_system_architecture.jpg`: System architecture diagram.
  - `metrics.png`: Visual summary of evaluation metrics.

- **main.ipynb**
  Jupyter notebook for running the evaluation pipeline, generating metrics, and visualizing results.

- **pyproject.toml / poetry.lock**
  Python dependencies and environment configuration (uses Poetry).

---

## Local Execution Guide

### 1. Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- An OpenAI API key (set as `OPENAI_API_KEY` in your environment)

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/synvert-group/llm_application_evaluation.git
cd llm_application_evaluation
poetry install
```

### 3. Environment Setup

Create a `.env` file in the project root (or export manually):

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Prepare Data

- Place your PDF documents in `data/raw/` (some are already included).
- The evaluation questions and ground truths are in `data/questions_groundtruth.json`.

### 5. Run the Evaluation

You can run the evaluation pipeline and generate metrics using the provided Jupyter notebook:

```bash
poetry shell
jupyter notebook main.ipynb
```

Follow the notebook cells to:
- Initialize the application and vector store
- Parse and embed documents
- Run Q&A with different retrieval/reranking configurations
- Evaluate results using RAGAS metrics
