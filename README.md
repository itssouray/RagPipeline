# RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) system built from scratch.

## Features

- PDF Loading
- Recursive Chunking
- SentenceTransformer Embeddings
- FAISS Vector Store
- Semantic Retrieval
- GPT-4.1 Response Generation
- Persistent Vector Store

## Architecture

```text
PDF
 ↓
Loader
 ↓
Chunker
 ↓
Embeddings
 ↓
FAISS
 ↓
Retriever
 ↓
GPT-4.1
 ↓
Answer
```

## Project Structure

```text
src/
├── loaders/
├── chunkers/
├── embeddings/
├── vectorstores/
├── retrievers/
├── generators/
├── pipelines/
├── models/
└── config/
```

## Run

```bash
pip install -r requirements.txt

python main.py
```
