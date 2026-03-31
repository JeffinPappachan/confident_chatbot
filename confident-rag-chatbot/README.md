# Confident Group RAG Chatbot

A polished Retrieval-Augmented Generation (RAG) chatbot for **Confident Group**, built with **FastAPI**, **LangChain**, **FAISS**, **sentence-transformers**, **Groq/OpenAI/Ollama**, and **Streamlit**.

The system ingests company documents, builds a FAISS index, combines semantic and keyword retrieval, reranks results, protects the prompt context window, and returns grounded answers with source citations.

## Architecture

```text
User
-> Streamlit UI
-> FastAPI API
-> Query Rewriter
-> Embedding
-> Hybrid Retrieval (FAISS + BM25)
-> Reranker
-> Context Window Guard
-> Prompt Builder
-> LLM
-> Answer + Sources
```

## Project Structure

```text
confident-rag-chatbot/
|- app/
|  |- __init__.py
|  |- config.py
|  |- document_loader.py
|  |- embeddings.py
|  |- hybrid_retriever.py
|  |- llm_client.py
|  |- logger.py
|  |- main.py
|  |- query_rewriter.py
|  |- rag_pipeline.py
|  |- reranker.py
|  `- vector_store.py
|- data/
|  |- confident_docs/
|  `- faiss_index/
|- logs/
|- scripts/
|  `- ingest_documents.py
|- ui/
|  |- __init__.py
|  `- chat_ui.py
|- requirements.txt
|- README.md
`- run.sh
```

## Features

- FastAPI backend with `POST /chat` and `GET /health`
- Streamlit chat UI with preserved chat history
- Query rewriting for vague follow-up questions
- Hybrid retrieval using FAISS plus BM25
- Lightweight reranking with startup warmup and lazy fallback
- Context window guard to prevent prompt overflow
- Structured source citations in API/UI responses
- Query logging to `logs/query_logs.json`
- PDF ingestion with chunk metadata for source filename and page number
- Optional website ingestion support
- Groq, OpenAI-compatible, and Ollama provider support

## Prerequisites

- Python 3.11 recommended
- `uv` installed
- A configured LLM provider

## Setup

### 1. Create the virtual environment

```bash
uv venv .venv
```

### 2. Activate it in PowerShell

```powershell
cd confident-rag-chatbot
. .\.venv\Scripts\Activate.ps1
```

You can also skip activation and run everything with `uv run`.

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Add documents

Place Confident Group PDF files in:

```text
data/confident_docs/
```

Example files:

- `confident_group_overview.pdf`
- `confident_projects.pdf`
- `confident_services.pdf`
- `confident_faq.pdf`

## Environment Variables

Create a `.env` file in the project root.

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=llama-3.3-70b-versatile
BACKEND_URL=http://127.0.0.1:8000

# Retrieval
HYBRID_RETRIEVAL_TOP_K=20
HYBRID_VECTOR_TOP_K=15
HYBRID_BM25_TOP_K=15
RERANKER_TOP_K=5
ENABLE_QUERY_REWRITE=true
ENABLE_RERANKER=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
MAX_CONTEXT_TOKENS=2200

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Optional OpenAI-compatible provider
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key
# OPENAI_API_BASE=https://api.openai.com/v1

# Optional Ollama
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.1
# OLLAMA_BASE_URL=http://localhost:11434

# Optional website ingestion
# WEBSITE_URLS=https://www.confident-group.com,https://example.com/about

# Optional Hugging Face cache location if disk space is limited
# HF_HOME=E:\hf_cache
```

## Ingest Documents

```bash
uv run python scripts/ingest_documents.py
```

This will:

- load PDF documents from `data/confident_docs`
- optionally load website content from `WEBSITE_URLS`
- split the text into chunks
- add chunk metadata including source and page
- generate embeddings
- save the FAISS index to `data/faiss_index`

## Run the Backend

```bash
uv run uvicorn app.main:app --reload
```

Backend URLs:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

On startup, the app warms up the reranker model so the first user request is faster.

## Run the Streamlit UI

In a second terminal:

```bash
uv run streamlit run ui/chat_ui.py
```

UI URL:

- `http://localhost:8501`

## API Contract

### Request

```json
POST /chat
{
  "question": "What services does Confident Group provide?"
}
```

### Response

```json
{
  "answer": "Confident Group provides ...",
  "sources": [
    {
      "document": "confident_services.pdf",
      "page": 2
    }
  ]
}
```

## Example Questions

- What services does Confident Group provide?
- What residential projects does Confident Group offer?
- What can you tell me about Confident Group's completed projects?
- Where are Confident Group projects located?
- What amenities are available in their properties?
- Who founded Confident Group?
- What villa projects are mentioned in the documents?
- What about their villas?
- Do they offer premium housing options?

## Expected Behavior

- Answers are grounded only in the ingested context
- If the information is missing, the assistant says it is not available
- Citations are returned separately in the `sources` field
- Query logs are written without breaking the chatbot if logging fails
- Re-run ingestion whenever documents are added or replaced

## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

Run commands from the project root:

```bash
uv run streamlit run ui/chat_ui.py
```

### First request is slow

The backend warms up the reranker on startup. If the model still needs to download the first time, keep the server running until startup completes.

### Low disk space during model download

Use a larger Hugging Face cache path:

```env
HF_HOME=E:\hf_cache
```

### `500 Internal Server Error` from `/chat`

Common causes:

- missing or invalid API key
- backend not restarted after changing `.env`
- FAISS index not created yet

Fix:

```bash
uv run python scripts/ingest_documents.py
uv run uvicorn app.main:app --reload
```

## Notes

- Groq uses an OpenAI-compatible chat completions API
- Ollama remains available as a local fallback
- The project is structured for local demos, internal reviews, and further production hardening
