# Confident Group RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for **Confident Group** built with **FastAPI**, **Streamlit**, **LangChain**, **FAISS**, **sentence-transformers**, and provider-backed LLM inference through **Groq**, **OpenAI-compatible APIs**, or **Ollama**.

The current system ingests PDF documents from `data/confident_docs`, optionally pulls text from websites, creates a FAISS index, combines vector search with BM25 keyword retrieval, reranks the retrieved chunks, guards the final prompt size, and returns grounded answers with structured source citations.

## Architecture

```text
User
-> Streamlit UI
-> FastAPI API
-> Query Rewriter
-> Hybrid Retrieval (FAISS + BM25)
-> Cross-Encoder Reranker
-> Context Window Guard
-> Prompt Builder
-> LLM Provider
-> Answer + Sources
```

## What The System Does

- `POST /chat` accepts a user question and returns an `answer` plus `sources`
- `GET /health` exposes a simple health check
- Loads PDFs from `data/confident_docs`
- Optionally ingests website pages from `WEBSITE_URLS`
- Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings by default
- Merges FAISS semantic search with BM25 keyword matches
- Optionally rewrites vague follow-up queries before retrieval
- Optionally reranks retrieved chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Prevents oversized prompts with a context-window guard
- Logs query activity to `logs/query_logs.json`
- Returns source metadata separately from the generated answer

## Project Structure

```text
confident-rag-chatbot/
|- app/
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
|- scripts/
|  `- ingest_documents.py
|- ui/
|  `- chat_ui.py
|- requirements.txt
|- README.md
`- run.sh
```

## Requirements

- Python 3.11 recommended
- `uv` installed
- At least one configured LLM provider:
  - Groq
  - OpenAI-compatible API
  - Ollama

## Setup

### 1. Create and activate a virtual environment

PowerShell:

```powershell
cd confident-rag-chatbot
uv venv .venv
. .\.venv\Scripts\Activate.ps1
```

You can also skip activation and use `uv run ...` for every command.

### 2. Install dependencies

```powershell
uv pip install -r requirements.txt
```

### 3. Add source documents

Place company PDFs in:

```text
data/confident_docs/
```

The project already expects files such as:

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

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=4
HYBRID_RETRIEVAL_TOP_K=20
HYBRID_VECTOR_TOP_K=15
HYBRID_BM25_TOP_K=15
RERANKER_TOP_K=5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
MAX_CONTEXT_TOKENS=2200
LLM_TIMEOUT_SECONDS=60
QUERY_REWRITE_TIMEOUT_SECONDS=20
ENABLE_QUERY_REWRITE=true
ENABLE_RERANKER=true

# Optional log override
# LOG_PATH=logs/query_logs.json

# Optional website ingestion
# WEBSITE_URLS=https://www.confident-group.com,https://example.com/about

# Optional OpenAI-compatible provider
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key
# OPENAI_API_BASE=https://api.openai.com/v1

# Optional Groq base override
# GROQ_API_BASE=https://api.groq.com/openai/v1

# Optional Ollama
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.1
# OLLAMA_BASE_URL=http://localhost:11434

# Optional Hugging Face cache override
# HF_HOME=E:\hf_cache
```

Notes:

- `.env` entries should not contain spaces around `=`
- Restart the backend after changing provider or model settings
- `MODEL_NAME` is used for Groq and OpenAI-compatible chat completions
- `OLLAMA_MODEL` is used only when `LLM_PROVIDER=ollama`

## Build The Index

Run ingestion after adding or replacing documents:

```powershell
uv run python scripts/ingest_documents.py
```

This process:

- loads PDF files from `data/confident_docs`
- optionally fetches website content from `WEBSITE_URLS`
- splits content into chunks
- normalizes metadata to preserve source and page information
- assigns a stable `chunk_id` to each chunk
- creates embeddings
- writes the FAISS index to `data/faiss_index`

## Run The Backend

```powershell
uv run uvicorn app.main:app --reload
```

Available endpoints:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/chat`
- `http://127.0.0.1:8000/docs`

Startup behavior:

- the RAG pipeline loads the FAISS index during app lifespan initialization
- the reranker model is warmed up on startup
- if reranker warmup fails, the app still starts and falls back gracefully at runtime

## Run The Streamlit UI

In a second terminal:

```powershell
uv run streamlit run ui/chat_ui.py
```

UI URL:

- `http://localhost:8501`

The UI sends requests to `BACKEND_URL` and shows cited sources under each assistant response.

## API Contract

### Request

```json
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

## Retrieval And Answer Flow

1. The incoming question is validated by the FastAPI API.
2. The query rewriter optionally rewrites vague follow-up questions.
3. The hybrid retriever combines FAISS similarity search and BM25 keyword search.
4. The reranker optionally sorts the retrieved chunks by relevance.
5. A context guard trims the final chunk list to fit the configured token budget.
6. The LLM receives the system prompt plus grounded context and returns an answer.
7. The API returns the answer and deduplicated source metadata.

## Helper Script

`run.sh` installs dependencies and runs ingestion:

```bash
./run.sh
```

It is mainly useful in Unix-like environments. On Windows, run the `uv` commands directly.

## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

Run commands from the project root:

```powershell
uv run streamlit run ui/chat_ui.py
```

### `500 Internal Server Error` from `/chat`

Common causes:

- missing or invalid provider API key
- unsupported `LLM_PROVIDER`
- backend not restarted after editing `.env`
- FAISS index not built yet
- no PDF files available during ingestion

Typical recovery steps:

```powershell
uv run python scripts/ingest_documents.py
uv run uvicorn app.main:app --reload
```

### First request is slow

The reranker model is warmed up at startup, but the initial model download can still take time the first time the project runs.

### Website ingestion fails

Check that URLs in `WEBSITE_URLS` are reachable and return readable HTML content.

### Low disk space during model download

Set a larger Hugging Face cache path:

```env
HF_HOME=E:\hf_cache
```

## Notes

- Answers are designed to stay grounded in the ingested context
- If the answer is missing from the retrieved context, the assistant is instructed to say the information is not available
- Source citations are returned as structured metadata, not embedded in the answer text
- Query logging should not break the chatbot if logging fails
- Re-run ingestion whenever source documents change
