# Confident Group RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot for **Confident Group**, built with **FastAPI**, **LangChain**, **FAISS**, **sentence-transformers**, **Groq**, and **Streamlit**.

The application ingests company documents, creates embeddings, stores them in a FAISS vector index, retrieves relevant context for user questions, and sends the retrieved context to a Groq-hosted LLM to generate grounded answers.

## Project Structure

```text
confident-rag-chatbot/
|- app/
|  |- __init__.py
|  |- config.py
|  |- document_loader.py
|  |- embeddings.py
|  |- main.py
|  |- rag_pipeline.py
|  `- vector_store.py
|- data/
|  |- confident_docs/
|  `- faiss_index/
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

- FastAPI backend with a `POST /chat` endpoint
- LangChain-based document loading and chunking
- Sentence-transformer embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- FAISS vector store for semantic search
- Groq as the default LLM provider
- Optional Ollama support for local LLM inference
- Streamlit chat UI with conversation history
- PDF ingestion from `data/confident_docs`
- Optional website ingestion using BeautifulSoup

## Prerequisites

- Python 3.11 recommended
- `uv` installed
- A Groq API key

## 1. Create and Use the Virtual Environment

From the project root:

```bash
uv venv .venv
```

Activate it if you want:

### PowerShell

```powershell
cd confident-rag-chatbot
. .\.venv\Scripts\Activate.ps1
```

You can also skip manual activation and run everything through `uv run`.

## 2. Install Dependencies

```bash
uv pip install -r requirements.txt
```

## 3. Add Company Documents

Place Confident Group PDF files in:

```text
data/confident_docs/
```

Example files used in this project:

- `confident_group_overview.pdf`
- `confident_projects.pdf`
- `confident_services.pdf`
- `confident_faq.pdf`

## 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=llama-3.3-70b-versatile
BACKEND_URL=http://127.0.0.1:8000

# Optional settings
# CHUNK_SIZE=1000
# CHUNK_OVERLAP=200
# TOP_K_RESULTS=4

# Optional Ollama fallback
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.1
# OLLAMA_BASE_URL=http://localhost:11434

# Optional website ingestion
# WEBSITE_URLS=https://www.confident-group.com,https://example.com/about
```

Important:

- Use `GROQ_API_KEY=...` with no spaces around `=`
- Restart the backend after changing `.env`

## 5. Build the Vector Database

Run the ingestion pipeline after adding PDFs:

```bash
uv run python scripts/ingest_documents.py
```

This will:

- load PDF documents from `data/confident_docs`
- optionally load website content from `WEBSITE_URLS`
- split the text into chunks
- generate embeddings
- save the FAISS index to `data/faiss_index`

## 6. Run the FastAPI Backend

```bash
uv run uvicorn app.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## 7. Run the Streamlit UI

Open a second terminal in the same project directory and run:

```bash
uv run streamlit run ui/chat_ui.py
```

UI URL:

```text
http://localhost:8501
```

The chat title is:

```text
Confident Group AI Assistant
```

## API Example

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
  "response": "..."
}
```

## Example Questions for Demo

- What services does Confident Group provide?
- Where are Confident Group projects located?
- What amenities are available in their projects?
- What types of residential projects does Confident Group develop?
- Who founded Confident Group?
- What types of homes do they offer?

## Expected Behavior

- The assistant answers using retrieved document context
- If the answer is not present in the ingested documents, it should say it does not know
- Re-run ingestion whenever you add or update documents

## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

Run Streamlit from the project root:

```bash
uv run streamlit run ui/chat_ui.py
```

### `500 Internal Server Error` from `/chat`

Common causes:

- `GROQ_API_KEY` is missing or invalid
- the backend was not restarted after updating `.env`
- the FAISS index was not created yet

Fix:

```bash
uv run python scripts/ingest_documents.py
uv run uvicorn app.main:app --reload
```

### Backend works but answers are poor

- Verify the PDF files contain the information you are asking about
- Re-run ingestion after replacing or updating PDFs
- Ask questions that match the document content closely

## Notes

- Groq uses an OpenAI-compatible chat completions API
- Ollama support remains available as an optional local fallback
- The current setup is designed for local development and demos
