# GigaCorp Customer Support Assistant

> A small RAG (Retrieval-Augmented Generation) demo using FastAPI, Chroma, and a local FAQ.

## What this project is

This repository contains a simple customer support assistant that:
- Loads a FAQ text file (`gigacorp_faq.txt`) into a Chroma vector store (via `ingest.py`).
- Exposes a FastAPI web service (`app.py`) with a web UI in `static/` and a chat API (`/api/chat`).
- Uses a configurable LLM provider (Groq or OpenAI) when an API key is available; otherwise it returns a safe fallback response.

## Files of interest
- `app.py` — FastAPI application and chat endpoint.
- `ingest.py` — script to ingest `gigacorp_faq.txt` into `chroma_db/`.
- `gigacorp_faq.txt` — plaintext FAQ used as the knowledge base.
- `chroma_db/` — persistent Chroma database (created by `ingest.py`).
- `static/` — simple web UI (`index.html`, `style.css`).
- `tests/` — basic tests (run with `pytest`).

## Requirements

- Python 3.10+ recommended
- Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment

Create a `.env` file or export environment variables. Common options:

- `GROQ_API_KEY` — (optional) API key for Groq provider. If omitted, the app will return a documented fallback.
- `PORT` — (optional) port for the FastAPI server (default: `8000`).

## Ingest data (build the vector DB)

Run the ingestion script to populate `chroma_db/` from the FAQ:

```bash
python ingest.py
```

This creates/updates files under `chroma_db/` (a local SQLite-backed Chroma store).

## Run the application

Start the service (defaults to port 8000):

```bash
# Windows PowerShell
python app.py
```

Open the web UI at: http://localhost:8000

### API endpoints

- `POST /api/chat` — chat endpoint. JSON payload:

```json
{
  "message": "User question text",
  "history": [
    {"role": "user", "content": "previous message"},
    {"role": "assistant", "content": "assistant reply"}
  ]
}
```

- `GET /api/faq` — returns the raw FAQ text.

## Tests

Run tests with:

```bash
pytest -q
```

## Troubleshooting

- If embeddings or the Chroma store fail to load, ensure `chroma_db/` is present (run `ingest.py`).
- If the LLM provider errors due to missing API key, supply `GROQ_API_KEY` or change provider code in `app.py`.

## Notes

This project is a demo and intentionally minimal. The `app.py` file runs an embedded Uvicorn server when executed directly and defaults to port `8000`.
