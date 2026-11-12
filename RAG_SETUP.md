# Research Assistant (Gemini) — quick setup

1. `pip install -r requirements.txt`
2. Get a free API key at https://aistudio.google.com/apikey
3. Either copy `.env.example` to `.env` and set `GEMINI_API_KEY`, or paste the key into the dashboard's
   **💬 Research assistant** page (it stays in memory only).
4. `streamlit run dashboard_app.py`
5. Open **💬 Research assistant**, then click **Build / refresh search index** (once).
6. Ask a question, or tap one of the example questions.

## Models
| Purpose | Default | Alternatives |
|---|---|---|
| Embeddings (768-dim) | `gemini-embedding-2` | `gemini-embedding-001` |
| Answer writing | `gemini-3.5-flash` | `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite` |

Google retires and renames models over time. If you see a "model not found" message, pick another model in the
settings panel or edit `GEMINI_RAG_CHAT_MODEL` / `GEMINI_RAG_EMBEDDING_MODEL` in `.env`
(current list: https://ai.google.dev/gemini-api/docs/models).

If you switch the embedding model, rebuild the index. If you edit `data/literature_corpus.json`, rebuild too.

## Command-line tools
- `python test_rag_corpus.py` — validates the corpus (no API call).
- `python build_rag_index.py` — builds the index from the terminal.
- `python evaluate_rag_retrieval.py` — Hit@3 and MRR@5 retrieval scores on a labelled question set.
