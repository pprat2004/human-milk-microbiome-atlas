"""
Grounded literature RAG for the Human Milk Microbiome Atlas (Google Gemini edition).

Indexing:
  - Gemini embeddings (default: gemini-embedding-2, 768 dimensions)
  - Local cosine-similarity vector search (NumPy)
Generation:
  - Gemini generateContent API (default: gemini-3.5-flash; configurable)

The API key is read from GEMINI_API_KEY (or GOOGLE_API_KEY) or passed in from the
dashboard session. It is never written to the index or source files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data" / "literature_corpus.json"
DEFAULT_STORE = ROOT / "rag_store"

# Model names can be overridden from .env (see .env.example) if Google retires or renames them.
DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_RAG_EMBEDDING_MODEL", "gemini-embedding-2")
DEFAULT_CHAT_MODEL = os.getenv("GEMINI_RAG_CHAT_MODEL", "gemini-3.5-flash")
EMBEDDING_DIMENSIONS = 768  # Matryoshka-truncated size: compact, fast and accurate for a small corpus.
EMBEDDING_MODEL_CHOICES = ["gemini-embedding-2", "gemini-embedding-001"]
CHAT_MODEL_CHOICES = ["gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite"]


class RAGError(RuntimeError):
    """Raised when the RAG pipeline cannot safely complete a step."""


def get_client(api_key: str | None = None):
    """Create a Gemini client from the supplied key or the environment."""
    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RAGError(
            "Gemini API key is missing. Set GEMINI_API_KEY or enter it in the dashboard."
        )
    try:
        from google import genai
    except ImportError as exc:
        raise RAGError(
            "The Google Gen AI SDK is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return genai.Client(api_key=key)


def _with_retry(call, attempts: int = 4):
    """Run an API call, retrying politely on rate limits and temporary server errors."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # SDK raises APIError subclasses with a numeric .code
            code = getattr(exc, "code", None)
            if code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise RAGError(_friendly_error(exc)) from exc


def _friendly_error(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    text = str(exc)
    if code in (400, 401, 403) and ("API key" in text or "API_KEY" in text or code in (401, 403)):
        return "Gemini rejected the API key. Check that it is valid and the Gemini API is enabled for it."
    if code == 404:
        return ("Gemini could not find the selected model. Pick another model in the settings "
                "or update GEMINI_RAG_CHAT_MODEL / GEMINI_RAG_EMBEDDING_MODEL in .env.")
    if code == 429:
        return "Gemini rate limit reached. Wait a minute and try again."
    return f"Gemini API error: {text[:300]}"


def load_corpus(corpus_path: str | Path = DEFAULT_CORPUS) -> list[dict[str, Any]]:
    path = Path(corpus_path)
    if not path.exists():
        raise RAGError(f"Corpus file not found: {path}")
    try:
        docs = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RAGError(f"Could not read corpus: {exc}") from exc

    if not isinstance(docs, list) or not docs:
        raise RAGError("Corpus must be a non-empty JSON list.")

    seen = set()
    required = {"doc_id", "title", "content", "doc_type"}
    for doc in docs:
        missing = required - set(doc)
        if missing:
            raise RAGError(f"Corpus document is missing fields: {sorted(missing)}")
        if doc["doc_id"] in seen:
            raise RAGError(f"Duplicate document ID: {doc['doc_id']}")
        seen.add(doc["doc_id"])
        if not str(doc["content"]).strip():
            raise RAGError(f"Document has empty content: {doc['doc_id']}")
    return docs


def corpus_fingerprint(docs: list[dict[str, Any]]) -> str:
    canonical = json.dumps(docs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + int(max_chars * 0.55):
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_documents(
    docs: list[dict[str, Any]],
    max_chars: int = 1800,
    overlap: int = 180,
) -> list[dict[str, Any]]:
    """Split each source into concise chunks while preserving bibliographic metadata."""
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        raw_sections = [
            part.strip()
            for part in re.split(r"\n\s*\n", str(doc["content"]))
            if part.strip()
        ]
        blocks: list[str] = []
        current = ""
        for section in raw_sections:
            for piece in _split_long_text(section, max_chars, overlap):
                if not current:
                    current = piece
                elif len(current) + len(piece) + 2 <= max_chars:
                    current += "\n\n" + piece
                else:
                    blocks.append(current)
                    current = piece
        if current:
            blocks.append(current)

        for number, block in enumerate(blocks, start=1):
            header = (
                f"Title: {doc['title']}\n"
                f"Authors: {doc.get('authors', 'Not listed')}\n"
                f"Year: {doc.get('year', 'n.d.')}\n"
                f"Source type: {doc['doc_type']}\n"
                f"Topics: {', '.join(doc.get('topics', []))}\n"
            )
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_C{number:02d}",
                    "doc_id": doc["doc_id"],
                    "chunk_number": number,
                    "title": doc["title"],
                    "authors": doc.get("authors"),
                    "year": doc.get("year"),
                    "journal": doc.get("journal"),
                    "doi": doc.get("doi"),
                    "url": doc.get("url"),
                    "doc_type": doc["doc_type"],
                    "access_type": doc.get("access_type"),
                    "topics": doc.get("topics", []),
                    "text": header + "\n" + block,
                }
            )
    return chunks


def _format_for_model(texts: list[str], model: str, task: str, titles: list[str] | None) -> list[str]:
    """gemini-embedding-2 reads the task from the text itself; embedding-001 uses task_type."""
    if "embedding-2" not in model:
        return texts
    if task == "query":
        return [f"task: search result | query: {t}" for t in texts]
    titles = titles or ["none"] * len(texts)
    return [f"title: {ti} | text: {t}" for ti, t in zip(titles, texts)]


def _embed_batch(
    client,
    texts: list[str],
    model: str,
    task: str = "document",
    titles: list[str] | None = None,
) -> np.ndarray:
    """Embed texts with Gemini and return L2-normalised float32 vectors."""
    from google.genai import types

    contents = _format_for_model(texts, model, task, titles)
    if "embedding-2" in model:
        config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS)
    else:
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        )
    response = _with_retry(
        lambda: client.models.embed_content(model=model, contents=contents, config=config)
    )
    vectors = np.asarray([item.values for item in response.embeddings], dtype=np.float32)
    if vectors.shape[0] != len(texts):
        raise RAGError("Gemini returned a different number of embeddings than requested.")
    # Truncated Matryoshka vectors are not unit length, so normalise before cosine search.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def build_index(
    api_key: str | None = None,
    corpus_path: str | Path = DEFAULT_CORPUS,
    store_dir: str | Path = DEFAULT_STORE,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Create/update the persistent vector index using real model embeddings."""
    client = get_client(api_key)
    docs = load_corpus(corpus_path)
    chunks = chunk_documents(docs)
    if not chunks:
        raise RAGError("No chunks were produced from the corpus.")

    all_vectors = []
    texts = [chunk["text"] for chunk in chunks]
    titles = [chunk["title"] for chunk in chunks]
    for start in range(0, len(texts), batch_size):
        all_vectors.append(
            _embed_batch(
                client,
                texts[start : start + batch_size],
                embedding_model,
                task="document",
                titles=titles[start : start + batch_size],
            )
        )
    matrix = np.vstack(all_vectors).astype(np.float32)

    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    vectors_path = store / "vectors.npz"
    chunks_path = store / "chunks.json"
    manifest_path = store / "manifest.json"

    np.savez_compressed(vectors_path, embeddings=matrix)
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model,
        "corpus_sha256": corpus_fingerprint(docs),
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "vector_dimensions": int(matrix.shape[1]),
        "provider": "Google Gemini API",
        "retrieval": "cosine similarity over L2-normalised Gemini embeddings",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_index(
    store_dir: str | Path = DEFAULT_STORE,
    corpus_path: str | Path = DEFAULT_CORPUS,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    store = Path(store_dir)
    vectors_path = store / "vectors.npz"
    chunks_path = store / "chunks.json"
    manifest_path = store / "manifest.json"
    if not all(p.exists() for p in [vectors_path, chunks_path, manifest_path]):
        raise RAGError("Vector index is not built yet. Use 'Build / refresh index' first.")
    try:
        docs = load_corpus(corpus_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        with np.load(vectors_path) as npz:
            vectors = np.asarray(npz["embeddings"], dtype=np.float32)
    except Exception as exc:
        raise RAGError(f"Could not load vector index: {exc}") from exc

    current_hash = corpus_fingerprint(docs)
    if manifest.get("corpus_sha256") != current_hash:
        raise RAGError("The literature corpus changed after indexing. Rebuild the index.")
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise RAGError("Vector index dimensions do not match chunk metadata.")
    return vectors, chunks, manifest


def retrieve(
    query: str,
    api_key: str | None = None,
    top_k: int = 5,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    store_dir: str | Path = DEFAULT_STORE,
    corpus_path: str | Path = DEFAULT_CORPUS,
) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        raise RAGError("Enter a research question before searching.")
    vectors, chunks, manifest = load_index(store_dir, corpus_path)
    if manifest.get("embedding_model") != embedding_model:
        raise RAGError(
            "The selected embedding model does not match the saved index. Rebuild the index."
        )
    client = get_client(api_key)
    qvec = _embed_batch(client, [query], embedding_model, task="query")[0]
    if qvec.shape[0] != vectors.shape[1]:
        raise RAGError("Embedding size does not match the saved index. Rebuild the index.")
    scores = vectors @ qvec
    count = min(max(int(top_k), 1), len(chunks))
    order = np.argsort(scores)[::-1][:count]

    results = []
    for rank, idx in enumerate(order, start=1):
        item = dict(chunks[int(idx)])
        item["score"] = float(scores[int(idx)])
        item["rank"] = rank
        results.append(item)
    return results


def generate_grounded_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    api_key: str | None = None,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> str:
    if not retrieved_chunks:
        raise RAGError("No source passages were retrieved; cannot generate an answer.")
    client = get_client(api_key)

    allowed_ids = sorted({item["doc_id"] for item in retrieved_chunks})
    context_parts = []
    for item in retrieved_chunks:
        context_parts.append(
            f"[{item['doc_id']}] {item['title']} ({item.get('year') or 'n.d.'})\n"
            f"Source type: {item['doc_type']}\n"
            f"URL: {item.get('url') or 'Internal project record; no public URL'}\n"
            f"Retrieved passage:\n{item['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    instructions = f"""
You are the literature research assistant for a human-milk microbiome research project.
Answer only from the retrieved passages below. The retrieved corpus contains concise,
curated summaries of papers and one clearly labelled internal project-results note; it is
not a full-text archive. Never imply that you read an entire paper if the retrieved passage
only contains a summary.

Evidence rules:
1. Cite every substantive empirical or methodological claim with one or more allowed IDs:
   {", ".join(allowed_ids)}. Use citations exactly as [D01] or [P01].
2. Do not invent data, sample sizes, results, methods, authors, or citations.
3. Distinguish published literature (doc_type peer_reviewed_article/review) from the
   internal project note (doc_type internal_project_note). Never describe P01 as peer reviewed.
4. Separate human-milk microbiome evidence from infant stool/gut microbiome evidence.
5. Separate associations from causal effects. Do not infer a country-specific or clinical
   result from the current MAMI/PRIME data where metadata or design do not support it.
6. If the retrieved context is insufficient, say exactly what is missing and do not fill gaps
   from general model knowledge.
7. Include a concise "Sources used" section listing the cited document IDs and titles.
8. Do not diagnose, recommend treatment, or provide individual medical advice.
Use clear, technically accurate language suitable for a student research project.
"""
    prompt = f"""Research question:
{question}

Retrieved evidence:
{context}

Write a direct answer with a brief synthesis, limitations, and a Sources used section.
"""
    from google.genai import types

    response = _with_retry(
        lambda: client.models.generate_content(
            model=chat_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                max_output_tokens=4096,  # leaves room for the model's internal reasoning
            ),
        )
    )
    answer = (getattr(response, "text", None) or "").strip()
    if not answer:
        raise RAGError(
            "Gemini returned no text (the response may have been filtered). Try rephrasing the question."
        )
    return answer


def ask_rag(
    question: str,
    api_key: str | None = None,
    top_k: int = 5,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chat_model: str = DEFAULT_CHAT_MODEL,
    store_dir: str | Path = DEFAULT_STORE,
    corpus_path: str | Path = DEFAULT_CORPUS,
) -> tuple[str, list[dict[str, Any]]]:
    retrieved = retrieve(
        query=question,
        api_key=api_key,
        top_k=top_k,
        embedding_model=embedding_model,
        store_dir=store_dir,
        corpus_path=corpus_path,
    )
    answer = generate_grounded_answer(
        question=question,
        retrieved_chunks=retrieved,
        api_key=api_key,
        chat_model=chat_model,
    )
    return answer, retrieved
