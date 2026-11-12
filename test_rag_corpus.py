import json
from pathlib import Path
from rag_engine import load_corpus, chunk_documents, corpus_fingerprint

ROOT = Path(__file__).resolve().parent
docs = load_corpus()
chunks = chunk_documents(docs)
assert len(docs) >= 8, "Expected at least 8 literature/project documents."
assert len(chunks) >= len(docs), "Every source should create at least one retrieval chunk."
assert len({d["doc_id"] for d in docs}) == len(docs), "Document IDs must be unique."
assert any(d["doc_type"] == "internal_project_note" for d in docs)
assert all(d.get("url") or d["doc_type"] == "internal_project_note" for d in docs)
print(f"Corpus validation passed: {len(docs)} documents, {len(chunks)} chunks.")
print("Fingerprint:", corpus_fingerprint(docs))
