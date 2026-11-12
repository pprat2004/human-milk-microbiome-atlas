# Research assistant: implementation status

## Included
- 8 curated peer-reviewed literature summaries plus 1 internal MAMI/PRIME project-results note.
- Stable document IDs, bibliographic metadata, topic tags and source URLs.
- Document chunking with overlap.
- Gemini embeddings (`gemini-embedding-2`, 768 dimensions) with local cosine-similarity search in NumPy.
- Evidence-grounded answers from Gemini (`gemini-3.5-flash` by default) through the Gemini API.
- Prompt rules for citation, source-type separation, uncertainty, non-causal interpretation, and keeping
  human-milk evidence apart from infant-stool evidence.
- Retrieval evaluation set with Hit@3 and MRR@5 script, and corpus tests that need no API call.

## Needs your Gemini API key
- Building the index and asking questions make live Gemini calls.
- Run `python evaluate_rag_retrieval.py` after indexing to measure retrieval quality.
