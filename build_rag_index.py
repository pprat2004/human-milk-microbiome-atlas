"""Build the Gemini vector index for the literature library from the command line."""
import argparse
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_engine import DEFAULT_EMBEDDING_MODEL, build_index


def main():
    parser = argparse.ArgumentParser(description="Build the human-milk literature RAG vector index (Gemini).")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    manifest = build_index(embedding_model=args.embedding_model)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
