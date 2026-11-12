import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from pathlib import Path
from rag_engine import retrieve

ROOT = Path(__file__).resolve().parent
EVAL_FILE = ROOT / "rag" / "retrieval_eval_questions.json"

def main():
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    results = []
    for item in questions:
        hits = retrieve(item["question"], top_k=5)
        ranked_ids = [hit["doc_id"] for hit in hits]
        relevant = set(item["relevant_doc_ids"])
        first_relevant_rank = next((i + 1 for i, doc_id in enumerate(ranked_ids) if doc_id in relevant), None)
        results.append({
            "question": item["question"],
            "relevant_doc_ids": sorted(relevant),
            "retrieved_doc_ids": ranked_ids,
            "hit_at_3": any(doc_id in relevant for doc_id in ranked_ids[:3]),
            "reciprocal_rank": 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        })
    hit3 = sum(r["hit_at_3"] for r in results) / len(results)
    mrr = sum(r["reciprocal_rank"] for r in results) / len(results)
    output = {"n_questions": len(results), "hit_at_3": hit3, "mrr_at_5": mrr, "details": results}
    out_path = ROOT / "rag" / "retrieval_evaluation_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
