"""
src/retrieval/evaluate_dense.py
-------------------------------
Executes the Dense Vector Retrieval benchmark using BAAI/bge-small-en-v1.5
on the frozen development dataset (data/evaluation/dev.reviewed.jsonl)
against all 9,800 Python 3.12 documentation chunks.

Generates:
  1. data/evaluation/dense_dev_benchmark_report.json (Overall and category metrics + BM25 comparison)
  2. data/evaluation/dense_dev_predictions.jsonl (Top-20 retrieved candidates per query)
"""

import json
import time
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.metrics import evaluate_query, aggregate_metrics

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "chunks.jsonl"
DEV_PATH = PROJECT_ROOT / "data" / "evaluation" / "dev.reviewed.jsonl"
DENSE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "dense_bge_index"
BM25_REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bm25_dev_benchmark_report.json"

REPORT_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "dense_dev_benchmark_report.json"
PREDICTIONS_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "dense_dev_predictions.jsonl"


def load_chunks() -> List[Dict[str, Any]]:
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_dev_queries() -> List[Dict[str, Any]]:
    queries = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def run_dense_benchmark(
    model_name: str = "BAAI/bge-small-en-v1.5",
    top_k: int = 20,
    use_cache: bool = True
) -> Dict[str, Any]:
    print("=" * 75)
    print(f"Starting Dense Retrieval Benchmark ({model_name})")
    print("=" * 75)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} corpus chunks from {CHUNKS_PATH}")

    retriever: DenseRetriever
    meta_cache = DENSE_INDEX_PATH.with_suffix(".meta.pkl.gz")
    embed_cache = DENSE_INDEX_PATH.with_suffix(".embeddings.npz")

    if use_cache and meta_cache.exists() and embed_cache.exists():
        print(f"Loading cached dense index from {DENSE_INDEX_PATH}...")
        t0 = time.time()
        retriever = DenseRetriever.load(DENSE_INDEX_PATH)
        print(f"Dense index loaded in {time.time() - t0:.2f}s (Embeddings shape: {retriever.embeddings.shape})")
    else:
        print("Generating dense embeddings for all chunks...")
        t0 = time.time()
        retriever = DenseRetriever(model_name=model_name, batch_size=64)
        retriever.index_corpus(chunks, verbose=True)
        print(f"Saving pre-computed dense index to {DENSE_INDEX_PATH}...")
        retriever.save(DENSE_INDEX_PATH)
        print(f"Dense index built and cached in {time.time() - t0:.2f}s")

    chunk_to_doc = {c["chunk_id"]: c.get("doc_id", c.get("relative_path", "")) for c in chunks}

    queries = load_dev_queries()
    print(f"Loaded {len(queries)} evaluation queries from {DEV_PATH}")

    # Run batch retrieval
    t_start = time.time()
    batch_predictions = retriever.batch_retrieve(queries, top_k=top_k)
    t_total = time.time() - t_start

    per_query_results = []
    predictions = []
    category_evals = defaultdict(list)

    for q in queries:
        qid = q["query_id"]
        cat = q["category"]
        question = q["question"]
        target_cids = set(q.get("relevant_chunk_ids", []))
        candidates = batch_predictions.get(qid, [])
        retrieved_cids = [c["chunk_id"] for c in candidates]

        eval_metrics = evaluate_query(
            query_record=q,
            retrieved_cids=retrieved_cids,
            k_values=[1, 3, 5, 10, 20],
            chunk_to_doc_id=chunk_to_doc
        )
        per_query_results.append(eval_metrics)
        category_evals[cat].append(eval_metrics)

        predictions.append({
            "query_id": qid,
            "category": cat,
            "question": question,
            "target_chunk_ids": list(target_cids),
            "retrieved_candidates": [
                {
                    "rank": c["rank"],
                    "chunk_id": c["chunk_id"],
                    "score": c["score"],
                    "is_hit": c["chunk_id"] in target_cids,
                    "doc_path": c["doc_path"],
                    "preview": c["preview"]
                }
                for c in candidates
            ]
        })

    qps = len(queries) / t_total if t_total > 0 else 0
    overall_metrics = aggregate_metrics(per_query_results)

    category_summary = {}
    for cat, evals in sorted(category_evals.items()):
        category_summary[cat] = {
            "query_count": len(evals),
            "metrics": aggregate_metrics(evals)
        }

    # Load BM25 baseline for comparison if available
    bm25_metrics = None
    if BM25_REPORT_PATH.exists():
        with open(BM25_REPORT_PATH, "r", encoding="utf-8") as f:
            bm25_data = json.load(f)
            bm25_metrics = bm25_data.get("overall_metrics", {})

    report = {
        "benchmark_metadata": {
            "model_type": "dense_vector",
            "model_name": model_name,
            "device": retriever.device,
            "embedding_dimension": int(retriever.embeddings.shape[1]),
            "corpus_chunks_count": len(chunks),
            "benchmark_dataset": "dev.reviewed.jsonl (frozen v1.0.0)",
            "total_queries_evaluated": len(queries),
            "evaluation_duration_seconds": round(t_total, 3),
            "queries_per_second": round(qps, 1)
        },
        "overall_metrics": overall_metrics,
        "category_metrics": category_summary,
        "bm25_baseline_comparison": {
            k: {
                "dense": overall_metrics.get(k, 0.0),
                "bm25": bm25_metrics.get(k, 0.0) if bm25_metrics else None,
                "delta": round(overall_metrics.get(k, 0.0) - bm25_metrics.get(k, 0.0), 4) if bm25_metrics and k in bm25_metrics else None
            }
            for k in ["hit@1", "hit@5", "hit@10", "hit@20", "recall@5", "recall@10", "mrr", "ndcg@10"]
        } if bm25_metrics else {}
    }

    print(f"\nWriting dense predictions to {PREDICTIONS_OUT_PATH}...")
    with open(PREDICTIONS_OUT_PATH, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Writing dense benchmark report to {REPORT_OUT_PATH}...")
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Display comparison table
    print("\n" + "=" * 78)
    print("DENSE RETRIEVAL (BGE-small-en-v1.5) vs BM25 BASELINE COMPARISON")
    print("=" * 78)
    print(f"{'Metric':<18} {'BM25':>10} {'Dense':>10} {'Delta':>10} {'Rel Change':>12}")
    print("-" * 78)
    for m_key in ["hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "recall@5", "recall@10", "mrr", "ndcg@10", "doc_hit@5"]:
        dense_val = overall_metrics.get(m_key, 0.0)
        bm25_val = bm25_metrics.get(m_key, 0.0) if bm25_metrics else 0.0
        delta = dense_val - bm25_val
        rel_change = (delta / bm25_val * 100) if bm25_val > 0 else 0.0
        print(f"{m_key:<18} {bm25_val:>10.4f} {dense_val:>10.4f} {delta:>+10.4f} {rel_change:>+11.1f}%")

    print("\n" + "=" * 78)
    print("CATEGORY BREAKDOWN: DENSE vs BM25 (MRR & Hit@10)")
    print("=" * 78)
    print(f"{'Category':<24} {'Count':<6} {'BM25 MRR':<10} {'Dense MRR':<10} {'BM25 H@10':<11} {'Dense H@10':<11}")
    print("-" * 78)
    bm25_cat_metrics = bm25_data.get("category_metrics", {}) if bm25_metrics else {}
    for cat, data in category_summary.items():
        cnt = data["query_count"]
        d_m = data["metrics"]
        b_m = bm25_cat_metrics.get(cat, {}).get("metrics", {})
        d_mrr = f"{d_m.get('mrr', 0):.4f}"
        b_mrr = f"{b_m.get('mrr', 0):.4f}" if b_m else "N/A"
        d_h10 = f"{d_m.get('hit@10', 0)*100:.1f}%"
        b_h10 = f"{b_m.get('hit@10', 0)*100:.1f}%" if b_m else "N/A"
        print(f"{cat:<24} {cnt:<6} {b_mrr:<10} {d_mrr:<10} {b_h10:<11} {d_h10:<11}")
    print("=" * 78)

    return report


if __name__ == "__main__":
    run_dense_benchmark()

