"""
src/retrieval/evaluate_bm25.py
------------------------------
Executes the BM25 retrieval benchmark on the frozen development dataset
(data/evaluation/dev.reviewed.jsonl) against the 9,800 Python 3.12 chunks.

Generates:
  1. data/evaluation/bm25_dev_benchmark_report.json (Overall and category-level metrics)
  2. data/evaluation/bm25_dev_predictions.jsonl (Top-20 retrieved candidates per query)
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

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.metrics import evaluate_query, aggregate_metrics
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "chunks.jsonl"
DEV_PATH = PROJECT_ROOT / "data" / "evaluation" / "dev.reviewed.jsonl"
INDEX_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "bm25_index.pkl.gz"

REPORT_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bm25_dev_benchmark_report.json"
PREDICTIONS_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bm25_dev_predictions.jsonl"


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


def run_benchmark(top_k: int = 20, use_cache: bool = True) -> Dict[str, Any]:
    print("=" * 70)
    print("Starting BM25 Sparse Retrieval Baseline Benchmark")
    print("=" * 70)

    # 1. Load or build index
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} corpus chunks from {CHUNKS_PATH}")

    retriever: BM25Retriever
    if use_cache and INDEX_CACHE_PATH.exists():
        print(f"Loading cached BM25 index from {INDEX_CACHE_PATH}...")
        t0 = time.time()
        retriever = BM25Retriever.load(INDEX_CACHE_PATH)
        print(f"Loaded index in {time.time() - t0:.2f}s")
    else:
        print("Building new BM25 index over chunks...")
        t0 = time.time()
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index_corpus(chunks, verbose=True)
        print(f"Saving index cache to {INDEX_CACHE_PATH}...")
        retriever.save(INDEX_CACHE_PATH)
        print(f"Index built and cached in {time.time() - t0:.2f}s")

    # Mapping chunk_id -> doc_id for document-level metrics
    chunk_to_doc = {c["chunk_id"]: c.get("doc_id", c.get("relative_path", "")) for c in chunks}

    # 2. Load evaluation dataset
    queries = load_dev_queries()
    print(f"Loaded {len(queries)} evaluation queries from {DEV_PATH}")

    # 3. Retrieve and evaluate
    t_start = time.time()
    per_query_results = []
    predictions = []
    category_evals = defaultdict(list)

    for q in queries:
        qid = q["query_id"]
        question = q["question"]
        cat = q["category"]
        target_cids = set(q.get("relevant_chunk_ids", []))

        # Retrieve top candidates
        candidates = retriever.retrieve(question, top_k=top_k)
        retrieved_cids = [c["chunk_id"] for c in candidates]

        # Evaluate query metrics
        eval_metrics = evaluate_query(
            query_record=q,
            retrieved_cids=retrieved_cids,
            k_values=[1, 3, 5, 10, 20],
            chunk_to_doc_id=chunk_to_doc
        )
        per_query_results.append(eval_metrics)
        category_evals[cat].append(eval_metrics)

        # Store predictions
        prediction_record = {
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
        }
        predictions.append(prediction_record)

    t_total = time.time() - t_start
    qps = len(queries) / t_total if t_total > 0 else 0

    # 4. Aggregate metrics
    overall_metrics = aggregate_metrics(per_query_results)
    category_summary = {}
    for cat, evals in sorted(category_evals.items()):
        category_summary[cat] = {
            "query_count": len(evals),
            "metrics": aggregate_metrics(evals)
        }

    # Identify failure queries (Hit@10 == 0)
    hard_queries = [
        {
            "query_id": qe["query_id"],
            "category": qe["category"],
            "mrr": qe["mrr"],
            "top_retrieved": next(p["retrieved_candidates"][:3] for p in predictions if p["query_id"] == qe["query_id"])
        }
        for qe in per_query_results if qe.get("hit@10", 0) == 0
    ]

    report = {
        "benchmark_metadata": {
            "model_type": "sparse_lexical",
            "retriever": "BM25Okapi",
            "k1": 1.5,
            "b": 0.75,
            "tokenization": "code_and_text_aware (dotted expressions, snake_case subwords, alphanumeric)",
            "corpus_chunks_count": len(chunks),
            "benchmark_dataset": "dev.reviewed.jsonl (frozen v1.0.0)",
            "total_queries_evaluated": len(queries),
            "evaluation_duration_seconds": round(t_total, 3),
            "queries_per_second": round(qps, 1)
        },
        "overall_metrics": overall_metrics,
        "category_metrics": category_summary,
        "zero_hit_at_10_count": len(hard_queries),
        "zero_hit_at_10_queries": hard_queries
    }

    # 5. Write reports
    print(f"\nWriting predictions to {PREDICTIONS_OUT_PATH}...")
    with open(PREDICTIONS_OUT_PATH, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Writing summary report to {REPORT_OUT_PATH}...")
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 6. Display formatted results
    print("\n" + "=" * 75)
    print("BM25 RETRIEVAL BASELINE PERFORMANCE SUMMARY")
    print("=" * 75)
    print(f"Queries Evaluated: {len(queries)} | Corpus Chunks: {len(chunks)} | Latency: {t_total*1000/len(queries):.1f} ms/query")
    print("-" * 75)
    print(f"{'Metric':<20} {'Score':>10}")
    print("-" * 35)
    key_metrics = [
        ("Hit@1", overall_metrics.get("hit@1", 0)),
        ("Hit@3", overall_metrics.get("hit@3", 0)),
        ("Hit@5", overall_metrics.get("hit@5", 0)),
        ("Hit@10", overall_metrics.get("hit@10", 0)),
        ("Hit@20", overall_metrics.get("hit@20", 0)),
        ("Recall@1", overall_metrics.get("recall@1", 0)),
        ("Recall@5", overall_metrics.get("recall@5", 0)),
        ("Recall@10", overall_metrics.get("recall@10", 0)),
        ("Recall@20", overall_metrics.get("recall@20", 0)),
        ("Precision@1", overall_metrics.get("precision@1", 0)),
        ("Precision@5", overall_metrics.get("precision@5", 0)),
        ("Precision@10", overall_metrics.get("precision@10", 0)),
        ("MRR", overall_metrics.get("mrr", 0)),
        ("MAP@20", overall_metrics.get("map_at_20", 0)),
        ("nDCG@5", overall_metrics.get("ndcg@5", 0)),
        ("nDCG@10", overall_metrics.get("ndcg@10", 0)),
        ("nDCG@20", overall_metrics.get("ndcg@20", 0)),
        ("Doc_Hit@1", overall_metrics.get("doc_hit@1", 0)),
        ("Doc_Hit@5", overall_metrics.get("doc_hit@5", 0)),
    ]
    for name, val in key_metrics:
        pct = f"({val*100:.1f}%)" if "hit" in name.lower() or "recall" in name.lower() or "precision" in name.lower() else ""
        print(f"{name:<20} {val:>10.4f}  {pct}")

    print("\n" + "=" * 75)
    print(f"{'Category':<24} {'Count':<6} {'Hit@1':<10} {'Hit@5':<10} {'Hit@10':<10} {'MRR':<8} {'nDCG@10':<8}")
    print("-" * 75)
    for cat, data in category_summary.items():
        cnt = data["query_count"]
        m = data["metrics"]
        h1 = f"{m.get('hit@1', 0)*100:.1f}%"
        h5 = f"{m.get('hit@5', 0)*100:.1f}%"
        h10 = f"{m.get('hit@10', 0)*100:.1f}%"
        mrr = f"{m.get('mrr', 0):.4f}"
        ndcg = f"{m.get('ndcg@10', 0):.4f}"
        print(f"{cat:<24} {cnt:<6} {h1:<10} {h5:<10} {h10:<10} {mrr:<8} {ndcg:<8}")
    print("=" * 75)

    return report


if __name__ == "__main__":
    run_benchmark()
