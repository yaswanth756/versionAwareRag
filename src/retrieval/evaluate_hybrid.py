"""
src/retrieval/evaluate_hybrid.py
--------------------------------
Executes the Hybrid Retrieval benchmark (BM25 + Dense BGE-small fusion)
on the frozen development dataset (data/evaluation/dev.reviewed.jsonl)
against all 9,800 Python 3.12 documentation chunks.

Evaluates:
  1. Reciprocal Rank Fusion (RRF) across various k_rrf and candidate pool sizes.
  2. Convex score normalization (Linear Combination) across various alpha weights.
  3. Full 3-way comparative analysis: BM25 vs Dense vs Hybrid.

Generates:
  - data/evaluation/hybrid_dev_benchmark_report.json
  - data/evaluation/hybrid_dev_predictions.jsonl
"""

import json
import time
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
    convex_score_fusion,
)
from src.retrieval.metrics import evaluate_query, aggregate_metrics

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "chunks.jsonl"
DEV_PATH = PROJECT_ROOT / "data" / "evaluation" / "dev.reviewed.jsonl"
BM25_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "bm25_index.pkl.gz"
DENSE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "dense_bge_index"

BM25_REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bm25_dev_benchmark_report.json"
DENSE_REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "dense_dev_benchmark_report.json"

REPORT_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "hybrid_dev_benchmark_report.json"
PREDICTIONS_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "hybrid_dev_predictions.jsonl"


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


def evaluate_candidate_dict(
    queries: List[Dict[str, Any]],
    batch_predictions: Dict[str, List[Dict[str, Any]]],
    chunk_to_doc: Dict[str, str],
    k_values: List[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    per_query_results = []
    category_evals = defaultdict(list)

    for q in queries:
        qid = q["query_id"]
        candidates = batch_predictions.get(qid, [])
        retrieved_cids = [c["chunk_id"] for c in candidates]

        eval_metrics = evaluate_query(
            query_record=q,
            retrieved_cids=retrieved_cids,
            k_values=k_values,
            chunk_to_doc_id=chunk_to_doc,
        )
        per_query_results.append(eval_metrics)
        category_evals[q["category"]].append(eval_metrics)

    overall = aggregate_metrics(per_query_results)
    return overall, per_query_results, category_evals


def run_hybrid_benchmark():
    print("=" * 80)
    print("HYBRID RETRIEVAL BENCHMARK: BM25 + DENSE BGE-SMALL-EN-V1.5 FUSION")
    print("=" * 80)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} corpus chunks from {CHUNKS_PATH}")
    chunk_to_doc = {c["chunk_id"]: c.get("doc_id", c.get("relative_path", "")) for c in chunks}

    queries = load_dev_queries()
    print(f"Loaded {len(queries)} evaluation queries from {DEV_PATH}")

    # 1. Load single retrievers
    print("Loading BM25 index...")
    bm25_retriever = BM25Retriever.load(BM25_INDEX_PATH)
    print("Loading Dense index...")
    dense_retriever = DenseRetriever.load(DENSE_INDEX_PATH)
    hybrid_retriever = HybridRetriever(bm25_retriever, dense_retriever)

    # Pre-fetch candidate pools at max candidate pool size = 100 for fast grid exploration
    print("\nPre-fetching candidates (pool_size=100) from BM25 and Dense...")
    t0 = time.time()
    bm25_pools_100 = bm25_retriever.batch_retrieve(queries, top_k=100)
    dense_pools_100 = dense_retriever.batch_retrieve(queries, top_k=100)
    print(f"Pre-fetched pools for {len(queries)} queries in {time.time() - t0:.2f}s")

    # 2. Grid Search over parameters
    print("\nRunning Parameter Grid Search...")
    grid_configs = [
        # (method, pool_size, k_rrf, w_bm25, w_dense, alpha, label)
        ("rrf", 20, 20, 1.0, 1.0, 0.5, "RRF (pool=20, k=20)"),
        ("rrf", 20, 60, 1.0, 1.0, 0.5, "RRF (pool=20, k=60)"),
        ("rrf", 50, 20, 1.0, 1.0, 0.5, "RRF (pool=50, k=20)"),
        ("rrf", 50, 40, 1.0, 1.0, 0.5, "RRF (pool=50, k=40)"),
        ("rrf", 50, 60, 1.0, 1.0, 0.5, "RRF (pool=50, k=60)"),
        ("rrf", 50, 100, 1.0, 1.0, 0.5, "RRF (pool=50, k=100)"),
        ("rrf", 100, 20, 1.0, 1.0, 0.5, "RRF (pool=100, k=20)"),
        ("rrf", 100, 40, 1.0, 1.0, 0.5, "RRF (pool=100, k=40)"),
        ("rrf", 100, 60, 1.0, 1.0, 0.5, "RRF (pool=100, k=60)"),
        ("rrf", 50, 60, 0.7, 1.0, 0.5, "Weighted RRF (w_bm25=0.7, w_dense=1.0)"),
        ("rrf", 50, 60, 1.0, 0.7, 0.5, "Weighted RRF (w_bm25=1.0, w_dense=0.7)"),
        ("linear", 50, 60, 1.0, 1.0, 0.3, "Linear Normalization (alpha=0.3)"),
        ("linear", 50, 60, 1.0, 1.0, 0.4, "Linear Normalization (alpha=0.4)"),
        ("linear", 50, 60, 1.0, 1.0, 0.5, "Linear Normalization (alpha=0.5)"),
        ("linear", 50, 60, 1.0, 1.0, 0.6, "Linear Normalization (alpha=0.6)"),
        ("linear", 50, 60, 1.0, 1.0, 0.7, "Linear Normalization (alpha=0.7)"),
    ]

    grid_results = []
    best_config = None
    best_score = -1.0
    best_predictions = None
    best_per_query = None
    best_cat_evals = None

    for method, pool_size, k_rrf, w_bm25, w_dense, alpha, label in grid_configs:
        batch_preds = {}
        for q in queries:
            qid = q["query_id"]
            b_cand = bm25_pools_100[qid][:pool_size]
            d_cand = dense_pools_100[qid][:pool_size]

            if method == "rrf":
                ranked_lists = [(b_cand, w_bm25), (d_cand, w_dense)]
                raw_fused = reciprocal_rank_fusion(ranked_lists, k_rrf=k_rrf, top_k=20)
                formatted = []
                for item in raw_fused:
                    m_ranks = item.get("method_ranks", {})
                    m_scores = item.get("method_scores", {})
                    formatted.append({
                        "rank": item["rank"],
                        "chunk_id": item["chunk_id"],
                        "score": item["score"],
                        "doc_path": item["doc_path"],
                        "doc_title": item["doc_title"],
                        "breadcrumbs": item["breadcrumbs"],
                        "preview": item["preview"],
                        "bm25_rank": m_ranks.get(0),
                        "dense_rank": m_ranks.get(1),
                        "bm25_score": m_scores.get(0),
                        "dense_score": m_scores.get(1),
                    })
                batch_preds[qid] = formatted
            else:
                batch_preds[qid] = convex_score_fusion(
                    bm25_candidates=b_cand,
                    dense_candidates=d_cand,
                    alpha=alpha,
                    top_k=20,
                )

        overall, per_q, cat_evals = evaluate_candidate_dict(queries, batch_preds, chunk_to_doc)
        grid_entry = {
            "label": label,
            "method": method,
            "pool_size": pool_size,
            "k_rrf": k_rrf,
            "weight_bm25": w_bm25,
            "weight_dense": w_dense,
            "alpha": alpha,
            "hit@1": overall.get("hit@1", 0.0),
            "hit@5": overall.get("hit@5", 0.0),
            "hit@10": overall.get("hit@10", 0.0),
            "hit@20": overall.get("hit@20", 0.0),
            "mrr": overall.get("mrr", 0.0),
            "map_at_20": overall.get("map_at_20", 0.0),
            "ndcg@10": overall.get("ndcg@10", 0.0),
        }
        grid_results.append(grid_entry)

        # Composite selection metric: emphasize Hit@10 and nDCG@10
        composite_score = (
            overall.get("hit@10", 0.0) * 0.4
            + overall.get("ndcg@10", 0.0) * 0.4
            + overall.get("hit@20", 0.0) * 0.2
        )
        if composite_score > best_score:
            best_score = composite_score
            best_config = grid_entry
            best_predictions = batch_preds
            best_per_query = per_q
            best_cat_evals = cat_evals

    print("\n" + "=" * 98)
    print("HYBRID FUSION PARAMETER GRID SEARCH RESULTS")
    print("=" * 98)
    print(f"{'Configuration':<42} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'Hit@20':>7} {'MRR':>7} {'MAP@20':>7} {'nDCG@10':>8}")
    print("-" * 98)
    for res in grid_results:
        star = " *" if res["label"] == best_config["label"] else ""
        print(f"{res['label'] + star:<42} {res['hit@1']:>7.4f} {res['hit@5']:>7.4f} {res['hit@10']:>7.4f} {res['hit@20']:>7.4f} {res['mrr']:>7.4f} {res['map_at_20']:>7.4f} {res['ndcg@10']:>8.4f}")
    print("=" * 98)
    print(f"* Best performing configuration: {best_config['label']}")

    # 3. Compile best overall and category metrics
    best_overall = aggregate_metrics(best_per_query)
    category_summary = {}
    for cat, evals in sorted(best_cat_evals.items()):
        category_summary[cat] = {
            "query_count": len(evals),
            "metrics": aggregate_metrics(evals),
        }

    # 4. Load baseline reports for 3-way comparison
    bm25_report = {}
    dense_report = {}
    if BM25_REPORT_PATH.exists():
        with open(BM25_REPORT_PATH, "r", encoding="utf-8") as f:
            bm25_report = json.load(f)
    if DENSE_REPORT_PATH.exists():
        with open(DENSE_REPORT_PATH, "r", encoding="utf-8") as f:
            dense_report = json.load(f)

    bm25_overall = bm25_report.get("overall_metrics", {})
    dense_overall = dense_report.get("overall_metrics", {})
    bm25_cats = bm25_report.get("category_metrics", {})
    dense_cats = dense_report.get("category_metrics", {})

    three_way_comparison = {}
    comparison_keys = [
        "hit@1", "hit@3", "hit@5", "hit@10", "hit@20",
        "recall@5", "recall@10", "mrr", "map_at_20", "ndcg@10",
        "doc_hit@1", "doc_hit@5", "doc_hit@10"
    ]
    for k in comparison_keys:
        h_val = best_overall.get(k, 0.0)
        b_val = bm25_overall.get(k, 0.0)
        d_val = dense_overall.get(k, 0.0)
        delta_bm25 = round(h_val - b_val, 4)
        delta_dense = round(h_val - d_val, 4)
        three_way_comparison[k] = {
            "bm25": b_val,
            "dense": d_val,
            "hybrid": h_val,
            "delta_vs_bm25": delta_bm25,
            "delta_vs_dense": delta_dense,
            "rel_gain_vs_bm25_pct": round((delta_bm25 / b_val * 100) if b_val > 0 else 0.0, 2),
            "rel_gain_vs_dense_pct": round((delta_dense / d_val * 100) if d_val > 0 else 0.0, 2),
        }

    category_three_way = {}
    for cat in category_summary.keys():
        h_m = category_summary[cat]["metrics"]
        b_m = bm25_cats.get(cat, {}).get("metrics", {})
        d_m = dense_cats.get(cat, {}).get("metrics", {})
        category_three_way[cat] = {
            "query_count": category_summary[cat]["query_count"],
            "bm25_hit@10": b_m.get("hit@10", 0.0),
            "dense_hit@10": d_m.get("hit@10", 0.0),
            "hybrid_hit@10": h_m.get("hit@10", 0.0),
            "bm25_mrr": b_m.get("mrr", 0.0),
            "dense_mrr": d_m.get("mrr", 0.0),
            "hybrid_mrr": h_m.get("mrr", 0.0),
            "bm25_ndcg@10": b_m.get("ndcg@10", 0.0),
            "dense_ndcg@10": d_m.get("ndcg@10", 0.0),
            "hybrid_ndcg@10": h_m.get("ndcg@10", 0.0),
        }

    # 5. Format and save predictions
    predictions_records = []
    for q in queries:
        qid = q["query_id"]
        cat = q["category"]
        question = q["question"]
        target_cids = set(q.get("relevant_chunk_ids", []))
        cands = best_predictions.get(qid, [])
        predictions_records.append({
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
                    "bm25_rank": c.get("bm25_rank"),
                    "dense_rank": c.get("dense_rank"),
                    "bm25_score": c.get("bm25_score"),
                    "dense_score": c.get("dense_score"),
                    "doc_path": c.get("doc_path", ""),
                    "preview": c.get("preview", ""),
                }
                for c in cands
            ],
        })

    print(f"\nWriting hybrid predictions to {PREDICTIONS_OUT_PATH}...")
    with open(PREDICTIONS_OUT_PATH, "w", encoding="utf-8") as f:
        for p in predictions_records:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 6. Format and save comprehensive benchmark report
    # Also find best pure RRF configuration for reference
    best_rrf_config = max(
        [r for r in grid_results if r["method"] == "rrf"],
        key=lambda x: x["hit@10"] * 0.4 + x["ndcg@10"] * 0.4 + x["hit@20"] * 0.2
    )

    final_report = {
        "benchmark_metadata": {
            "model_type": "hybrid_bm25_plus_dense",
            "lexical_model": "BM25Okapi (k1=1.5, b=0.75)",
            "dense_model": "BAAI/bge-small-en-v1.5",
            "benchmark_dataset": "dev.reviewed.jsonl (frozen v1.0.0)",
            "total_queries_evaluated": len(queries),
            "corpus_chunks_count": len(chunks),
            "primary_configuration": best_config,
            "best_pure_rrf_configuration": best_rrf_config,
        },
        "grid_search_summary": grid_results,
        "overall_metrics": best_overall,
        "category_metrics": category_summary,
        "three_way_comparison": three_way_comparison,
        "category_three_way_comparison": category_three_way,
    }

    print(f"Writing hybrid benchmark report to {REPORT_OUT_PATH}...")
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    # 7. Print 3-Way Head-to-Head Comparison Table
    print("\n" + "=" * 94)
    print("3-WAY HEAD-TO-HEAD RETRIEVAL BENCHMARK: BM25 vs DENSE vs HYBRID")
    print("=" * 94)
    print(f"{'Metric':<14} {'BM25':>10} {'Dense':>10} {'Hybrid':>10} {'Gain vs BM25':>17} {'Gain vs Dense':>17}")
    print("-" * 94)
    for m in comparison_keys:
        row = three_way_comparison.get(m, {})
        b_val = row.get("bm25", 0.0)
        d_val = row.get("dense", 0.0)
        h_val = row.get("hybrid", 0.0)
        gain_b = f"{row.get('delta_vs_bm25', 0.0):+0.4f} ({row.get('rel_gain_vs_bm25_pct', 0.0):+5.1f}%)"
        gain_d = f"{row.get('delta_vs_dense', 0.0):+0.4f} ({row.get('rel_gain_vs_dense_pct', 0.0):+5.1f}%)"
        print(f"{m:<14} {b_val:>10.4f} {d_val:>10.4f} {h_val:>10.4f} {gain_b:>17} {gain_d:>17}")
    print("=" * 94)

    # 8. Print Category Breakdown Table
    print("\n" + "=" * 90)
    print("CATEGORY BREAKDOWN: Hit@10 & MRR ACROSS RETRIEVAL SYSTEMS")
    print("=" * 90)
    print(f"{'Category':<24} {'Count':<6} {'BM25 H@10':<11} {'Dense H@10':<11} {'Hyb H@10':<11} {'BM25 MRR':<10} {'Dense MRR':<10} {'Hyb MRR':<10}")
    print("-" * 90)
    for cat, data in category_three_way.items():
        cnt = data["query_count"]
        bh10 = f"{data['bm25_hit@10']*100:.1f}%"
        dh10 = f"{data['dense_hit@10']*100:.1f}%"
        hh10 = f"{data['hybrid_hit@10']*100:.1f}%"
        bmrr = f"{data['bm25_mrr']:.4f}"
        dmrr = f"{data['dense_mrr']:.4f}"
        hmrr = f"{data['hybrid_mrr']:.4f}"
        print(f"{cat:<24} {cnt:<6} {bh10:<11} {dh10:<11} {hh10:<11} {bmrr:<10} {dmrr:<10} {hmrr:<10}")
    print("=" * 90)

    return final_report


if __name__ == "__main__":
    run_hybrid_benchmark()
