"""
src/retrieval/evaluate_candidate_pool_expansion.py
-------------------------------------------------
Executes Stage 5C: Candidate Pool Expansion Experiment.
Evaluates the impact of candidate pool size:
    K in [20, 30, 40, 50]
while holding the optimal blending weight fixed at beta = 0.6:
    S_final = 0.6 * norm(S_ce) + 0.4 * norm(S_hybrid)

Evaluates:
  - Hybrid Top-20 -> BGE reranking
  - Hybrid Top-30 -> BGE reranking
  - Hybrid Top-40 -> BGE reranking
  - Hybrid Top-50 -> BGE reranking

Generates:
  - data/evaluation/bge_candidate_pool_expansion_report.json
  - data/evaluation/bge_candidate_pool_expansion_summary.json
  - data/evaluation/bge_candidate_pool_expansion_predictions.jsonl
  - data/evaluation/hybrid_top50_dev_predictions.jsonl
"""

import json
import time
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import convex_score_fusion, minmax_normalize
from src.retrieval.metrics import evaluate_query, aggregate_metrics

# File paths
DEV_PATH = PROJECT_ROOT / "data" / "evaluation" / "dev.reviewed.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "chunks.jsonl"
BM25_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "bm25_index.pkl.gz"
DENSE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "dense_bge_index"

PRIOR_RERANK_PREDS_PATH = PROJECT_ROOT / "data" / "evaluation" / "reranker_dev_predictions.jsonl"
HYBRID_TOP50_PREDS_PATH = PROJECT_ROOT / "data" / "evaluation" / "hybrid_top50_dev_predictions.jsonl"
SCORED_PAIRS_CACHE_PATH = PROJECT_ROOT / "data" / "evaluation" / "bge_top50_scored_pairs.json"

REPORT_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bge_candidate_pool_expansion_report.json"
SUMMARY_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bge_candidate_pool_expansion_summary.json"
PREDICTIONS_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bge_candidate_pool_expansion_predictions.jsonl"

REMOTE_SERVER_URL = "https://tropical-module-feet-continuity.trycloudflare.com"


def load_dev_queries() -> List[Dict[str, Any]]:
    queries = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_chunks() -> Dict[str, Dict[str, Any]]:
    chunks = {}
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    return chunks


def generate_hybrid_top50(queries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Generates Hybrid top-50 candidate pools across all dev queries."""
    if HYBRID_TOP50_PREDS_PATH.exists():
        print(f"Loading cached Hybrid top-50 candidate pools from {HYBRID_TOP50_PREDS_PATH}...")
        pools = {}
        with open(HYBRID_TOP50_PREDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pools[rec["query_id"]] = rec["retrieved_candidates"]
        if len(pools) == len(queries) and all(len(c) == 50 for c in pools.values()):
            return pools

    print("Generating Hybrid top-50 candidate pools (BM25 + Dense alpha=0.7)...")
    t0 = time.time()
    bm25 = BM25Retriever.load(BM25_INDEX_PATH)
    dense = DenseRetriever.load(DENSE_INDEX_PATH)

    bm25_pools = bm25.batch_retrieve(queries, top_k=50)
    dense_pools = dense.batch_retrieve(queries, top_k=50)

    hybrid_pools = {}
    records = []
    for q in queries:
        qid = q["query_id"]
        cands = convex_score_fusion(bm25_pools[qid], dense_pools[qid], alpha=0.7, top_k=50)
        hybrid_pools[qid] = cands
        records.append({
            "query_id": qid,
            "category": q["category"],
            "question": q["question"],
            "retrieved_candidates": cands,
        })

    with open(HYBRID_TOP50_PREDS_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated and saved Hybrid top-50 pools in {time.time() - t0:.2f}s.")
    return hybrid_pools


def fetch_bge_scores(
    queries: List[Dict[str, Any]],
    hybrid_pools: Dict[str, List[Dict[str, Any]]],
    chunk_lookup: Dict[str, Dict[str, Any]],
    remote_url: str = REMOTE_SERVER_URL,
) -> Dict[str, Dict[str, float]]:
    """
    Loads pre-scored pairs and queries remote Kaggle GPU server for any missing candidate chunks.
    Returns: Mapping of query_id -> {chunk_id: ce_score}.
    """
    scores_cache: Dict[str, Dict[str, float]] = defaultdict(dict)

    # 1. Load from cache file if exists
    if SCORED_PAIRS_CACHE_PATH.exists():
        with open(SCORED_PAIRS_CACHE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for qid, c_scores in raw.items():
                scores_cache[qid].update({cid: float(s) for cid, s in c_scores.items()})

    # 2. Load from Stage 5 reranker predictions (2,800 pairs already scored)
    if PRIOR_RERANK_PREDS_PATH.exists():
        with open(PRIOR_RERANK_PREDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["query_id"]
                for c in rec["retrieved_candidates"]:
                    if "ce_score" in c and c["chunk_id"] not in scores_cache[qid]:
                        scores_cache[qid][c["chunk_id"]] = float(c["ce_score"])

    # 3. Identify missing pairs
    missing_items = []
    total_missing_pairs = 0
    for q in queries:
        qid = q["query_id"]
        q_text = q["question"]
        cands = hybrid_pools.get(qid, [])

        unscored_cands = [c for c in cands if c["chunk_id"] not in scores_cache[qid]]
        if unscored_cands:
            passages = []
            for c in unscored_cands:
                cid = c["chunk_id"]
                text = chunk_lookup.get(cid, {}).get("searchable_text", c.get("preview", ""))
                passages.append(text)
            missing_items.append({
                "query_id": qid,
                "query": q_text,
                "passages": passages,
                "chunk_ids": [c["chunk_id"] for c in unscored_cands],
            })
            total_missing_pairs += len(passages)

    print(f"Total candidate pairs needed: {len(queries) * 50}")
    print(f"Already cached pairs: {sum(len(v) for v in scores_cache.values())}")
    print(f"Pairs to score on Kaggle GPU: {total_missing_pairs}")

    if missing_items:
        batch_chunk_size = 10  # 10 queries per HTTP request (~300-500 pairs)
        print(f"Scoring {total_missing_pairs} pairs via {remote_url} in batches of {batch_chunk_size} queries...")
        t0 = time.time()

        for idx in range(0, len(missing_items), batch_chunk_size):
            chunk = missing_items[idx:idx + batch_chunk_size]
            payload = {
                "items": [
                    {"query_id": it["query_id"], "query": it["query"], "passages": it["passages"]}
                    for it in chunk
                ]
            }

            resp = requests.post(f"{remote_url}/batch_rerank", json=payload, timeout=120)
            resp.raise_for_status()
            batch_resp = resp.json()["results"]

            for it in chunk:
                qid = it["query_id"]
                q_scores = batch_resp[qid]
                for cid, score in zip(it["chunk_ids"], q_scores):
                    scores_cache[qid][cid] = float(score)

            completed_pairs = sum(len(it["passages"]) for it in missing_items[:idx + len(chunk)])
            pct = (completed_pairs / total_missing_pairs) * 100
            print(f"  Progress: {completed_pairs}/{total_missing_pairs} pairs scored ({pct:.1f}%)...")

        print(f"Remote GPU inference completed in {time.time() - t0:.2f}s.")

        # Save cache
        with open(SCORED_PAIRS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(scores_cache, f, indent=2)
        print(f"Saved full score cache to {SCORED_PAIRS_CACHE_PATH}")

    return scores_cache


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


def run_candidate_pool_expansion(k_values: List[int] = None, beta: float = 0.6) -> Dict[str, Any]:
    if k_values is None:
        k_values = [20, 30, 40, 50]

    print("=" * 90)
    print("STAGE 5C: CANDIDATE POOL EXPANSION EXPERIMENT (BGE-RERANKER-LARGE)")
    print(f"Evaluating candidate pool sizes: K in {k_values} with fixed beta = {beta}")
    print("=" * 90)

    queries = load_dev_queries()
    chunk_lookup = load_chunks()
    chunk_to_doc = {c["chunk_id"]: c.get("metadata", {}).get("document_path", "") for c in chunk_lookup.values()}

    # 1. Generate / load Hybrid top-50 candidate pools
    hybrid_pools_50 = generate_hybrid_top50(queries)

    # 2. Fetch BGE-Large cross-encoder scores
    bge_scores = fetch_bge_scores(queries, hybrid_pools_50, chunk_lookup)

    # 3. Evaluate each K
    pool_results = {}
    all_predictions_records = []

    for k in k_values:
        t0 = time.time()
        k_preds = {}

        for q in queries:
            qid = q["query_id"]
            cands_k = hybrid_pools_50[qid][:k]

            q_ce_raw = [bge_scores[qid][c["chunk_id"]] for c in cands_k]
            q_orig_raw = [float(c.get("score", 0.0)) for c in cands_k]

            ce_norm = np.array(minmax_normalize(q_ce_raw), dtype=np.float32)
            orig_norm = np.array(minmax_normalize(q_orig_raw), dtype=np.float32)
            final_scores = (beta * ce_norm + (1.0 - beta) * orig_norm).tolist()

            ranked_indices = sorted(range(len(cands_k)), key=lambda idx: final_scores[idx], reverse=True)

            ranked_cands = []
            for rank, idx in enumerate(ranked_indices, start=1):
                base_c = cands_k[idx]
                ranked_cands.append({
                    "rank": rank,
                    "chunk_id": base_c["chunk_id"],
                    "score": round(float(final_scores[idx]), 6),
                    "ce_score": round(float(q_ce_raw[idx]), 6),
                    "original_rank": base_c.get("rank"),
                    "original_score": base_c.get("score"),
                    "doc_path": base_c.get("doc_path", ""),
                    "preview": base_c.get("preview", ""),
                })
            k_preds[qid] = ranked_cands

        t_blend = time.time() - t0
        eval_k_values = [1, 3, 5, 10, 20]
        if k not in eval_k_values:
            eval_k_values.append(k)
        eval_k_values = sorted(eval_k_values)

        overall, per_q, cat_evals = evaluate_candidate_dict(queries, k_preds, chunk_to_doc, k_values=eval_k_values)

        cat_summary = {}
        for cat, evals in sorted(cat_evals.items()):
            cat_summary[cat] = {
                "query_count": len(evals),
                "metrics": aggregate_metrics(evals),
            }

        pool_results[str(k)] = {
            "candidate_pool_size": k,
            "blend_beta": beta,
            "blending_seconds": round(t_blend, 4),
            "overall_metrics": overall,
            "category_metrics": cat_summary,
        }

        # Build prediction records for this K
        for q in queries:
            qid = q["query_id"]
            cat = q["category"]
            question = q["question"]
            target_cids = set(q.get("relevant_chunk_ids", []))
            ranked = k_preds[qid]

            all_predictions_records.append({
                "candidate_pool_size": k,
                "blend_beta": beta,
                "query_id": qid,
                "category": cat,
                "question": question,
                "target_chunk_ids": list(target_cids),
                "retrieved_candidates": [
                    {
                        "rank": c["rank"],
                        "chunk_id": c["chunk_id"],
                        "score": c["score"],
                        "ce_score": c["ce_score"],
                        "is_hit": c["chunk_id"] in target_cids,
                        "original_rank": c.get("original_rank"),
                        "original_score": c.get("original_score"),
                        "doc_path": c.get("doc_path", ""),
                    }
                    for c in ranked
                ],
            })

    # Historical baselines for comparison
    baselines = {
        "hybrid_top20": {"mrr": 0.6184, "hit@1": 0.4929, "hit@5": 0.7786, "hit@10": 0.8786, "hit@20": 0.9214, "ndcg@10": 0.6627},
        "minilm_top20_beta_0.7": {"mrr": 0.6492, "hit@1": 0.5357, "hit@5": 0.8214, "hit@10": 0.8857, "hit@20": 0.9214, "ndcg@10": 0.6897},
        "bge_top20_beta_0.6": pool_results["20"]["overall_metrics"],
    }

    # Best pool size
    best_mrr_k = int(max(k_values, key=lambda k: pool_results[str(k)]["overall_metrics"]["mrr"]))
    best_h1_k = int(max(k_values, key=lambda k: pool_results[str(k)]["overall_metrics"]["hit@1"]))
    best_ndcg10_k = int(max(k_values, key=lambda k: pool_results[str(k)]["overall_metrics"]["ndcg@10"]))

    summary_data = {
        "experiment_name": "Stage 5C: Candidate Pool Expansion Experiment",
        "model_name": "BAAI/bge-reranker-large",
        "fixed_beta": beta,
        "evaluated_pool_sizes": k_values,
        "best_configurations": {
            "best_by_mrr": {
                "pool_size": best_mrr_k,
                "mrr": pool_results[str(best_mrr_k)]["overall_metrics"]["mrr"],
                "gain_vs_k20": round(pool_results[str(best_mrr_k)]["overall_metrics"]["mrr"] - pool_results["20"]["overall_metrics"]["mrr"], 4),
            },
            "best_by_hit1": {
                "pool_size": best_h1_k,
                "hit@1": pool_results[str(best_h1_k)]["overall_metrics"]["hit@1"],
                "gain_vs_k20": round(pool_results[str(best_h1_k)]["overall_metrics"]["hit@1"] - pool_results["20"]["overall_metrics"]["hit@1"], 4),
            },
            "best_by_ndcg10": {
                "pool_size": best_ndcg10_k,
                "ndcg@10": pool_results[str(best_ndcg10_k)]["overall_metrics"]["ndcg@10"],
                "gain_vs_k20": round(pool_results[str(best_ndcg10_k)]["overall_metrics"]["ndcg@10"] - pool_results["20"]["overall_metrics"]["ndcg@10"], 4),
            },
        },
        "comparison_table": [
            {
                "pool_size": k,
                "hit@1": pool_results[str(k)]["overall_metrics"].get("hit@1"),
                "hit@3": pool_results[str(k)]["overall_metrics"].get("hit@3"),
                "hit@5": pool_results[str(k)]["overall_metrics"].get("hit@5"),
                "hit@10": pool_results[str(k)]["overall_metrics"].get("hit@10"),
                "hit@20": pool_results[str(k)]["overall_metrics"].get("hit@20"),
                f"hit@{k}": pool_results[str(k)]["overall_metrics"].get(f"hit@{k}"),
                "recall@5": pool_results[str(k)]["overall_metrics"].get("recall@5"),
                "recall@10": pool_results[str(k)]["overall_metrics"].get("recall@10"),
                "mrr": pool_results[str(k)]["overall_metrics"].get("mrr"),
                "map_at_20": pool_results[str(k)]["overall_metrics"].get("map_at_20"),
                "ndcg@5": pool_results[str(k)]["overall_metrics"].get("ndcg@5"),
                "ndcg@10": pool_results[str(k)]["overall_metrics"].get("ndcg@10"),
            }
            for k in k_values
        ],
    }

    full_report = {
        "metadata": {
            "experiment": "Stage 5C: Candidate Pool Expansion Experiment",
            "model": "BAAI/bge-reranker-large",
            "fixed_beta": beta,
            "evaluated_pool_sizes": k_values,
            "total_queries": len(queries),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": summary_data,
        "detailed_results": pool_results,
    }

    # Write files
    print(f"\nWriting full expansion report to {REPORT_OUT_PATH}...")
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"Writing summary report to {SUMMARY_OUT_PATH}...")
    with open(SUMMARY_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"Writing predictions to {PREDICTIONS_OUT_PATH}...")
    with open(PREDICTIONS_OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_predictions_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 90)
    print("STAGE 5C EXPERIMENT COMPLETE")
    print("=" * 90)
    print(f"Best K by MRR:     K = {best_mrr_k} (MRR: {summary_data['best_configurations']['best_by_mrr']['mrr']:.4f})")
    print(f"Best K by Hit@1:   K = {best_h1_k} (Hit@1: {summary_data['best_configurations']['best_by_hit1']['hit@1']:.4f})")
    print(f"Best K by nDCG@10: K = {best_ndcg10_k} (nDCG@10: {summary_data['best_configurations']['best_by_ndcg10']['ndcg@10']:.4f})")

    return full_report


if __name__ == "__main__":
    run_candidate_pool_expansion()

