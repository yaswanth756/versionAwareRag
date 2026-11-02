"""
src/retrieval/evaluate_bge_beta_sweep.py
---------------------------------------
Executes Stage 5B: BGE-Reranker Score Blending Experiment.
Systematically evaluates blending weights:
    beta in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for BAAI/bge-reranker-large against the frozen Python 3.12 documentation
development dataset (140 queries).

Formula:
    S_final = beta * norm(S_ce) + (1 - beta) * norm(S_hybrid)
where:
    norm(scores) is min-max normalization per query candidate list.
    When scores tie, stable sort preserves first-stage Hybrid candidate rank order.

Generates:
    - data/evaluation/bge_beta_sweep_report.json
    - data/evaluation/bge_beta_sweep_predictions.jsonl
    - data/evaluation/bge_beta_sweep_summary.json
"""

import json
import time
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import minmax_normalize
from src.retrieval.metrics import evaluate_query, aggregate_metrics

# File paths
DEV_PATH = PROJECT_ROOT / "data" / "evaluation" / "dev.reviewed.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "chunks.jsonl"
HYBRID_PREDS_PATH = PROJECT_ROOT / "data" / "evaluation" / "hybrid_dev_predictions.jsonl"
RERANK_PREDS_PATH = PROJECT_ROOT / "data" / "evaluation" / "reranker_dev_predictions.jsonl"

REPORT_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bge_beta_sweep_report.json"
PREDICTIONS_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bge_beta_sweep_predictions.jsonl"
SUMMARY_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bge_beta_sweep_summary.json"

ROOT_REPORT_OUT_PATH = PROJECT_ROOT / "bge_beta_sweep_report.json"
ROOT_PREDICTIONS_OUT_PATH = PROJECT_ROOT / "bge_beta_sweep_predictions.jsonl"
ROOT_SUMMARY_OUT_PATH = PROJECT_ROOT / "bge_beta_sweep_summary.json"

# Historical baselines
HISTORICAL_BASELINES = {
    "hybrid": {
        "mrr": 0.6184,
        "hit@1": 0.4929,
        "hit@3": 0.7214,
        "hit@5": 0.7786,
        "hit@10": 0.8786,
        "hit@20": 0.9214,
        "recall@5": 0.7464,
        "recall@10": 0.8429,
        "map_at_20": 0.6026,
        "ndcg@5": 0.6290,
        "ndcg@10": 0.6627,
    },
    "minilm_blended_beta_0.7": {
        "mrr": 0.6492,
        "hit@1": 0.5357,
        "hit@3": 0.7286,
        "hit@5": 0.8214,
        "hit@10": 0.8857,
        "hit@20": 0.9214,
        "recall@5": 0.7857,
        "recall@10": 0.8536,
        "map_at_20": 0.6340,
        "ndcg@5": 0.6658,
        "ndcg@10": 0.6897,
    },
    "bge_blended_beta_0.7_prior": {
        "mrr": 0.6402,
        "hit@1": 0.5143,
        "hit@3": 0.7286,
        "hit@5": 0.8000,
        "hit@10": 0.8571,
        "hit@20": 0.9214,
        "recall@5": 0.7714,
        "recall@10": 0.8250,
        "map_at_20": 0.6268,
        "ndcg@5": 0.6558,
        "ndcg@10": 0.6748,
    },
}


def load_dev_queries() -> List[Dict[str, Any]]:
    queries = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_chunk_to_doc() -> Dict[str, str]:
    chunk_to_doc = {}
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_to_doc[c["chunk_id"]] = c.get("metadata", {}).get("document_path", "")
    return chunk_to_doc


def load_candidate_data() -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, float]]]:
    # Hybrid candidate pools
    hybrid_preds: Dict[str, List[Dict[str, Any]]] = {}
    with open(HYBRID_PREDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            hybrid_preds[rec["query_id"]] = rec["retrieved_candidates"]

    # BGE Cross-Encoder scores
    bge_ce_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(RERANK_PREDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["query_id"]
            for cand in rec["retrieved_candidates"]:
                bge_ce_scores[qid][cand["chunk_id"]] = float(cand["ce_score"])

    return hybrid_preds, bge_ce_scores


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


def run_beta_sweep(betas: List[float] = None) -> Dict[str, Any]:
    if betas is None:
        betas = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("=" * 90)
    print("STAGE 5B: BGE-RERANKER SCORE BLENDING EXPERIMENT (BETA SWEEP)")
    print("=" * 90)

    queries = load_dev_queries()
    chunk_to_doc = load_chunk_to_doc()
    hybrid_preds, bge_ce_scores = load_candidate_data()

    print(f"Loaded {len(queries)} dev queries.")
    print(f"Evaluating {len(betas)} beta values: {betas}")

    sweep_results = {}
    all_predictions_records = []
    per_beta_predictions: Dict[float, Dict[str, List[Dict[str, Any]]]] = {}

    for beta in betas:
        t0 = time.time()
        beta_preds: Dict[str, List[Dict[str, Any]]] = {}

        for q in queries:
            qid = q["query_id"]
            cands = hybrid_preds.get(qid, [])

            # Extract raw scores
            q_ce_raw = [bge_ce_scores[qid].get(c["chunk_id"], 0.0) for c in cands]
            q_orig_raw = [float(c.get("score", 0.0)) for c in cands]

            # Compute normalized blended score
            if beta == 0.0:
                final_scores = q_orig_raw
            elif beta == 1.0:
                final_scores = q_ce_raw
            else:
                ce_norm = np.array(minmax_normalize(q_ce_raw), dtype=np.float32)
                orig_norm = np.array(minmax_normalize(q_orig_raw), dtype=np.float32)
                final_scores = (beta * ce_norm + (1.0 - beta) * orig_norm).tolist()

            # Deterministic stable sort preserves hybrid order on ties
            ranked_indices = sorted(range(len(cands)), key=lambda idx: final_scores[idx], reverse=True)

            ranked_cands = []
            for rank, idx in enumerate(ranked_indices, start=1):
                base_c = cands[idx]
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
            beta_preds[qid] = ranked_cands

        t_blend = time.time() - t0
        per_beta_predictions[beta] = beta_preds

        # Evaluate metrics
        overall, per_q, cat_evals = evaluate_candidate_dict(queries, beta_preds, chunk_to_doc)

        cat_summary = {}
        for cat, evals in sorted(cat_evals.items()):
            cat_summary[cat] = {
                "query_count": len(evals),
                "metrics": aggregate_metrics(evals),
            }

        sweep_results[str(beta)] = {
            "beta": beta,
            "blending_duration_seconds": round(t_blend, 4),
            "overall_metrics": overall,
            "category_metrics": cat_summary,
        }

        # Build prediction records for this beta
        for q in queries:
            qid = q["query_id"]
            cat = q["category"]
            question = q["question"]
            target_cids = set(q.get("relevant_chunk_ids", []))
            ranked = beta_preds[qid]

            all_predictions_records.append({
                "beta": beta,
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

    # Find best betas
    best_mrr_beta = max(betas, key=lambda b: sweep_results[str(b)]["overall_metrics"]["mrr"])
    best_ndcg10_beta = max(betas, key=lambda b: sweep_results[str(b)]["overall_metrics"]["ndcg@10"])
    best_hit1_beta = max(betas, key=lambda b: sweep_results[str(b)]["overall_metrics"]["hit@1"])
    best_hit5_beta = max(betas, key=lambda b: sweep_results[str(b)]["overall_metrics"]["hit@5"])

    summary_data = {
        "experiment_name": "Stage 5B: BGE-Reranker Score Blending Experiment (Beta Sweep)",
        "model_name": "BAAI/bge-reranker-large",
        "dataset": "dev.reviewed.jsonl (frozen v1.0.0, 140 queries)",
        "candidate_pool": "hybrid_dev_predictions.jsonl (top-20 chunks, 2800 pairs)",
        "hardware_inference": "NVIDIA Tesla T4 GPU (Kaggle via Cloudflare tunnel, 222.35s total neural inference)",
        "best_configurations": {
            "best_by_mrr": {
                "beta": best_mrr_beta,
                "mrr": sweep_results[str(best_mrr_beta)]["overall_metrics"]["mrr"],
                "gain_vs_hybrid": round(sweep_results[str(best_mrr_beta)]["overall_metrics"]["mrr"] - HISTORICAL_BASELINES["hybrid"]["mrr"], 4),
                "gain_vs_beta_0.7": round(sweep_results[str(best_mrr_beta)]["overall_metrics"]["mrr"] - sweep_results["0.7"]["overall_metrics"]["mrr"], 4),
            },
            "best_by_ndcg10": {
                "beta": best_ndcg10_beta,
                "ndcg@10": sweep_results[str(best_ndcg10_beta)]["overall_metrics"]["ndcg@10"],
                "gain_vs_hybrid": round(sweep_results[str(best_ndcg10_beta)]["overall_metrics"]["ndcg@10"] - HISTORICAL_BASELINES["hybrid"]["ndcg@10"], 4),
                "gain_vs_beta_0.7": round(sweep_results[str(best_ndcg10_beta)]["overall_metrics"]["ndcg@10"] - sweep_results["0.7"]["overall_metrics"]["ndcg@10"], 4),
            },
            "best_by_hit1": {
                "beta": best_hit1_beta,
                "hit@1": sweep_results[str(best_hit1_beta)]["overall_metrics"]["hit@1"],
                "gain_vs_hybrid": round(sweep_results[str(best_hit1_beta)]["overall_metrics"]["hit@1"] - HISTORICAL_BASELINES["hybrid"]["hit@1"], 4),
                "gain_vs_beta_0.7": round(sweep_results[str(best_hit1_beta)]["overall_metrics"]["hit@1"] - sweep_results["0.7"]["overall_metrics"]["hit@1"], 4),
            },
            "best_by_hit5": {
                "beta": best_hit5_beta,
                "hit@5": sweep_results[str(best_hit5_beta)]["overall_metrics"]["hit@5"],
            },
        },
        "historical_baselines": HISTORICAL_BASELINES,
        "beta_comparison_table": [
            {
                "beta": b,
                "hit@1": sweep_results[str(b)]["overall_metrics"]["hit@1"],
                "hit@3": sweep_results[str(b)]["overall_metrics"]["hit@3"],
                "hit@5": sweep_results[str(b)]["overall_metrics"]["hit@5"],
                "hit@10": sweep_results[str(b)]["overall_metrics"]["hit@10"],
                "hit@20": sweep_results[str(b)]["overall_metrics"]["hit@20"],
                "recall@5": sweep_results[str(b)]["overall_metrics"]["recall@5"],
                "recall@10": sweep_results[str(b)]["overall_metrics"]["recall@10"],
                "mrr": sweep_results[str(b)]["overall_metrics"]["mrr"],
                "map_at_20": sweep_results[str(b)]["overall_metrics"]["map_at_20"],
                "ndcg@5": sweep_results[str(b)]["overall_metrics"]["ndcg@5"],
                "ndcg@10": sweep_results[str(b)]["overall_metrics"]["ndcg@10"],
            }
            for b in betas
        ],
    }

    full_report = {
        "metadata": {
            "experiment": "Stage 5B: BGE-Reranker Score Blending Experiment",
            "model": "BAAI/bge-reranker-large",
            "evaluated_queries": len(queries),
            "evaluated_betas": betas,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": summary_data,
        "detailed_sweep_results": sweep_results,
    }

    # Write files
    print(f"\nWriting full sweep report to {REPORT_OUT_PATH}...")
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    with open(ROOT_REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"Writing summary report to {SUMMARY_OUT_PATH}...")
    with open(SUMMARY_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    with open(ROOT_SUMMARY_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"Writing all sweep predictions to {PREDICTIONS_OUT_PATH}...")
    with open(PREDICTIONS_OUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_predictions_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(ROOT_PREDICTIONS_OUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_predictions_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Also save separate predictions per beta configuration
    for b in betas:
        per_beta_path = PROJECT_ROOT / "data" / "evaluation" / f"bge_beta_{b}_predictions.jsonl"
        with open(per_beta_path, "w", encoding="utf-8") as f:
            for rec in all_predictions_records:
                if rec["beta"] == b:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 90)
    print("STAGE 5B EXPERIMENT COMPLETE")
    print("=" * 90)
    print(f"Best Beta by MRR:     beta = {best_mrr_beta} (MRR: {summary_data['best_configurations']['best_by_mrr']['mrr']:.4f})")
    print(f"Best Beta by nDCG@10: beta = {best_ndcg10_beta} (nDCG@10: {summary_data['best_configurations']['best_by_ndcg10']['ndcg@10']:.4f})")
    print(f"Best Beta by Hit@1:   beta = {best_hit1_beta} (Hit@1: {summary_data['best_configurations']['best_by_hit1']['hit@1']:.4f})")

    return full_report


if __name__ == "__main__":
    run_beta_sweep()

