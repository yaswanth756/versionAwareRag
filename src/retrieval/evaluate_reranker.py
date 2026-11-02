"""
src/retrieval/evaluate_reranker.py
----------------------------------
Executes the Stage 5 Cross-Encoder Neural Reranking benchmark on top of the
first-stage retrieval candidates across the frozen development dataset
(data/evaluation/dev.reviewed.jsonl).

Evaluates:
  1. Pure Cross-Encoder Reranking (blend_beta=1.0) on Hybrid candidates.
  2. Blended Cross-Encoder Reranking (blend_beta=0.7, 70% CE + 30% Hybrid prior).
  3. Ablations: Reranking on BM25 alone and Dense alone.
  4. Comprehensive multi-pipeline comparison: BM25 vs Dense vs Hybrid vs Pure Reranker vs Blended Reranker.

Generates:
  - data/evaluation/reranker_dev_benchmark_report.json
  - data/evaluation/reranker_dev_predictions.jsonl
"""

import json
import time
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.remote_reranker import RemoteCrossEncoderReranker
from src.retrieval.metrics import evaluate_query, aggregate_metrics

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "chunks.jsonl"
DEV_PATH = PROJECT_ROOT / "data" / "evaluation" / "dev.reviewed.jsonl"

BM25_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "bm25_dev_predictions.jsonl"
DENSE_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "dense_dev_predictions.jsonl"
HYBRID_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "hybrid_dev_predictions.jsonl"

BM25_REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "bm25_dev_benchmark_report.json"
DENSE_REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "dense_dev_benchmark_report.json"
HYBRID_REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "hybrid_dev_benchmark_report.json"

REPORT_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "reranker_dev_benchmark_report.json"
PREDICTIONS_OUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "reranker_dev_predictions.jsonl"


def load_chunks() -> Dict[str, Dict[str, Any]]:
    chunks = {}
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks[chunk["chunk_id"]] = chunk
    return chunks


def load_dev_queries() -> List[Dict[str, Any]]:
    queries = []
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_predictions(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    preds = {}
    if not path.exists():
        return preds
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            preds[record["query_id"]] = record["retrieved_candidates"]
    return preds


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


def run_reranker_benchmark(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size: int = 32,
    top_k: int = 20,
    remote_url: Optional[str] = None,
) -> Dict[str, Any]:
    header_model = f"REMOTE GPU ({remote_url})" if remote_url else model_name
    print("=" * 90)
    print(f"STAGE 5: CROSS-ENCODER NEURAL RERANKING BENCHMARK ({header_model})")
    print("=" * 90)

    chunk_lookup = load_chunks()
    print(f"Loaded {len(chunk_lookup)} corpus chunks from {CHUNKS_PATH}")
    chunk_to_doc = {cid: c.get("doc_id", c.get("relative_path", "")) for cid, c in chunk_lookup.items()}

    queries = load_dev_queries()
    print(f"Loaded {len(queries)} evaluation queries from {DEV_PATH}")

    # Load candidates from hybrid predictions
    hybrid_preds = load_predictions(HYBRID_PREDICTIONS_PATH)
    if not hybrid_preds:
        raise FileNotFoundError(f"Hybrid predictions not found at {HYBRID_PREDICTIONS_PATH}. Run evaluate_hybrid.py first.")
    print(f"Loaded hybrid candidate pools for {len(hybrid_preds)} queries from {HYBRID_PREDICTIONS_PATH}")

    # Initialize Reranker (Remote or Local)
    if remote_url:
        print(f"Initializing Remote GPU CrossEncoder client connected to: {remote_url}")
        reranker = RemoteCrossEncoderReranker(api_url=remote_url, batch_size=batch_size)
    else:
        reranker = CrossEncoderReranker(model_name=model_name, batch_size=batch_size)
        print(f"Initialized local CrossEncoder on device: {reranker.device}")

    # 1. Pure Cross-Encoder Reranking (blend_beta=1.0)
    total_pairs = sum(len(cands) for cands in hybrid_preds.values())
    print(f"\n1. Executing Pure Cross-Encoder Reranking (beta=1.0) across {len(queries)} queries ({total_pairs} pairs)...")
    t0 = time.time()
    reranked_pure = reranker.batch_rerank(
        queries=queries,
        candidate_dict=hybrid_preds,
        chunk_lookup=chunk_lookup,
        text_field="searchable_text",
        top_k=top_k,
        blend_beta=1.0,
    )
    t_rerank = time.time() - t0
    pairs_per_sec = total_pairs / t_rerank if t_rerank > 0 else 0
    ms_per_query = (t_rerank / len(queries) * 1000) if queries else 0
    print(f"Pure reranking completed in {t_rerank:.2f}s ({pairs_per_sec:.1f} pairs/sec, {ms_per_query:.1f} ms/query)")

    overall_pure, per_q_pure, cat_pure = evaluate_candidate_dict(queries, reranked_pure, chunk_to_doc)

    # 2. Blended Cross-Encoder Reranking (beta=0.7: 70% CE + 30% Hybrid)
    print("\n2. Executing Blended Score Reranking (beta=0.7: 70% CrossEncoder + 30% Hybrid Prior)...")
    reranked_blended = reranker.batch_rerank(
        queries=queries,
        candidate_dict=hybrid_preds,
        chunk_lookup=chunk_lookup,
        text_field="searchable_text",
        top_k=top_k,
        blend_beta=0.7,
    )
    overall_blended, per_q_blended, cat_blended = evaluate_candidate_dict(queries, reranked_blended, chunk_to_doc)

    # 3. Ablation: Reranking on BM25 and Dense candidates
    print("\n3. Running ablations on single-retriever candidate pools...")
    bm25_preds = load_predictions(BM25_PREDICTIONS_PATH)
    reranked_bm25 = reranker.batch_rerank(queries, bm25_preds, chunk_lookup, top_k=top_k, blend_beta=1.0) if bm25_preds else {}
    overall_bm25_reranked, _, _ = evaluate_candidate_dict(queries, reranked_bm25, chunk_to_doc) if reranked_bm25 else ({}, [], {})

    dense_preds = load_predictions(DENSE_PREDICTIONS_PATH)
    reranked_dense = reranker.batch_rerank(queries, dense_preds, chunk_lookup, top_k=top_k, blend_beta=1.0) if dense_preds else {}
    overall_dense_reranked, _, _ = evaluate_candidate_dict(queries, reranked_dense, chunk_to_doc) if reranked_dense else ({}, [], {})

    # 4. Save predictions (saving blended configuration as primary, which achieves highest overall performance)
    predictions_records = []
    for q in queries:
        qid = q["query_id"]
        cat = q["category"]
        question = q["question"]
        target_cids = set(q.get("relevant_chunk_ids", []))
        cands = reranked_blended.get(qid, [])
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
                    "ce_score": c.get("ce_score"),
                    "is_hit": c["chunk_id"] in target_cids,
                    "original_rank": c.get("original_rank"),
                    "original_score": c.get("original_score"),
                    "bm25_rank": c.get("bm25_rank"),
                    "dense_rank": c.get("dense_rank"),
                    "doc_path": c.get("doc_path", ""),
                    "preview": c.get("preview", ""),
                }
                for c in cands
            ],
        })

    print(f"\nWriting reranker predictions (blended optimal) to {PREDICTIONS_OUT_PATH}...")
    with open(PREDICTIONS_OUT_PATH, "w", encoding="utf-8") as f:
        for p in predictions_records:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 5. Load baseline reports for comparative tables
    bm25_report = {}
    dense_report = {}
    hybrid_report = {}
    if BM25_REPORT_PATH.exists():
        with open(BM25_REPORT_PATH, "r", encoding="utf-8") as f:
            bm25_report = json.load(f)
    if DENSE_REPORT_PATH.exists():
        with open(DENSE_REPORT_PATH, "r", encoding="utf-8") as f:
            dense_report = json.load(f)
    if HYBRID_REPORT_PATH.exists():
        with open(HYBRID_REPORT_PATH, "r", encoding="utf-8") as f:
            hybrid_report = json.load(f)

    bm25_overall = bm25_report.get("overall_metrics", {})
    dense_overall = dense_report.get("overall_metrics", {})
    hybrid_overall = hybrid_report.get("overall_metrics", {})

    category_summary_blended = {}
    for cat, evals in sorted(cat_blended.items()):
        category_summary_blended[cat] = {
            "query_count": len(evals),
            "metrics": aggregate_metrics(evals),
        }

    category_summary_pure = {}
    for cat, evals in sorted(cat_pure.items()):
        category_summary_pure[cat] = {
            "query_count": len(evals),
            "metrics": aggregate_metrics(evals),
        }

    # 5-way comparison table
    comparison_keys = [
        "hit@1", "hit@3", "hit@5", "hit@10", "hit@20",
        "recall@5", "recall@10", "mrr", "map_at_20", "ndcg@10",
        "doc_hit@1", "doc_hit@5", "doc_hit@10"
    ]
    five_way_comparison = {}
    for k in comparison_keys:
        b_val = bm25_overall.get(k, 0.0)
        d_val = dense_overall.get(k, 0.0)
        h_val = hybrid_overall.get(k, 0.0)
        p_val = overall_pure.get(k, 0.0)
        bl_val = overall_blended.get(k, 0.0)

        delta_bm25 = round(bl_val - b_val, 4)
        delta_hybrid = round(bl_val - h_val, 4)

        five_way_comparison[k] = {
            "bm25": b_val,
            "dense": d_val,
            "hybrid": h_val,
            "pure_reranked": p_val,
            "blended_reranked": bl_val,
            "gain_blended_vs_bm25": delta_bm25,
            "gain_blended_vs_hybrid": delta_hybrid,
            "rel_gain_blended_vs_bm25_pct": round((delta_bm25 / b_val * 100) if b_val > 0 else 0.0, 2),
            "rel_gain_blended_vs_hybrid_pct": round((delta_hybrid / h_val * 100) if h_val > 0 else 0.0, 2),
        }

    final_report = {
        "benchmark_metadata": {
            "model_type": "cross_encoder_neural_reranker",
            "model_name": getattr(reranker, "model_name", model_name),
            "device": getattr(reranker, "device", "unknown"),
            "benchmark_dataset": "dev.reviewed.jsonl (frozen v1.0.0)",
            "total_queries_evaluated": len(queries),
            "total_pairs_scored": total_pairs,
            "reranking_duration_seconds": round(t_rerank, 3),
            "pairs_per_second": round(pairs_per_sec, 1),
            "ms_per_query": round(ms_per_query, 1),
            "first_stage_input": "hybrid_bm25_plus_dense_predictions",
            "optimal_configuration": {
                "fusion_mode": "blended_score_interpolation",
                "blend_beta": 0.7,
                "description": "70% Cross-Encoder relevance score + 30% first-stage Hybrid score prior"
            }
        },
        "blended_reranked_metrics": overall_blended,
        "pure_reranked_metrics": overall_pure,
        "category_metrics_blended": category_summary_blended,
        "category_metrics_pure": category_summary_pure,
        "five_way_pipeline_comparison": five_way_comparison,
        "ablation_pipeline_comparison": {
            "bm25_raw": bm25_overall,
            "bm25_plus_reranker": overall_bm25_reranked,
            "dense_raw": dense_overall,
            "dense_plus_reranker": overall_dense_reranked,
            "hybrid_raw": hybrid_overall,
            "hybrid_pure_reranker": overall_pure,
            "hybrid_blended_reranker": overall_blended,
        },
    }

    print(f"Writing reranker benchmark report to {REPORT_OUT_PATH}...")
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    # 6. Display 5-Way Comparison Table
    print("\n" + "=" * 116)
    print("5-WAY END-TO-END PIPELINE BENCHMARK: BM25 vs DENSE vs HYBRID vs PURE RERANK vs BLENDED RERANK")
    print("=" * 116)
    print(f"{'Metric':<13} {'BM25':>8} {'Dense':>8} {'Hybrid':>8} {'Pure CE':>9} {'Blended':>10} {'Gain vs BM25':>18} {'Gain vs Hybrid':>18}")
    print("-" * 116)
    for m in comparison_keys:
        row = five_way_comparison.get(m, {})
        b_val = row.get("bm25", 0.0)
        d_val = row.get("dense", 0.0)
        h_val = row.get("hybrid", 0.0)
        p_val = row.get("pure_reranked", 0.0)
        bl_val = row.get("blended_reranked", 0.0)
        gain_b = f"{row.get('gain_blended_vs_bm25', 0.0):+0.4f} ({row.get('rel_gain_blended_vs_bm25_pct', 0.0):+5.1f}%)"
        gain_h = f"{row.get('gain_blended_vs_hybrid', 0.0):+0.4f} ({row.get('rel_gain_blended_vs_hybrid_pct', 0.0):+5.1f}%)"
        print(f"{m:<13} {b_val:>8.4f} {d_val:>8.4f} {h_val:>8.4f} {p_val:>9.4f} {bl_val:>10.4f} {gain_b:>18} {gain_h:>18}")
    print("=" * 116)

    # 7. Display Category Breakdown for Blended Reranker
    print("\n" + "=" * 105)
    print("CATEGORY BREAKDOWN: Hit@1 & MRR (HYBRID vs BLENDED RERANKER)")
    print("=" * 105)
    print(f"{'Category':<24} {'Count':<6} {'Hyb H@1':<10} {'Blended H@1':<13} {'Hyb MRR':<10} {'Blended MRR':<13} {'MRR Gain':>12}")
    print("-" * 105)
    hybrid_cats = hybrid_report.get("category_metrics", {})
    for cat, data in category_summary_blended.items():
        cnt = data["query_count"]
        bl_m = data["metrics"]
        h_m = hybrid_cats.get(cat, {}).get("metrics", {})
        hh1 = f"{h_m.get('hit@1', 0)*100:.1f}%"
        blh1 = f"{bl_m.get('hit@1', 0)*100:.1f}%"
        hmrr = h_m.get("mrr", 0.0)
        blmrr = bl_m.get("mrr", 0.0)
        gain_mrr = f"{blmrr - hmrr:+0.4f} ({(blmrr - hmrr)/hmrr*100:+.1f}%)" if hmrr > 0 else "N/A"
        print(f"{cat:<24} {cnt:<6} {hh1:<10} {blh1:<13} {hmrr:<10.4f} {blmrr:<13.4f} {gain_mrr:>12}")
    print("=" * 105)

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cross-Encoder Reranker on Development Benchmark.")
    parser.add_argument("--model-name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Local model name if running locally.")
    parser.add_argument("--remote-url", type=str, default=None, help="Public URL of Kaggle/Cloud GPU server (e.g. https://xxx.trycloudflare.com).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    args = parser.parse_args()

    run_reranker_benchmark(
        model_name=args.model_name,
        batch_size=args.batch_size,
        remote_url=args.remote_url,
    )
