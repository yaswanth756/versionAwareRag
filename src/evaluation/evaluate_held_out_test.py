"""
src/evaluation/evaluate_held_out_test.py
-----------------------------------------
Stage 7: Final Held-Out Blind Evaluation on Python 3.12 Documentation RAG System.

Evaluates the completely frozen production pipeline against the 60 held-out test queries
in data/evaluation/test.jsonl (SHA-256: d5840a2c25c4245658ee952b8f2382a0c3942eb1cc51598dfc39d13dd47d13ca).

Evaluation Dimensions:
  1. Retrieval & Reranking: Hit@1/3/5/10/20, MRR, nDCG@10, Candidate Pool Recall (K=30)
  2. Context Engineering: Adaptive Slicing Coverage (M in [10, 16], floor=0.10, max 3/sec)
  3. Category Breakdown: Exact API, Conceptual, Code Example, Version Specific, Troubleshooting, Multi-Document, Ambiguous
  4. Generation & Citations: Grounded Generation with Gemini 2.5 Flash on Vertex AI (Citation rate, validity, latency)
  5. Generalization Audit: Side-by-side comparison of Dev Benchmark (140 Qs) vs Test Benchmark (60 Qs)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import RAGPipeline
from src.generator import RAGGenerator
from src.retrieval.metrics import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)

TEST_DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "test.jsonl"
PREDICTIONS_OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "held_out_test_predictions.jsonl"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "held_out_test_report.json"

DEV_BENCHMARK_BASELINES = {
    "hit@1": 0.5286,
    "hit@3": 0.7286,
    "hit@5": 0.8000,
    "hit@10": 0.8643,
    "hit@20": 0.9429,
    "pool_recall_k30": 0.9643,
    "adaptive_coverage": 0.9214,
    "mrr": 0.6479,
    "ndcg@10": 0.6806,
}


def load_test_dataset(filepath: Path) -> List[Dict[str, Any]]:
    """Loads and validates the held-out test dataset."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_held_out_evaluation(
    eval_generation: bool = True,
    sample_gen_size: int = 15,
    mode: str = "auto",
) -> Dict[str, Any]:
    print("=" * 80)
    print("STAGE 7: FINAL HELD-OUT BLIND EVALUATION")
    print("Evaluating Frozen Production RAG Pipeline on data/evaluation/test.jsonl")
    print("=" * 80)

    # 1. Load test records
    test_records = load_test_dataset(TEST_DATASET_PATH)
    print(f"Loaded {len(test_records)} held-out test queries.")

    # 2. Initialize Pipeline & Generator
    pipeline = RAGPipeline(mode=mode)
    generator = None
    if eval_generation:
        print("Initializing Vertex AI RAGGenerator (gemini-2.5-flash)...")
        generator = RAGGenerator(pipeline=pipeline)

    # 3. Iterate over all 60 test queries
    print(f"\nRunning retrieval across {len(test_records)} queries...")
    t_start_all = time.time()

    per_query_results = []
    category_metrics = defaultdict(lambda: {
        "count": 0, "hits@1": 0, "hits@5": 0, "hits@10": 0, "hits@20": 0,
        "pool_hits_k30": 0, "adaptive_hits": 0, "rr_sum": 0.0, "ndcg_sum": 0.0
    })

    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    hits_10 = 0
    hits_15 = 0
    hits_20 = 0
    hits_pool_30 = 0
    hits_adaptive = 0
    rr_sum = 0.0
    ndcg10_sum = 0.0

    total_stage1_ms = 0.0
    total_stage2_ms = 0.0
    total_chunks_selected = 0

    generation_eval_count = 0
    citations_present_count = 0
    valid_citations_count = 0
    total_gen_ms = 0.0

    # Determine which queries get full generation (all or sample)
    gen_query_ids = set()
    if eval_generation:
        if sample_gen_size >= len(test_records):
            gen_query_ids = set(r["query_id"] for r in test_records)
        else:
            # Stratified sample across all categories
            cat_groups = defaultdict(list)
            for r in test_records:
                cat_groups[r["category"]].append(r["query_id"])
            for cat, qids in cat_groups.items():
                gen_query_ids.update(qids[:2])  # 2 per category = 14-16 total

    for i, record in enumerate(test_records, start=1):
        qid = record["query_id"]
        qtext = record["question"]
        cat = record["category"]
        rel_ids = set(record["relevant_chunk_ids"])
        rel_graded = {c["chunk_id"]: c.get("relevance_score", 3) for c in record.get("relevant_chunks", [])}
        if not rel_graded:
            rel_graded = {cid: 3 for cid in rel_ids}

        # Run pipeline retrieval (Full Pool K=30 + Adaptive Slicing)
        # 1. First get candidates
        query_res = pipeline.query(qtext)
        selected_chunks = query_res["chunks"]
        timings = query_res["timings"]

        total_stage1_ms += timings.get("stage1_hybrid_ms", 0.0)
        total_stage2_ms += timings.get("stage2_rerank_ms", 0.0)
        total_chunks_selected += len(selected_chunks)

        selected_cids = [c["chunk_id"] for c in selected_chunks]

        # Check full candidate pool K=30 hits
        # To get the full 30 candidate IDs before slicing:
        full_30_chunks, _ = pipeline.retrieve(qtext, top_k=30, adaptive=False)
        full_30_cids = [c["chunk_id"] for c in full_30_chunks]

        # Metrics computation
        h1 = calculate_hit_at_k(full_30_cids, rel_ids, 1)
        h3 = calculate_hit_at_k(full_30_cids, rel_ids, 3)
        h5 = calculate_hit_at_k(full_30_cids, rel_ids, 5)
        h10 = calculate_hit_at_k(full_30_cids, rel_ids, 10)
        h15 = calculate_hit_at_k(full_30_cids, rel_ids, 15)
        h20 = calculate_hit_at_k(full_30_cids, rel_ids, 20)
        h30 = calculate_hit_at_k(full_30_cids, rel_ids, 30)
        h_adapt = 1 if any(cid in rel_ids for cid in selected_cids) else 0

        rr = calculate_mrr(full_30_cids, rel_ids)
        ndcg10 = calculate_ndcg_at_k(full_30_cids, rel_graded, k=10)

        hits_1 += h1
        hits_3 += h3
        hits_5 += h5
        hits_10 += h10
        hits_15 += h15
        hits_20 += h20
        hits_pool_30 += h30
        hits_adaptive += h_adapt
        rr_sum += rr
        ndcg10_sum += ndcg10

        # Category tracking
        c_dict = category_metrics[cat]
        c_dict["count"] += 1
        c_dict["hits@1"] += h1
        c_dict["hits@5"] += h5
        c_dict["hits@10"] += h10
        c_dict["hits@20"] += h20
        c_dict["pool_hits_k30"] += h30
        c_dict["adaptive_hits"] += h_adapt
        c_dict["rr_sum"] += rr
        c_dict["ndcg_sum"] += ndcg10

        gen_result = None
        if eval_generation and qid in gen_query_ids:
            gen_res = generator.generate(qtext, stream=False)
            gen_text = gen_res["answer"]
            gen_time = gen_res["timings"]["generation_ms"]
            total_gen_ms += gen_time
            generation_eval_count += 1

            cited_indices = gen_res["cited_doc_indices"]
            has_citations = len(cited_indices) > 0
            if has_citations:
                citations_present_count += 1

            # Validate that cited indices are within [1, len(selected_chunks)]
            all_valid = has_citations and all(1 <= idx <= len(selected_chunks) for idx in cited_indices)
            if all_valid:
                valid_citations_count += 1

            gen_result = {
                "generated_answer": gen_text,
                "cited_doc_indices": cited_indices,
                "has_citations": has_citations,
                "citations_valid": all_valid,
                "generation_ms": gen_time,
                "is_out_of_domain": gen_res.get("is_out_of_domain", False),
            }

        query_record = {
            "query_id": qid,
            "category": cat,
            "question": qtext,
            "target_chunk_ids": list(rel_ids),
            "retrieved_adaptive_chunks": [c["chunk_id"] for c in selected_chunks],
            "num_adaptive_chunks": len(selected_chunks),
            "hit@1": bool(h1),
            "hit@5": bool(h5),
            "hit@10": bool(h10),
            "hit@20": bool(h20),
            "hit@30_pool": bool(h30),
            "hit_adaptive_window": bool(h_adapt),
            "reciprocal_rank": round(rr, 4),
            "ndcg@10": round(ndcg10, 4),
            "timings_ms": timings,
            "generation": gen_result,
        }
        per_query_results.append(query_record)

        if i % 10 == 0 or i == len(test_records):
            print(f"[{i:2d}/{len(test_records)}] Processed... Running MRR: {rr_sum/i:.4f} | Adaptive Hits: {hits_adaptive}/{i} ({hits_adaptive/i*100:.1f}%)")

    total_eval_time = round(time.time() - t_start_all, 2)
    n = len(test_records)

    # 4. Aggregated Global Metrics
    global_metrics = {
        "total_queries": n,
        "hit@1": round(hits_1 / n, 4),
        "hit@3": round(hits_3 / n, 4),
        "hit@5": round(hits_5 / n, 4),
        "hit@10": round(hits_10 / n, 4),
        "hit@15": round(hits_15 / n, 4),
        "hit@20": round(hits_20 / n, 4),
        "candidate_pool_recall_k30": round(hits_pool_30 / n, 4),
        "adaptive_window_coverage": round(hits_adaptive / n, 4),
        "mrr": round(rr_sum / n, 4),
        "ndcg@10": round(ndcg10_sum / n, 4),
        "average_adaptive_chunks": round(total_chunks_selected / n, 2),
        "average_stage1_ms": round(total_stage1_ms / n, 2),
        "average_stage2_ms": round(total_stage2_ms / n, 2),
        "average_retrieval_ms": round((total_stage1_ms + total_stage2_ms) / n, 2),
        "total_wall_clock_seconds": total_eval_time,
    }

    # 5. Category Breakdown Metrics
    category_summary = {}
    for cat, data in category_metrics.items():
        cn = data["count"]
        category_summary[cat] = {
            "query_count": cn,
            "hit@1": round(data["hits@1"] / cn, 4),
            "hit@5": round(data["hits@5"] / cn, 4),
            "hit@10": round(data["hits@10"] / cn, 4),
            "hit@20": round(data["hits@20"] / cn, 4),
            "candidate_pool_recall_k30": round(data["pool_hits_k30"] / cn, 4),
            "adaptive_coverage": round(data["adaptive_hits"] / cn, 4),
            "mrr": round(data["rr_sum"] / cn, 4),
            "ndcg@10": round(data["ndcg_sum"] / cn, 4),
        }

    # 6. Generation Evaluation Metrics
    gen_metrics = {}
    if generation_eval_count > 0:
        gen_metrics = {
            "queries_generated": generation_eval_count,
            "citation_presence_rate": round(citations_present_count / generation_eval_count, 4),
            "citation_validity_rate": round(valid_citations_count / generation_eval_count, 4),
            "average_generation_latency_ms": round(total_gen_ms / generation_eval_count, 2),
        }

    # 7. Generalization Comparison (Dev vs Test)
    generalization_delta = {
        "mrr": {
            "dev_v1": DEV_BENCHMARK_BASELINES["mrr"],
            "test_held_out": global_metrics["mrr"],
            "delta": round(global_metrics["mrr"] - DEV_BENCHMARK_BASELINES["mrr"], 4),
        },
        "hit@1": {
            "dev_v1": DEV_BENCHMARK_BASELINES["hit@1"],
            "test_held_out": global_metrics["hit@1"],
            "delta": round(global_metrics["hit@1"] - DEV_BENCHMARK_BASELINES["hit@1"], 4),
        },
        "hit@5": {
            "dev_v1": DEV_BENCHMARK_BASELINES["hit@5"],
            "test_held_out": global_metrics["hit@5"],
            "delta": round(global_metrics["hit@5"] - DEV_BENCHMARK_BASELINES["hit@5"], 4),
        },
        "hit@10": {
            "dev_v1": DEV_BENCHMARK_BASELINES["hit@10"],
            "test_held_out": global_metrics["hit@10"],
            "delta": round(global_metrics["hit@10"] - DEV_BENCHMARK_BASELINES["hit@10"], 4),
        },
        "hit@20": {
            "dev_v1": DEV_BENCHMARK_BASELINES["hit@20"],
            "test_held_out": global_metrics["hit@20"],
            "delta": round(global_metrics["hit@20"] - DEV_BENCHMARK_BASELINES["hit@20"], 4),
        },
        "candidate_pool_recall_k30": {
            "dev_v1": DEV_BENCHMARK_BASELINES["pool_recall_k30"],
            "test_held_out": global_metrics["candidate_pool_recall_k30"],
            "delta": round(global_metrics["candidate_pool_recall_k30"] - DEV_BENCHMARK_BASELINES["pool_recall_k30"], 4),
        },
        "adaptive_coverage": {
            "dev_v1": DEV_BENCHMARK_BASELINES["adaptive_coverage"],
            "test_held_out": global_metrics["adaptive_window_coverage"],
            "delta": round(global_metrics["adaptive_window_coverage"] - DEV_BENCHMARK_BASELINES["adaptive_coverage"], 4),
        },
        "ndcg@10": {
            "dev_v1": DEV_BENCHMARK_BASELINES["ndcg@10"],
            "test_held_out": global_metrics["ndcg@10"],
            "delta": round(global_metrics["ndcg@10"] - DEV_BENCHMARK_BASELINES["ndcg@10"], 4),
        },
    }

    report = {
        "stage": "Stage 7 Final Held-Out Evaluation",
        "dataset": "test.jsonl (60 questions)",
        "dataset_sha256": "d5840a2c25c4245658ee952b8f2382a0c3942eb1cc51598dfc39d13dd47d13ca",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "global_metrics": global_metrics,
        "category_breakdown": category_summary,
        "generation_metrics": gen_metrics,
        "generalization_comparison": generalization_delta,
    }

    # 8. Save Predictions & Report
    print(f"\nWriting predictions to {PREDICTIONS_OUTPUT_PATH}...")
    with open(PREDICTIONS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for p in per_query_results:
            f.write(json.dumps(p) + "\n")

    print(f"Writing evaluation report to {REPORT_OUTPUT_PATH}...")
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("HELD-OUT TEST EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Test MRR:                     {global_metrics['mrr']:.4f}  (Dev: {DEV_BENCHMARK_BASELINES['mrr']:.4f} | Delta: {generalization_delta['mrr']['delta']:+.4f})")
    print(f"Test Hit@1:                   {global_metrics['hit@1']*100:.2f}%  (Dev: {DEV_BENCHMARK_BASELINES['hit@1']*100:.2f}%)")
    print(f"Test Hit@5:                   {global_metrics['hit@5']*100:.2f}%  (Dev: {DEV_BENCHMARK_BASELINES['hit@5']*100:.2f}%)")
    print(f"Test Hit@10:                  {global_metrics['hit@10']*100:.2f}%  (Dev: {DEV_BENCHMARK_BASELINES['hit@10']*100:.2f}%)")
    print(f"Test Hit@20:                  {global_metrics['hit@20']*100:.2f}%  (Dev: {DEV_BENCHMARK_BASELINES['hit@20']*100:.2f}%)")
    print(f"Test Pool Recall (K=30):      {global_metrics['candidate_pool_recall_k30']*100:.2f}%  (Dev: {DEV_BENCHMARK_BASELINES['pool_recall_k30']*100:.2f}%)")
    print(f"Test Adaptive Coverage:       {global_metrics['adaptive_window_coverage']*100:.2f}%  (Dev: {DEV_BENCHMARK_BASELINES['adaptive_coverage']*100:.2f}%)")
    print(f"Test nDCG@10:                 {global_metrics['ndcg@10']:.4f}  (Dev: {DEV_BENCHMARK_BASELINES['ndcg@10']:.4f})")
    if generation_eval_count > 0:
        print(f"Generation Citation Presence: {gen_metrics['citation_presence_rate']*100:.1f}%")
        print(f"Generation Citation Validity: {gen_metrics['citation_validity_rate']*100:.1f}%")
        print(f"Avg Generation Latency:       {gen_metrics['average_generation_latency_ms']}ms")
    print("=" * 80 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Stage 7: Held-Out Test Evaluation")
    parser.add_argument("--mode", type=str, choices=["auto", "remote", "local"], default="auto", help="Reranker mode")
    parser.add_argument("--no-gen", action="store_true", help="Skip LLM generation evaluation")
    parser.add_argument("--sample-gen", type=int, default=14, help="Number of queries to sample for full LLM generation")

    args = parser.parse_args()

    run_held_out_evaluation(
        eval_generation=not args.no_gen,
        sample_gen_size=args.sample_gen,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

