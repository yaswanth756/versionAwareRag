"""
src/pipeline.py
---------------
Production-Ready End-to-End RAG Retrieval Pipeline for Python 3.12 Documentation.

Architecture:
  1. Stage 1: Hybrid Retrieval (BM25 + Dense BGE-Small, alpha=0.7, candidate_pool_size=30)
  2. Stage 2: Cross-Encoder Neural Reranking (BAAI/bge-reranker-large remote or MiniLM local fallback)
  3. Score Blending: S_final = 0.6 * norm(S_ce) + 0.4 * norm(S_hybrid)
  4. Diversity & Deduplication: Max 3 chunks per document section (prevents section monopolization)
  5. Adaptive Score Slicing: Dynamic context window (min 10, max 16, score_floor=0.10) for high recall without token bloat
  6. Context Engineering: U-shaped prompt assembly ("Lost in the Middle" mitigation) + citation anchors

Usage:
  # Programmatic
  from src.pipeline import RAGPipeline
  rag = RAGPipeline()
  result = rag.query("What parameters does itertools.batched() accept?", top_k=10)
  print(result["llm_context_prompt"])

  # CLI
  python -m src.pipeline "What are the default tolerances in math.isclose()?"
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import convex_score_fusion, minmax_normalize
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.remote_reranker import RemoteCrossEncoderReranker

# Default paths
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "chunks.jsonl"
DEFAULT_BM25_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "bm25_index.pkl.gz"
DEFAULT_DENSE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "dense_bge_index"
DEFAULT_REMOTE_URL = "https://tropical-module-feet-continuity.trycloudflare.com"


class RAGPipeline:
    """
    Unified end-to-end RAG retrieval pipeline with two-stage hybrid retrieval,
    neural cross-encoder reranking, and context engineering.
    """

    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        bm25_index_path: Path = DEFAULT_BM25_INDEX_PATH,
        dense_index_path: Path = DEFAULT_DENSE_INDEX_PATH,
        remote_url: Optional[str] = DEFAULT_REMOTE_URL,
        mode: str = "auto",  # 'auto', 'remote', or 'local'
        candidate_pool_size: int = 30,  # Proven optimal in Stage 5C
        hybrid_alpha: float = 0.7,      # Proven optimal in Stage 4
        blend_beta: float = 0.6,        # Proven optimal in Stage 5B
        max_chunks_per_section: Optional[int] = 3,  # Prevents section monopolization
        min_chunks: int = 10,           # Baseline minimum context chunks
        max_chunks: int = 16,           # Maximum context cap
        score_floor: float = 0.10,      # Dynamic relevance threshold
        adaptive_slicing: bool = True,  # Enables adaptive score-based context slicing
    ):
        self.chunks_path = Path(chunks_path)
        self.bm25_index_path = Path(bm25_index_path)
        self.dense_index_path = Path(dense_index_path)
        self.remote_url = remote_url
        self.mode = mode
        self.candidate_pool_size = candidate_pool_size
        self.hybrid_alpha = hybrid_alpha
        self.blend_beta = blend_beta
        self.max_chunks_per_section = max_chunks_per_section
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
        self.score_floor = score_floor
        self.adaptive_slicing = adaptive_slicing

        self.chunk_lookup: Dict[str, Dict[str, Any]] = {}
        self.bm25: Optional[BM25Retriever] = None
        self.dense: Optional[DenseRetriever] = None
        self.reranker = None
        self.reranker_backend = "none"

        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Loads corpus chunks, retrieval indexes, and initializes the reranker."""
        print("=" * 70)
        print("INITIALIZING RAG PIPELINE (Python 3.12 Documentation)")
        print("=" * 70)

        # 1. Load corpus chunks
        t0 = time.time()
        print(f"Loading corpus chunks from {self.chunks_path}...")
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                self.chunk_lookup[c["chunk_id"]] = c
        print(f"Loaded {len(self.chunk_lookup)} chunks ({time.time() - t0:.2f}s).")

        # 2. Load BM25 index
        t0 = time.time()
        print(f"Loading BM25 index from {self.bm25_index_path}...")
        self.bm25 = BM25Retriever.load(self.bm25_index_path)
        print(f"BM25 index ready ({time.time() - t0:.2f}s).")

        # 3. Load Dense index
        t0 = time.time()
        print(f"Loading Dense BGE-small index from {self.dense_index_path}...")
        self.dense = DenseRetriever.load(self.dense_index_path)
        print(f"Dense index ready ({time.time() - t0:.2f}s).")

        # 4. Initialize Reranker
        self._setup_reranker()

        print("=" * 70)
        slicing_desc = f"Adaptive [{self.min_chunks}-{self.max_chunks}, floor={self.score_floor}]" if self.adaptive_slicing else "Fixed"
        print(f"PIPELINE READY | Pool: K={self.candidate_pool_size} | Blend: beta={self.blend_beta} | Slicing: {slicing_desc} | Reranker: {self.reranker_backend}")
        print("=" * 70 + "\n")

    def _setup_reranker(self):
        """Connects to remote Kaggle GPU reranker or falls back to local MiniLM."""
        remote_healthy = False
        if self.mode in ("auto", "remote") and self.remote_url:
            try:
                resp = requests.get(f"{self.remote_url.rstrip('/')}/health", timeout=3)
                if resp.status_code == 200:
                    remote_healthy = True
                    print(f"Connected to remote GPU reranker at {self.remote_url}")
            except Exception:
                remote_healthy = False

        if remote_healthy and self.mode in ("auto", "remote"):
            self.reranker = RemoteCrossEncoderReranker(api_url=self.remote_url)
            self.reranker_backend = f"Remote GPU (BAAI/bge-reranker-large on Tesla T4)"
        else:
            if self.mode == "remote":
                raise ConnectionError(f"Could not reach remote reranker at {self.remote_url}")
            print("Remote GPU not connected. Loading local reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
            self.reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            self.reranker_backend = f"Local CPU/MPS (ms-marco-MiniLM-L-6-v2)"

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_chunks: Optional[int] = None,
        max_chunks: Optional[int] = None,
        score_floor: Optional[float] = None,
        adaptive: Optional[bool] = None,
        max_chunks_per_section: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Executes two-stage retrieval:
          1. First-Stage: Hybrid retrieval (Top-30)
          2. Second-Stage: Cross-Encoder neural reranking with beta=0.6 score blending
          3. Diversity Deduplication: Cap chunks from the same document section (default: max 3)
          4. Adaptive Score Slicing: Dynamically selects context size (min 10, max 16, score_floor >= 0.10)
             or exact top_k if specified / adaptive=False.

        Returns:
          Tuple of (selected_chunks, timing_dict).
        """
        timings = {}

        # 1. First-Stage Hybrid Retrieval
        t_stage1 = time.time()
        bm25_cands = self.bm25.retrieve(query, top_k=self.candidate_pool_size)
        dense_cands = self.dense.retrieve(query, top_k=self.candidate_pool_size)
        hybrid_cands = convex_score_fusion(
            bm25_candidates=bm25_cands,
            dense_candidates=dense_cands,
            alpha=self.hybrid_alpha,
            top_k=self.candidate_pool_size,
        )
        timings["stage1_hybrid_ms"] = round((time.time() - t_stage1) * 1000, 2)

        # 2. Second-Stage Cross-Encoder Reranking
        t_stage2 = time.time()
        passages = [
            self.chunk_lookup.get(c["chunk_id"], {}).get("searchable_text", c.get("preview", ""))
            for c in hybrid_cands
        ]

        # Extract raw cross-encoder scores
        if isinstance(self.reranker, RemoteCrossEncoderReranker):
            payload = {"query": query, "passages": passages}
            resp = requests.post(f"{self.reranker.api_url}/rerank", json=payload, timeout=self.reranker.timeout)
            resp.raise_for_status()
            raw_ce_scores = np.array(resp.json()["scores"], dtype=np.float32)
        else:
            pairs = [(query, p) for p in passages]
            raw_ce_scores = self.reranker._load_model().predict(
                pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            raw_ce_scores = np.asarray(raw_ce_scores, dtype=np.float32).reshape(-1)

        # 3. Normalized Score Blending (beta = 0.6)
        ce_norm = np.array(minmax_normalize(raw_ce_scores.tolist()), dtype=np.float32)
        orig_raw = [float(c.get("score", 0.0)) for c in hybrid_cands]
        orig_norm = np.array(minmax_normalize(orig_raw), dtype=np.float32)
        final_scores = self.blend_beta * ce_norm + (1.0 - self.blend_beta) * orig_norm

        # 4. Sort and select candidates (with Section Deduplication and Adaptive Score Slicing)
        sorted_indices = sorted(range(len(hybrid_cands)), key=lambda idx: final_scores[idx], reverse=True)
        timings["stage2_rerank_ms"] = round((time.time() - t_stage2) * 1000, 2)
        timings["total_latency_ms"] = round(timings["stage1_hybrid_ms"] + timings["stage2_rerank_ms"], 2)

        eff_max_sec = max_chunks_per_section if max_chunks_per_section is not None else self.max_chunks_per_section
        use_adaptive = adaptive if adaptive is not None else (top_k is None and self.adaptive_slicing)
        eff_min = min_chunks if min_chunks is not None else self.min_chunks
        eff_max = max_chunks if max_chunks is not None else self.max_chunks
        eff_floor = score_floor if score_floor is not None else self.score_floor

        section_counts = defaultdict(int)
        selected_indices = []
        for idx in sorted_indices:
            cand = hybrid_cands[idx]
            cid = cand["chunk_id"]
            chunk_rec = self.chunk_lookup.get(cid, {})
            crumbs = tuple(chunk_rec.get("breadcrumbs", cand.get("breadcrumbs", [])))
            doc = chunk_rec.get("relative_path", cand.get("doc_path", ""))
            sec_key = (doc, crumbs)

            # Diversity check: skip if section already reached max allowed
            if eff_max_sec is not None and section_counts[sec_key] >= eff_max_sec:
                continue

            # Cutoff check: Fixed vs Adaptive
            if not use_adaptive:
                target_k = top_k if top_k is not None else eff_min
                if len(selected_indices) >= target_k:
                    break
            else:
                # 1. Guarantee minimum chunks (eff_min)
                # 2. Beyond eff_min, include only if score >= eff_floor, up to eff_max
                if len(selected_indices) >= eff_min:
                    if len(selected_indices) >= eff_max or final_scores[idx] < eff_floor:
                        break

            section_counts[sec_key] += 1
            selected_indices.append(idx)

        ranked_results = []
        for rank, idx in enumerate(selected_indices, start=1):
            cand = hybrid_cands[idx]
            cid = cand["chunk_id"]
            chunk_record = self.chunk_lookup.get(cid, {})

            # Version added extraction
            version_added = ""
            for vd in chunk_record.get("version_directives", []):
                if vd.get("directive_type") == "added":
                    version_added = vd.get("target_version", "")
                    break

            entry = {
                "rank": rank,
                "chunk_id": cid,
                "final_score": round(float(final_scores[idx]), 6),
                "ce_score": round(float(raw_ce_scores[idx]), 6),
                "hybrid_score": round(float(cand.get("score", 0.0)), 6),
                "doc_path": chunk_record.get("relative_path", cand.get("doc_path", "")),
                "doc_title": chunk_record.get("heading", cand.get("doc_title", "")),
                "breadcrumbs": chunk_record.get("breadcrumbs", cand.get("breadcrumbs", [])),
                "version_added": version_added,
                "text": chunk_record.get("searchable_text", cand.get("preview", "")),
                "preview": cand.get("preview", ""),
            }
            ranked_results.append(entry)

        return ranked_results, timings

    def format_context(
        self,
        chunks: List[Dict[str, Any]],
        order_strategy: str = "u_shaped",
    ) -> str:
        """
        Formats retrieved chunks into clean, citation-ready XML blocks.
        Implements 'Lost in the Middle' U-shaped ordering:
          - Best chunk placed at top.
          - Second-best chunk placed at bottom.
          - Intermediate chunks placed in middle.
        """
        if not chunks:
            return "No relevant documentation found."

        ordered = list(chunks)
        if order_strategy == "u_shaped" and len(chunks) >= 3:
            # Reorder: [1, 3, 5, ..., 6, 4, 2]
            left, right = [], []
            for i, c in enumerate(chunks):
                if i % 2 == 0:
                    left.append(c)
                else:
                    right.insert(0, c)
            ordered = left + right

        formatted_blocks = []
        for c in ordered:
            crumbs = " > ".join(c.get("breadcrumbs", [])) or c.get("doc_title", "")
            ver = f" | Version Added: {c['version_added']}" if c.get("version_added") else ""
            block = (
                f'<document index="{c["rank"]}" id="{c["chunk_id"]}" path="{c["doc_path"]}">\n'
                f'  <metadata>Section: {crumbs}{ver}</metadata>\n'
                f'  <content>\n{c["text"].strip()}\n  </content>\n'
                f'</document>'
            )
            formatted_blocks.append(block)

        context_str = (
            "You are an expert Python 3.12 documentation assistant. Answer the user's question using ONLY "
            "the official documentation excerpts below. Cite your source documents using their path or document index.\n\n"
            "<official_python_312_documentation>\n"
            + "\n\n".join(formatted_blocks)
            + "\n</official_python_312_documentation>"
        )
        return context_str

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        min_chunks: Optional[int] = None,
        max_chunks: Optional[int] = None,
        score_floor: Optional[float] = None,
        adaptive: Optional[bool] = None,
        order_strategy: str = "u_shaped",
        max_chunks_per_section: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end query function:
          - Runs two-stage retrieval (Hybrid Top-30 -> Reranker -> Adaptive Slicing)
          - Deduplicates over-represented sections (max 3 chunks/section)
          - Dynamically sizes context (10 to 16 chunks based on score floor)
          - Formats citation-ready context for the LLM
          - Returns structured response with timings and chunk metadata
        """
        chunks, timings = self.retrieve(
            query_text,
            top_k=top_k,
            min_chunks=min_chunks,
            max_chunks=max_chunks,
            score_floor=score_floor,
            adaptive=adaptive,
            max_chunks_per_section=max_chunks_per_section,
        )
        context_prompt = self.format_context(chunks, order_strategy=order_strategy)

        return {
            "query": query_text,
            "num_chunks": len(chunks),
            "top_k": top_k,
            "timings": timings,
            "reranker_backend": self.reranker_backend,
            "chunks": chunks,
            "llm_context_prompt": context_prompt,
        }


def main():
    parser = argparse.ArgumentParser(description="Python 3.12 RAG Production Retrieval Pipeline")
    parser.add_argument("query", type=str, help="Python technical question or search query")
    parser.add_argument("--top-k", type=int, default=None, help="Fixed number of chunks (disables adaptive slicing if set)")
    parser.add_argument("--mode", type=str, choices=["auto", "remote", "local"], default="auto", help="Reranker mode")
    parser.add_argument("--show-prompt", action="store_true", help="Print the generated LLM context prompt")

    args = parser.parse_args()

    pipeline = RAGPipeline(mode=args.mode)
    res = pipeline.query(args.query, top_k=args.top_k)

    print("\n" + "=" * 80)
    print(f"QUERY: {res['query']}")
    print(f"BACKEND: {res['reranker_backend']}")
    print(f"TIMINGS: Stage 1 (Hybrid): {res['timings']['stage1_hybrid_ms']}ms | Stage 2 (Rerank): {res['timings']['stage2_rerank_ms']}ms | Total: {res['timings']['total_latency_ms']}ms")
    print("=" * 80)

    for c in res["chunks"]:
        crumbs = " > ".join(c.get("breadcrumbs", [])) or c.get("doc_title", "")
        print(f"\n[Rank #{c['rank']}] Score: {c['final_score']} (CE: {c['ce_score']} | Hyb: {c['hybrid_score']})")
        print(f"Doc: {c['doc_path']} ({crumbs})")
        print(f"ID:  {c['chunk_id']}")
        preview_lines = c["text"].strip().split("\n")[:4]
        for line in preview_lines:
            print(f"  {line}")

    if args.show_prompt:
        print("\n" + "=" * 80)
        print("GENERATED LLM CONTEXT PROMPT:")
        print("=" * 80)
        print(res["llm_context_prompt"])


if __name__ == "__main__":
    main()
