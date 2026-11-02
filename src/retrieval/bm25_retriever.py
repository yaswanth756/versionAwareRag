"""
src/retrieval/bm25_retriever.py
-------------------------------
Lexical BM25 retrieval module customized for Python documentation & code.
Features:
  - Code-aware tokenization: preserves dotted expressions (module.class.func),
    extracts snake_case sub-tokens, and handles programming syntax.
  - Fast indexing and scoring via rank_bm25.BM25Okapi.
  - Serialization / deserialization for cached index loading.
"""

import gzip
import pickle
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize_code_and_text(text: str) -> List[str]:
    """
    Tokenizes technical text and code:
    1. Extracts dotted compound identifiers (e.g., 'itertools.batched', 'os.path.join').
    2. Extracts standard alphanumeric and underscore words.
    3. Decomposes snake_case compound terms into individual subwords.
    """
    text_lower = text.lower()

    # 1. Dotted compound paths (at least 2 segments separated by a period)
    dotted = re.findall(r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]+)+\b", text_lower)

    # 2. Words, identifiers, and numbers
    words = re.findall(r"\b[a-z0-9_]+\b", text_lower)

    # 3. Subwords from snake_case identifiers (length > 3 with underscore)
    subwords = []
    for w in words:
        if "_" in w and len(w) > 3:
            parts = [p for p in w.split("_") if len(p) > 1]
            if len(parts) > 1:
                subwords.extend(parts)

    return words + dotted + subwords


class BM25Retriever:
    """
    BM25 retriever built on rank_bm25.BM25Okapi.
    Indexes Python 3.12 documentation chunks and returns scored top-k candidate chunks.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or tokenize_code_and_text
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_lookup: Dict[str, Dict[str, Any]] = {}
        self.is_indexed: bool = False

    def index_corpus(
        self,
        chunks: List[Dict[str, Any]],
        text_field: str = "searchable_text",
        verbose: bool = False
    ) -> "BM25Retriever":
        """
        Builds the BM25 index over the provided corpus of chunks.
        """
        t0 = time.time()
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunk_lookup = {c["chunk_id"]: c for c in chunks}

        # Tokenize all chunks
        tokenized_corpus = []
        for c in chunks:
            text = c.get(text_field, "")
            tokenized_corpus.append(self.tokenizer(text))

        t1 = time.time()
        if verbose:
            print(f"Tokenized {len(chunks)} chunks in {t1 - t0:.2f}s")

        # Initialize rank_bm25
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.is_indexed = True

        t2 = time.time()
        if verbose:
            print(f"BM25 index built in {t2 - t1:.2f}s (Total: {t2 - t0:.2f}s)")

        return self

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Scores all chunks for the query and returns the top-k highest scoring chunks.
        """
        if not self.is_indexed or self.bm25 is None:
            raise RuntimeError("BM25Retriever must be indexed before calling retrieve().")

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        # Get scores across all chunks
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices sorted in descending order
        num_candidates = min(top_k, len(scores))
        if num_candidates <= 0:
            return []

        # Use argpartition for fast top-k if corpus is large, then sort top slice
        if len(scores) > num_candidates * 4:
            top_part = np.argpartition(scores, -num_candidates)[-num_candidates:]
            top_indices = top_part[np.argsort(-scores[top_part])]
        else:
            top_indices = np.argsort(-scores)[:num_candidates]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            cid = self.chunk_ids[idx]
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "chunk_id": cid,
                "score": round(score, 4),
                "doc_path": chunk.get("relative_path", ""),
                "doc_title": chunk.get("doc_title", ""),
                "breadcrumbs": chunk.get("breadcrumbs", []),
                "preview": chunk.get("searchable_text", "")[:180].replace("\n", " ").strip(),
            })

        return results

    def batch_retrieve(
        self,
        queries: List[Dict[str, Any]],
        top_k: int = 20,
        query_text_key: str = "question"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Executes retrieval for a batch of query records.
        Returns a dict mapping query_id to list of retrieved chunk candidate dicts.
        """
        results: Dict[str, List[Dict[str, Any]]] = {}
        for q in queries:
            qid = q.get("query_id", "")
            q_text = q.get(query_text_key, "")
            results[qid] = self.retrieve(q_text, top_k=top_k)
        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Saves retriever state (BM25Okapi model, chunk metadata, and parameters) to a compressed pickle file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "k1": self.k1,
            "b": self.b,
            "bm25": self.bm25,
            "chunk_ids": self.chunk_ids,
            "chunks": self.chunks,
            "is_indexed": self.is_indexed,
        }
        with gzip.open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: Union[str, Path], tokenizer: Optional[Callable[[str], List[str]]] = None) -> "BM25Retriever":
        """
        Loads a previously saved BM25Retriever from disk.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"BM25 index file not found at {filepath}")

        with gzip.open(filepath, "rb") as f:
            state = pickle.load(f)

        retriever = cls(k1=state["k1"], b=state["b"], tokenizer=tokenizer)
        retriever.bm25 = state["bm25"]
        retriever.chunk_ids = state["chunk_ids"]
        retriever.chunks = state["chunks"]
        retriever.chunk_lookup = {c["chunk_id"]: c for c in state["chunks"]}
        retriever.is_indexed = state["is_indexed"]
        return retriever

