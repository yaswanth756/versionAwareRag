"""
src/retrieval/dense_retriever.py
--------------------------------
Dense semantic vector retrieval module using SentenceTransformers (BAAI/bge-small-en-v1.5).
Features:
  - Hardware acceleration: auto-selects Apple Silicon MPS (Metal), CUDA, or CPU.
  - BGE query instruction prefixing for asymmetric passage retrieval.
  - Unit L2-normalized embeddings for fast cosine similarity via dot product.
  - Fast vector search using vectorized NumPy matrix-vector multiplication.
  - Serialization: caches pre-computed chunk embeddings for instant subsequent loads.
"""

import os
import gzip
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Force offline mode to use local cached weights immediately without network timeouts
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class DenseRetriever:
    """
    Dense vector retriever using SentenceTransformer models (e.g. BAAI/bge-small-en-v1.5).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 64,
        query_instruction: str = "Represent this sentence for searching relevant passages: ",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.device = device or self._detect_device()

        self.model = None
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, D)
        self.chunk_ids: List[str] = []
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_lookup: Dict[str, Dict[str, Any]] = {}
        self.is_indexed: bool = False

    @staticmethod
    def _detect_device() -> str:
        """Auto-detects the best hardware accelerator (MPS, CUDA, or CPU)."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self):
        """Lazy-loads the SentenceTransformer model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs={"local_files_only": True}
            )

    def index_corpus(
        self,
        chunks: List[Dict[str, Any]],
        text_field: str = "searchable_text",
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> "DenseRetriever":
        """
        Generates vector embeddings for all corpus chunks and stores them.
        """
        self._load_model()
        t0 = time.time()
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunk_lookup = {c["chunk_id"]: c for c in chunks}

        texts = [c.get(text_field, "") for c in chunks]
        bs = batch_size or self.batch_size

        if verbose:
            print(f"Encoding {len(texts)} chunks in batches of {bs} on {self.device}...")

        # Encode passages without query instruction prefix
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=verbose,
            normalize_embeddings=True,  # Unit L2 norm for dot product cosine similarity
            convert_to_numpy=True,
        )

        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.is_indexed = True

        t1 = time.time()
        if verbose:
            print(f"Corpus embeddings generated: shape {self.embeddings.shape} in {t1 - t0:.2f}s")

        return self

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Encodes query with BGE search prefix and computes cosine similarity against all chunks.
        """
        if not self.is_indexed or self.embeddings is None:
            raise RuntimeError("DenseRetriever must be indexed before calling retrieve().")

        self._load_model()

        # Format query with asymmetric instruction prefix for BGE
        formatted_query = f"{self.query_instruction}{query.strip()}"

        # Encode query to normalized 1D vector
        query_embedding = self.model.encode(
            formatted_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

        # Compute cosine similarities via matrix-vector dot product (both unit-normalized)
        # (N, D) @ (D,) -> (N,)
        scores = np.dot(self.embeddings, query_embedding)

        num_candidates = min(top_k, len(scores))
        if num_candidates <= 0:
            return []

        # Fast top-k selection
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
        Batch encodes all queries and retrieves top candidates for each.
        """
        if not self.is_indexed or self.embeddings is None:
            raise RuntimeError("DenseRetriever must be indexed before calling batch_retrieve().")

        self._load_model()
        formatted_queries = [
            f"{self.query_instruction}{q.get(query_text_key, '').strip()}"
            for q in queries
        ]

        # Batch encode queries
        query_embeddings = self.model.encode(
            formatted_queries,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

        # Matrix multiplication: (num_queries, D) @ (D, num_chunks) -> (num_queries, num_chunks)
        all_scores = np.dot(query_embeddings, self.embeddings.T)

        results: Dict[str, List[Dict[str, Any]]] = {}
        for q_idx, q in enumerate(queries):
            qid = q.get("query_id", "")
            scores = all_scores[q_idx]
            num_candidates = min(top_k, len(scores))
            top_indices = np.argsort(-scores)[:num_candidates]

            q_results = []
            for rank, idx in enumerate(top_indices, start=1):
                score = float(scores[idx])
                cid = self.chunk_ids[idx]
                chunk = self.chunks[idx]
                q_results.append({
                    "rank": rank,
                    "chunk_id": cid,
                    "score": round(score, 4),
                    "doc_path": chunk.get("relative_path", ""),
                    "doc_title": chunk.get("doc_title", ""),
                    "breadcrumbs": chunk.get("breadcrumbs", []),
                    "preview": chunk.get("searchable_text", "")[:180].replace("\n", " ").strip(),
                })
            results[qid] = q_results

        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Saves dense index (embeddings, chunk metadata, configuration) to disk.
        Embeddings are saved in compressed numpy format for minimal disk space.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        meta_path = filepath.with_suffix(".meta.pkl.gz")
        embed_path = filepath.with_suffix(".embeddings.npz")

        # Save metadata
        meta = {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "query_instruction": self.query_instruction,
            "chunk_ids": self.chunk_ids,
            "chunks": self.chunks,
            "is_indexed": self.is_indexed,
        }
        with gzip.open(meta_path, "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save embeddings
        if self.embeddings is not None:
            np.savez_compressed(embed_path, embeddings=self.embeddings)

    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[str] = None) -> "DenseRetriever":
        """
        Loads pre-computed embeddings and metadata from disk.
        """
        filepath = Path(filepath)
        meta_path = filepath.with_suffix(".meta.pkl.gz")
        embed_path = filepath.with_suffix(".embeddings.npz")

        if not meta_path.exists() or not embed_path.exists():
            raise FileNotFoundError(f"Dense index files not found at {meta_path} / {embed_path}")

        with gzip.open(meta_path, "rb") as f:
            meta = pickle.load(f)

        retriever = cls(
            model_name=meta["model_name"],
            device=device or meta.get("device"),
            batch_size=meta.get("batch_size", 64),
            query_instruction=meta.get("query_instruction", ""),
        )
        retriever.chunk_ids = meta["chunk_ids"]
        retriever.chunks = meta["chunks"]
        retriever.chunk_lookup = {c["chunk_id"]: c for c in meta["chunks"]}
        retriever.is_indexed = meta["is_indexed"]

        # Load embeddings
        loaded_npz = np.load(embed_path)
        retriever.embeddings = loaded_npz["embeddings"]

        return retriever

