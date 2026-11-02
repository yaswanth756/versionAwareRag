"""
src/config.py
-------------
Centralized configuration management for the Version-Aware RAG pipeline.
Defines directory layouts, version targets, and chunking/retrieval parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# Dynamically discover project root: two levels up from src/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PathConfig:
    """Immutable project filesystem layout."""
    root_dir: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    indexes_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "indexes")
    evaluation_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "evaluation")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not exist."""
        for path in [
            self.processed_data_dir,
            self.indexes_dir,
            self.evaluation_dir,
            self.reports_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class VersionConfig:
    """Version metadata and targeting settings."""
    primary_version: str = "3.12"
    supported_versions: Tuple[str, ...] = ("3.10", "3.11", "3.12")
    raw_html_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "data"
        / "raw"
        / "python-3.12"
        / "python-3.12-docs-html"
    )


@dataclass(frozen=True)
class ChunkingConfig:
    """Hyperparameters for structural document chunking."""
    target_chunk_tokens: int = 500
    soft_ceiling_tokens: int = 525
    chunk_overlap_tokens: int = 60
    min_chunk_tokens: int = 40       # Drop meaningless micro-chunks (nav bars, headers)
    preserve_code_blocks: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval and reranking hyperparameters."""
    top_k_candidates: int = 15      # Stage 1: Initial recall set (BM25 + Dense)
    top_k_final: int = 4            # Stage 2: Context passed to generator after reranking
    rrf_k: int = 60                 # Reciprocal Rank Fusion constant
    similarity_threshold: float = 0.65


@dataclass(frozen=True)
class PipelineConfig:
    """Master pipeline configuration combining all modules."""
    paths: PathConfig = field(default_factory=PathConfig)
    version: VersionConfig = field(default_factory=VersionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


# Global default instance for easy imports across src/
config = PipelineConfig()


if __name__ == "__main__":
    # Smoke test: Validate paths and ensure runtime folders exist
    config.paths.ensure_directories()
    print(" Configuration initialized successfully!")
    print(f"• Project Root:     {config.paths.root_dir}")
    print(f"• Raw HTML Source:  {config.version.raw_html_dir}")
    print(f"• Raw Dir Exists?   {config.version.raw_html_dir.exists()}")
    print(f"• Primary Version:  {config.version.primary_version}")
    print(f"• Chunk Target:     {config.chunking.target_chunk_tokens} tokens")