"""
src/chunker.py
--------------
Production-grade structural chunker with:
1. Stage 1 Context Header Optimization: Deduplicates identical (type, version) pairs.
2. Stage 2 Token Budget Policy: 500-token normal target, 525-token soft ceiling, explicit category metadata.
3. Stage 3 AST-Aware Safe Recipe Splitting: Checks dependencies, preserves monolithic scripts intact,
   and records source_block_id / fragment metadata for 100% exact reconstruction.
4. Stage 4 Grammar Preservation: Preserves PEG grammar specifications atomically.
5. Globally unique chunk IDs: {doc_id}::{unit_id}::{idx}.
"""

import ast
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config


@dataclass
class DocumentChunk:
    """Production chunk representation ready for dense vector and sparse BM25 indexing."""
    chunk_id: str
    doc_id: str
    relative_path: str
    symbol_name: Optional[str]
    heading: str
    breadcrumbs: List[str]
    python_version: str
    version_directives: List[Dict[str, str]]
    code_blocks: List[str]
    searchable_text: str
    raw_content: str
    token_count: int
    # Classification metadata
    chunk_size_category: str = "normal"  # "normal" (<=500), "tolerance_accepted" (501-525), "intentionally_oversized" (>525)
    is_oversized: bool = False
    overflow_reason: Optional[str] = None
    # Fragment reconstruction metadata (for split recipes)
    source_block_id: Optional[str] = None
    fragment_index: Optional[int] = None
    total_fragments: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StructuralChunker:
    """High-precision, production-grade chunker for technical documentation."""

    def __init__(
        self,
        target_tokens: int = config.chunking.target_chunk_tokens,
        soft_ceiling_tokens: int = config.chunking.soft_ceiling_tokens,
        min_tokens: int = config.chunking.min_chunk_tokens,
        tokenizer_model: str = "cl100k_base",
    ):
        self.target_tokens = target_tokens
        self.soft_ceiling_tokens = soft_ceiling_tokens
        self.min_tokens = min_tokens
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Exact BPE token count."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, disallowed_special=()))

    @staticmethod
    def deduplicate_breadcrumbs(breadcrumbs: List[str]) -> List[str]:
        """Remove sequential duplicate breadcrumb items."""
        if not breadcrumbs:
            return []
        cleaned = []
        for crumb in breadcrumbs:
            crumb_clean = crumb.strip()
            if not crumb_clean:
                continue
            if not cleaned or cleaned[-1].lower() != crumb_clean.lower():
                cleaned.append(crumb_clean)
        return cleaned

    def _build_context_header(
        self,
        title: str,
        breadcrumbs: List[str],
        python_version: str,
        version_directives: List[Dict[str, str]],
        signature: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Constructs context header and returns (header_string, header_token_count).
        Stage 1: Deduplicates only identical (type, version) pairs without losing distinct directives.
        """
        header_parts = [f"[Document: {title} | Python Version: {python_version}]"]

        clean_crumbs = self.deduplicate_breadcrumbs(breadcrumbs)
        if clean_crumbs:
            header_parts.append(f"[Path: {' > '.join(clean_crumbs)}]")

        if version_directives:
            seen_directives: Set[Tuple[str, str]] = set()
            unique_directive_strs = []
            for d in version_directives:
                if isinstance(d, dict) and "directive_type" in d and "target_version" in d:
                    key = (d["directive_type"].lower().strip(), d["target_version"].strip())
                    if key not in seen_directives:
                        seen_directives.add(key)
                        unique_directive_strs.append(f"{d['directive_type'].upper()} {d['target_version']}")
            if unique_directive_strs:
                header_parts.append(f"[Version Directives: {', '.join(unique_directive_strs)}]")

        if signature:
            header_parts.append(f"Signature: {signature}")

        header_str = "\n".join(header_parts) + "\n\n"
        return header_str, self.count_tokens(header_str)

    def _split_oversized_prose(self, text: str, max_tokens: int) -> List[str]:
        """Splits an oversized prose paragraph using sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sub_paras: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            s_tokens = self.count_tokens(sentence)

            if s_tokens > max_tokens:
                if current_chunk:
                    sub_paras.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                chars_per_token = max(1, len(sentence) // s_tokens)
                slice_size = max_tokens * chars_per_token
                for i in range(0, len(sentence), slice_size):
                    sub_paras.append(sentence[i : i + slice_size])
                continue

            if current_tokens + s_tokens > max_tokens and current_chunk:
                sub_paras.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = s_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += s_tokens

        if current_chunk:
            sub_paras.append(" ".join(current_chunk))

        return sub_paras

    def partition_recipe_code_block(
        self,
        code: str,
        unit_id: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Stage 3: AST-aware partitioner for recipe collections.
        Checks dependencies, verifies exact reconstruction, and falls back to atomic preservation.
        """
        # Stage 4: Do not split grammar specifications
        if "grammar" in unit_id or "sec_full-grammar" in unit_id or "sec_abstract-grammar" in unit_id:
            return None

        # Do not split scripts with main entrypoints
        if "if __name__ == '__main__':" in code or 'if __name__ == "__main__":' in code or "def main(" in code:
            return None

        try:
            tree = ast.parse(code)
        except Exception:
            # GDB trace, CLI output, REPL output, or invalid Python: keep atomic
            return None

        defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if len(defs) <= 1:
            return None

        # Detect inter-function dependencies
        def_names = {d.name: d for d in defs}
        dep_graph: Dict[str, Set[str]] = {}
        for d in defs:
            refs = {n.id for n in ast.walk(d) if isinstance(n, ast.Name)}
            dep_graph[d.name] = refs.intersection(def_names.keys()) - {d.name}

        # Cluster definitions by dependency
        clusters: List[List[ast.AST]] = []
        assigned: Set[str] = set()

        for d in defs:
            if d.name in assigned:
                continue
            # Build cluster of this definition and any definition it depends on or depends on it
            cluster = [d]
            assigned.add(d.name)
            # Find all connected definitions in this block
            to_visit = list(dep_graph.get(d.name, set()))
            while to_visit:
                curr_name = to_visit.pop()
                if curr_name in def_names and curr_name not in assigned:
                    assigned.add(curr_name)
                    cluster.append(def_names[curr_name])
                    to_visit.extend(dep_graph.get(curr_name, set()))
            # Sort cluster in original source order
            cluster.sort(key=lambda node: node.lineno)
            clusters.append(cluster)

        # If everything collapsed into one big cluster, we cannot safely partition
        if len(clusters) <= 1:
            return None

        lines = code.splitlines(keepends=True)
        first_def_line = defs[0].lineno - 1
        preamble = "".join(lines[:first_def_line])

        # Group clusters into fragments respecting the target token budget
        fragments_data: List[Dict[str, Any]] = []
        source_block_id = hashlib.sha256(code.encode("utf-8")).hexdigest()

        # Build fragments along contiguous line spans
        cur_line = 0
        for i, cluster in enumerate(clusters):
            # start line is the start of this cluster (or line 0 for the first cluster to include preamble)
            start_line = cur_line
            # end line is the start line of the next cluster, or end of file
            if i + 1 < len(clusters):
                next_cluster_first_def = clusters[i + 1][0]
                end_line = next_cluster_first_def.lineno - 1
            else:
                end_line = len(lines)

            raw_slice = "".join(lines[start_line:end_line])
            cur_line = end_line

            # Display version includes shared preamble (imports) if not already the first fragment
            if i > 0 and preamble.strip():
                display_code = f"{preamble.rstrip()}\n\n{raw_slice.strip()}"
            else:
                display_code = raw_slice.strip()

            fragments_data.append({
                "raw_slice": raw_slice,
                "display_code": display_code,
                "token_count": self.count_tokens(display_code),
            })

        # CRITICAL VALIDATION: Verify exact character-for-character reconstruction invariant
        reconstructed = "".join(f["raw_slice"] for f in fragments_data)
        if reconstructed != code:
            # Safety fallback: if reconstruction fails, keep block atomic!
            return None

        total_frags = len(fragments_data)
        for idx, f in enumerate(fragments_data):
            f["source_block_id"] = source_block_id
            f["fragment_index"] = idx
            f["total_fragments"] = total_frags

        return fragments_data

    def chunk_unit(
        self,
        record: Dict[str, Any],
        doc_meta: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Convert a record into DocumentChunks."""
        if doc_meta is None:
            doc_id = record.get("document_id") or record.get("doc_id", "unknown")
            relative_path = record.get("source_path") or record.get("relative_path", "")
            title = record.get("title") or record.get("heading") or Path(relative_path).stem
            python_version = record.get("python_version", config.version.primary_version)
            unit_id = record.get("unit_id") or f"{doc_id}_chunk"
            symbol_name = record.get("symbol_name")
            signature = record.get("signature")
            breadcrumbs = record.get("breadcrumbs") or [title]
            version_directives = record.get("version_directives", [])
            code_blocks = record.get("code_blocks", [])
            prose_text = record.get("prose_text") or record.get("text", "")
        else:
            doc_id = doc_meta["doc_id"]
            relative_path = doc_meta["relative_path"]
            title = doc_meta["title"]
            python_version = doc_meta["python_version"]
            unit_id = record["unit_id"]
            symbol_name = record.get("symbol_name")
            signature = record.get("signature")
            breadcrumbs = record.get("breadcrumbs") or [title]
            version_directives = record.get("version_directives", [])
            code_blocks = record.get("code_blocks", [])
            prose_text = record.get("prose_text", "")

        context_header, header_tokens = self._build_context_header(
            title=title,
            breadcrumbs=breadcrumbs,
            python_version=python_version,
            version_directives=version_directives,
            signature=signature,
        )
        effective_budget = max(100, self.soft_ceiling_tokens - header_tokens)

        # Elements: List of (type, formatted_text, token_count, overflow_reason, frag_meta)
        elements: List[Tuple[str, str, int, Optional[str], Optional[Dict[str, Any]]]] = []

        if prose_text.strip():
            raw_paras = [p.strip() for p in prose_text.split("\n\n") if p.strip()]
            for p in raw_paras:
                p_tokens = self.count_tokens(p)
                if p_tokens > effective_budget:
                    sub_paras = self._split_oversized_prose(p, max_tokens=effective_budget)
                    for sp in sub_paras:
                        elements.append(("prose", sp, self.count_tokens(sp), None, None))
                else:
                    elements.append(("prose", p, p_tokens, None, None))

        for cb in code_blocks:
            cb_str = cb.strip()
            if not cb_str:
                continue
            cb_tokens = self.count_tokens(f"```python\n{cb_str}\n```")

            # Check if this large code block can be safely partitioned as a recipe
            if cb_tokens > self.target_tokens:
                recipe_frags = self.partition_recipe_code_block(cb, unit_id=unit_id)
                if recipe_frags:
                    for rf in recipe_frags:
                        formatted_frag = f"```python\n{rf['display_code']}\n```"
                        frag_tokens = self.count_tokens(formatted_frag)
                        frag_meta = {
                            "source_block_id": rf["source_block_id"],
                            "fragment_index": rf["fragment_index"],
                            "total_fragments": rf["total_fragments"],
                        }
                        elements.append(("code_fragment", formatted_frag, frag_tokens, None, frag_meta))
                    continue

                # If cannot be safely split, determine appropriate overflow_reason
                if "grammar" in unit_id or "sec_full-grammar" in unit_id or "sec_abstract-grammar" in unit_id:
                    overflow_reason = "atomic_grammar_specification"
                elif "if __name__ == '__main__':" in cb or 'if __name__ == "__main__":' in cb or "def main(" in cb:
                    overflow_reason = "atomic_monolithic_script"
                elif cb.startswith("$") or cb.startswith("(gdb)") or cb.startswith(">>>") or cb.startswith("BaseException"):
                    overflow_reason = "atomic_terminal_or_repl_session"
                else:
                    overflow_reason = "atomic_code_example"

                formatted_cb = f"```python\n{cb_str}\n```"
                elements.append(("code", formatted_cb, cb_tokens, overflow_reason, None))
            else:
                formatted_cb = f"```python\n{cb_str}\n```"
                elements.append(("code", formatted_cb, cb_tokens, None, None))

        if not elements:
            return []

        chunks: List[DocumentChunk] = []
        current_elements: List[Tuple[str, str, int, Optional[str], Optional[Dict[str, Any]]]] = []
        current_tokens = 0
        part_idx = 0

        def flush_chunk(
            items: List[Tuple[str, str, int, Optional[str], Optional[Dict[str, Any]]]],
            idx: int,
        ) -> Optional[DocumentChunk]:
            if not items:
                return None
            body_text = "\n\n".join(it[1] for it in items)
            chunk_codes = [
                it[1].replace("```python\n", "").replace("\n```", "")
                for it in items
                if it[0] in ("code", "code_fragment")
            ]
            searchable_text = context_header + body_text
            total_tokens = self.count_tokens(searchable_text)

            # Stage 2: Token budget classification
            if total_tokens <= self.target_tokens:
                category = "normal"
                is_oversized = False
                reason = None
            elif total_tokens <= self.soft_ceiling_tokens:
                category = "tolerance_accepted"
                is_oversized = False
                reason = None
            else:
                category = "intentionally_oversized"
                is_oversized = True
                reasons = [it[3] for it in items if it[3]]
                if reasons:
                    reason = reasons[0]
                elif any("grammar" in c.lower() for c in breadcrumbs) or "grammar" in unit_id:
                    reason = "atomic_grammar_specification"
                elif chunk_codes:
                    reason = "atomic_monolithic_code"
                else:
                    reason = "oversized_prose_block"

            frag_items = [it for it in items if it[0] == "code_fragment" and it[4]]
            source_block_id = frag_items[0][4].get("source_block_id") if frag_items else None
            fragment_index = frag_items[0][4].get("fragment_index") if frag_items else None
            total_fragments = frag_items[0][4].get("total_fragments") if frag_items else None

            return DocumentChunk(
                chunk_id=f"{doc_id}::{unit_id}::{idx}",
                doc_id=doc_id,
                relative_path=relative_path,
                symbol_name=symbol_name,
                heading=breadcrumbs[-1] if breadcrumbs else title,
                breadcrumbs=self.deduplicate_breadcrumbs(breadcrumbs),
                python_version=python_version,
                version_directives=version_directives,
                code_blocks=chunk_codes,
                searchable_text=searchable_text,
                raw_content=body_text,
                token_count=total_tokens,
                chunk_size_category=category,
                is_oversized=is_oversized,
                overflow_reason=reason,
                source_block_id=source_block_id,
                fragment_index=fragment_index,
                total_fragments=total_fragments,
            )

        for el in elements:
            el_type, el_text, el_tokens, el_reason, el_meta = el
            if current_tokens + el_tokens > effective_budget and current_elements:
                chunk = flush_chunk(current_elements, part_idx)
                if chunk:
                    chunks.append(chunk)
                    part_idx += 1
                current_elements = [el]
                current_tokens = el_tokens
            else:
                current_elements.append(el)
                current_tokens += el_tokens

        if current_elements:
            chunk = flush_chunk(current_elements, part_idx)
            if chunk:
                chunks.append(chunk)

        filtered_chunks: List[DocumentChunk] = []
        for chk in chunks:
            if chk.token_count >= self.min_tokens or chk.symbol_name or chk.version_directives or chk.code_blocks:
                filtered_chunks.append(chk)
            else:
                if filtered_chunks:
                    merged = filtered_chunks[-1]
                    merged.raw_content += f"\n\n{chk.raw_content}"
                    merged.searchable_text = context_header + merged.raw_content
                    merged.token_count = self.count_tokens(merged.searchable_text)
                    if merged.token_count <= self.target_tokens:
                        merged.chunk_size_category = "normal"
                        merged.is_oversized = False
                        merged.overflow_reason = None
                    elif merged.token_count <= self.soft_ceiling_tokens:
                        merged.chunk_size_category = "tolerance_accepted"
                        merged.is_oversized = False
                        merged.overflow_reason = None
                    else:
                        merged.chunk_size_category = "intentionally_oversized"
                        merged.is_oversized = True
                        merged.overflow_reason = "oversized_merged_prose"

        return filtered_chunks

    def process_corpus(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """Stream JSONL documents and produce production-grade chunks."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total_chunks = 0
        total_tokens = 0
        total_code_blocks_retained = 0
        docs_processed = 0

        with input_file.open("r", encoding="utf-8") as in_f, output_file.open("w", encoding="utf-8") as out_f:
            for line in tqdm(in_f, desc="Chunking Documents", unit="doc"):
                doc = json.loads(line)
                docs_processed += 1
                doc_meta = {
                    "doc_id": doc["doc_id"],
                    "relative_path": doc["relative_path"],
                    "title": doc["title"],
                    "python_version": doc["python_version"],
                }

                doc_unit_counts: Dict[str, int] = {}
                for unit in doc.get("units", []):
                    base_unit_id = unit.get("unit_id", "unit")
                    doc_unit_counts[base_unit_id] = doc_unit_counts.get(base_unit_id, 0) + 1
                    occurrence = doc_unit_counts[base_unit_id]

                    unit_to_process = unit
                    if occurrence > 1:
                        unit_to_process = dict(unit)
                        unit_to_process["unit_id"] = f"{base_unit_id}_ovl{occurrence}"

                    chunks = self.chunk_unit(unit_to_process, doc_meta=doc_meta)
                    for chunk in chunks:
                        total_chunks += 1
                        total_tokens += chunk.token_count
                        total_code_blocks_retained += len(chunk.code_blocks)
                        out_f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        return {
            "docs_processed": docs_processed,
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "code_blocks_retained": total_code_blocks_retained,
            "avg_tokens_per_chunk": total_tokens // max(1, total_chunks),
        }


if __name__ == "__main__":
    chunker = StructuralChunker()
    input_file = config.paths.processed_data_dir / "python-3.12" / "parsed_documents.jsonl"
    output_file = config.paths.processed_data_dir / "python-3.12" / "chunks.jsonl"

    print("🚀 Running hardened structural chunker (Step 2B)...")
    stats = chunker.process_corpus(input_file, output_file)

    print("\n✅ Step 2B Chunking Complete!")
    print(f"• Documents Processed:   {stats['docs_processed']}")
    print(f"• Total Chunks Created:  {stats['total_chunks']:,}")
    print(f"• Exact Tokens:          {stats['total_tokens']:,}")
    print(f"• Code Blocks Retained:  {stats['code_blocks_retained']:,}")
    print(f"• Average Chunk Size:    {stats['avg_tokens_per_chunk']} tokens")
    print(f"• Saved To:              {output_file}")
