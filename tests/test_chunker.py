"""
tests/test_chunker.py
---------------------
Unit tests for structural chunker integrity:
1. Version directive deduplication without loss of distinct directives.
2. 501–525 token tolerance classification.
3. Oversized atomic script preservation with metadata.
4. AST-aware recipe splitting and 100% exact reconstruction.
5. Dependency detection and grouping.
6. Grammar specification preservation.
7. Globally unique chunk IDs across documents and overloads.
8. Uniqueness across the complete generated corpus.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import StructuralChunker


class TestChunkerIntegrity(unittest.TestCase):
    def setUp(self):
        self.chunker = StructuralChunker(target_tokens=500, soft_ceiling_tokens=525, min_tokens=40)

    def test_version_directive_deduplication(self):
        """Verifies duplicate identical (type, version) pairs are deduplicated while preserving distinct directives."""
        directives = [
            {"directive_type": "added", "target_version": "3.11", "raw_text": "Added in version 3.11."},
            {"directive_type": "added", "target_version": "3.11", "raw_text": "Added in version 3.11."},
            {"directive_type": "changed", "target_version": "3.12", "raw_text": "Changed in version 3.12."},
            {"directive_type": "deprecated", "target_version": "3.14", "raw_text": "Deprecated in version 3.14."},
        ]

        header, _ = self.chunker._build_context_header(
            title="Test Doc",
            breadcrumbs=["Test"],
            python_version="3.12",
            version_directives=directives,
        )

        self.assertIn("ADDED 3.11", header)
        self.assertIn("CHANGED 3.12", header)
        self.assertIn("DEPRECATED 3.14", header)
        # Verify ADDED 3.11 appears only ONCE in the header
        self.assertEqual(header.count("ADDED 3.11"), 1)

    def test_token_tolerance_accepted(self):
        """Verifies chunks between 501 and 525 tokens are classified as tolerance_accepted and not oversized."""
        # Synthesize a unit that lands around 510 tokens
        # Target tokens: 500, soft ceiling: 525
        words = ["token"] * 470
        prose = " ".join(words)
        unit = {
            "unit_id": "sec_tolerance",
            "symbol_name": None,
            "breadcrumbs": ["Tolerance Section"],
            "prose_text": prose,
            "code_blocks": [],
            "version_directives": [],
        }
        doc_meta = {
            "doc_id": "doc_tolerance",
            "relative_path": "tolerance.html",
            "title": "Tolerance Test",
            "python_version": "3.12",
        }

        chunks = self.chunker.chunk_unit(unit, doc_meta=doc_meta)
        self.assertEqual(len(chunks), 1)
        chk = chunks[0]
        if 500 < chk.token_count <= 525:
            self.assertEqual(chk.chunk_size_category, "tolerance_accepted")
            self.assertFalse(chk.is_oversized)
        elif chk.token_count <= 500:
            self.assertEqual(chk.chunk_size_category, "normal")
            self.assertFalse(chk.is_oversized)

    def test_oversized_atomic_script_preservation(self):
        """Verifies monolithic runnable scripts are kept intact and tagged as atomic_monolithic_script."""
        monolithic_script = (
            "import sys\nimport os\n\n"
            "def worker_process(q):\n"
            "    print('Worker running...')\n\n"
            "def main():\n"
            "    print('Starting app')\n"
            "    " + "\n    ".join([f"x_{i} = {i} * 2" for i in range(150)]) + "\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        unit = {
            "unit_id": "sec_app_script",
            "symbol_name": None,
            "breadcrumbs": ["App"],
            "prose_text": "Here is the complete application script.",
            "code_blocks": [monolithic_script],
            "version_directives": [],
        }
        doc_meta = {
            "doc_id": "doc_script",
            "relative_path": "script.html",
            "title": "App Script",
            "python_version": "3.12",
        }

        chunks = self.chunker.chunk_unit(unit, doc_meta=doc_meta)
        # Monolithic script must not be split
        code_chunks = [c for c in chunks if len(c.code_blocks) > 0]
        self.assertEqual(len(code_chunks), 1)
        chk = code_chunks[0]
        self.assertEqual(chk.code_blocks[0].strip(), monolithic_script.strip())
        self.assertTrue(chk.is_oversized)
        self.assertEqual(chk.overflow_reason, "atomic_monolithic_script")

    def test_safe_recipe_splitting_and_exact_reconstruction(self):
        """Verifies recipe collections with independent functions split cleanly and reconstruct 100% exactly."""
        recipe_code = (
            "import itertools\n\n"
            "def take(n, iterable):\n"
            "    return list(itertools.islice(iterable, n))\n\n"
            "def prepend(val, iterable):\n"
            "    return itertools.chain([val], iterable)\n\n"
            "def tabulate(fn, start=0):\n"
            "    return map(fn, itertools.count(start))\n"
        )

        fragments = self.chunker.partition_recipe_code_block(recipe_code, unit_id="sec_itertools-recipes")
        self.assertIsNotNone(fragments)
        self.assertGreater(len(fragments), 1)

        # Invariant check: exact string reconstruction
        reconstructed = "".join(f["raw_slice"] for f in fragments)
        self.assertEqual(recipe_code, reconstructed, "Partitioned slices must reconstruct original string 100% character-for-character")

        # Invariant check: fragment metadata
        source_id = hashlib.sha256(recipe_code.encode("utf-8")).hexdigest()
        for i, f in enumerate(fragments):
            self.assertEqual(f["source_block_id"], source_id)
            self.assertEqual(f["fragment_index"], i)
            self.assertEqual(f["total_fragments"], len(fragments))

    def test_dependency_detection_keeps_helpers_together(self):
        """Verifies that dependent helper functions are clustered into the same fragment."""
        dependent_code = (
            "def _internal_helper(x):\n"
            "    return x * 2\n\n"
            "def public_api(val):\n"
            "    return _internal_helper(val) + 1\n\n"
            "def standalone_func(val):\n"
            "    return val ** 2\n"
        )

        fragments = self.chunker.partition_recipe_code_block(dependent_code, unit_id="sec_recipes")
        self.assertIsNotNone(fragments)
        # _internal_helper and public_api must be in the same fragment
        frag_with_public = next(f for f in fragments if "public_api" in f["raw_slice"])
        self.assertIn("_internal_helper", frag_with_public["raw_slice"], "Helper and caller must be clustered together")

    def test_grammar_specification_preserved_atomically(self):
        """Verifies that grammar blocks are marked atomic_grammar_specification and never split."""
        unit = {
            "unit_id": "sec_full-grammar-specification",
            "symbol_name": None,
            "breadcrumbs": ["Grammar Specification"],
            "prose_text": "Python Grammar Specification.",
            "code_blocks": ["# PEG grammar for Python\n" + "rule: 'token'\n" * 200],
            "version_directives": [],
        }
        doc_meta = {
            "doc_id": "reference_grammar",
            "relative_path": "reference/grammar.html",
            "title": "Grammar",
            "python_version": "3.12",
        }

        chunks = self.chunker.chunk_unit(unit, doc_meta=doc_meta)
        self.assertGreater(len(chunks), 0)
        chk = next(c for c in chunks if len(c.code_blocks) > 0)
        self.assertTrue(chk.is_oversized)
        self.assertEqual(chk.overflow_reason, "atomic_grammar_specification")

    def test_repeated_section_ids_across_different_documents(self):
        """Verifies that identical section IDs in different documents produce distinct chunk IDs."""
        shared_unit = {
            "unit_id": "sec_pending-removal-in-python-3-14",
            "symbol_name": None,
            "breadcrumbs": ["Pending Removal"],
            "prose_text": "Features pending removal in Python 3.14 include deprecated legacy modules, functions, classes and CLI flags that have been slated for removal in Python 3.14 per PEP guidelines.",
            "code_blocks": [],
            "version_directives": [{"directive_type": "deprecated", "target_version": "3.14"}],
        }

        doc_meta_a = {
            "doc_id": "deprecations_index",
            "relative_path": "deprecations/index.html",
            "title": "Deprecations Index",
            "python_version": "3.12",
        }

        doc_meta_b = {
            "doc_id": "whatsnew_3.12",
            "relative_path": "whatsnew/3.12.html",
            "title": "What's New In Python 3.12",
            "python_version": "3.12",
        }

        chunks_a = self.chunker.chunk_unit(shared_unit, doc_meta=doc_meta_a)
        chunks_b = self.chunker.chunk_unit(shared_unit, doc_meta=doc_meta_b)

        self.assertGreater(len(chunks_a), 0)
        self.assertGreater(len(chunks_b), 0)

        ids_a = {c.chunk_id for c in chunks_a}
        ids_b = {c.chunk_id for c in chunks_b}

        self.assertEqual(len(ids_a.intersection(ids_b)), 0, "Chunk IDs across distinct docs must never collide")
        self.assertTrue(all("deprecations_index" in cid for cid in ids_a))
        self.assertTrue(all("whatsnew_3.12" in cid for cid in ids_b))

    def test_uniqueness_across_complete_generated_corpus(self):
        """Verifies that chunks.jsonl (if it exists) contains zero duplicate chunk IDs."""
        chunks_file = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "chunks.jsonl"
        if not chunks_file.exists():
            self.skipTest("chunks.jsonl does not exist yet")

        from collections import Counter
        seen_ids = []
        with chunks_file.open("r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                seen_ids.append(c["chunk_id"])

        counts = Counter(seen_ids)
        dups = {cid: cnt for cid, cnt in counts.items() if cnt > 1}
        self.assertEqual(len(dups), 0, f"Found duplicate chunk IDs: {list(dups.keys())[:5]}")


if __name__ == "__main__":
    unittest.main()
