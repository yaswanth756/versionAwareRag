"""
tests/test_evaluation_dataset.py
--------------------------------
Automated invariant verification for the Grounded RAG Evaluation Dataset.
Verifies JSONL validity, ID uniqueness, chunk referential integrity,
verbatim evidence substring containment, split orthogonality, and version labels.
"""

import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "chunks.jsonl"


class TestEvaluationDataset(unittest.TestCase):
    """Test suite for evaluation dataset integrity."""

    @classmethod
    def setUpClass(cls):
        cls.chunks = {}
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                cls.chunks[c["chunk_id"]] = c

        cls.dev_records = []
        with open(EVAL_DIR / "dev.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                cls.dev_records.append(json.loads(line))

        cls.test_records = []
        with open(EVAL_DIR / "test.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                cls.test_records.append(json.loads(line))

        cls.all_records = cls.dev_records + cls.test_records

    def test_record_counts_and_splits(self):
        """Verify total question count and 70/30 split."""
        self.assertEqual(len(self.all_records), 200)
        self.assertEqual(len(self.dev_records), 140)
        self.assertEqual(len(self.test_records), 60)

    def test_query_id_uniqueness(self):
        """Verify all query IDs are unique across both dev and test."""
        qids = [r["query_id"] for r in self.all_records]
        self.assertEqual(len(qids), len(set(qids)))

    def test_question_uniqueness(self):
        """Verify no duplicate or empty questions exist."""
        questions = [r["question"].strip().lower() for r in self.all_records]
        self.assertEqual(len(questions), len(set(questions)))
        for q in questions:
            self.assertTrue(len(q) > 0)

    def test_no_data_leakage(self):
        """Verify strict disjointness between development and test sets."""
        dev_ids = {r["query_id"] for r in self.dev_records}
        test_ids = {r["query_id"] for r in self.test_records}
        self.assertEqual(len(dev_ids.intersection(test_ids)), 0)

        dev_questions = {r["question"].strip().lower() for r in self.dev_records}
        test_questions = {r["question"].strip().lower() for r in self.test_records}
        self.assertEqual(len(dev_questions.intersection(test_questions)), 0)

    def test_category_coverage(self):
        """Verify representation of all 7 mandatory evaluation categories."""
        expected_categories = {
            "exact_api_lookup",
            "conceptual",
            "code_example",
            "version_specific",
            "troubleshooting",
            "multi_document",
            "ambiguous_terminology",
        }
        all_categories = {r["category"] for r in self.all_records}
        self.assertEqual(all_categories, expected_categories)

        # Check that both dev and test have all 7 categories
        self.assertEqual({r["category"] for r in self.dev_records}, expected_categories)
        self.assertEqual({r["category"] for r in self.test_records}, expected_categories)

    def test_chunk_referential_integrity(self):
        """Verify that every referenced chunk ID exists in chunks.jsonl."""
        for r in self.all_records:
            qid = r["query_id"]
            self.assertTrue(len(r["relevant_chunks"]) > 0, f"{qid} has no relevant chunks")
            for rc in r["relevant_chunks"]:
                cid = rc["chunk_id"]
                self.assertIn(cid, self.chunks, f"Chunk {cid} referenced in {qid} does not exist in corpus")

    def test_evidence_verbatim_containment(self):
        """Verify that all evidence text passages exist verbatim in referenced chunk searchable text."""
        for r in self.all_records:
            qid = r["query_id"]
            for rc in r["relevant_chunks"]:
                cid = rc["chunk_id"]
                evidence = rc["evidence_text"]
                self.assertTrue(bool(evidence.strip()), f"Empty evidence text in {qid} for chunk {cid}")
                chunk_text = self.chunks[cid]["searchable_text"]
                self.assertIn(
                    evidence,
                    chunk_text,
                    f"Evidence text in {qid} not found in chunk {cid} searchable text",
                )

    def test_relevance_scores_and_metadata(self):
        """Verify validity of relevance scores, python versions, and answers."""
        for r in self.all_records:
            qid = r["query_id"]
            self.assertTrue(bool(r["expected_answer"].strip()), f"Empty expected answer in {qid}")
            for rc in r["relevant_chunks"]:
                cid = rc["chunk_id"]
                score = rc["relevance_score"]
                self.assertIn(score, (0, 1, 2, 3), f"Invalid relevance score {score} in {qid}")
                self.assertEqual(rc["python_version"], "3.12", f"Mismatched version in {qid}")
                self.assertEqual(
                    rc["doc_path"],
                    self.chunks[cid]["relative_path"],
                    f"Mismatched doc_path in {qid} for chunk {cid}",
                )

    def test_metadata_file_integrity(self):
        """Verify metadata.json and validation_report.json exist and match."""
        meta_file = EVAL_DIR / "metadata.json"
        report_file = EVAL_DIR / "validation_report.json"
        self.assertTrue(meta_file.exists())
        self.assertTrue(report_file.exists())

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["total_questions"], 200)
        self.assertEqual(meta["random_seed"], 42)
        self.assertEqual(meta["split_counts"]["dev"], 140)
        self.assertEqual(meta["split_counts"]["test"], 60)

        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["statistics"]["failed_checks_count"], 0)


if __name__ == "__main__":
    unittest.main()

