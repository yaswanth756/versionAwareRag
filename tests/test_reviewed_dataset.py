"""
tests/test_reviewed_dataset.py
------------------------------
Validation test suite for the human-quality audited and repaired dev dataset:
  - data/evaluation/dev.reviewed.jsonl
  - data/evaluation/human_review_report.json
  - data/evaluation/review_decisions.jsonl

Verifies:
1. Exact record count (140) and query ID uniqueness.
2. 100% referential integrity with data/processed/python-3.12/chunks.jsonl.
3. Every evidence excerpt is a verbatim substring of referenced chunk searchable_text.
4. No duplicate chunk IDs within any query's relevant_chunk_ids.
5. All relevance scores are valid integers (0, 1, 2, or 3).
6. Non-empty questions and expected answers.
7. Original dev.jsonl and test.jsonl are completely untouched (matching SHA-256 hashes).
8. 140 review decisions matching the dev records 1-to-1.
9. Q001 retains human decision ACCEPT.
"""

import json
import hashlib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "python-3.12" / "chunks.jsonl"
DEV_REVIEWED_FILE = EVAL_DIR / "dev.reviewed.jsonl"
REPORT_FILE = EVAL_DIR / "human_review_report.json"
DECISIONS_FILE = EVAL_DIR / "review_decisions.jsonl"
ORIG_DEV_FILE = EVAL_DIR / "dev.jsonl"
ORIG_TEST_FILE = EVAL_DIR / "test.jsonl"
META_FILE = EVAL_DIR / "metadata.json"

EXPECTED_DEV_SHA256 = "8f0203e048a1b1b59087cf0c507d80fde4c34c652bd8df7ad7416d127c00648c"
EXPECTED_TEST_SHA256 = "d5840a2c25c4245658ee952b8f2382a0c3942eb1cc51598dfc39d13dd47d13ca"
EXPECTED_FROZEN_DEV_REVIEWED_SHA256 = "cdb72a100a39ed26e5baac060a637d3939eb00ac7e2b6a009266945482abfd92"


class TestReviewedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = {}
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                cls.chunks[c["chunk_id"]] = c

        cls.reviewed_records = []
        with open(DEV_REVIEWED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                cls.reviewed_records.append(json.loads(line))

        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            cls.report = json.load(f)

        cls.decisions = []
        with open(DECISIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                cls.decisions.append(json.loads(line))

    def test_original_files_untouched(self):
        """Verify original dev.jsonl and test.jsonl were never modified."""
        with open(ORIG_DEV_FILE, "rb") as f:
            dev_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(dev_hash, EXPECTED_DEV_SHA256, "Original dev.jsonl hash mismatch! File was modified.")

        with open(ORIG_TEST_FILE, "rb") as f:
            test_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(test_hash, EXPECTED_TEST_SHA256, "Original test.jsonl hash mismatch! File was modified.")

    def test_record_count_and_uniqueness(self):
        """Verify exactly 140 questions in dev.reviewed.jsonl and unique query IDs."""
        self.assertEqual(len(self.reviewed_records), 140)
        qids = [r["query_id"] for r in self.reviewed_records]
        self.assertEqual(len(qids), len(set(qids)))
        self.assertTrue(all(qid.startswith("Q") for qid in qids))

    def test_chunk_referential_integrity(self):
        """Verify all referenced chunk IDs exist in chunks.jsonl and match relevant_chunk_ids."""
        for r in self.reviewed_records:
            qid = r["query_id"]
            rc_ids = [rc["chunk_id"] for rc in r["relevant_chunks"]]
            self.assertEqual(rc_ids, r["relevant_chunk_ids"], f"Mismatch in chunk IDs list for {qid}")
            # Ensure no duplicates within query
            self.assertEqual(len(rc_ids), len(set(rc_ids)), f"Duplicate chunk IDs in {qid}")
            for cid in rc_ids:
                self.assertIn(cid, self.chunks, f"Chunk {cid} referenced in {qid} does not exist in corpus!")

    def test_evidence_substring_containment(self):
        """Verify every evidence text excerpt is an exact substring of chunk searchable_text."""
        for r in self.reviewed_records:
            qid = r["query_id"]
            for rc in r["relevant_chunks"]:
                cid = rc["chunk_id"]
                ev = rc["evidence_text"]
                self.assertTrue(len(ev) > 0, f"Empty evidence text in {qid} for chunk {cid}")
                chunk_text = self.chunks[cid]["searchable_text"]
                self.assertIn(ev, chunk_text, f"Evidence text for {qid} not found in chunk {cid} searchable_text!")

    def test_relevance_scores_and_metadata(self):
        """Verify scores are in {0, 1, 2, 3} and metadata fields match chunk data."""
        for r in self.reviewed_records:
            qid = r["query_id"]
            for rc in r["relevant_chunks"]:
                score = rc["relevance_score"]
                self.assertIn(score, [0, 1, 2, 3], f"Invalid relevance score {score} in {qid}")
                cid = rc["chunk_id"]
                self.assertEqual(rc["doc_path"], self.chunks[cid]["relative_path"])
                self.assertEqual(rc["python_version"], self.chunks[cid]["python_version"])

    def test_questions_and_answers_nonempty(self):
        """Verify questions and expected answers are non-empty and well-formed."""
        for r in self.reviewed_records:
            qid = r["query_id"]
            self.assertTrue(len(r["question"].strip()) > 10, f"Question too short in {qid}")
            self.assertTrue(len(r["expected_answer"].strip()) > 15, f"Expected answer too short in {qid}")

    def test_review_decisions_alignment(self):
        """Verify 140 review decisions match dev records 1:1."""
        self.assertEqual(len(self.decisions), 140)
        dec_qids = [d["query_id"] for d in self.decisions]
        rec_qids = [r["query_id"] for r in self.reviewed_records]
        self.assertEqual(dec_qids, rec_qids)

        for d in self.decisions:
            self.assertIn(d["final_recommendation"], ["ACCEPT", "REVISE_AND_ACCEPT", "REJECT", "NEEDS_HUMAN_REVIEW"])
            self.assertTrue(len(d["reason"].strip()) > 0)

    def test_q001_decision_preserved(self):
        """Verify Q001 retains human decision ACCEPT."""
        q001_dec = next(d for d in self.decisions if d["query_id"] == "Q001")
        self.assertEqual(q001_dec["final_recommendation"], "ACCEPT")

    def test_report_completeness(self):
        """Verify human review report has expected structure and non-zero counts."""
        self.assertEqual(self.report["total_questions_audited"], 140)
        self.assertEqual(self.report["rejected_questions_count"], 0)
        self.assertGreater(self.report["total_revised_and_accepted_count"], 0)
        self.assertGreater(self.report["questions_with_corrected_evidence_count"], 0)
        self.assertEqual(len(self.report["category_wise_statistics"]), 7)
        self.assertEqual(len(self.report["modified_query_ids"]), len(self.report["before_after_details"]))

    def test_frozen_benchmark_sha256(self):
        """Verify dev.reviewed.jsonl exact frozen SHA-256 checksum."""
        with open(DEV_REVIEWED_FILE, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(actual_sha, EXPECTED_FROZEN_DEV_REVIEWED_SHA256,
                         "dev.reviewed.jsonl checksum changed! Frozen benchmark must not be modified.")

    def test_user_approvals_recorded(self):
        """Verify explicit user approval is recorded ONLY for 6 corrected questions and Q001, not all 140."""
        target_approved = {"Q111", "Q113", "Q114", "Q116", "Q118", "Q119"}
        found_approved = set()
        audit_accepted_count = 0
        pre_accepted_count = 0

        for d in self.decisions:
            qid = d["query_id"]
            if qid in target_approved:
                found_approved.add(qid)
                self.assertEqual(d.get("user_approval"), "APPROVED")
                self.assertEqual(d.get("approval_type"), "EXPLICIT_USER_APPROVAL")
                self.assertEqual(d.get("user_approval_date"), "2026-10-31")
                self.assertIn("Explicit User Approval Granted: 2026-10-31", d["reason"])
            elif qid == "Q001":
                pre_accepted_count += 1
                self.assertEqual(d.get("user_approval"), "ACCEPTED")
                self.assertEqual(d.get("approval_type"), "USER_PRE_ACCEPTED")
            else:
                audit_accepted_count += 1
                self.assertEqual(d.get("user_approval"), "AUDIT_ACCEPTED")
                self.assertEqual(d.get("approval_type"), "AUDIT_ACCEPTED")
                self.assertIsNone(d.get("user_approval_date"))

        self.assertEqual(found_approved, target_approved)
        self.assertEqual(len(found_approved), 6)
        self.assertEqual(pre_accepted_count, 1)
        self.assertEqual(audit_accepted_count, 133)

    def test_benchmark_freeze_metadata(self):
        """Verify metadata.json and human_review_report.json record freeze version, status, date, hash, and distinguish user vs audit approvals."""
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertIn("benchmark_freeze", meta)
        bf = meta["benchmark_freeze"]
        self.assertEqual(bf["benchmark_version"], "1.0.0")
        self.assertEqual(bf["benchmark_status"], "FROZEN")
        self.assertEqual(bf["freeze_date"], "2026-10-31")
        self.assertEqual(bf["record_count"], 140)
        self.assertEqual(bf["sha256"], EXPECTED_FROZEN_DEV_REVIEWED_SHA256)
        self.assertEqual(set(bf["explicit_user_approved_questions"]), {"Q111", "Q113", "Q114", "Q116", "Q118", "Q119"})
        self.assertEqual(bf["user_pre_accepted_questions"], ["Q001"])
        self.assertEqual(bf["approval_summary"]["explicit_user_approved_count"], 6)
        self.assertEqual(bf["approval_summary"]["user_pre_accepted_count"], 1)
        self.assertEqual(bf["approval_summary"]["audit_accepted_count"], 133)

        # Report checks
        self.assertEqual(self.report["benchmark_version"], "1.0.0")
        self.assertEqual(self.report["benchmark_status"], "FROZEN")
        self.assertEqual(self.report["freeze_date"], "2026-10-31")
        self.assertEqual(self.report["benchmark_sha256"], EXPECTED_FROZEN_DEV_REVIEWED_SHA256)
        self.assertEqual(self.report["benchmark_record_count"], 140)
        self.assertEqual(set(self.report["explicit_user_approved_questions"]), {"Q111", "Q113", "Q114", "Q116", "Q118", "Q119"})
        self.assertEqual(self.report["user_pre_accepted_questions"], ["Q001"])
        self.assertEqual(self.report["approval_distribution"]["explicit_user_approved_count"], 6)
        self.assertEqual(self.report["approval_distribution"]["user_pre_accepted_count"], 1)
        self.assertEqual(self.report["approval_distribution"]["audit_accepted_count"], 133)


if __name__ == "__main__":
    unittest.main()

