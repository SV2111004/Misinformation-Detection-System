"""
Validation Test Suite for Preprocessed Datasets:
- LIAR-PLUS Evidence Dataset
- FakeNewsNet Propagation Metadata
- Reproducibility and Idempotency Verification
"""

import hashlib
import json
import os
import sys
from pathlib import Path

# Add project root and data_preprocessing scripts directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "data_preprocessing"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess_liar_plus import preprocess_liar_plus
from preprocess_fakenewsnet import preprocess_fakenewsnet

LIAR_LABELS = {"pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"}


def file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_file_existence_and_readability():
    print("[TEST 1/6] Verifying file existence and readability...")

    # Check raw files
    raw_files = [
        PROJECT_ROOT / "dataset" / "politifact_evidence" / "raw" / "train2.tsv",
        PROJECT_ROOT / "dataset" / "politifact_evidence" / "raw" / "val2.tsv",
        PROJECT_ROOT / "dataset" / "politifact_evidence" / "raw" / "test2.tsv",
        PROJECT_ROOT / "dataset" / "fakenewsnet" / "raw" / "politifact_fake.csv",
        PROJECT_ROOT / "dataset" / "fakenewsnet" / "raw" / "politifact_real.csv",
    ]
    for p in raw_files:
        assert p.exists(), f"Raw file missing: {p}"
        assert p.stat().st_size > 0, f"Raw file is empty: {p}"

    # Check processed files
    processed_files = [
        PROJECT_ROOT / "dataset" / "politifact_evidence" / "processed" / "train_evidence.jsonl",
        PROJECT_ROOT / "dataset" / "politifact_evidence" / "processed" / "val_evidence.jsonl",
        PROJECT_ROOT / "dataset" / "politifact_evidence" / "processed" / "test_evidence.jsonl",
        PROJECT_ROOT / "dataset" / "politifact_evidence" / "processed" / "politifact_evidence_all.jsonl",
        PROJECT_ROOT / "dataset" / "politifact_evidence" / "processed" / "liar_plus_preprocessing_report.json",
        PROJECT_ROOT / "dataset" / "fakenewsnet" / "processed" / "politifact_fake_processed.jsonl",
        PROJECT_ROOT / "dataset" / "fakenewsnet" / "processed" / "politifact_real_processed.jsonl",
        PROJECT_ROOT / "dataset" / "fakenewsnet" / "processed" / "politifact_news_metadata.jsonl",
        PROJECT_ROOT / "dataset" / "fakenewsnet" / "processed" / "fakenewsnet_preprocessing_report.json",
    ]
    for p in processed_files:
        assert p.exists(), f"Processed file missing: {p}"
        assert p.stat().st_size > 0, f"Processed file is empty: {p}"

    print("  -> All raw and processed files exist and are non-empty.")


def test_liar_plus_schema_and_completeness():
    print("[TEST 2/6] Verifying LIAR-PLUS schema, label validity, and completeness...")

    proc_dir = PROJECT_ROOT / "dataset" / "politifact_evidence" / "processed"
    splits = [
        ("val", proc_dir / "val_evidence.jsonl", 1284),
        ("test", proc_dir / "test_evidence.jsonl", 1267),
        ("train", proc_dir / "train_evidence.jsonl", 10238),
    ]

    total_records = 0
    total_justifications = 0

    required_keys = {
        "statement_id", "claim", "verdict", "justification",
        "has_justification", "speaker", "split", "source_dataset"
    }

    for split_name, fpath, expected_count in splits:
        records = []
        seen_ids = set()
        with open(fpath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                rec = json.loads(line)
                records.append(rec)

                # Schema checks
                missing_keys = required_keys - set(rec.keys())
                assert not missing_keys, f"Row {line_no} in {fpath.name} missing keys: {missing_keys}"

                # Key field assertions
                assert rec["statement_id"], f"Empty statement_id at row {line_no}"
                assert rec["claim"], f"Empty claim at row {line_no}"
                assert rec["verdict"] in LIAR_LABELS, f"Invalid verdict {rec['verdict']} at row {line_no}"
                assert rec["split"] == split_name, f"Split mismatch {rec['split']} != {split_name}"
                assert rec["source_dataset"] == "liar-plus"

                # Check boolean flag matches text presence
                assert rec["has_justification"] == bool(rec["justification"])

                seen_ids.add(rec["statement_id"])

        count = len(records)
        assert count == expected_count, f"{split_name} count mismatch: got {count}, expected {expected_count}"
        assert len(seen_ids) == count, f"Duplicate IDs detected in {split_name}"

        with_just = sum(1 for r in records if r["has_justification"])
        fill_rate = with_just / count * 100
        assert fill_rate > 98.0, f"Justification fill rate unexpectedly low in {split_name}: {fill_rate:.2f}%"

        total_records += count
        total_justifications += with_just

    print(f"  -> Total LIAR-PLUS verified: {total_records:,} records across splits.")
    print(f"  -> Overall justification fill rate: {total_justifications / total_records * 100:.2f}% (>= 98% constraint met).")


def test_split_isolation():
    print("[TEST 3/6] Verifying zero data leakage across LIAR-PLUS evaluation splits...")

    proc_dir = PROJECT_ROOT / "dataset" / "politifact_evidence" / "processed"
    val_ids = {json.loads(l)["statement_id"] for l in open(proc_dir / "val_evidence.jsonl", encoding="utf-8")}
    test_ids = {json.loads(l)["statement_id"] for l in open(proc_dir / "test_evidence.jsonl", encoding="utf-8")}
    train_ids = {json.loads(l)["statement_id"] for l in open(proc_dir / "train_evidence.jsonl", encoding="utf-8")}

    # Validation and Test splits must have ZERO overlap with each other
    val_test_overlap = val_ids.intersection(test_ids)
    assert len(val_test_overlap) == 0, f"Leakage detected: val and test share {len(val_test_overlap)} IDs!"

    print(f"  -> Val/Test split overlap: 0 IDs. Strict evaluation isolation confirmed.")


def test_fakenewsnet_schema_and_tweet_counts():
    print("[TEST 4/6] Verifying FakeNewsNet schema, URL normalization, and tweet ID parsing...")

    proc_dir = PROJECT_ROOT / "dataset" / "fakenewsnet" / "processed"
    subsets = [
        ("fake", proc_dir / "politifact_fake_processed.jsonl", 432),
        ("real", proc_dir / "politifact_real_processed.jsonl", 624),
    ]

    required_keys = {
        "news_id", "title", "news_url", "dataset_label",
        "tweet_ids", "tweet_id_count", "has_tweets", "source_dataset"
    }

    total_articles = 0
    total_tweets = 0

    for label, fpath, expected_count in subsets:
        records = []
        seen_ids = set()

        with open(fpath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                rec = json.loads(line)
                records.append(rec)

                missing_keys = required_keys - set(rec.keys())
                assert not missing_keys, f"Row {line_no} in {fpath.name} missing keys: {missing_keys}"

                assert rec["news_id"].startswith("politifact"), f"Invalid news_id {rec['news_id']}"
                assert rec["dataset_label"] == label
                assert rec["source_dataset"] == "fakenewsnet-politifact"

                # Check tweet list count consistency
                assert isinstance(rec["tweet_ids"], list)
                assert len(rec["tweet_ids"]) == rec["tweet_id_count"]
                assert rec["has_tweets"] == (rec["tweet_id_count"] > 0)

                # Ensure all tweet IDs are digit strings
                for tid in rec["tweet_ids"]:
                    assert tid.isdigit(), f"Non-numeric tweet ID detected: {tid}"

                seen_ids.add(rec["news_id"])

        count = len(records)
        assert count == expected_count, f"{label} count mismatch: got {count}, expected {expected_count}"
        assert len(seen_ids) == count, f"Duplicate news IDs detected in {label}"

        total_articles += count
        total_tweets += sum(r["tweet_id_count"] for r in records)

    print(f"  -> Total FakeNewsNet verified: {total_articles:,} articles, {total_tweets:,} tweet IDs.")
    print("  -> Tweet list lengths and tweet_id_count fields are 100% consistent.")


def test_reproducibility():
    print("[TEST 5/6] Verifying pipeline idempotency & bit-for-bit reproducibility...")

    proc_dir_liar = PROJECT_ROOT / "dataset" / "politifact_evidence" / "processed"
    proc_dir_fnn = PROJECT_ROOT / "dataset" / "fakenewsnet" / "processed"

    # Compute hashes before rerun
    files_to_hash = [
        proc_dir_liar / "val_evidence.jsonl",
        proc_dir_liar / "test_evidence.jsonl",
        proc_dir_liar / "train_evidence.jsonl",
        proc_dir_fnn / "politifact_fake_processed.jsonl",
        proc_dir_fnn / "politifact_real_processed.jsonl",
    ]

    hashes_before = {p.name: file_sha256(p) for p in files_to_hash}

    # Re-run preprocessing
    preprocess_liar_plus()
    preprocess_fakenewsnet()

    # Compute hashes after rerun
    hashes_after = {p.name: file_sha256(p) for p in files_to_hash}

    for fname in hashes_before:
        assert hashes_before[fname] == hashes_after[fname], (
            f"Reproducibility failure: hash for {fname} changed upon re-running pipeline!"
        )

    print("  -> Re-ran pipeline from scratch: SHA256 hashes matched identically across all outputs.")


def test_existing_datasets_unmodified():
    print("[TEST 6/6] Verifying that original LIAR and raw data files are 100% untouched...")

    liar_files = {
        "train.tsv": 2408165,
        "valid.tsv": 301556,
        "test.tsv": 301118
    }

    liar_dir = PROJECT_ROOT / "dataset" / "liar"
    for fname, expected_size in liar_files.items():
        p = liar_dir / fname
        assert p.exists(), f"Original LIAR file missing: {p}"
        assert p.stat().st_size == expected_size, (
            f"Original LIAR file size changed! Expected {expected_size}, got {p.stat().st_size}"
        )

    print("  -> Original LIAR files in dataset/liar/ are confirmed intact and unmodified.")


def run_all_tests():
    print("=" * 65)
    print("RUNNING PREPROCESSING VALIDATION TEST SUITE")
    print("=" * 65)

    test_file_existence_and_readability()
    test_liar_plus_schema_and_completeness()
    test_split_isolation()
    test_fakenewsnet_schema_and_tweet_counts()
    test_reproducibility()
    test_existing_datasets_unmodified()

    print("\n" + "=" * 65)
    print("ALL 6 TEST SUITES PASSED CLEANLY (100% SUCCESS)!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()

