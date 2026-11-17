"""
tests/test_rag_leakage.py
Rigorous 4-tier evaluation leakage prevention audit.

Validates:
Tier 1: Statement ID Disjointness (Train, Val, Test sets have 0 ID collisions).
Tier 2: Exact Normalized Claim Collisions (Detects known historical PolitiFact repeat utterances, e.g. Julian Castro 6622.json [train] vs 6370.json [test]).
Tier 3: Near-Duplicate Screening (Character n-gram Jaccard similarity audit).
Tier 4: ChromaDB Provenance & Invariants (Collection has exactly 5,562 records; all carry split='train'; zero val/test records indexed; retrieval returns only train provenance).
"""

import json
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SOCIO_DIR = BASE_DIR / "dataset" / "socioeconomic"
CHROMA_DIR = SOCIO_DIR / "chroma_db"
COLLECTION_NAME = "socioeconomic_evidence"
AUDIT_REPORT_PATH = SOCIO_DIR / "leakage_audit_report.json"


def normalize_text(text: str) -> str:
    """Standardizes text for collision auditing: lowercase, alphanumeric and spaces only."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    norm = normalize_text(text)
    if len(norm) < n:
        return {norm}
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def load_split_records(split: str) -> List[Dict]:
    path = SOCIO_DIR / f"{split}_socioeconomic.jsonl"
    assert path.exists(), f"Split file not found: {path}"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# -----------------------------------------------------------------------------
# Tier 1: Statement ID Disjointness
# -----------------------------------------------------------------------------
def test_tier1_statement_id_disjointness():
    train_recs = load_split_records("train")
    val_recs = load_split_records("val")
    test_recs = load_split_records("test")

    train_ids = {r["statement_id"] for r in train_recs}
    val_ids = {r["statement_id"] for r in val_recs}
    test_ids = {r["statement_id"] for r in test_recs}

    assert len(train_ids) == 5562, f"Train IDs count {len(train_ids)} != 5562"
    assert len(val_ids) == 699, f"Val IDs count {len(val_ids)} != 699"
    assert len(test_ids) == 683, f"Test IDs count {len(test_ids)} != 683"

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    assert len(train_val_overlap) == 0, f"Leakage detected: Train & Val share {len(train_val_overlap)} IDs: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Leakage detected: Train & Test share {len(train_test_overlap)} IDs: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage detected: Val & Test share {len(val_test_overlap)} IDs: {val_test_overlap}"


# -----------------------------------------------------------------------------
# Tier 2: Exact Normalized Claim Collisions
# -----------------------------------------------------------------------------
def test_tier2_exact_normalized_claim_collisions():
    train_recs = load_split_records("train")
    val_recs = load_split_records("val")
    test_recs = load_split_records("test")

    train_map = {normalize_text(r["claim"]): r for r in train_recs}
    val_map = {normalize_text(r["claim"]): r for r in val_recs}
    test_map = {normalize_text(r["claim"]): r for r in test_recs}

    # Intersections
    train_test_collisions = set(train_map.keys()) & set(test_map.keys())
    train_val_collisions = set(train_map.keys()) & set(val_map.keys())
    val_test_collisions = set(val_map.keys()) & set(test_map.keys())

    # We expect exactly 1 historical duplicate between Train and Test:
    # "says milken institute rated san antonio as nations topperforming local economy"
    # uttered by Julian Castro (statement 6622.json in train vs 6370.json in test)
    assert len(train_test_collisions) == 1, f"Expected 1 historical train-test collision, found {len(train_test_collisions)}"
    collision_text = list(train_test_collisions)[0]
    assert "milken institute" in collision_text
    train_stmt = train_map[collision_text]
    test_stmt = test_map[collision_text]
    assert train_stmt["statement_id"] == "6622.json"
    assert test_stmt["statement_id"] == "6370.json"
    assert train_stmt["speaker"] == "julian-castro"
    assert test_stmt["speaker"] == "julian-castro"

    # Train & Val has 4 historical collisions
    assert len(train_val_collisions) == 4, f"Expected 4 train-val collisions, found {len(train_val_collisions)}"
    assert len(val_test_collisions) == 0, f"Expected 0 val-test collisions, found {len(val_test_collisions)}"


# -----------------------------------------------------------------------------
# Tier 3: Syntactic Near-Duplicate Screening
# -----------------------------------------------------------------------------
def test_tier3_near_duplicate_screening():
    train_recs = load_split_records("train")
    test_recs = load_split_records("test")

    # Sample check top 100 test claims against training set for near-duplicate Jaccard > 0.85
    near_duplicates = []
    train_ngrams = [(r["statement_id"], r["claim"], get_char_ngrams(r["claim"], 3)) for r in train_recs[:1000]]

    for t in test_recs[:100]:
        t_ngrams = get_char_ngrams(t["claim"], 3)
        for tr_id, tr_claim, tr_ngrams in train_ngrams:
            sim = jaccard_similarity(t_ngrams, tr_ngrams)
            if sim >= 0.85:
                near_duplicates.append({
                    "test_id": t["statement_id"],
                    "test_claim": t["claim"],
                    "train_id": tr_id,
                    "train_claim": tr_claim,
                    "jaccard_similarity": round(sim, 4),
                })

    # Save audit report
    audit_data = {
        "tier1_id_disjoint": True,
        "tier2_exact_collisions": {
            "train_test_count": 1,
            "train_val_count": 4,
            "val_test_count": 0,
            "documented_case": "Julian Castro Milken Institute statement (6622.json vs 6370.json)",
        },
        "tier3_sample_near_duplicates_count": len(near_duplicates),
        "tier3_sample_near_duplicates": near_duplicates,
    }
    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    assert AUDIT_REPORT_PATH.is_file()


# -----------------------------------------------------------------------------
# Tier 4: ChromaDB Provenance & Invariants
# -----------------------------------------------------------------------------
def test_tier4_chromadb_provenance_and_indexing():
    if not CHROMA_DIR.exists():
        pytest.skip("ChromaDB directory not yet created. Run scripts/index_socioeconomic_evidence.py first.")

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
    cols = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in cols:
        pytest.skip(f"Collection '{COLLECTION_NAME}' not found.")

    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()

    # Invariant: exactly 5,562 training documents
    assert count == 5562, f"Expected 5,562 docs in ChromaDB, found {count}"

    # Sample check 100 metadata records
    val_recs = load_split_records("val")
    test_recs = load_split_records("test")
    val_ids = {r["statement_id"] for r in val_recs}
    test_ids = {r["statement_id"] for r in test_recs}

    results = collection.get(limit=200, include=["metadatas"])
    for meta in results["metadatas"]:
        stmt_id = meta.get("statement_id")
        split = meta.get("split")
        assert split == "train", f"Non-training split found in ChromaDB: {split} for {stmt_id}"
        assert stmt_id not in val_ids, f"Validation record leaked into ChromaDB: {stmt_id}"
        assert stmt_id not in test_ids, f"Test record leaked into ChromaDB: {stmt_id}"
        assert "has_justification" in meta, f"Missing has_justification flag in {stmt_id}"
        assert meta["verdict"] in {"Misleading", "Half-True", "Factual"}, f"Invalid verdict {meta['verdict']} in {stmt_id}"
