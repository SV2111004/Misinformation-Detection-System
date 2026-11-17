"""
tests/test_canonical_dataset_integrity.py
Validates the canonical Socio-Economic & Domestic Public Welfare Policy dataset integrity.

Invariants checked:
1. Total records == 6,944
2. Train records == 5,562 (5,504 with justification, 58 missing)
3. Validation records == 699 (698 with justification, 1 missing)
4. Test records == 683 (676 with justification, 7 missing)
5. Total with justification == 6,878 (99.05%)
6. Label schema strictly within {"Misleading", "Half-True", "Factual"}
7. Numeric label_id mappings: 0=Misleading, 1=Half-True, 2=Factual
8. Consistent schema fields across all JSONL records
9. Non-empty claim text for every record
10. Missing justifications cleanly flagged with has_justification metadata
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SOCIO_DIR = BASE_DIR / "dataset" / "socioeconomic"

EXPECTED_COUNTS = {
    "train": {"total": 5562, "just": 5504, "missing": 58},
    "val": {"total": 699, "just": 698, "missing": 1},
    "test": {"total": 683, "just": 676, "missing": 7},
}

EXPECTED_OVERALL_TOTAL = 6944
EXPECTED_TOTAL_JUSTIFICATIONS = 6878

LABEL_TO_ID = {
    "Misleading": 0,
    "Half-True": 1,
    "Factual": 2,
}

REQUIRED_FIELDS = {
    "statement_id",
    "claim",
    "verdict",
    "label",
    "label_id",
    "subject",
    "speaker",
    "has_justification",
    "split",
    "source_dataset",
}


def load_split(split_name: str):
    path = SOCIO_DIR / f"{split_name}_socioeconomic.jsonl"
    assert path.exists(), f"Split file not found: {path}"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_split_files_exist():
    for split in ["train", "val", "test"]:
        p = SOCIO_DIR / f"{split}_socioeconomic.jsonl"
        assert p.is_file(), f"Missing split file: {p}"


def test_dataset_record_counts():
    total_recs = 0
    total_just = 0

    for split, exp in EXPECTED_COUNTS.items():
        recs = load_split(split)
        assert len(recs) == exp["total"], f"Split {split} count {len(recs)} != {exp['total']}"
        total_recs += len(recs)

        just_count = sum(1 for r in recs if (r.get("justification") or "").strip())
        assert just_count == exp["just"], f"Split {split} justification count {just_count} != {exp['just']}"
        total_just += just_count

    assert total_recs == EXPECTED_OVERALL_TOTAL, f"Overall records {total_recs} != {EXPECTED_OVERALL_TOTAL}"
    assert total_just == EXPECTED_TOTAL_JUSTIFICATIONS, f"Overall justifications {total_just} != {EXPECTED_TOTAL_JUSTIFICATIONS}"


def test_label_schema_and_mapping():
    for split in ["train", "val", "test"]:
        recs = load_split(split)
        for r in recs:
            label_name = r.get("label")
            verdict = r.get("verdict")
            label_id = r.get("label_id")

            assert label_name in LABEL_TO_ID, f"Invalid label '{label_name}' in {r.get('statement_id')}"
            assert verdict == label_name, f"verdict '{verdict}' does not match label '{label_name}'"
            assert label_id == LABEL_TO_ID[label_name], f"label_id {label_id} != {LABEL_TO_ID[label_name]}"


def test_required_fields_and_non_empty_claims():
    for split in ["train", "val", "test"]:
        recs = load_split(split)
        for r in recs:
            for field in REQUIRED_FIELDS:
                assert field in r, f"Missing field '{field}' in statement {r.get('statement_id')}"

            claim = r.get("claim", "")
            assert isinstance(claim, str) and len(claim.strip()) > 3, f"Empty or invalid claim: {claim}"
            assert r.get("split") == split, f"Split mismatch {r.get('split')} != {split}"
            assert r.get("source_dataset") == "liar-plus", f"Unexpected source_dataset: {r.get('source_dataset')}"

            # Check has_justification consistency
            just_str = (r.get("justification") or "").strip()
            assert r.get("has_justification") == bool(just_str)


def test_dataset_summary_json_consistency():
    summary_path = SOCIO_DIR / "dataset_summary.json"
    assert summary_path.is_file(), f"Missing summary: {summary_path}"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary.get("total_records") == 6944
    assert summary.get("overall_label_distribution") == {
        "Misleading": 1838,
        "Half-True": 2684,
        "Factual": 2422,
    }
    assert summary.get("justification_coverage", {}).get("count") == 6878
