"""
Preprocess the official LIAR-PLUS dataset into clean, validated JSONL splits
for evidence-based claim verification without introducing data leakage.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def get_project_root() -> Path:
    # scripts/data_preprocessing/preprocess_liar_plus.py -> 3 levels up to project root
    return Path(__file__).resolve().parents[2]


def safe_float(val: str) -> Optional[float]:
    try:
        return float(val.strip())
    except (ValueError, AttributeError):
        return None


def clean_text(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = val.strip()
    if not s or s.lower() == "none":
        return None
    return s


def preprocess_liar_plus(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    project_root = get_project_root()
    if raw_dir is None:
        raw_dir = project_root / "dataset" / "politifact_evidence" / "raw"
    if output_dir is None:
        output_dir = project_root / "dataset" / "politifact_evidence" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        {"split": "val", "filename": "val2.tsv", "out_filename": "val_evidence.jsonl"},
        {"split": "test", "filename": "test2.tsv", "out_filename": "test_evidence.jsonl"},
        {"split": "train", "filename": "train2.tsv", "out_filename": "train_evidence.jsonl"},
    ]

    report = {
        "dataset_name": "LIAR-PLUS",
        "splits": {},
        "total_records": 0,
        "total_with_justification": 0,
        "total_malformed_rows": 0,
    }

    all_records: List[Dict[str, Any]] = []

    for s_info in splits:
        split_name = s_info["split"]
        raw_path = raw_dir / s_info["filename"]
        out_path = output_dir / s_info["out_filename"]

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw LIAR-PLUS file not found: {raw_path}")

        records: List[Dict[str, Any]] = []
        malformed: List[Dict[str, Any]] = []
        seen_ids = set()
        duplicate_ids = 0

        with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            for row_idx, row in enumerate(reader):
                if len(row) != 16:
                    malformed.append({
                        "row_index": row_idx,
                        "col_count": len(row),
                        "snippet": " | ".join(row[:4]) if row else ""
                    })
                    continue

                stmt_id = row[1].strip()
                if stmt_id in seen_ids:
                    duplicate_ids += 1
                seen_ids.add(stmt_id)

                verdict = row[2].strip()
                claim = row[3].strip()
                subject = clean_text(row[4])
                speaker = clean_text(row[5])
                speaker_job = clean_text(row[6])
                state_info = clean_text(row[7])
                party_affiliation = clean_text(row[8])

                credit_history = {
                    "barely_true": safe_float(row[9]),
                    "false": safe_float(row[10]),
                    "half_true": safe_float(row[11]),
                    "mostly_true": safe_float(row[12]),
                    "pants_fire": safe_float(row[13]),
                }

                context = clean_text(row[14])
                justification = clean_text(row[15])

                item = {
                    "statement_id": stmt_id,
                    "claim": claim,
                    "verdict": verdict,
                    "justification": justification,
                    "has_justification": bool(justification),
                    "speaker": speaker,
                    "subject": subject,
                    "speaker_job": speaker_job,
                    "state_info": state_info,
                    "party_affiliation": party_affiliation,
                    "credit_history": credit_history,
                    "context": context,
                    "split": split_name,
                    "source_dataset": "liar-plus"
                }

                records.append(item)

        # Write split-specific jsonl
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        all_records.extend(records)

        with_just = sum(1 for r in records if r["has_justification"])

        report["splits"][split_name] = {
            "source_file": s_info["filename"],
            "output_file": s_info["out_filename"],
            "total_parsed_rows": len(records) + len(malformed),
            "valid_records": len(records),
            "malformed_rows_count": len(malformed),
            "malformed_rows_details": malformed,
            "unique_statement_ids": len(seen_ids),
            "duplicate_statement_ids": duplicate_ids,
            "records_with_justification": with_just,
            "justification_fill_rate": round(with_just / len(records) * 100, 2) if records else 0,
        }

        report["total_records"] += len(records)
        report["total_with_justification"] += with_just
        report["total_malformed_rows"] += len(malformed)

    # Write combined jsonl with split preserved
    all_out_path = output_dir / "politifact_evidence_all.jsonl"
    with open(all_out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report["combined_output_file"] = "politifact_evidence_all.jsonl"
    report["overall_justification_fill_rate"] = round(
        report["total_with_justification"] / report["total_records"] * 100, 2
    ) if report["total_records"] else 0

    # Write summary report
    summary_path = output_dir / "liar_plus_preprocessing_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = preprocess_liar_plus()
    print("=== LIAR-PLUS PREPROCESSING COMPLETE ===")
    for split_k, s_data in rep["splits"].items():
        print(f"[{split_k.upper()}] Valid: {s_data['valid_records']:,} | Malformed: {s_data['malformed_rows_count']} | Justifications: {s_data['records_with_justification']:,} ({s_data['justification_fill_rate']}%)")
    print(f"[TOTAL] Valid Records: {rep['total_records']:,} ({rep['overall_justification_fill_rate']}% with justification)")

