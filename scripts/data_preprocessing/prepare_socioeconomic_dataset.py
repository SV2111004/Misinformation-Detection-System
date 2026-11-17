"""
Data Preparation Script: Filter LIAR-PLUS dataset into the Socio-Economic & Public Welfare domain
with the approved 3-Class truthfulness scheme (Misleading / Half-True / Factual).

Outputs clean JSONL splits into dataset/socioeconomic/:
- train_socioeconomic.jsonl
- val_socioeconomic.jsonl
- test_socioeconomic.jsonl
- dataset_summary.json
"""

import os
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "dataset" / "politifact_evidence" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "socioeconomic"

# Approved consolidated socio-economic & public welfare domain tags
SOCIO_ECONOMIC_TAGS: Set[str] = {
    "economy", "taxes", "jobs", "federal-budget", "state-budget", 
    "state-finances", "deficit", "debt", "financial-regulation", 
    "labor", "banking", "corporations", "small-business", "poverty", 
    "income", "job-accomplishments", "workers", "stimulus", "trade", 
    "market-regulation", "housing", "unions", "city-budget",
    "health-care", "medicare", "medicaid", "public-health", 
    "abortion", "drugs", "food-safety", "social-security", "retirement"
}

# Approved 3-class mapping
LABEL_MAP_3CLASS = {
    "pants-fire": "Misleading",
    "false": "Misleading",
    "barely-true": "Half-True",
    "half-true": "Half-True",
    "mostly-true": "Factual",
    "true": "Factual"
}

LABEL_TO_ID = {
    "Misleading": 0,
    "Half-True": 1,
    "Factual": 2
}

def parse_subjects(subj_str: str) -> List[str]:
    if not subj_str:
        return []
    return [s.strip().lower() for s in subj_str.split(",") if s.strip()]

def matches_socioeconomic(record: Dict[str, Any]) -> bool:
    subjs = parse_subjects(record.get("subject", ""))
    return any(s in SOCIO_ECONOMIC_TAGS for s in subjs)

def process_splits():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    splits = [
        {"split": "train", "in_file": "train_evidence.jsonl", "out_file": "train_socioeconomic.jsonl"},
        {"split": "val", "in_file": "val_evidence.jsonl", "out_file": "val_socioeconomic.jsonl"},
        {"split": "test", "in_file": "test_evidence.jsonl", "out_file": "test_socioeconomic.jsonl"},
    ]
    
    summary = {
        "domain_name": "Socio-Economic & Domestic Public Welfare Policy",
        "label_scheme": "3-Class Tertiary Truthfulness",
        "labels": ["Misleading", "Half-True", "Factual"],
        "label_ids": LABEL_TO_ID,
        "splits": {},
        "total_records": 0,
        "overall_label_distribution": {},
        "justification_coverage": {}
    }
    
    total_recs = 0
    all_3class_counts = Counter()
    all_6class_counts = Counter()
    total_justifications = 0
    
    for s_info in splits:
        split_name = s_info["split"]
        in_path = INPUT_DIR / s_info["in_file"]
        out_path = OUTPUT_DIR / s_info["out_file"]
        
        split_records = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                
                if matches_socioeconomic(record):
                    orig_verdict = record.get("verdict", "").strip().lower()
                    if orig_verdict not in LABEL_MAP_3CLASS:
                        continue
                    
                    label_3class = LABEL_MAP_3CLASS[orig_verdict]
                    label_id = LABEL_TO_ID[label_3class]
                    
                    record["original_verdict"] = orig_verdict
                    record["verdict"] = label_3class
                    record["label"] = label_3class
                    record["label_id"] = label_id
                    record["domain"] = "socioeconomic_policy"
                    
                    split_records.append(record)
                    all_3class_counts[label_3class] += 1
                    all_6class_counts[orig_verdict] += 1
                    if record.get("has_justification"):
                        total_justifications += 1
        
        # Write filtered JSONL
        with open(out_path, "w", encoding="utf-8") as f_out:
            for rec in split_records:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
        # Split stats
        s_count = len(split_records)
        s_just = sum(1 for r in split_records if r.get("has_justification"))
        s_dist = Counter(r["label"] for r in split_records)
        summary["splits"][split_name] = {
            "record_count": s_count,
            "justification_count": s_just,
            "justification_percentage": round((s_just / s_count * 100), 2) if s_count else 0,
            "label_distribution": dict(s_dist)
        }
        total_recs += s_count
        print(f"[{split_name.upper()}] Processed: {s_count} records (Justifications: {s_just})")
        print(f"  Distribution: Misleading={s_dist['Misleading']}, Half-True={s_dist['Half-True']}, Factual={s_dist['Factual']}")

    summary["total_records"] = total_recs
    summary["overall_label_distribution"] = dict(all_3class_counts)
    summary["overall_6class_original_distribution"] = dict(all_6class_counts)
    summary["justification_coverage"] = {
        "count": total_justifications,
        "percentage": round((total_justifications / total_recs * 100), 2) if total_recs else 0
    }
    
    summary_path = OUTPUT_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)
        
    print(f"\nCompleted! Total Socio-Economic Records: {total_recs}")
    print(f"Summary written to: {summary_path}")

if __name__ == "__main__":
    process_splits()
