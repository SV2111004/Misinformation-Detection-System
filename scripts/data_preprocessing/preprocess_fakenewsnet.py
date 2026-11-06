"""
Preprocess the official FakeNewsNet base index CSV files (politifact_fake.csv and politifact_real.csv)
into structured, validated JSONL datasets for propagation modeling.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Increase CSV field size limit for large tweet_ids fields
csv.field_size_limit(2147483647)


def get_project_root() -> Path:
    # scripts/data_preprocessing/preprocess_fakenewsnet.py -> 3 levels up to project root
    return Path(__file__).resolve().parents[2]


def clean_str(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = val.strip()
    return s if s else None


def parse_tweet_ids(raw_tweet_ids: str) -> List[str]:
    if not raw_tweet_ids or not raw_tweet_ids.strip():
        return []
    # According to FakeNewsNet documentation, tweet IDs are separated by tab \t
    # Fallback to whitespace splitting if no tab is present
    cleaned = raw_tweet_ids.strip()
    if "\t" in cleaned:
        tokens = cleaned.split("\t")
    else:
        tokens = cleaned.split()
    return [t.strip() for t in tokens if t.strip()]


def preprocess_fakenewsnet(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    project_root = get_project_root()
    if raw_dir is None:
        raw_dir = project_root / "dataset" / "fakenewsnet" / "raw"
    if output_dir is None:
        output_dir = project_root / "dataset" / "fakenewsnet" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        {"filename": "politifact_fake.csv", "label": "fake", "out_filename": "politifact_fake_processed.jsonl"},
        {"filename": "politifact_real.csv", "label": "real", "out_filename": "politifact_real_processed.jsonl"},
    ]

    report = {
        "dataset_name": "FakeNewsNet-PolitiFact",
        "subsets": {},
        "total_news_records": 0,
        "total_tweet_ids": 0,
    }

    all_records: List[Dict[str, Any]] = []

    for src_info in sources:
        label = src_info["label"]
        raw_path = raw_dir / src_info["filename"]
        out_path = output_dir / src_info["out_filename"]

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw FakeNewsNet file not found: {raw_path}")

        records: List[Dict[str, Any]] = []
        malformed: List[Dict[str, Any]] = []
        seen_ids = set()
        duplicate_ids = 0
        missing_urls = 0
        missing_titles = 0
        tweet_counts: List[int] = []

        with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = None
            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    header = row
                    continue

                if len(row) != 4:
                    malformed.append({
                        "row_index": row_idx,
                        "col_count": len(row),
                        "snippet": " | ".join(row[:2]) if row else ""
                    })
                    continue

                news_id = row[0].strip()
                if not news_id:
                    malformed.append({
                        "row_index": row_idx,
                        "col_count": len(row),
                        "snippet": "Empty news_id"
                    })
                    continue

                if news_id in seen_ids:
                    duplicate_ids += 1
                seen_ids.add(news_id)

                news_url = clean_str(row[1])
                if news_url is None:
                    missing_urls += 1

                title = clean_str(row[2])
                if title is None:
                    missing_titles += 1

                tweet_id_list = parse_tweet_ids(row[3])
                t_count = len(tweet_id_list)
                tweet_counts.append(t_count)

                item = {
                    "news_id": news_id,
                    "title": title or "",
                    "news_url": news_url,
                    "dataset_label": label,
                    "tweet_ids": tweet_id_list,
                    "tweet_id_count": t_count,
                    "has_tweets": t_count > 0,
                    "source_dataset": "fakenewsnet-politifact"
                }

                records.append(item)

        # Write label-specific JSONL
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        all_records.extend(records)

        total_tweets_subset = sum(tweet_counts)
        non_zero_counts = [c for c in tweet_counts if c > 0]
        tweet_counts.sort()

        report["subsets"][label] = {
            "source_file": src_info["filename"],
            "output_file": src_info["out_filename"],
            "total_records": len(records),
            "malformed_rows_count": len(malformed),
            "unique_news_ids": len(seen_ids),
            "duplicate_news_ids": duplicate_ids,
            "missing_urls_count": missing_urls,
            "missing_titles_count": missing_titles,
            "total_tweet_ids": total_tweets_subset,
            "articles_with_tweets_count": len(non_zero_counts),
            "articles_with_tweets_pct": round(len(non_zero_counts) / len(records) * 100, 2) if records else 0,
            "tweet_count_stats": {
                "min": min(tweet_counts) if tweet_counts else 0,
                "max": max(tweet_counts) if tweet_counts else 0,
                "mean": round(total_tweets_subset / len(records), 2) if records else 0,
                "median": tweet_counts[len(tweet_counts) // 2] if tweet_counts else 0,
            }
        }

        report["total_news_records"] += len(records)
        report["total_tweet_ids"] += total_tweets_subset

    # Write combined JSONL
    combined_out_path = output_dir / "politifact_news_metadata.jsonl"
    with open(combined_out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report["combined_output_file"] = "politifact_news_metadata.jsonl"

    # Write report
    summary_path = output_dir / "fakenewsnet_preprocessing_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = preprocess_fakenewsnet()
    print("=== FAKENEWSNET PREPROCESSING COMPLETE ===")
    for label_k, s_data in rep["subsets"].items():
        stats = s_data["tweet_count_stats"]
        print(f"[{label_k.upper()}] Articles: {s_data['total_records']:,} | Tweets: {s_data['total_tweet_ids']:,} | Mean: {stats['mean']} | Median: {stats['median']} | Missing URLs: {s_data['missing_urls_count']}")
    print(f"[TOTAL] Total News Articles: {rep['total_news_records']:,} | Total Referenced Tweets: {rep['total_tweet_ids']:,}")

