"""
Master Pipeline Runner for Preprocessing Newly Acquired Datasets:
1. LIAR-PLUS (Evidence Retrieval & Claim Verification)
2. FakeNewsNet (Social Context & Propagation Analysis)
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from preprocess_liar_plus import preprocess_liar_plus
from preprocess_fakenewsnet import preprocess_fakenewsnet


def run_pipeline():
    print("=" * 65)
    print("MISINFORMATION DETECTION SYSTEM - DATA PREPROCESSING PIPELINE")
    print("=" * 65)

    print("\n>>> STEP 1: Preprocessing LIAR-PLUS Evidence Dataset...")
    liar_rep = preprocess_liar_plus()
    print(f"    Processed {liar_rep['total_records']:,} records across {len(liar_rep['splits'])} splits.")
    print(f"    Justification fill rate: {liar_rep['overall_justification_fill_rate']}%")

    print("\n>>> STEP 2: Preprocessing FakeNewsNet Propagation Metadata...")
    fnn_rep = preprocess_fakenewsnet()
    print(f"    Processed {fnn_rep['total_news_records']:,} news records.")
    print(f"    Parsed {fnn_rep['total_tweet_ids']:,} unique tweet-ID engagements.")

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    return liar_rep, fnn_rep


if __name__ == "__main__":
    run_pipeline()

