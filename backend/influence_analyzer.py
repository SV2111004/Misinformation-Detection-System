"""
Influence Analyzer Module for Political Misinformation Detection System.

Analyzes historical social engagement indicators from the FakeNewsNet-PolitiFact benchmark.
Provides:
- Lexical matching of claims against 1,056 historical news headlines via TF-IDF
- Deterministic Twitter Snowflake ID decoding for millisecond timestamps
- Safe handling of legacy (pre-Nov 2010) and malformed IDs
- Empirical percentile calculations among fake-labeled (N=432) and real-labeled (N=624) stories
- Observed temporal propagation metrics (first/last tweet timestamp, span, daily distribution)
"""

import datetime
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Standard Twitter Snowflake constants
TWITTER_EPOCH_MS = 1288834974657  # Nov 4, 2010 01:42:54.657 UTC
SNOWFLAKE_MIN_ID = 30000000000     # First Snowflake IDs began ~30 billion (Nov 2010)
MAX_VALID_YEAR = 2026              # Bound for valid decodable timestamps

INFLUENCE_DISCLAIMER_TEXT = (
    "These indicators describe historical tweet IDs recorded for matching benchmark stories. "
    "They do not measure the reach or influence of the newly submitted claim. "
    "Tweet-ID timestamps are decoded only where valid. The dataset does not contain "
    "follower counts, impressions, or retweet relationship graphs."
)


def get_default_fakenewsnet_file() -> Path:
    """Returns the default path to processed FakeNewsNet metadata JSONL."""
    return (
        Path(__file__).resolve().parent.parent
        / "dataset"
        / "fakenewsnet"
        / "processed"
        / "politifact_news_metadata.jsonl"
    )


def extract_domain(url: Optional[str]) -> Optional[str]:
    """Safely extracts domain name from URL string."""
    if not url:
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "http://" + cleaned
    try:
        parsed = urlparse(cleaned)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc if netloc else None
    except Exception:
        return None


def decode_snowflake_id(tweet_id_str: str) -> Tuple[Optional[datetime.datetime], str]:
    """
    Validates and decodes a tweet ID.
    Returns (datetime_utc, status_category) where status_category is one of:
    - 'valid': Standard Twitter Snowflake ID decoded to UTC datetime
    - 'legacy': Pre-Snowflake sequential ID (pre-Nov 2010, < 30,000,000,000)
    - 'malformed': Non-digit, empty, or negative ID string
    """
    s = str(tweet_id_str).strip()
    if not s or not s.isdigit():
        return None, "malformed"

    try:
        num = int(s)
    except ValueError:
        return None, "malformed"

    if num < SNOWFLAKE_MIN_ID:
        return None, "legacy"

    # Decode using Twitter Snowflake formula: (id >> 22) + epoch_ms
    ts_ms = (num >> 22) + TWITTER_EPOCH_MS
    try:
        dt = datetime.datetime.fromtimestamp(ts_ms / 1000.0, datetime.timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None, "legacy"

    if 2010 <= dt.year <= MAX_VALID_YEAR:
        return dt, "valid"
    else:
        return None, "legacy"


def analyze_temporal_metrics(tweet_ids: List[str]) -> Dict[str, Any]:
    """
    Computes temporal propagation metrics for a list of tweet IDs.
    Calculates timestamps strictly from validated Twitter Snowflake IDs.
    """
    total_count = len(tweet_ids)
    if total_count == 0:
        return {
            "has_temporal_data": False,
            "total_recorded_ids": 0,
            "valid_snowflake_ids": 0,
            "legacy_ids_count": 0,
            "malformed_ids_count": 0,
            "message": "No tweet IDs recorded for this historical story.",
        }

    valid_dts: List[datetime.datetime] = []
    legacy_count = 0
    malformed_count = 0

    for tid in tweet_ids:
        dt, status = decode_snowflake_id(tid)
        if status == "valid" and dt is not None:
            valid_dts.append(dt)
        elif status == "legacy":
            legacy_count += 1
        else:
            malformed_count += 1

    valid_count = len(valid_dts)
    if valid_count == 0:
        return {
            "has_temporal_data": False,
            "total_recorded_ids": total_count,
            "valid_snowflake_ids": 0,
            "legacy_ids_count": legacy_count,
            "malformed_ids_count": malformed_count,
            "message": (
                "Recorded tweet IDs are legacy sequential IDs (pre-Nov 2010) or invalid; "
                "creation timestamps cannot be derived via Snowflake decoding."
            ),
        }

    valid_dts.sort()
    t_min = valid_dts[0]
    t_max = valid_dts[-1]
    span_hours = round((t_max - t_min).total_seconds() / 3600.0, 2)
    span_days = round(span_hours / 24.0, 2)

    # First 24-hour volume and percentage
    cutoff_24h = t_min + datetime.timedelta(hours=24)
    first_24h_count = sum(1 for dt in valid_dts if dt <= cutoff_24h)
    first_24h_pct = round((first_24h_count / valid_count) * 100.0, 1)

    # Peak daily activity among recorded IDs
    day_counts = Counter(dt.strftime("%Y-%m-%d") for dt in valid_dts)
    peak_daily_counts = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "has_temporal_data": True,
        "total_recorded_ids": total_count,
        "valid_snowflake_ids": valid_count,
        "legacy_ids_count": legacy_count,
        "malformed_ids_count": malformed_count,
        "earliest_recorded_tweet_utc": t_min.isoformat(),
        "latest_recorded_tweet_utc": t_max.isoformat(),
        "observed_span_hours": span_hours,
        "observed_span_days": span_days,
        "first_24h_tweets_count": first_24h_count,
        "first_24h_percentage": first_24h_pct,
        "peak_daily_counts": peak_daily_counts,
    }


class InfluenceAnalyzer:
    """
    In-memory TF-IDF retriever and statistical evaluator for FakeNewsNet political news stories.
    """

    def __init__(
        self,
        metadata_file: Optional[Path] = None,
        max_features: int = 15000,
        sublinear_tf: bool = True,
    ):
        self.metadata_file = Path(metadata_file) if metadata_file else get_default_fakenewsnet_file()
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf

        self.records: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self.fake_counts: List[int] = []
        self.real_counts: List[int] = []
        self.all_counts: List[int] = []

        self._is_indexed = False

    def load_and_index(self) -> "InfluenceAnalyzer":
        """Loads processed FakeNewsNet metadata JSONL and fits TF-IDF matrix."""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"FakeNewsNet processed file not found: {self.metadata_file}")

        self.records = []
        self.fake_counts = []
        self.real_counts = []
        self.all_counts = []

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    rec = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                title = (rec.get("title") or "").strip()
                if not title:
                    continue

                tweet_ids = rec.get("tweet_ids") or []
                count = rec.get("tweet_id_count", len(tweet_ids))
                rec["tweet_id_count"] = count
                rec["tweet_ids"] = tweet_ids

                label = rec.get("dataset_label", "unknown")
                if label == "fake":
                    self.fake_counts.append(count)
                elif label == "real":
                    self.real_counts.append(count)
                self.all_counts.append(count)

                self.records.append(rec)

        self.fake_counts.sort()
        self.real_counts.sort()
        self.all_counts.sort()

        if self.records:
            corpus = [r["title"] for r in self.records]
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.max_features,
                sublinear_tf=self.sublinear_tf,
                stop_words="english",
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

        self._is_indexed = True
        logger.info(f"InfluenceAnalyzer indexed {len(self.records)} stories ({len(self.fake_counts)} fake, {len(self.real_counts)} real).")
        return self

    def calc_percentile(self, value: int, category: str = "fake") -> float:
        """
        Calculates the empirical percentile of a tweet count within a category.
        Percentile represents the percentage of benchmark stories with tweet_count <= value.
        """
        counts = self.fake_counts if category == "fake" else self.real_counts
        if not counts:
            return 0.0
        # Weak percentile: fraction of benchmark stories with count <= value
        less_equal = sum(1 for c in counts if c <= value)
        return round((less_equal / len(counts)) * 100.0, 1)

    def search(
        self,
        claim: str,
        top_k: int = 3,
        min_score: float = 0.08,
    ) -> List[Dict[str, Any]]:
        """
        Search historical stories textually similar to the user-supplied claim.

        :param claim: User claim text.
        :param top_k: Maximum number of matched stories to return (bounded between 1 and 20).
        :param min_score: Minimum cosine similarity score threshold (default: >= 0.08).
        :return: List of ranked match dictionaries with influence and temporal metrics.
        """
        if not self._is_indexed:
            self.load_and_index()

        text = claim.strip()
        if not text:
            raise ValueError("Claim cannot be empty or whitespace.")

        if top_k < 1:
            raise ValueError("top_k must be a positive integer >= 1.")

        top_k = min(top_k, 20)

        if not self.records or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # Vectorize claim query
        q_vec = self.vectorizer.transform([text])
        sims = (self.tfidf_matrix * q_vec.T).toarray().ravel()

        # Filter indices by min_score
        candidate_indices = [idx for idx, s in enumerate(sims) if s >= min_score]
        if not candidate_indices:
            return []

        # Deterministic sorting: descending by similarity, tie-breaking by news_id
        candidate_indices.sort(
            key=lambda idx: (-sims[idx], str(self.records[idx].get("news_id") or ""))
        )
        top_indices = candidate_indices[:top_k]

        results = []
        for idx in top_indices:
            rec = self.records[idx]
            tweet_ids = rec.get("tweet_ids", [])
            tweet_count = rec.get("tweet_id_count", len(tweet_ids))
            domain = extract_domain(rec.get("news_url"))
            temporal_data = analyze_temporal_metrics(tweet_ids)

            results.append({
                "news_id": rec.get("news_id"),
                "title": rec.get("title"),
                "dataset_label": rec.get("dataset_label"),
                "news_url": rec.get("news_url"),
                "domain": domain,
                "similarity_score": round(float(sims[idx]), 4),
                "observed_tweet_count": tweet_count,
                "percentile_among_fake": self.calc_percentile(tweet_count, "fake"),
                "percentile_among_real": self.calc_percentile(tweet_count, "real"),
                "temporal_metrics": temporal_data,
            })

        return results

    def get_index_stats(self) -> Dict[str, Any]:
        """Returns statistics about the current index state and FakeNewsNet distributions."""
        if not self._is_indexed:
            self.load_and_index()

        def compute_subset_stats(counts: List[int]) -> Dict[str, Any]:
            if not counts:
                return {}
            n = len(counts)
            return {
                "count": n,
                "total_tweets": sum(counts),
                "min": counts[0],
                "p25": counts[int(n * 0.25)],
                "median": counts[n // 2],
                "mean": round(sum(counts) / n, 2),
                "p75": counts[int(n * 0.75)],
                "p90": counts[int(n * 0.90)],
                "p95": counts[int(n * 0.95)],
                "max": counts[-1],
                "zero_tweets_count": sum(1 for c in counts if c == 0),
            }

        return {
            "is_indexed": self._is_indexed,
            "total_stories": len(self.records),
            "fake_stories_count": len(self.fake_counts),
            "real_stories_count": len(self.real_counts),
            "total_tweet_ids": sum(self.all_counts),
            "fake_tweet_stats": compute_subset_stats(self.fake_counts),
            "real_tweet_stats": compute_subset_stats(self.real_counts),
            "vocab_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
        }

    def get_benchmark_stats(self) -> Dict[str, Any]:
        """Alias for get_index_stats."""
        return self.get_index_stats()


# Lazy singleton instance for API usage
_analyzer_instance: Optional[InfluenceAnalyzer] = None


def get_influence_analyzer() -> InfluenceAnalyzer:
    """Returns the singleton InfluenceAnalyzer instance, initializing it if needed."""
    global _analyzer_instance
    if _analyzer_instance is None or not _analyzer_instance._is_indexed:
        analyzer = InfluenceAnalyzer()
        analyzer.load_and_index()
        _analyzer_instance = analyzer
    return _analyzer_instance


def reset_influence_analyzer():
    """Resets the singleton instance (useful for testing)."""
    global _analyzer_instance
    _analyzer_instance = None
