"""
Automated Test Suite for Influence Analysis Module and API Endpoints:
- Twitter Snowflake ID decoding math and boundary conditions
- Legacy ID (< Nov 2010) and malformed ID handling
- Temporal propagation calculations on controlled test data
- InfluenceAnalyzer TF-IDF indexing, exact title retrieval, and deterministic ranking
- Similarity threshold filtering and empty results for unrelated queries
- Percentile calculations against fake and real distributions
- API integration: POST /influence/analyze and GET /influence/stats
- Validation errors on POST /influence/analyze
- Regression testing of existing endpoints (/, /health, /predict, /compare, /evidence/search, /evidence/stats)
"""

import datetime
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.influence_analyzer import (
    InfluenceAnalyzer,
    TWITTER_EPOCH_MS,
    SNOWFLAKE_MIN_ID,
    INFLUENCE_DISCLAIMER_TEXT,
    decode_snowflake_id,
    analyze_temporal_metrics,
    extract_domain,
    get_influence_analyzer,
    reset_influence_analyzer,
)
from backend.main import app

client = TestClient(app)


# ==============================================================================
# 1. Twitter Snowflake Decoding & Legacy ID Unit Tests
# ==============================================================================

def test_snowflake_decoding_valid_ids():
    """
    Verify Twitter Snowflake decoding formula on known tweet IDs:
    (id >> 22) + 1288834974657 ms = UTC timestamp.
    """
    # Sample tweet ID from FakeNewsNet row 0: '937349434668498944'
    tweet_id = "937349434668498944"
    dt, status = decode_snowflake_id(tweet_id)
    assert status == "valid"
    assert dt is not None
    assert dt.year == 2017
    assert dt.month == 12
    assert dt.day == 3
    assert dt.hour == 15
    assert dt.minute == 54

    # Sample tweet ID from 2018: '967132259869487105'
    dt2, status2 = decode_snowflake_id("967132259869487105")
    assert status2 == "valid"
    assert dt2 is not None
    assert dt2.year == 2018
    assert dt2.month == 2
    assert dt2.day == 23


def test_snowflake_decoding_legacy_and_malformed():
    """
    Verify that legacy pre-Snowflake IDs (< 30,000,000,000) and malformed strings
    are safely classified and not erroneously decoded.
    """
    # Legacy sequential ID (pre-Nov 2010)
    legacy_id = "89585682"
    dt_legacy, status_legacy = decode_snowflake_id(legacy_id)
    assert status_legacy == "legacy"
    assert dt_legacy is None

    # Small integer ID
    dt_small, status_small = decode_snowflake_id("123456789")
    assert status_small == "legacy"
    assert dt_small is None

    # Non-digit string
    dt_alpha, status_alpha = decode_snowflake_id("not_a_tweet_id")
    assert status_alpha == "malformed"
    assert dt_alpha is None

    # Empty and whitespace
    dt_empty, status_empty = decode_snowflake_id("")
    assert status_empty == "malformed"

    dt_space, status_space = decode_snowflake_id("   ")
    assert status_space == "malformed"

    # Negative integer
    dt_neg, status_neg = decode_snowflake_id("-937349434668498944")
    assert status_neg == "malformed"


# ==============================================================================
# 2. Temporal Metrics Calculation Unit Tests
# ==============================================================================

def test_temporal_metrics_controlled_data():
    """
    Verify temporal calculations (span, first 24h count, daily breakdown)
    using controlled timestamps generated from known offsets.
    """
    # Synthesize 3 snowflake IDs with known 12-hour intervals
    base_ms = int(datetime.datetime(2018, 5, 1, 10, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    
    def ms_to_id(ms):
        shifted = (ms - TWITTER_EPOCH_MS) << 22
        return str(shifted + 100)  # add sequence/worker bits

    id_t0 = ms_to_id(base_ms)
    id_t12 = ms_to_id(base_ms + 12 * 3600 * 1000)  # +12 hours (within 24h)
    id_t36 = ms_to_id(base_ms + 36 * 3600 * 1000)  # +36 hours (after 24h)

    # Include 1 legacy ID and 1 malformed ID to verify safe categorization
    test_ids = [id_t0, id_t12, id_t36, "89585682", "invalid_id_xyz"]

    metrics = analyze_temporal_metrics(test_ids)
    assert metrics["has_temporal_data"] is True
    assert metrics["total_recorded_ids"] == 5
    assert metrics["valid_snowflake_ids"] == 3
    assert metrics["legacy_ids_count"] == 1
    assert metrics["malformed_ids_count"] == 1

    assert metrics["observed_span_hours"] == 36.0
    assert metrics["observed_span_days"] == 1.5
    assert metrics["first_24h_tweets_count"] == 2
    assert metrics["first_24h_percentage"] == round(2 / 3 * 100, 1)


def test_temporal_metrics_empty_and_legacy_only():
    """
    Verify temporal metrics when no IDs exist or only legacy IDs exist.
    """
    # Empty list
    res_empty = analyze_temporal_metrics([])
    assert res_empty["has_temporal_data"] is False
    assert res_empty["total_recorded_ids"] == 0

    # Legacy IDs only
    res_legacy = analyze_temporal_metrics(["89585682", "12345678"])
    assert res_legacy["has_temporal_data"] is False
    assert res_legacy["total_recorded_ids"] == 2
    assert res_legacy["valid_snowflake_ids"] == 0
    assert res_legacy["legacy_ids_count"] == 2


# ==============================================================================
# 3. Domain Extraction Helper Tests
# ==============================================================================

def test_extract_domain():
    """Verify safe domain extraction from URLs."""
    assert extract_domain("speedtalk.com/forum/viewtopic.php?t=51650") == "speedtalk.com"
    assert extract_domain("http://www.nytimes.com/article/123") == "nytimes.com"
    assert extract_domain("https://abcnews.go.com/Politics") == "abcnews.go.com"
    assert extract_domain(None) is None
    assert extract_domain("") is None
    assert extract_domain("   ") is None


# ==============================================================================
# 4. InfluenceAnalyzer Corpus & Retrieval Tests
# ==============================================================================

@pytest.fixture(scope="module")
def analyzer():
    """Module-level fixture for loaded InfluenceAnalyzer."""
    reset_influence_analyzer()
    inst = get_influence_analyzer()
    return inst


def test_influence_analyzer_corpus_stats(analyzer):
    """
    Verify that FakeNewsNet metadata is loaded with exact corpus counts.
    """
    stats = analyzer.get_index_stats()
    assert stats["is_indexed"] is True
    assert stats["total_stories"] == 1056
    assert stats["fake_stories_count"] == 432
    assert stats["real_stories_count"] == 624
    assert stats["total_tweet_ids"] == 583516
    assert stats["fake_tweet_stats"]["median"] == 79
    assert stats["real_tweet_stats"]["median"] == 8
    assert stats["vocab_size"] > 5000


def test_influence_analyzer_exact_retrieval(analyzer):
    """
    Verify exact-title retrieval returns high similarity and deterministic ranking.
    """
    query = "BREAKING: First NFL Team Declares Bankruptcy Over Kneeling Thugs"
    results = analyzer.search(query, top_k=3, min_score=0.08)
    assert len(results) >= 1
    top = results[0]
    assert top["news_id"] == "politifact15014"
    assert top["dataset_label"] == "fake"
    assert top["similarity_score"] > 0.85
    assert top["observed_tweet_count"] == 163
    assert top["domain"] == "speedtalk.com"
    assert 0.0 <= top["percentile_among_fake"] <= 100.0
    assert 0.0 <= top["percentile_among_real"] <= 100.0
    assert top["temporal_metrics"]["has_temporal_data"] is True


def test_influence_analyzer_unrelated_query_empty(analyzer):
    """
    Verify that an unrelated claim with no matching tokens returns an empty list.
    """
    unrelated = "zxqwkj1982734zzpq mnxbvcxz9999 zzyyxxww9911"
    results = analyzer.search(unrelated, top_k=5, min_score=0.08)
    assert results == []


def test_influence_analyzer_min_score_boundary(analyzer):
    """
    Verify that raising min_score filters out lower similarity matches.
    """
    query = "First NFL Team Declares Bankruptcy"
    low_thresh = analyzer.search(query, top_k=10, min_score=0.05)
    high_thresh = analyzer.search(query, top_k=10, min_score=0.99)
    assert len(low_thresh) >= len(high_thresh)


def test_percentile_calculations(analyzer):
    """
    Verify percentile calculation boundaries:
    - 0 tweets should map to a valid lower-tier percentile
    - Extremely high tweet counts (e.g. 30,000) should reach 100.0%
    """
    p_zero_fake = analyzer.calc_percentile(0, "fake")
    p_zero_real = analyzer.calc_percentile(0, "real")
    assert 0.0 <= p_zero_fake <= 20.0
    assert 20.0 <= p_zero_real <= 45.0

    p_huge_fake = analyzer.calc_percentile(50000, "fake")
    assert p_huge_fake == 100.0


# ==============================================================================
# 5. FastAPI Endpoints: POST /influence/analyze & GET /influence/stats
# ==============================================================================

def test_api_influence_stats():
    """Verify GET /influence/stats contract."""
    res = client.get("/influence/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["total_stories"] == 1056
    assert data["fake_stories_count"] == 432
    assert data["real_stories_count"] == 624
    assert data["total_tweet_ids"] == 583516


def test_api_influence_analyze_success():
    """Verify POST /influence/analyze on matching political headline."""
    payload = {
        "claim": "BREAKING: First NFL Team Declares Bankruptcy Over Kneeling Thugs",
        "top_k": 2,
        "min_score": 0.08,
    }
    res = client.post("/influence/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == payload["claim"]
    assert data["result_count"] >= 1
    assert len(data["results"]) <= 2
    assert data["disclaimer"] == INFLUENCE_DISCLAIMER_TEXT
    assert "benchmark_baselines" in data
    assert data["benchmark_baselines"]["fake_median_tweets"] == 79.0
    assert data["benchmark_baselines"]["real_median_tweets"] == 8.0

    item = data["results"][0]
    assert "news_id" in item
    assert "title" in item
    assert "dataset_label" in item
    assert "observed_tweet_count" in item
    assert "percentile_among_fake" in item
    assert "percentile_among_real" in item
    assert "temporal_metrics" in item


def test_api_influence_analyze_empty_results():
    """Verify POST /influence/analyze returns 200 with result_count=0 for unrelated query."""
    payload = {
        "claim": "Quantum chromodynamics plasma quark gluon viscosity",
        "top_k": 3,
        "min_score": 0.08,
    }
    res = client.post("/influence/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["result_count"] == 0
    assert data["results"] == []


def test_api_influence_analyze_validation_errors():
    """Verify HTTP status codes for invalid inputs on POST /influence/analyze."""
    # 1. Empty string -> 422
    assert client.post("/influence/analyze", json={"claim": ""}).status_code == 422

    # 2. Whitespace only -> 400
    res_space = client.post("/influence/analyze", json={"claim": "    "})
    assert res_space.status_code == 400
    assert "at least 3 non-whitespace" in res_space.json()["detail"]

    # 3. Too short (< 3 chars) -> 400
    res_short = client.post("/influence/analyze", json={"claim": "ab"})
    assert res_short.status_code == 400

    # 4. Invalid top_k (< 1) -> 422
    assert client.post("/influence/analyze", json={"claim": "Valid claim", "top_k": 0}).status_code == 422

    # 5. Invalid top_k (> 20) -> 422
    assert client.post("/influence/analyze", json={"claim": "Valid claim", "top_k": 25}).status_code == 422


# ==============================================================================
# 6. Regression Testing: Ensure Existing Endpoints Are Unaffected
# ==============================================================================

def test_regression_existing_endpoints():
    """Verify that /, /health, /predict, /compare, and /evidence endpoints remain intact."""
    # 1. GET /
    assert client.get("/").status_code == 200

    # 2. GET /health
    res_h = client.get("/health")
    assert res_h.status_code == 200
    assert res_h.json()["status"] == "healthy"

    # 3. POST /predict
    test_claim = "The state unemployment rate dropped below 5 percent."
    res_p = client.post("/predict", json={"claim": test_claim})
    assert res_p.status_code == 200

    # 4. POST /compare
    res_c = client.post("/compare", json={"claim": test_claim})
    assert res_c.status_code == 200
    assert "baseline" in res_c.json() and "distilbert" in res_c.json()

    # 5. POST /evidence/search
    res_e = client.post("/evidence/search", json={"claim": test_claim, "top_k": 3})
    assert res_e.status_code == 200
    assert "results" in res_e.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

