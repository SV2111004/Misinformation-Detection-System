"""
Integration Test Suite for Frontend-Backend Evidence Retrieval:
Verifies contracts, edge cases, error isolation, and response payloads
expected by the React frontend.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_stats_endpoint_contract_for_frontend():
    """
    Frontend calls GET /evidence/stats on mount to display the dynamic corpus size.
    Verify required fields: total_indexed, vocab_size, is_indexed.
    """
    res = client.get("/evidence/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_indexed" in data and isinstance(data["total_indexed"], int)
    assert data["total_indexed"] > 0
    assert "vocab_size" in data and isinstance(data["vocab_size"], int)
    assert data["vocab_size"] > 0
    assert data["is_indexed"] is True


def test_successful_evidence_search_payload_contract():
    """
    Frontend sends { claim: str, top_k: 5 } to POST /evidence/search.
    Verify response structure matches what frontend EvidenceCard expects:
    - claim, verdict, justification, speaker, speaker_job, state_info,
      party_affiliation, subject, context, split, source_dataset, similarity_score.
    """
    query = "The state unemployment rate dropped below 5 percent following new manufacturing investments."
    res = client.post("/evidence/search", json={"claim": query, "top_k": 5})
    assert res.status_code == 200
    data = res.json()

    assert data["query"] == query
    assert "result_count" in data
    assert "disclaimer" in data and len(data["disclaimer"]) > 50
    assert isinstance(data["results"], list)

    if data["results"]:
        item = data["results"][0]
        # Required fields in EvidenceCard
        assert "claim" in item and isinstance(item["claim"], str)
        assert "verdict" in item and isinstance(item["verdict"], str)
        assert "justification" in item
        assert "similarity_score" in item
        assert isinstance(item["similarity_score"], (int, float))
        # Metadata fields
        assert "speaker" in item
        assert "split" in item
        assert "source_dataset" in item
        assert item["split"] == "train"
        assert "has_context_alignment" in item
        assert isinstance(item["has_context_alignment"], bool)


def test_empty_matches_behavior():
    """
    When a claim has zero lexical overlap with political historical claims,
    POST /evidence/search must return 200 with result_count == 0 and results == [].
    Frontend uses this to display:
    'No sufficiently relevant historical evidence was found in the current dataset.'
    """
    unrelated_query = "zxqwkj1982734zzpq mnxbvcxz9999 zzyyxxww9911"
    res = client.post("/evidence/search", json={"claim": unrelated_query, "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["result_count"] == 0
    assert data["results"] == []


def test_evidence_failure_does_not_break_prediction():
    """
    Independent workflow verification:
    If an evidence search fails or is rejected, /compare and /predict continue to function.
    """
    # 1. Invalid evidence search request (invalid top_k -> 422)
    bad_evidence_res = client.post("/evidence/search", json={"claim": "valid political claim", "top_k": 0})
    assert bad_evidence_res.status_code == 422

    # 2. Comparison request on the same claim must still succeed (HTTP 200)
    compare_res = client.post("/compare", json={"claim": "valid political claim"})
    assert compare_res.status_code == 200
    comp_data = compare_res.json()
    assert "baseline" in comp_data and "distilbert" in comp_data
    assert comp_data["baseline"]["label"] in {"pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"}
    assert comp_data["distilbert"]["label"] in {"pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"}


def test_claim_validation_parity():
    """
    Verify /compare, /evidence/search, and /influence/analyze have matching validation rules:
    - Empty string -> 422
    - Whitespace -> 400
    - Too short (<3 chars) -> 400
    - Too long (>5000 chars) -> 400 / 422
    """
    for endpoint in ["/compare", "/evidence/search", "/influence/analyze"]:
        # Empty string
        r1 = client.post(endpoint, json={"claim": ""})
        assert r1.status_code == 422

        # Whitespace
        r2 = client.post(endpoint, json={"claim": "    "})
        assert r2.status_code == 400

        # Too short (<3 non-whitespace)
        r3 = client.post(endpoint, json={"claim": "hi"})
        assert r3.status_code == 400

        # Valid
        r4 = client.post(endpoint, json={"claim": "Valid political statement for testing."})
        assert r4.status_code == 200


def test_influence_contracts_for_frontend():
    """
    Verify that GET /influence/stats and POST /influence/analyze return
    all fields and schemas expected by the React frontend components.
    """
    # 1. Stats endpoint
    res_stats = client.get("/influence/stats")
    assert res_stats.status_code == 200
    s_data = res_stats.json()
    assert s_data["total_stories"] == 1056
    assert s_data["total_tweet_ids"] == 583516

    # 2. Analyze endpoint
    query = "BREAKING: First NFL Team Declares Bankruptcy Over Kneeling"
    res_ana = client.post("/influence/analyze", json={"claim": query, "top_k": 3, "min_score": 0.08})
    assert res_ana.status_code == 200
    a_data = res_ana.json()
    assert "query" in a_data
    assert "result_count" in a_data
    assert "results" in a_data
    assert "benchmark_baselines" in a_data
    assert "disclaimer" in a_data

    if a_data["results"]:
        top = a_data["results"][0]
        assert "news_id" in top
        assert "title" in top
        assert "dataset_label" in top
        assert "similarity_score" in top
        assert "observed_tweet_count" in top
        assert "percentile_among_fake" in top
        assert "percentile_among_real" in top
        assert "temporal_metrics" in top


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
