"""
Comprehensive Automated Test Suite for Evidence Retrieval Module and API:
- EvidenceRetriever indexing and split filtering
- Retrieval ranking on controlled mock records
- Query validation and edge cases
- Metadata preservation (claims, verdicts, justifications, speakers)
- Behavior when no records match or index is empty
- API integration: POST /evidence/search and GET /evidence/stats
- Regression testing of existing endpoints (/, /health, /predict, /compare)
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

from backend.evidence_retriever import (
    EvidenceRetriever,
    DISCLAIMER_TEXT,
    get_evidence_retriever,
    reset_evidence_retriever,
)
from backend.main import app
from fastapi.testclient import TestClient

VALID_LIAR_LABELS = {"barely-true", "false", "half-true", "mostly-true", "pants-fire", "true"}


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def api_client():
    """Provides a reusable FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture(scope="module")
def real_train_retriever():
    """Provides a retriever pre-indexed with default train-only split."""
    reset_evidence_retriever()
    retriever = EvidenceRetriever(include_splits=("train",))
    retriever.load_and_index()
    return retriever


# ==============================================================================
# 1. Indexing & Split Filtering Unit Tests
# ==============================================================================

def test_evidence_retriever_train_split_indexing(real_train_retriever):
    """
    Verify that the default retriever correctly indexes only train split records
    and excludes val and test splits to prevent evaluation data leakage.
    """
    stats = real_train_retriever.get_index_stats()
    assert stats["is_indexed"] is True
    assert stats["total_examined"] == 12789
    assert stats["total_indexed"] == 10154
    assert stats["total_excluded"] == 2635
    assert stats["exclusion_reasons"]["excluded_split"] == 2551  # 1284 val + 1267 test
    assert stats["exclusion_reasons"]["missing_justification"] == 84
    assert stats["included_splits"] == ["train"]

    # Verify that every indexed record belongs exclusively to the train split
    for rec in real_train_retriever.records:
        assert rec["split"] == "train", f"Found non-train record in index: {rec['statement_id']}"
        assert rec.get("claim"), f"Indexed record with empty claim: {rec['statement_id']}"
        assert rec.get("justification"), f"Indexed record with empty justification: {rec['statement_id']}"


def test_split_isolation_mock_records():
    """
    Verify split isolation on controlled mock records.
    Only records matching include_splits must be indexed.
    """
    mock_records = [
        {"statement_id": "m1.json", "claim": "Inflation dropped by 3 percent", "justification": "BLS numbers confirm drop.", "split": "train"},
        {"statement_id": "m2.json", "claim": "Gas prices doubled in one month", "justification": "EIA reports indicate only 5% rise.", "split": "val"},
        {"statement_id": "m3.json", "claim": "State created 50000 manufacturing jobs", "justification": "State audit shows only 12000 jobs.", "split": "test"},
        {"statement_id": "m4.json", "claim": "Tax revenues reached record levels", "justification": "", "split": "train"},  # missing justification
        {"statement_id": "m5.json", "claim": "", "justification": "Some justification text", "split": "train"},  # missing claim
    ]

    # Test 1: Train split only (default)
    retriever_train = EvidenceRetriever(include_splits=("train",))
    retriever_train.index_records(mock_records)
    assert retriever_train.total_indexed == 1
    assert len(retriever_train.records) == 1
    assert retriever_train.records[0]["statement_id"] == "m1.json"
    assert retriever_train.exclusion_reasons["excluded_split"] == 2
    assert retriever_train.exclusion_reasons["missing_justification"] == 1
    assert retriever_train.exclusion_reasons["missing_claim"] == 1

    # Test 2: Include all splits
    retriever_all = EvidenceRetriever(include_splits=("train", "val", "test"))
    retriever_all.index_records(mock_records)
    assert retriever_all.total_indexed == 3
    indexed_ids = {r["statement_id"] for r in retriever_all.records}
    assert indexed_ids == {"m1.json", "m2.json", "m3.json"}

    # Test 3: Exclude test split explicitly
    retriever_no_test = EvidenceRetriever(include_splits=("train", "val", "test"), exclude_splits={"test"})
    retriever_no_test.index_records(mock_records)
    assert retriever_no_test.total_indexed == 2
    indexed_ids_no_test = {r["statement_id"] for r in retriever_no_test.records}
    assert indexed_ids_no_test == {"m1.json", "m2.json"}


# ==============================================================================
# 2. Retrieval Ranking Unit Tests
# ==============================================================================

def test_controlled_ranking_precision():
    """
    Verify TF-IDF cosine ranking precision using distinctive controlled claims.
    """
    mock_records = [
        {
            "statement_id": "tax_cut.json",
            "claim": "The federal corporate income tax cuts stimulated corporate investment and lowered national debt.",
            "verdict": "false",
            "justification": "Congressional Budget Office reported corporate cuts added 1.9 trillion to debt.",
            "speaker": "politician-alpha",
            "split": "train",
        },
        {
            "statement_id": "clean_energy.json",
            "claim": "Renewable solar and wind energy subsidies created thousands of domestic green jobs.",
            "verdict": "mostly-true",
            "justification": "Department of Energy census confirms 12 percent annual solar job expansion.",
            "speaker": "politician-beta",
            "split": "train",
        },
        {
            "statement_id": "health_care.json",
            "claim": "Emergency room visits increased significantly after universal health coverage implementation.",
            "verdict": "barely-true",
            "justification": "Kaiser Family Foundation research demonstrated emergency visits plateaued.",
            "speaker": "politician-gamma",
            "split": "train",
        },
    ]

    retriever = EvidenceRetriever(include_splits=("train",))
    retriever.index_records(mock_records)

    # Query matching topic 1: corporate taxes
    results_tax = retriever.search("corporate income tax cut deficit debt", top_k=2)
    assert len(results_tax) >= 1
    assert results_tax[0]["statement_id"] == "tax_cut.json"
    assert results_tax[0]["verdict"] == "false"
    assert results_tax[0]["similarity_score"] > 0.0

    # Query matching topic 2: renewable energy
    results_energy = retriever.search("renewable solar green energy subsidies", top_k=2)
    assert len(results_energy) >= 1
    assert results_energy[0]["statement_id"] == "clean_energy.json"
    assert results_energy[0]["verdict"] == "mostly-true"

    # Query matching topic 3: emergency health care
    results_health = retriever.search("universal health coverage hospital emergency room", top_k=2)
    assert len(results_health) >= 1
    assert results_health[0]["statement_id"] == "health_care.json"
    assert results_health[0]["verdict"] == "barely-true"


def test_exact_near_duplicate_and_unrelated_retrieval():
    """
    Controlled quality test for exact matches, near duplicates, and completely unrelated claims.
    """
    mock_records = [
        {
            "statement_id": "rec_exact.json",
            "claim": "Wisconsin added thirty thousand private sector manufacturing jobs last year.",
            "verdict": "mostly-true",
            "justification": "Department of Workforce Development monthly employment reports confirm 30,000 jobs.",
            "speaker": "governor-alpha",
            "split": "train",
        },
        {
            "statement_id": "rec_other.json",
            "claim": "National highway trust fund faces multi billion dollar shortfall.",
            "verdict": "true",
            "justification": "Congressional transportation committee audit found 12 billion shortfall.",
            "speaker": "senator-beta",
            "split": "train",
        },
    ]

    retriever = EvidenceRetriever(include_splits=("train",))
    retriever.index_records(mock_records)

    # 1. Exact-match query
    exact_q = "Wisconsin added thirty thousand private sector manufacturing jobs last year."
    res_exact = retriever.search(exact_q, top_k=1)
    assert len(res_exact) == 1
    assert res_exact[0]["statement_id"] == "rec_exact.json"
    assert res_exact[0]["similarity_score"] > 0.99

    # 2. Near-duplicate query (minor rephrasing)
    near_q = "Wisconsin created thirty thousand manufacturing jobs in private sector."
    res_near = retriever.search(near_q, top_k=1)
    assert len(res_near) == 1
    assert res_near[0]["statement_id"] == "rec_exact.json"
    assert res_near[0]["similarity_score"] > 0.60

    # 3. Unrelated query
    unrelated_q = "Astronomers discover water vapor on distant exoplanet."
    res_unrelated = retriever.search(unrelated_q, top_k=5)
    assert res_unrelated == []


def test_duplicate_statement_ids_and_missing_justifications():
    """
    Verify that duplicate statement IDs and missing justifications are filtered safely.
    """
    records = [
        {"statement_id": "id1.json", "claim": "State tax revenue reached record level.", "justification": "Treasury figures show record surplus.", "split": "train"},
        {"statement_id": "id1.json", "claim": "Duplicate ID with different claim text.", "justification": "Another justification.", "split": "train"},
        {"statement_id": "id2.json", "claim": "Gasoline prices increased five cents.", "justification": "", "split": "train"},
        {"statement_id": "id3.json", "claim": "Public university tuition doubled.", "justification": "none", "split": "train"},
    ]

    retriever = EvidenceRetriever(include_splits=("train",))
    retriever.index_records(records)

    assert retriever.total_examined == 4
    assert retriever.total_indexed == 1
    assert len(retriever.records) == 1
    assert retriever.records[0]["statement_id"] == "id1.json"
    assert retriever.records[0]["claim"] == "State tax revenue reached record level."
    assert retriever.exclusion_reasons["duplicate_id"] == 1
    assert retriever.exclusion_reasons["missing_justification"] == 2


def test_stable_tie_breaking_order():
    """
    Verify that ties in similarity score break deterministically by statement_id.
    """
    records = [
        {"statement_id": "zeta_claim.json", "claim": "Economic growth accelerated rapidly.", "justification": "Justification A.", "split": "train"},
        {"statement_id": "alpha_claim.json", "claim": "Economic growth accelerated rapidly.", "justification": "Justification B.", "split": "train"},
    ]

    retriever = EvidenceRetriever(include_splits=("train",))
    retriever.index_records(records)

    results = retriever.search("Economic growth accelerated rapidly.", top_k=2)
    assert len(results) == 2
    assert results[0]["similarity_score"] == results[1]["similarity_score"]
    assert results[0]["statement_id"] == "alpha_claim.json"
    assert results[1]["statement_id"] == "zeta_claim.json"


def test_top_k_boundary_values():
    """
    Verify top_k boundary values: top_k=1, top_k=5, top_k=20, and capped top_k>20.
    """
    mock_records = [
        {"statement_id": f"s_{i:02d}.json", "claim": f"Healthcare policy statement number {i} regarding medicare and insurance.", "justification": f"Justification for item {i}", "split": "train"}
        for i in range(25)
    ]
    retriever = EvidenceRetriever(include_splits=("train",))
    retriever.index_records(mock_records)

    # top_k = 1
    res_1 = retriever.search("healthcare policy medicare insurance", top_k=1)
    assert len(res_1) == 1

    # top_k = 5 (default)
    res_5 = retriever.search("healthcare policy medicare insurance", top_k=5)
    assert len(res_5) == 5

    # top_k = 20
    res_20 = retriever.search("healthcare policy medicare insurance", top_k=20)
    assert len(res_20) == 20

    # top_k > 20 is capped to 20
    res_50 = retriever.search("healthcare policy medicare insurance", top_k=50)
    assert len(res_50) == 20


# ==============================================================================
# 3. Query Validation & Edge Cases
# ==============================================================================

def test_retriever_query_validation_and_edge_cases(real_train_retriever):
    """
    Test edge cases: empty strings, whitespace, zero top_k, gibberish queries, empty index.
    """
    # Empty claim raises ValueError
    with pytest.raises(ValueError, match="Claim cannot be empty"):
        real_train_retriever.search("")

    # Whitespace claim raises ValueError
    with pytest.raises(ValueError, match="Claim cannot be empty"):
        real_train_retriever.search("   \t\n   ")

    # Invalid top_k raises ValueError
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        real_train_retriever.search("Valid claim text", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        real_train_retriever.search("Valid claim text", top_k=-5)

    # top_k > 20 is capped to 20
    results_capped = real_train_retriever.search("unemployment rate in the united states", top_k=50)
    assert len(results_capped) <= 20

    # Completely unmatching gibberish terms return empty list
    gibberish_results = real_train_retriever.search("zxqwkj1982734zzpq mnxbvcxz9999")
    assert gibberish_results == []

    # Empty index behavior
    empty_retriever = EvidenceRetriever(include_splits=("train",))
    empty_retriever.index_records([])
    empty_results = empty_retriever.search("Any query whatsoever")
    assert empty_results == []


# ==============================================================================
# 4. Metadata Preservation & Quality Verification
# ==============================================================================

def test_metadata_fields_preservation(real_train_retriever):
    """
    Verify that search results preserve all original metadata fields without mutation.
    """
    results = real_train_retriever.search("unemployment and jobs economy", top_k=5)
    assert len(results) > 0

    required_fields = {
        "statement_id",
        "claim",
        "verdict",
        "justification",
        "speaker",
        "speaker_job",
        "state_info",
        "party_affiliation",
        "subject",
        "context",
        "split",
        "source_dataset",
        "similarity_score",
    }

    for item in results:
        # Check all expected fields are present
        for field in required_fields:
            assert field in item, f"Missing field '{field}' in retrieval result"

        # Check types and constraints
        assert isinstance(item["statement_id"], str) and item["statement_id"]
        assert isinstance(item["claim"], str) and item["claim"]
        assert isinstance(item["justification"], str) and len(item["justification"]) > 0
        assert item["verdict"] in VALID_LIAR_LABELS, f"Invalid verdict: {item['verdict']}"
        assert item["split"] == "train"
        assert item["source_dataset"] == "liar-plus"
        assert 0.0 <= item["similarity_score"] <= 1.0


# ==============================================================================
# 5. FastAPI Integration Tests (POST /evidence/search & GET /evidence/stats)
# ==============================================================================

def test_api_evidence_search_success(api_client):
    """
    Verify POST /evidence/search returns 200, valid records, and mandatory disclaimer.
    """
    payload = {
        "claim": "Barack Obama reduced the unemployment rate during his presidency.",
        "top_k": 3,
    }
    response = api_client.post("/evidence/search", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == payload["claim"]
    assert data["result_count"] == len(data["results"])
    assert len(data["results"]) == 3
    assert data["disclaimer"] == DISCLAIMER_TEXT
    assert "historical claims and their published justifications" in data["disclaimer"]
    assert "must not be interpreted as classifier ground truth or confirmed proof" in data["disclaimer"]

    # Verify score descending order
    scores = [r["similarity_score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_api_evidence_stats_endpoint(api_client):
    """
    Verify GET /evidence/stats returns accurate index metadata.
    """
    response = api_client.get("/evidence/stats")
    assert response.status_code == 200

    stats = response.json()
    assert stats["is_indexed"] is True
    assert stats["total_indexed"] == 10154
    assert stats["total_excluded"] == 2635
    assert stats["included_splits"] == ["train"]
    assert stats["vocab_size"] > 10000


def test_api_evidence_search_validation_errors(api_client):
    """
    Verify proper HTTP error codes for invalid inputs on POST /evidence/search.
    """
    # 1. Empty string -> 422 (Pydantic min_length=1)
    res = api_client.post("/evidence/search", json={"claim": ""})
    assert res.status_code == 422

    # 2. Whitespace only -> 400
    res = api_client.post("/evidence/search", json={"claim": "    "})
    assert res.status_code == 400
    assert "at least 3 non-whitespace" in res.json()["detail"]

    # 3. Too short (< 3 chars) -> 400
    res = api_client.post("/evidence/search", json={"claim": "ab"})
    assert res.status_code == 400
    assert "at least 3 non-whitespace" in res.json()["detail"]

    # 4. Exceeds max length (> 5000 chars) -> 400 or 422
    res = api_client.post("/evidence/search", json={"claim": "x" * 5001})
    assert res.status_code in (400, 422)

    # 5. Invalid top_k (< 1) -> 422
    res = api_client.post("/evidence/search", json={"claim": "Valid claim text", "top_k": 0})
    assert res.status_code == 422

    # 6. Invalid top_k (> 20) -> 422
    res = api_client.post("/evidence/search", json={"claim": "Valid claim text", "top_k": 25})
    assert res.status_code == 422


# ==============================================================================
# 6. Regression Testing: Existing Endpoints
# ==============================================================================

def test_regression_existing_endpoints(api_client):
    """
    Verify that existing endpoints (/, /health, /predict, /compare)
    continue to function without regression.
    """
    # GET /
    res_root = api_client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"
    assert "TF-IDF + Logistic Regression" in res_root.json()["models"]
    assert "DistilBERT" in res_root.json()["models"]

    # GET /health
    res_health = api_client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
    assert res_health.json()["distilbert_loaded"] is True
    assert res_health.json()["baseline_loaded"] is True
    assert res_health.json()["tfidf_loaded"] is True

    # POST /predict
    test_claim = "The state unemployment rate dropped below 5 percent."
    res_predict = api_client.post("/predict", json={"claim": test_claim})
    assert res_predict.status_code == 200
    data_predict = res_predict.json()
    assert data_predict["claim"] == test_claim
    assert data_predict["label"] in VALID_LIAR_LABELS
    assert 0.0 <= data_predict["confidence"] <= 1.0

    # POST /compare
    res_compare = api_client.post("/compare", json={"claim": test_claim})
    assert res_compare.status_code == 200
    data_compare = res_compare.json()
    assert data_compare["claim"] == test_claim
    assert "baseline" in data_compare and "distilbert" in data_compare
    assert data_compare["baseline"]["label"] in VALID_LIAR_LABELS
    assert data_compare["distilbert"]["label"] in VALID_LIAR_LABELS

    # Input validation on existing endpoints preserved
    res_bad = api_client.post("/predict", json={"claim": "  "})
    assert res_bad.status_code == 400

    res_bad_comp = api_client.post("/compare", json={"claim": "a"})
    assert res_bad_comp.status_code == 400


def test_alignment_flag_and_min_score_api(api_client):
    """
    Verify that search results contain the has_context_alignment boolean flag,
    and min_score threshold filtering works via POST /evidence/search.
    """
    res = api_client.post(
        "/evidence/search",
        json={
            "claim": "The state unemployment rate dropped below 5 percent following new manufacturing investments.",
            "top_k": 5,
            "min_score": 0.05,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) > 0

    for item in data["results"]:
        assert "has_context_alignment" in item
        assert isinstance(item["has_context_alignment"], bool)
        assert item["similarity_score"] >= 0.05

    # Very high min_score produces empty results
    res_high = api_client.post(
        "/evidence/search",
        json={
            "claim": "The state unemployment rate dropped below 5 percent.",
            "top_k": 5,
            "min_score": 0.99,
        },
    )
    assert res_high.status_code == 200
    assert res_high.json()["result_count"] == 0
    assert res_high.json()["results"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

