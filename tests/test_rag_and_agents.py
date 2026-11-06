"""
Unit and Integration Tests for RAG Engine, Specialized Agents, and Multi-Agent Orchestrator.

Tests:
1. RAG persistent indexing, hybrid search, citations, and grounded summary.
2. Row-alignment integrity checking and low-relevance empty behavior.
3. Agent A (Claim Analysis Agent) classification and confidence calibration.
4. Agent B (Evidence Research / RAG Agent) citation formatting and conflict detection.
5. Agent C (Historical Influence Agent) FakeNewsNet metrics and timestamp decoding.
6. Agent D (Verification & Risk Assessment Agent) synthesis and uncertainty quantification.
7. Agent E (Mitigation Recommendation Agent) responsible advisory generation.
8. Central Orchestrator parallel workflow, error isolation, and execution trace.
9. FastAPI endpoint POST /agent/analyze and GET /rag/stats.
10. Preservation of all existing API endpoints.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root and backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evidence_retriever import EvidenceRetriever, check_claim_justification_alignment
from backend.influence_analyzer import InfluenceAnalyzer, decode_snowflake_id
from backend.rag_engine import RAGEngine, get_rag_engine
from backend.agents import (
    ClaimAnalysisAgent,
    EvidenceRAGAgent,
    HistoricalInfluenceAgent,
    VerificationRiskAgent,
    MitigationRecommendationAgent,
)
from backend.orchestrator import OrchestrationWorkflow, get_orchestrator
from backend.main import app


@pytest.fixture(scope="session")
def api_client():
    """Shared FastAPI TestClient instance."""
    return TestClient(app)


# ============================================================
# 1. RAG ENGINE TESTS
# ============================================================

def test_rag_engine_initialization(tmp_path):
    """Verifies that RAGEngine initializes with persistent ChromaDB directory."""
    chroma_dir = tmp_path / "test_chroma"
    engine = RAGEngine(persist_dir=chroma_dir, collection_name="test_collection")
    engine.initialize_client()

    assert engine._is_ready is True
    assert engine.collection is not None
    assert engine.collection.count() == 0


def test_rag_engine_index_and_hybrid_search(tmp_path):
    """Tests indexing sample records and querying with hybrid semantic + lexical search."""
    sample_jsonl = tmp_path / "sample_evidence.jsonl"
    records = [
        {
            "statement_id": "stmt_001.json",
            "claim": "The state unemployment rate dropped below 4 percent following manufacturing investments.",
            "verdict": "mostly-true",
            "justification": "State labor department records show unemployment fell to 3.8 percent after factory openings.",
            "speaker": "John Doe",
            "speaker_job": "Governor",
            "state_info": "Ohio",
            "party_affiliation": "democrat",
            "subject": "economy,jobs",
            "context": "press release",
            "split": "train",
            "source_dataset": "liar-plus",
        },
        {
            "statement_id": "stmt_002.json",
            "claim": "Federal healthcare spending increased by 50 percent under the new budget resolution.",
            "verdict": "false",
            "justification": "Congressional Budget Office reports confirm healthcare spending grew by only 4.2 percent.",
            "speaker": "Jane Smith",
            "speaker_job": "Representative",
            "state_info": "Texas",
            "party_affiliation": "republican",
            "subject": "health-care,budget",
            "context": "floor speech",
            "split": "train",
            "source_dataset": "liar-plus",
        },
    ]

    import json
    with open(sample_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # In-memory lexical retriever for this test
    lexical_retriever = EvidenceRetriever()
    lexical_retriever.index_records(records)

    chroma_dir = tmp_path / "chroma_db"
    engine = RAGEngine(
        persist_dir=chroma_dir,
        collection_name="test_hybrid",
        processed_file=sample_jsonl,
        lexical_retriever=lexical_retriever,
        alpha=0.5,
    )
    build_stats = engine.build_index()

    assert build_stats["total_added"] == 2
    assert engine.collection.count() == 2

    # Query matching unemployment
    results = engine.search("unemployment rate dropped below four percent", top_k=2, min_score=0.01)
    assert len(results) >= 1
    top = results[0]
    assert top["statement_id"] == "stmt_001.json"
    assert "unemployment" in top["claim"].lower()
    assert top["verdict"] == "mostly-true"
    assert "semantic_score" in top
    assert "lexical_score" in top
    assert top["similarity_score"] > 0.0
    assert top["has_context_alignment"] is True


def test_rag_engine_low_relevance_empty_behavior(tmp_path):
    """Ensures non-matching queries return empty results when min_score is high."""
    chroma_dir = tmp_path / "chroma_empty"
    engine = RAGEngine(persist_dir=chroma_dir, collection_name="test_empty")
    engine.initialize_client()

    results = engine.search("completely unrelated query with high threshold", top_k=5, min_score=0.99)
    assert results == []

    summary = engine.generate_grounded_summary("unrelated query", results)
    assert summary["evidence_status"] == "insufficient"
    assert summary["citations_count"] == 0
    assert "Insufficient historical evidence" in summary["summary"]


def test_rag_engine_grounded_summary_conflict_detection():
    """Tests that grounded summary detects divergence when historical citations conflict."""
    engine = RAGEngine()
    conflicting_records = [
        {
            "statement_id": "c1.json",
            "claim": "Tax reform decreased average income.",
            "verdict": "false",
            "justification": "Bureau of Economic Analysis shows real median income grew by 1.8 percent.",
            "speaker": "Speaker A",
            "similarity_score": 0.45,
            "has_context_alignment": True,
        },
        {
            "statement_id": "c2.json",
            "claim": "Tax reform reduced deductions for low income earners.",
            "verdict": "mostly-true",
            "justification": "Joint Committee on Taxation confirmed standard deduction doubled but child tax exemptions were altered.",
            "speaker": "Speaker B",
            "similarity_score": 0.40,
            "has_context_alignment": True,
        },
    ]

    report = engine.generate_grounded_summary("Tax reform impacts on income", conflicting_records)
    assert report["has_conflicts"] is True
    assert "Divergence Detected" in report["summary"] or "conflicting" in report["summary"].lower()
    assert report["citations_count"] == 2
    assert report["verdict_distribution"] == {"false": 1, "mostly-true": 1}


# ============================================================
# 2. SPECIALIZED AGENTS TESTS
# ============================================================

def test_agent_a_claim_analysis():
    """Tests Agent A (Claim Analysis Agent) returns 6-class LIAR result and calibration note."""
    agent = ClaimAnalysisAgent()
    res = agent.run("The federal deficit reached record levels this fiscal year.")

    assert res["agent_id"] == "claim_analysis_agent"
    assert res["status"] == "completed"
    assert res["duration_ms"] > 0
    assert "data" in res
    assert res["data"]["label"] in {"pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"}
    assert 0.0 <= res["data"]["confidence"] <= 1.0
    assert "calibration_note" in res["data"]
    assert "confidence is not an independent proof" in res["disclaimer"].lower() or "statistical alignment" in res["disclaimer"].lower()


def test_agent_b_evidence_rag():
    """Tests Agent B (Evidence Research / RAG Agent) returns citations and grounded summary."""
    agent = EvidenceRAGAgent()
    res = agent.run("The state unemployment rate dropped below 5 percent.", top_k=3, min_score=0.01)

    assert res["agent_id"] == "evidence_rag_agent"
    assert res["status"] == "completed"
    assert "citations" in res["data"]
    assert "grounded_summary" in res["data"]
    assert "LIAR-PLUS" in res["disclaimer"]


def test_agent_c_historical_influence():
    """Tests Agent C (Historical Influence Agent) bounds metrics to benchmark stories."""
    agent = HistoricalInfluenceAgent()
    res = agent.run("BREAKING: First NFL Team Declares Bankruptcy Over Kneeling Thugs", top_k=2)

    assert res["agent_id"] == "historical_influence_agent"
    assert res["status"] == "completed"
    assert "matched_stories" in res["data"]
    assert "FakeNewsNet" in res["disclaimer"]
    assert "do NOT reflect social reach, impressions, follower networks" in res["disclaimer"]


def test_agent_d_verification_risk():
    """Tests Agent D (Verification & Risk Assessment Agent) synthesizes A, B, and C."""
    agent = VerificationRiskAgent()

    mock_a = {
        "status": "completed",
        "data": {"label": "false", "confidence": 0.88},
    }
    mock_b = {
        "status": "completed",
        "data": {
            "evidence_status": "grounded",
            "has_conflicts": False,
            "citations": [{"statement_id": "test_1", "verdict": "false", "similarity_score": 0.72}],
            "misaligned_citations_count": 0,
        },
    }
    mock_c = {
        "status": "completed",
        "data": {
            "matched_stories": [{"news_id": "p_1", "headline": "Test Headline", "recorded_tweet_count": 42}],
        },
    }

    res = agent.run(
        claim="Test political statement",
        claim_analysis=mock_a,
        evidence_rag=mock_b,
        influence_analysis=mock_c,
    )

    assert res["agent_id"] == "verification_risk_agent"
    assert res["status"] == "completed"
    data = res["data"]
    assert data["risk_level"] == "HIGH_MISINFORMATION_RISK"
    assert 0.0 <= data["uncertainty_score"] <= 1.0
    assert len(data["findings"]) >= 1
    assert any("Guardrail" in u for u in data["uncertainties"])


def test_agent_e_mitigation_recommendations():
    """Tests Agent E (Mitigation Recommendation Agent) provides responsible non-coercive advice."""
    agent = MitigationRecommendationAgent()
    mock_verification = {
        "data": {
            "risk_level": "HIGH_MISINFORMATION_RISK",
            "uncertainty_score": 0.25,
        }
    }

    res = agent.run(
        claim="Fabricated election voting machine claim",
        verification_data=mock_verification,
    )

    assert res["agent_id"] == "mitigation_agent"
    assert res["status"] == "completed"
    recs = res["data"]["recommendations"]
    assert len(recs) >= 2
    assert all("action" in r and "priority" in r and "details" in r for r in recs)
    assert "advisory" in res["disclaimer"].lower()
    assert res["data"]["executed_interventions_count"] == 0


# ============================================================
# 3. CENTRAL ORCHESTRATOR TESTS
# ============================================================

def test_orchestrator_full_workflow():
    """Verifies that the orchestrator coordinates all 5 agents and logs complete execution traces."""
    orch = OrchestrationWorkflow()
    report = orch.analyze_sync("The state unemployment rate dropped below 5 percent.", top_k_evidence=3, top_k_influence=2)

    assert report["workflow_status"] == "completed"
    assert report["total_duration_ms"] > 0
    assert "claim" in report
    assert "claim_analysis" in report
    assert "evidence_rag" in report
    assert "historical_influence" in report
    assert "verification_risk" in report
    assert "mitigation_recommendations" in report

    # Verify execution trace has all 5 agents
    trace = report["execution_trace"]
    assert len(trace) == 5
    agent_ids = [t["agent_id"] for t in trace]
    assert agent_ids == [
        "claim_analysis_agent",
        "evidence_rag_agent",
        "historical_influence_agent",
        "verification_risk_agent",
        "mitigation_agent",
    ]
    for t in trace:
        assert t["status"] in {"completed", "fallback"}
        assert t["duration_ms"] >= 0
        assert len(t["summary"]) > 0


def test_orchestrator_input_validation():
    """Verifies input validation on empty, too short, or too long claims."""
    orch = OrchestrationWorkflow()

    with pytest.raises(ValueError, match="too short"):
        orch.analyze_sync("a")

    with pytest.raises(ValueError, match="too short"):
        orch.analyze_sync("   ")

    with pytest.raises(ValueError, match="exceeds maximum"):
        orch.analyze_sync("x" * 5001)


def test_orchestrator_error_isolation():
    """Ensures that a failing sub-agent does not crash the entire orchestration pipeline."""
    class FailingAgent:
        def __init__(self):
            self.agent_id = "failing_agent"
            self.agent_name = "Failing Agent"
        def run(self, *args, **kwargs):
            raise RuntimeError("Simulated internal agent crash")

    orch = OrchestrationWorkflow(rag_agent=FailingAgent())
    # The rag_agent failure will be handled by OrchestrationWorkflow
    report = orch.analyze_sync("Federal budget spending increased.")
    assert "execution_trace" in report
    # The pipeline should complete and reflect partial failure
    assert report["workflow_status"] in {"completed", "completed_with_errors"}


# ============================================================
# 4. FASTAPI ENDPOINT INTEGRATION TESTS
# ============================================================

def test_api_agent_analyze_endpoint(api_client):
    """Tests POST /agent/analyze with valid claim payload."""
    response = api_client.post(
        "/agent/analyze",
        json={
            "claim": "The federal corporate tax rate was lowered to 21 percent.",
            "top_k_evidence": 3,
            "top_k_influence": 2,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["claim"] == "The federal corporate tax rate was lowered to 21 percent."
    assert "workflow_status" in data
    assert "claim_analysis" in data
    assert "evidence_rag" in data
    assert "historical_influence" in data
    assert "verification_risk" in data
    assert "mitigation_recommendations" in data
    assert "execution_trace" in data
    assert len(data["execution_trace"]) == 5


def test_api_agent_analyze_validation_errors(api_client):
    """Tests POST /agent/analyze rejects invalid inputs."""
    # Empty claim
    r1 = api_client.post("/agent/analyze", json={"claim": "  "})
    assert r1.status_code == 400

    # Short claim
    r2 = api_client.post("/agent/analyze", json={"claim": "ab"})
    assert r2.status_code == 400

    # Empty payload
    r3 = api_client.post("/agent/analyze", json={})
    assert r3.status_code == 422


def test_api_rag_stats_endpoint(api_client):
    """Tests GET /rag/stats returns collection and vector store details."""
    response = api_client.get("/rag/stats")
    assert response.status_code == 200
    data = response.json()

    assert "collection_name" in data
    assert "vector_store" in data
    assert "embedding_model" in data
    assert data["vector_store"].startswith("ChromaDB")


def test_api_preservation_of_existing_endpoints(api_client):
    """Verifies that all pre-existing endpoints continue working without regression."""
    # 1. GET /
    r_root = api_client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["status"] == "online"

    # 2. GET /health
    r_health = api_client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    # 3. POST /predict
    r_predict = api_client.post(
        "/predict",
        json={"claim": "The federal corporate tax rate is 21 percent."},
    )
    assert r_predict.status_code == 200
    assert "label" in r_predict.json()

    # 4. POST /compare
    r_compare = api_client.post(
        "/compare",
        json={"claim": "The federal corporate tax rate is 21 percent."},
    )
    assert r_compare.status_code == 200
    assert "baseline" in r_compare.json()
    assert "distilbert" in r_compare.json()

    # 5. POST /evidence/search
    r_evidence = api_client.post(
        "/evidence/search",
        json={"claim": "unemployment rate dropped below 5 percent", "top_k": 3},
    )
    assert r_evidence.status_code == 200
    assert "results" in r_evidence.json()

    # 6. GET /evidence/stats
    r_ev_stats = api_client.get("/evidence/stats")
    assert r_ev_stats.status_code == 200
    assert "total_indexed" in r_ev_stats.json()

    # 7. POST /influence/analyze
    r_influence = api_client.post(
        "/influence/analyze",
        json={"claim": "BREAKING: First NFL Team Declares Bankruptcy Over Kneeling Thugs", "top_k": 2},
    )
    assert r_influence.status_code == 200
    assert "results" in r_influence.json()

    # 8. GET /influence/stats
    r_inf_stats = api_client.get("/influence/stats")
    assert r_inf_stats.status_code == 200
    assert "total_stories" in r_inf_stats.json()

