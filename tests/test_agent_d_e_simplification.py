"""
Comprehensive Test Suite for Simplified Agent D & Agent E Workflow
Validates the 8 core evaluation cases required for project demonstration and viva.
"""

import pytest
from backend.agents import VerificationRiskAgent, MitigationRecommendationAgent


@pytest.fixture
def agent_d():
    return VerificationRiskAgent()


@pytest.fixture
def agent_e():
    return MitigationRecommendationAgent()


def test_case_1_agents_a_and_b_agree(agent_d, agent_e):
    """Case 1: Agent A and Agent B agree on Factual / True."""
    mock_a = {"status": "completed", "data": {"label": "Factual", "confidence": 0.90}}
    mock_b = {
        "status": "completed",
        "data": {
            "evidence_status": "grounded",
            "citations": [
                {
                    "statement_id": "c1",
                    "claim": "Medicare Part D enrollment rose.",
                    "verdict": "true",
                    "similarity_score": 0.85,
                },
                {
                    "statement_id": "c2",
                    "claim": "Medicare enrollment expanded under federal program.",
                    "verdict": "mostly-true",
                    "similarity_score": 0.80,
                },
            ],
        },
    }
    mock_c = {"status": "completed", "data": {"matched_stories": [{"headline": "Medicare record", "tweet_count": 50}]}}

    res_d = agent_d.run("Medicare Part D enrollment rose.", mock_a, mock_b, mock_c)
    data_d = res_d["data"]

    assert res_d["status"] == "completed"
    assert data_d["a_vs_b"]["status"] == "CONCURRENCE"
    assert data_d["a_vs_b"]["agreement"] == "Agreement"
    assert data_d["final_assessment"] == "Supported by Evidence"
    assert data_d["simple_risk_level"] == "Low"
    assert data_d["uncertainty_level"] == "Low"
    assert data_d["evidence_strength"] == "Strong"

    res_e = agent_e.run("Medicare Part D enrollment rose.", res_d)
    data_e = res_e["data"]
    assert data_e["sharing_recommendation"] == "Share with Context"
    assert len(data_e["immediate_action"]) > 0
    assert len(data_e["verification_steps"]) >= 3


def test_case_2_agents_a_and_b_disagree(agent_d, agent_e):
    """Case 2: Agent A and Agent B disagree (A says Factual, B says Misleading)."""
    mock_a = {"status": "completed", "data": {"label": "Factual", "confidence": 0.80}}
    mock_b = {
        "status": "completed",
        "data": {
            "evidence_status": "grounded",
            "citations": [
                {
                    "statement_id": "c3",
                    "claim": "Unemployment dropped to 0 percent.",
                    "verdict": "false",
                    "similarity_score": 0.88,
                }
            ],
        },
    }
    mock_c = {"status": "completed", "data": {"matched_stories": []}}

    res_d = agent_d.run("Unemployment dropped to 0 percent.", mock_a, mock_b, mock_c)
    data_d = res_d["data"]

    assert data_d["a_vs_b"]["status"] == "DISCREPANCY"
    assert data_d["a_vs_b"]["agreement"] == "Disagreement"
    # Empirical evidence takes precedence!
    assert data_d["final_assessment"] == "Contradicted by Evidence"
    assert data_d["simple_risk_level"] == "High"

    res_e = agent_e.run("Unemployment dropped to 0 percent.", res_d)
    data_e = res_e["data"]
    assert data_e["sharing_recommendation"] == "Do Not Share as Verified"


def test_case_3_agent_b_insufficient_evidence(agent_d, agent_e):
    """Case 3: Agent B has insufficient evidence (no matching fact-checks)."""
    mock_a = {"status": "completed", "data": {"label": "Misleading", "confidence": 0.85}}
    mock_b = {"status": "completed", "data": {"citations": [], "evidence_status": "insufficient"}}
    mock_c = {"status": "completed", "data": {"matched_stories": []}}

    res_d = agent_d.run("Obscure county expenditure statement.", mock_a, mock_b, mock_c)
    data_d = res_d["data"]

    assert data_d["a_vs_b"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert data_d["a_vs_b"]["agreement"] == "Insufficient Evidence"
    assert data_d["final_assessment"] in {"Needs Further Verification", "Insufficient Evidence"}
    assert data_d["uncertainty_level"] == "High"
    assert data_d["evidence_strength"] == "Insufficient"

    res_e = agent_e.run("Obscure county expenditure statement.", res_d)
    data_e = res_e["data"]
    assert data_e["sharing_recommendation"] == "Hold for Verification"
    assert "unverified" in data_e["immediate_action"].lower() or "hold" in data_e["immediate_action"].lower()


def test_case_4_agent_c_no_matching_stories(agent_d):
    """Case 4: Agent C has no matching historical stories."""
    mock_a = {"status": "completed", "data": {"label": "Factual", "confidence": 0.88}}
    mock_b = {
        "status": "completed",
        "data": {
            "evidence_status": "grounded",
            "citations": [{"statement_id": "c4", "claim": "State budget passed.", "verdict": "true", "similarity_score": 0.82}],
        },
    }
    mock_c = {"status": "completed", "data": {"matched_stories": []}}

    res_d = agent_d.run("State budget passed.", mock_a, mock_b, mock_c)
    data_d = res_d["data"]

    assert data_d["b_vs_c"]["status"] == "NO_HISTORICAL_DATA"
    assert data_d["b_vs_c"]["has_context"] is False
    assert data_d["b_vs_c"]["matched_stories_count"] == 0
    assert "no relevant historical" in data_d["b_vs_c"]["explanation"].lower()
    # Factuality risk is unaffected by Agent C
    assert data_d["risk_score_breakdown"]["social_influence_weight_pct"] == 0.0


def test_case_5_agent_c_data_unavailable(agent_d, agent_e):
    """Case 5: Agent C data is completely unavailable (offline / error)."""
    mock_a = {"status": "completed", "data": {"label": "Half-True", "confidence": 0.60}}
    mock_b = {
        "status": "completed",
        "data": {
            "evidence_status": "grounded",
            "citations": [{"statement_id": "c5", "claim": "Half of budget went to schools.", "verdict": "half-true", "similarity_score": 0.75}],
        },
    }
    mock_c = {"status": "UNAVAILABLE", "data": {}}

    res_d = agent_d.run("Half of budget went to schools.", mock_a, mock_b, mock_c)
    data_d = res_d["data"]

    assert res_d["status"] == "completed"
    assert data_d["b_vs_c"]["has_context"] is False

    res_e = agent_e.run("Half of budget went to schools.", res_d)
    assert res_e["status"] == "completed"
    assert "social media" in res_e["data"]["missing_context"].lower()


def test_case_6_agent_d_high_risk(agent_d, agent_e):
    """Case 6: Agent D reports High Risk."""
    mock_a = {"status": "completed", "data": {"label": "Misleading", "confidence": 0.95}}
    mock_b = {
        "status": "completed",
        "data": {
            "evidence_status": "grounded",
            "citations": [
                {"statement_id": "c6", "claim": "Taxes tripled overnight.", "verdict": "pants-fire", "similarity_score": 0.90},
                {"statement_id": "c7", "claim": "Tax rate surged 300 percent.", "verdict": "false", "similarity_score": 0.85},
            ],
        },
    }
    mock_c = {"status": "completed", "data": {"matched_stories": []}}

    res_d = agent_d.run("Taxes tripled overnight.", mock_a, mock_b, mock_c)
    data_d = res_d["data"]

    assert data_d["simple_risk_level"] == "High"
    assert data_d["final_risk_score"] >= 65.0

    res_e = agent_e.run("Taxes tripled overnight.", res_d)
    data_e = res_e["data"]
    assert data_e["sharing_recommendation"] == "Do Not Share as Verified"
    assert "not share" in data_e["immediate_action"].lower()


def test_case_7_agent_d_moderate_risk(agent_d, agent_e):
    """Case 7: Agent D reports Moderate Risk (partial context / mixed)."""
    mock_a = {"status": "completed", "data": {"label": "Half-True", "confidence": 0.65}}
    mock_b = {
        "status": "completed",
        "data": {
            "evidence_status": "grounded",
            "citations": [
                {"statement_id": "c8", "claim": "Spending rose but inflation accounted for most.", "verdict": "half-true", "similarity_score": 0.70}
            ],
        },
    }
    mock_c = {"status": "completed", "data": {"matched_stories": []}}

    res_d = agent_d.run("Spending rose last year.", mock_a, mock_b, mock_c)
    data_d = res_d["data"]

    assert data_d["simple_risk_level"] == "Moderate"
    assert 35.0 <= data_d["final_risk_score"] < 65.0

    res_e = agent_e.run("Spending rose last year.", res_d)
    data_e = res_e["data"]
    assert data_e["sharing_recommendation"] == "Hold for Verification"


def test_case_8_agent_d_low_risk_with_sufficient_evidence(agent_d, agent_e):
    """Case 8: Agent D reports Low Risk with sufficient evidence."""
    mock_a = {"status": "completed", "data": {"label": "Factual", "confidence": 0.92}}
    mock_b = {
        "status": "completed",
        "data": {
            "evidence_status": "grounded",
            "citations": [
                {"statement_id": "c9", "claim": "The bill passed by 60 votes.", "verdict": "true", "similarity_score": 0.92},
                {"statement_id": "c10", "claim": "Senate confirmed the vote count was 60.", "verdict": "true", "similarity_score": 0.88},
            ],
        },
    }
    mock_c = {"status": "completed", "data": {"matched_stories": []}}

    res_d = agent_d.run("The bill passed by 60 votes.", mock_a, mock_b, mock_c)
    data_d = res_d["data"]

    assert data_d["simple_risk_level"] == "Low"
    assert data_d["final_risk_score"] < 35.0
    assert data_d["evidence_strength"] == "Strong"

    res_e = agent_e.run("The bill passed by 60 votes.", res_d)
    data_e = res_e["data"]
    assert data_e["sharing_recommendation"] == "Share with Context"
