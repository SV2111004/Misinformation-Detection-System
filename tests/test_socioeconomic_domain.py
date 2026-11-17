"""
Tests for Socio-Economic & Domestic Public Welfare Domain (3-Class Implementation):
- Verifies dataset integrity in dataset/socioeconomic/
- Verifies 3-Class baseline model loading and inference
- Verifies /predict/socioeconomic and /domain/stats API endpoints
- Verifies multi-agent conflict handling with 3-class labels
"""

import json
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_socioeconomic_dataset_files():
    """Verify that filtered socio-economic dataset files exist and have valid structure."""
    data_dir = PROJECT_ROOT / "dataset" / "socioeconomic"
    assert data_dir.exists(), "dataset/socioeconomic/ must exist"
    
    summary_path = data_dir / "dataset_summary.json"
    assert summary_path.exists(), "dataset_summary.json must exist"
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    assert summary["domain_name"] == "Socio-Economic & Domestic Public Welfare Policy"
    assert summary["labels"] == ["Misleading", "Half-True", "Factual"]
    assert summary["total_records"] == 6944
    assert summary["splits"]["train"]["record_count"] == 5562
    assert summary["splits"]["val"]["record_count"] == 699
    assert summary["splits"]["test"]["record_count"] == 683
    assert summary["justification_coverage"]["percentage"] > 99.0

def test_socioeconomic_baseline_model_loading():
    """Verify that the 3-class baseline model and vectorizer load and predict correctly."""
    from backend.main import (
        socio_baseline_model,
        socio_tfidf_vectorizer,
        predict_socioeconomic_3class,
    )
    
    assert socio_baseline_model is not None, "socio_baseline_model must be loaded"
    assert socio_tfidf_vectorizer is not None, "socio_tfidf_vectorizer must be loaded"
    
    res = predict_socioeconomic_3class("The state budget deficit increased by 10 billion dollars.")
    assert "label" in res
    assert res["label"] in {"Misleading", "Half-True", "Factual"}
    assert 0.0 <= res["confidence"] <= 1.0
    assert "all_probabilities" in res
    assert len(res["all_probabilities"]) == 3

def test_socioeconomic_api_endpoints():
    """Verify FastAPI /predict/socioeconomic and /domain/stats endpoints."""
    from backend.main import app
    client = TestClient(app)
    
    # 1. /domain/stats
    res = client.get("/domain/stats")
    assert res.status_code == 200
    stats = res.json()
    assert stats["total_records"] == 6944
    assert stats["labels"] == ["Misleading", "Half-True", "Factual"]
    
    # 2. /predict/socioeconomic
    pred_res = client.post(
        "/predict/socioeconomic",
        json={"claim": "The unemployment rate dropped to 3.8 percent under the current administration."}
    )
    assert pred_res.status_code == 200
    pred = pred_res.json()
    assert pred["domain"] == "Socio-Economic & Domestic Public Welfare Policy"
    assert pred["label"] in {"Misleading", "Half-True", "Factual"}
    assert 0.0 <= pred["confidence"] <= 1.0
    assert "all_probabilities" in pred

def test_verification_risk_agent_with_3class_labels():
    """Verify that VerificationRiskAgent handles 3-class labels properly."""
    from backend.agents import VerificationRiskAgent
    agent = VerificationRiskAgent()
    
    # Concurrence on Misleading / False
    claim_analysis = {
        "status": "completed",
        "data": {"label": "Misleading", "confidence": 0.85}
    }
    evidence_rag = {
        "status": "completed",
        "data": {
            "citations": [{"statement_id": "s1.json", "verdict": "false"}],
            "evidence_status": "corroborated"
        }
    }
    influence_analysis = {"status": "completed", "data": {"matched_stories": []}}
    
    res = agent.run("Federal tax cut only benefited billionaires.", claim_analysis, evidence_rag, influence_analysis)
    assert res["status"] == "completed"
    assert res["data"]["risk_level"] == "HIGH_MISINFORMATION_RISK"
    assert any("misleading or non-factual" in f.lower() for f in res["data"]["findings"])
    
    # Concurrence on Factual / True
    claim_analysis_fact = {
        "status": "completed",
        "data": {"label": "Factual", "confidence": 0.90}
    }
    evidence_rag_fact = {
        "status": "completed",
        "data": {
            "citations": [{"statement_id": "s2.json", "verdict": "mostly-true"}],
            "evidence_status": "corroborated"
        }
    }
    res_fact = agent.run("Medicare spending increased in 2021.", claim_analysis_fact, evidence_rag_fact, influence_analysis)
    assert res_fact["data"]["risk_level"] == "LOW_MISINFORMATION_RISK"
    assert any("substantially factual" in f.lower() for f in res_fact["data"]["findings"])
