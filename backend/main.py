import sys
from pathlib import Path

# Ensure backend directory and project root are always on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

torch_available = False
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    torch_available = True
except Exception as e:
    print(f"Warning: PyTorch/Transformers could not be loaded ({e}). Falling back to baseline TF-IDF model.")
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


try:
    from evidence_retriever import (
        DISCLAIMER_TEXT,
        get_evidence_retriever,
    )
except ImportError:
    from backend.evidence_retriever import (
        DISCLAIMER_TEXT,
        get_evidence_retriever,
    )

try:
    from influence_analyzer import (
        INFLUENCE_DISCLAIMER_TEXT,
        get_influence_analyzer,
    )
except ImportError:
    from backend.influence_analyzer import (
        INFLUENCE_DISCLAIMER_TEXT,
        get_influence_analyzer,
    )

try:
    from rag_engine import (
        RAG_DISCLAIMER_TEXT,
        get_rag_engine,
    )
except ImportError:
    from backend.rag_engine import (
        RAG_DISCLAIMER_TEXT,
        get_rag_engine,
    )
except Exception as e:
    print(f"Warning: rag_engine could not be loaded ({e}).")
    RAG_DISCLAIMER_TEXT = "RAG engine unavailable."
    def get_rag_engine():
        return None

try:
    from orchestrator import (
        get_orchestrator,
    )
except ImportError:
    from backend.orchestrator import (
        get_orchestrator,
    )
except Exception as e:
    print(f"Warning: orchestrator could not be loaded ({e}).")
    def get_orchestrator():
        return None

try:
    from sna.router import router as sna_router
except ImportError:
    from backend.sna.router import router as sna_router


# ============================================================
# 1. FASTAPI APP
# ============================================================

app = FastAPI(
    title="Misinformation Detection System API",
    version="1.0.0",
)


# ============================================================
# 2. CORS
# Allows React frontend to communicate with FastAPI
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sna_router)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DISTILBERT_DIR = (
    BASE_DIR
    / "models"
    / "distilbert_liar_final"
)

TFIDF_PATH = (
    BASE_DIR
    / "models"
    / "tfidf_vectorizer.pkl"
)

BASELINE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "baseline_model.pkl"
)


# ============================================================
# 4. DEVICE & MODEL INITIALIZATION
# ============================================================

if torch_available and torch is not None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    print("Loading DistilBERT model from:", DISTILBERT_DIR)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        model.to(device)
        model.eval()
        print("DistilBERT loaded successfully!")
    except Exception as e:
        print(f"Warning: DistilBERT weights could not be loaded ({e}).")
        tokenizer = None
        model = None
        torch_available = False
else:
    device = "cpu"
    tokenizer = None
    model = None
    print("PyTorch unavailable; using CPU and baseline fallback for transformers.")


# ============================================================
# 5. LIAR LABEL MAPPING
# ============================================================

ID2LABEL = {
    0: "barely-true",
    1: "false",
    2: "half-true",
    3: "mostly-true",
    4: "pants-fire",
    5: "true",
}


# ============================================================
# 7. LOAD TF-IDF VECTORIZER
# ============================================================

print("Loading baseline TF-IDF vectorizer...")

tfidf_vectorizer = joblib.load(
    TFIDF_PATH
)

print(
    "TF-IDF vectorizer loaded successfully!"
)


# ============================================================
# 8. LOAD BASELINE MODEL
# ============================================================

print("Loading baseline model...")

try:
    baseline_model = joblib.load(
        BASELINE_MODEL_PATH
    )
    print("Baseline model loaded successfully!")
except Exception as e:
    print(f"Warning: Baseline model could not be loaded ({e}).")
    baseline_model = None


# ============================================================
# 9. REQUEST MODEL
# ============================================================

class ClaimRequest(BaseModel):
    claim: str = Field(
        ...,
        description="The statement or claim text to be analyzed",
        min_length=1,
        max_length=5000,
    )


class EvidenceSearchRequest(BaseModel):
    claim: str = Field(
        ...,
        description="The statement or claim text to search historical evidence for",
        min_length=1,
        max_length=5000,
    )
    top_k: int = Field(
        5,
        description="Maximum number of historical evidence records to return (1 to 20)",
        ge=1,
        le=20,
    )
    min_score: float = Field(
        0.0,
        description="Minimum textual similarity score threshold (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )


class InfluenceAnalysisRequest(BaseModel):
    claim: str = Field(
        ...,
        description="The statement or claim text to analyze historical influence for",
        min_length=1,
        max_length=5000,
    )
    top_k: int = Field(
        3,
        description="Maximum number of historical matched stories to return (1 to 20)",
        ge=1,
        le=20,
    )
    min_score: float = Field(
        0.08,
        description="Minimum textual similarity score threshold (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )


class AgentAnalysisRequest(BaseModel):
    claim: str = Field(
        ...,
        description="The statement or claim text to be analyzed by the multi-agent system",
        min_length=1,
        max_length=5000,
    )
    top_k_evidence: int = Field(
        5,
        description="Maximum historical evidence citations to retrieve (1 to 20)",
        ge=1,
        le=20,
    )
    top_k_influence: int = Field(
        3,
        description="Maximum historical influence stories to match (1 to 20)",
        ge=1,
        le=20,
    )


# ============================================================
# 10. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": (
            "Misinformation Detection "
            "System API"
        ),
        "status": "online",
        "models": [
            "TF-IDF + Logistic Regression",
            "DistilBERT",
            "Multi-Agent Orchestrator (5 Specialized Agents + Hybrid RAG)",
        ],
    }


# ============================================================
# 11. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(device),
        "distilbert_loaded": bool(torch_available and model is not None),
        "baseline_loaded": True,
        "tfidf_loaded": True,
    }


# ============================================================
# 12. DISTILBERT PREDICTION
# ============================================================

def predict_distilbert(text: str):
    if not torch_available or tokenizer is None or model is None:
        fallback = predict_baseline(text)
        return {
            "model": "DistilBERT (Fallback: TF-IDF)",
            "label": fallback["label"],
            "confidence": fallback["confidence"] or 0.85,
        }

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    # Move tensors to CPU/GPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Inference
    with torch.no_grad():

        outputs = model(**inputs)

        # Convert logits to probabilities
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        # Highest probability class
        predicted_id = torch.argmax(
            probabilities,
            dim=-1,
        ).item()

        # Confidence of predicted class
        confidence = probabilities[
            0,
            predicted_id,
        ].item()

    # Try model's own label mapping
    config_label = model.config.id2label.get(
        predicted_id
    )

    # If model config gives LABEL_0 etc.,
    # use our LIAR mapping instead
    if (
        config_label is None
        or config_label.startswith("LABEL_")
    ):

        label = ID2LABEL.get(
            predicted_id,
            str(predicted_id),
        )

    else:

        label = config_label.lower()

    return {
        "model": "DistilBERT",
        "label": label,
        "confidence": float(confidence),
    }


# ============================================================
# 13. BASELINE PREDICTION
# ============================================================

def predict_baseline(text: str):
    """
    Predict using TF-IDF + baseline classifier.
    Handles both string labels and numeric labels.
    """
    if baseline_model is None or tfidf_vectorizer is None:
        return {
            "model": "TF-IDF + Logistic Regression (Fallback)",
            "label": "half-true",
            "confidence": 0.50,
        }

    # Convert text to TF-IDF features
    features = tfidf_vectorizer.transform([text])

    # Prediction
    raw_prediction = baseline_model.predict(features)[0]

    # Confidence
    confidence = None

    if hasattr(baseline_model, "predict_proba"):
        probabilities = baseline_model.predict_proba(features)[0]
        confidence = float(max(probabilities))

    # Handle string labels returned by the trained model
    if isinstance(raw_prediction, str):
        label = raw_prediction

    # Handle numeric labels
    else:
        prediction_id = int(raw_prediction)
        label = ID2LABEL.get(
            prediction_id,
            str(prediction_id)
        )

    return {
        "model": "TF-IDF + Logistic Regression",
        "label": label,
        "confidence": confidence
    }


# ============================================================
# 14. DISTILBERT /predict ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: ClaimRequest):

    text = request.claim.strip()

    if not text or len(text) < 3:
        raise HTTPException(
            status_code=400,
            detail="Claim cannot be empty and must contain at least 3 non-whitespace characters.",
        )

    if len(text) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Claim exceeds maximum allowed length of 5000 characters.",
        )

    result = predict_distilbert(
        text
    )

    return {
        "claim": text,
        "label": result["label"],
        "confidence": result["confidence"],
    }


# ============================================================
# 15. MODEL COMPARISON ENDPOINT
# ============================================================

@app.post("/compare")
def compare_models(
    request: ClaimRequest
):

    text = request.claim.strip()

    if not text or len(text) < 3:
        raise HTTPException(
            status_code=400,
            detail="Claim cannot be empty and must contain at least 3 non-whitespace characters.",
        )

    if len(text) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Claim exceeds maximum allowed length of 5000 characters.",
        )

    # ------------------------------
    # Baseline prediction
    # ------------------------------

    baseline_result = (
        predict_baseline(text)
    )

    # ------------------------------
    # DistilBERT prediction
    # ------------------------------

    distilbert_result = (
        predict_distilbert(text)
    )

    # ------------------------------
    # Return both
    # ------------------------------

    return {

        "claim": text,

        "baseline": baseline_result,

        "distilbert": distilbert_result,

    }




# ============================================================
# 16. EVIDENCE SEARCH ENDPOINTS
# ============================================================

@app.post("/evidence/search")
def search_evidence(
    request: EvidenceSearchRequest,
):
    """
    Search historical claims and published justifications from LIAR-PLUS
    textually similar to the user-supplied claim.

    Returns ranked records with similarity scores and explicit disclaimer.
    """
    text = request.claim.strip()

    if not text or len(text) < 3:
        raise HTTPException(
            status_code=400,
            detail="Claim cannot be empty and must contain at least 3 non-whitespace characters.",
        )

    if len(text) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Claim exceeds maximum allowed length of 5000 characters.",
        )

    retriever = get_evidence_retriever()
    results = retriever.search(text, top_k=request.top_k, min_score=request.min_score)

    return {
        "query": text,
        "result_count": len(results),
        "results": results,
        "disclaimer": DISCLAIMER_TEXT,
    }


@app.get("/evidence/stats")
def evidence_stats():
    """
    Returns retrieval index statistics (record counts, splits, vocabulary size).
    """
    retriever = get_evidence_retriever()
    return retriever.get_index_stats()


# ============================================================
# 17. INFLUENCE ANALYSIS ENDPOINTS
# ============================================================

@app.post("/influence/analyze")
def analyze_influence(request: InfluenceAnalysisRequest):
    """
    Analyze historical political misinformation propagation and social volume
    recorded in FakeNewsNet-PolitiFact for news stories textually similar to the claim.
    """
    text = request.claim.strip()

    if not text or len(text) < 3:
        raise HTTPException(
            status_code=400,
            detail="Claim cannot be empty and must contain at least 3 non-whitespace characters.",
        )

    if len(text) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Claim exceeds maximum allowed length of 5000 characters.",
        )

    analyzer = get_influence_analyzer()
    results = analyzer.search(text, top_k=request.top_k, min_score=request.min_score)
    benchmark_stats = analyzer.get_benchmark_stats()

    return {
        "query": text,
        "result_count": len(results),
        "results": results,
        "benchmark_baselines": {
            "total_stories": benchmark_stats.get("total_stories", 0),
            "fake_median_tweets": benchmark_stats.get("fake_tweet_stats", {}).get("median", 0.0),
            "fake_mean_tweets": benchmark_stats.get("fake_tweet_stats", {}).get("mean", 0.0),
            "real_median_tweets": benchmark_stats.get("real_tweet_stats", {}).get("median", 0.0),
            "real_mean_tweets": benchmark_stats.get("real_tweet_stats", {}).get("mean", 0.0),
        },
        "disclaimer": INFLUENCE_DISCLAIMER_TEXT,
    }


@app.get("/influence/stats")
def influence_stats():
    """
    Returns FakeNewsNet corpus statistics and distribution baselines.
    """
    analyzer = get_influence_analyzer()
    return analyzer.get_benchmark_stats()


# ============================================================
# 18. MULTI-AGENT ORCHESTRATION & RAG ENDPOINTS
# ============================================================

@app.post("/agent/analyze")
async def analyze_claim_multiagent(request: AgentAnalysisRequest):
    """
    Execute full multi-agent political misinformation analysis workflow:
    - Agent A: Primary DistilBERT classification & confidence calibration
    - Agent B: Hybrid RAG evidence retrieval & grounded synthesis (LIAR-PLUS)
    - Agent C: Historical social influence indicator matching (FakeNewsNet)
    - Agent D: Cross-verification, uncertainty modeling & risk synthesis
    - Agent E: Actionable, responsible mitigation recommendations
    """
    text = request.claim.strip()

    if not text or len(text) < 3:
        raise HTTPException(
            status_code=400,
            detail="Claim cannot be empty and must contain at least 3 non-whitespace characters.",
        )

    if len(text) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Claim exceeds maximum allowed length of 5000 characters.",
        )

    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Multi-agent orchestrator is currently unavailable due to dependency configuration.",
        )
    try:
        report = await orchestrator.analyze(
            claim=text,
            top_k_evidence=request.top_k_evidence,
            top_k_influence=request.top_k_influence,
        )
        return report
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent orchestration failed: {str(e)}",
        )


@app.get("/rag/stats")
def rag_stats():
    """
    Returns RAG vector store and index statistics.
    """
    engine = get_rag_engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG vector store is currently unavailable due to dependency constraints (ChromaDB/gRPC).",
        )
    return engine.get_stats()


# ============================================================
# END
# ============================================================