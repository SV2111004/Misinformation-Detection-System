from pathlib import Path

import joblib
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


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
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


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
# 6. LOAD DISTILBERT
# ============================================================

print(
    "Loading DistilBERT model from:",
    DISTILBERT_DIR
)

tokenizer = AutoTokenizer.from_pretrained(
    str(DISTILBERT_DIR)
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(DISTILBERT_DIR)
)

model.to(device)

model.eval()

print("DistilBERT loaded successfully!")


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

baseline_model = joblib.load(
    BASELINE_MODEL_PATH
)

print(
    "Baseline model loaded successfully!"
)


# ============================================================
# 9. REQUEST MODEL
# ============================================================

class ClaimRequest(BaseModel):
    claim: str


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
        "distilbert_loaded": True,
        "baseline_loaded": True,
        "tfidf_loaded": True,
    }


# ============================================================
# 12. DISTILBERT PREDICTION
# ============================================================

def predict_distilbert(text: str):

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

    if not text:

        return {
            "error": (
                "Claim cannot be empty."
            )
        }

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

    if not text:

        return {
            "error": (
                "Claim cannot be empty."
            )
        }

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
# END
# ============================================================