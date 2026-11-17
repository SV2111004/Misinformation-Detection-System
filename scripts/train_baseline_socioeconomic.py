"""
Train and Evaluate the 3-Class TF-IDF + Logistic Regression Baseline Model
for the Socio-Economic & Public Welfare domain.

Saves:
- models/tfidf_socioeconomic_3class.pkl
- models/baseline_socioeconomic_3class.pkl
- models/baseline_socioeconomic_3class_metrics.json
"""

import json
import joblib
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset" / "socioeconomic"
MODELS_DIR = PROJECT_ROOT / "models"

def load_data(split_file: str) -> Tuple[List[str], List[str]]:
    filepath = DATA_DIR / split_file
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            texts.append(item["claim"])
            labels.append(item["label"])
    return texts, labels

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading Socio-Economic 3-Class Datasets...")
    X_train, y_train = load_data("train_socioeconomic.jsonl")
    X_val, y_val = load_data("val_socioeconomic.jsonl")
    X_test, y_test = load_data("test_socioeconomic.jsonl")
    
    print(f"Loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 1. TF-IDF Feature Extraction
    print("Fitting TF-IDF Vectorizer (ngram=(1,2), max_features=10,000)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)
    X_test_vec = tfidf.transform(X_test)
    
    # 2. Train Logistic Regression Baseline
    print("Training Logistic Regression Model (3-class: Misleading, Half-True, Factual)...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_vec, y_train)
    
    # 3. Evaluate on Validation Set
    val_pred = clf.predict(X_val_vec)
    val_acc = accuracy_score(y_val, val_pred)
    val_mf1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    val_wf1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)
    val_report = classification_report(y_val, val_pred, output_dict=True, zero_division=0)
    val_cm = confusion_matrix(y_val, val_pred, labels=clf.classes_).tolist()
    
    # 4. Evaluate on Test Set
    test_pred = clf.predict(X_test_vec)
    test_acc = accuracy_score(y_test, test_pred)
    test_mf1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
    test_wf1 = f1_score(y_test, test_pred, average="weighted", zero_division=0)
    test_report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    test_cm = confusion_matrix(y_test, test_pred, labels=clf.classes_).tolist()
    
    print("\n" + "="*70)
    print("VALIDATION METRICS:")
    print(f"Accuracy:    {val_acc*100:.2f}%")
    print(f"Macro F1:    {val_mf1:.4f}")
    print(f"Weighted F1: {val_wf1:.4f}")
    print("\nTEST METRICS:")
    print(f"Accuracy:    {test_acc*100:.2f}%")
    print(f"Macro F1:    {test_mf1:.4f}")
    print(f"Weighted F1: {test_wf1:.4f}")
    print("="*70)
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred, zero_division=0))
    
    # 5. Save Artifacts
    tfidf_out = MODELS_DIR / "tfidf_socioeconomic_3class.pkl"
    model_out = MODELS_DIR / "baseline_socioeconomic_3class.pkl"
    metrics_out = MODELS_DIR / "baseline_socioeconomic_3class_metrics.json"
    
    joblib.dump(tfidf, tfidf_out)
    joblib.dump(clf, model_out)
    
    metrics_data = {
        "model_type": "TF-IDF (1,2) + Balanced Logistic Regression",
        "domain": "Socio-Economic & Public Welfare Policy",
        "classes": list(clf.classes_),
        "validation": {
            "accuracy": round(val_acc, 4),
            "macro_f1": round(val_mf1, 4),
            "weighted_f1": round(val_wf1, 4),
            "confusion_matrix": val_cm,
            "report": val_report
        },
        "test": {
            "accuracy": round(test_acc, 4),
            "macro_f1": round(test_mf1, 4),
            "weighted_f1": round(test_wf1, 4),
            "confusion_matrix": test_cm,
            "report": test_report
        }
    }
    
    with open(metrics_out, "w", encoding="utf-8") as f_met:
        json.dump(metrics_data, f_met, indent=2)
        
    print(f"\nModels and metrics successfully saved to {MODELS_DIR}!")

if __name__ == "__main__":
    main()
