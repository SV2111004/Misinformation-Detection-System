"""
scripts/train_distilbert_socioeconomic.py
Fine-tunes DistilBERT on the canonical Socio-Economic 3-class dataset.

Labels:
    0: Misleading
    1: Half-True
    2: Factual

Features:
- Hugging Face DistilBertForSequenceClassification
- Class-weighted CrossEntropyLoss to address class distribution (26.5% / 38.6% / 34.9%)
- Reproducible random seeds
- Evaluation on validation split for model selection
- Final evaluation on held-out test split (683 claims)
- Exports checkpoint to models/distilbert_socioeconomic_3class/
- Saves evaluation metrics and classification report to models/distilbert_socioeconomic_3class_metrics.json
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SOCIO_DIR = BASE_DIR / "dataset" / "socioeconomic"
OUTPUT_DIR = BASE_DIR / "models" / "distilbert_socioeconomic_3class"
METRICS_PATH = BASE_DIR / "models" / "distilbert_socioeconomic_3class_metrics.json"

ID2LABEL = {0: "Misleading", 1: "Half-True", 2: "Factual"}
LABEL2ID = {"Misleading": 0, "Half-True": 1, "Factual": 2}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClaimDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 128):
        self.claims = []
        self.labels = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                self.claims.append(rec["claim"])
                self.labels.append(rec["label_id"])

        self.encodings = tokenizer(
            self.claims,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_class_weights(jsonl_path: Path) -> torch.Tensor:
    counts = [0, 0, 0]
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            counts[rec["label_id"]] += 1
            total += 1

    # Balanced class weights: total / (n_classes * count)
    weights = [total / (3.0 * c) for c in counts]
    logger.info(f"Class counts: {counts}, Calculated weights: {[round(w, 4) for w in weights]}")
    return torch.tensor(weights, dtype=torch.float)


def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if criterion:
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()
    rep = classification_report(
        all_labels,
        all_preds,
        target_names=["Misleading", "Half-True", "Factual"],
        output_dict=True,
        zero_division=0,
    )
    avg_loss = total_loss / len(all_labels) if criterion else 0.0

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "confusion_matrix": cm,
        "report": rep,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT 3-class model on Socioeconomic data")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=96, help="Max sequence length")
    parser.add_argument("--freeze_lower", action="store_true", default=True, help="Freeze lower 3 layers for fast CPU convergence")
    parser.add_argument("--quick_eval", action="store_true", help="Quick evaluation mode with subset")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths
    train_path = SOCIO_DIR / "train_socioeconomic.jsonl"
    val_path = SOCIO_DIR / "val_socioeconomic.jsonl"
    test_path = SOCIO_DIR / "test_socioeconomic.jsonl"

    logger.info("Loading tokenizer: distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    logger.info("Preparing datasets...")
    train_dataset = ClaimDataset(train_path, tokenizer, max_length=args.max_length)
    val_dataset = ClaimDataset(val_path, tokenizer, max_length=args.max_length)
    test_dataset = ClaimDataset(test_path, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)

    logger.info(f"Datasets loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Model configuration
    config = AutoConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        config=config,
    )

    if args.freeze_lower:
        logger.info("Freezing lower 3 transformer layers (0, 1, 2) for rapid CPU training...")
        for name, param in model.distilbert.transformer.layer.named_parameters():
            layer_num = int(name.split(".")[0])
            if layer_num < 3:
                param.requires_grad = False

    model.to(device)

    # Class-weighted loss
    class_weights = compute_class_weights(train_path).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    logger.info(f"Starting training: {args.epochs} epochs, {len(train_loader)} steps per epoch...")
    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                logger.info(f"Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation evaluation
        val_metrics = evaluate(model, val_loader, device, criterion)
        logger.info(f"Epoch {epoch} Val Results: Acc={val_metrics['accuracy']:.4f}, MacroF1={val_metrics['macro_f1']:.4f}, WeightedF1={val_metrics['weighted_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"--> New best model saved with Val Macro F1: {best_val_f1:.4f}")

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Final evaluation on test split
    logger.info("Running final evaluation on held-out test split (683 claims)...")
    test_metrics = evaluate(model, test_loader, device, criterion)
    logger.info(f"TEST RESULTS: Accuracy={test_metrics['accuracy']:.4f}, Macro F1={test_metrics['macro_f1']:.4f}, Weighted F1={test_metrics['weighted_f1']:.4f}")
    logger.info(f"Per-Class Test F1: Misleading={test_metrics['report']['Misleading']['f1-score']:.4f}, Half-True={test_metrics['report']['Half-True']['f1-score']:.4f}, Factual={test_metrics['report']['Factual']['f1-score']:.4f}")

    # Export model artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting model artifacts to: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save metrics JSON
    metrics_data = {
        "model_name": "DistilBERT Socio-Economic 3-Class",
        "base_model": "distilbert-base-uncased",
        "domain": "Socio-Economic & Domestic Public Welfare Policy",
        "label_mapping": ID2LABEL,
        "training_params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "frozen_layers": [0, 1, 2] if args.freeze_lower else [],
            "class_weights": [round(w.item(), 4) for w in class_weights.cpu()],
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "comparison_with_baseline": {
            "baseline_test_accuracy": 0.4348,
            "baseline_test_macro_f1": 0.4288,
            "distilbert_test_accuracy": test_metrics["accuracy"],
            "distilbert_test_macro_f1": test_metrics["macro_f1"],
            "accuracy_improvement": round(test_metrics["accuracy"] - 0.4348, 4),
            "macro_f1_improvement": round(test_metrics["macro_f1"] - 0.4288, 4),
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    logger.info(f"Metrics saved successfully to: {METRICS_PATH}")
    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
