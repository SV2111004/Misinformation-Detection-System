# Agentic AI-Based Misinformation Detection and Social Network Propagation Analysis

An AI-based system for detecting and analysing misinformation using Machine Learning and Transformer-based NLP models.

## 🔍 Project Overview

The current phase focuses on classifying claims from the **LIAR dataset** into six truthfulness categories.

Two models are implemented:

- **TF-IDF + Logistic Regression** — Classical ML baseline
- **DistilBERT** — Transformer-based NLP model

The models are integrated with a **FastAPI backend** and a **React.js frontend** to provide predictions, confidence scores, and model comparison.

## 🔄 Workflow

```text
User Claim
    ↓
React Frontend
    ↓
FastAPI Backend
    ↓
┌───────────────────┐
│ TF-IDF + LR       │
│ DistilBERT        │
└─────────┬─────────┘
          ↓
Prediction + Confidence
          ↓
Model Comparison

```

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **PyTorch**
- **Hugging Face Transformers**
- **FastAPI & Uvicorn**
- **React.js**
- **Git & GitHub**

## 📊 Dataset

**LIAR Dataset**

The dataset contains six truthfulness classes:

- `true`
- `mostly-true`
- `half-true`
- `barely-true`
- `false`
- `pants-fire`

The **TF-IDF + Logistic Regression** baseline achieved approximately **25.86% validation accuracy**.

## 🚀 Future Work

- Claim extraction
- RAG-based evidence retrieval and verification
- Agentic AI workflow
- Social network graph modelling
- Influence analysis
- Misinformation propagation simulation
- Integrated interactive dashboard

## 👥 Team

- **Janvi**
- **Akshat Saxena**
- **Shubhra Varshney**

**Jaypee Institute of Information Technology, Noida**
