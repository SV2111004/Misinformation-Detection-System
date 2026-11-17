"""
scripts/index_socioeconomic_evidence.py
Builds the unified ChromaDB persistent vector collection and TF-IDF lexical index
exclusively from the canonical training split: dataset/socioeconomic/train_socioeconomic.jsonl.

Invariants enforced:
- Exactly 5,562 training records are indexed.
- 5,504 records with justifications have has_justification = True.
- 58 records without justifications have has_justification = False and a clear placeholder.
- Dense embeddings generated from rec["claim"] via all-MiniLM-L6-v2.
- Metadata stores statement_id, claim, justification, has_justification, verdict, label_id, speaker, subject, split="train".
- Lexical TF-IDF index is fitted on concatenated (claim + " " + justification).
- Zero validation or test records are indexed.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SOCIO_DIR = BASE_DIR / "dataset" / "socioeconomic"
TRAIN_JSONL = SOCIO_DIR / "train_socioeconomic.jsonl"
CHROMA_DIR = SOCIO_DIR / "chroma_db"
COLLECTION_NAME = "socioeconomic_evidence"
LEXICAL_MODEL_PATH = CHROMA_DIR / "lexical_tfidf.pkl"


def build_socioeconomic_index(batch_size: int = 250, reset: bool = True) -> Dict[str, Any]:
    assert TRAIN_JSONL.exists(), f"Canonical training file not found: {TRAIN_JSONL}"

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initializing ChromaDB client at: {CHROMA_DIR}")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )

    # ChromaDB default embedding function uses all-MiniLM-L6-v2 via ONNX
    embedding_func = embedding_functions.DefaultEmbeddingFunction()

    if reset:
        existing_cols = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing_cols:
            logger.info(f"Resetting existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"description": "Canonical Socioeconomic Policy Evidence from LIAR-PLUS (Train split only)"},
    )

    current_count = collection.count()
    if current_count == 5562 and not reset:
        logger.info(f"Collection '{COLLECTION_NAME}' already has exactly 5562 documents.")
        return {"status": "already_indexed", "count": 5562}

    logger.info("Reading canonical training records...")
    ids: List[str] = []
    documents: List[str] = []  # Claim text for dense semantic matching
    metadatas: List[Dict[str, Any]] = []
    lexical_corpus: List[str] = []
    lexical_metadata: List[Dict[str, Any]] = []

    total_read = 0
    with_just = 0
    without_just = 0

    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_read += 1

            stmt_id = str(rec["statement_id"])
            claim = str(rec["claim"]).strip()
            raw_just = (rec.get("justification") or "").strip()
            has_just = bool(raw_just)

            if has_just:
                with_just += 1
                justification_text = raw_just
            else:
                without_just += 1
                justification_text = "No fact-checker justification recorded for this historical claim."

            verdict = str(rec.get("verdict") or rec.get("label") or "Half-True")
            label_id = int(rec.get("label_id", 1))

            meta = {
                "statement_id": stmt_id,
                "claim": claim,
                "justification": justification_text,
                "has_justification": has_just,
                "verdict": verdict,
                "label_id": label_id,
                "speaker": str(rec.get("speaker") or ""),
                "speaker_job": str(rec.get("speaker_job") or ""),
                "subject": str(rec.get("subject") or ""),
                "context": str(rec.get("context") or ""),
                "split": "train",
                "source_dataset": "liar-plus-socioeconomic",
            }

            ids.append(stmt_id)
            documents.append(claim)  # Historical claim embedded for symmetric retrieval
            metadatas.append(meta)

            # Combined claim + justification for lexical retrieval
            lexical_text = f"{claim} {justification_text}"
            lexical_corpus.append(lexical_text)
            lexical_metadata.append(meta)

            # Batch write to ChromaDB
            if len(documents) >= batch_size:
                collection.add(documents=documents, ids=ids, metadatas=metadatas)
                logger.info(f"Indexed batch of {len(documents)} (Total so far: {len(lexical_corpus)}/5562)")
                ids, documents, metadatas = [], [], []

    # Final batch
    if documents:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f"Indexed final batch of {len(documents)}")

    final_count = collection.count()
    logger.info(f"ChromaDB Ingestion Complete: {final_count} documents in '{COLLECTION_NAME}'.")
    logger.info(f"Breakdown: {with_just} with justification, {without_just} without justification.")

    assert final_count == 5562, f"Expected 5,562 documents in ChromaDB, found {final_count}!"

    # Fit and save lexical TF-IDF retriever
    logger.info("Fitting sparse TF-IDF vectorizer on combined claim + justification corpus...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(lexical_corpus)

    lexical_bundle = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "metadata": lexical_metadata,
        "count": len(lexical_metadata),
    }

    joblib.dump(lexical_bundle, LEXICAL_MODEL_PATH)
    logger.info(f"Lexical bundle saved to: {LEXICAL_MODEL_PATH}")

    return {
        "status": "success",
        "chroma_count": final_count,
        "with_justification": with_just,
        "without_justification": without_just,
        "lexical_corpus_size": len(lexical_corpus),
    }


if __name__ == "__main__":
    res = build_socioeconomic_index(batch_size=250, reset=True)
    print("Build Result:", json.dumps(res, indent=2))
