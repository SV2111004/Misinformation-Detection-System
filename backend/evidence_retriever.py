"""
Standalone Evidence Retrieval Module for Political Misinformation Claims.

Indexes processed LIAR-PLUS historical claims and their published justifications
using TF-IDF vectorization and cosine similarity matching.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "These results represent textually similar historical claims and their published justifications "
    "from the LIAR-PLUS dataset. High textual similarity does not indicate factual correctness or "
    "independent verification of the query claim. Retrieval results are contextual references and "
    "must not be interpreted as classifier ground truth or confirmed proof. Justifications reflect "
    "the source benchmark's extracted ruling text and have not been independently reverified by this module. "
    "Original veracity labels correspond to the historical claim record in the source dataset."
)


def check_claim_justification_alignment(claim: str, speaker: str, justification: str) -> bool:
    """
    Empirically verifies whether a justification exhibits semantic or contextual
    alignment with its associated claim and speaker, detecting upstream TSV row-shift anomalies.
    """
    if not claim or not justification:
        return False
    just_lower = justification.lower()
    speaker_clean = (speaker or "").lower().replace("-", " ")
    speaker_words = [
        w for w in speaker_clean.split()
        if len(w) > 2 and w not in {"the", "and", "for", "none", "rep", "sen"}
    ]
    if any(w in just_lower for w in speaker_words):
        return True

    stopwords = {
        "about", "their", "there", "which", "would", "could", "should",
        "other", "these", "those", "state", "after", "before", "during",
        "percent", "people", "years", "first", "being", "under", "than"
    }
    claim_words = [w.strip(".,;:\"'!?()[]{}") for w in claim.lower().split()]
    informative_words = [w for w in claim_words if len(w) > 4 and w not in stopwords]
    if any(w in just_lower for w in informative_words):
        return True

    return False


def get_default_processed_dir() -> Path:
    # backend/evidence_retriever.py -> parent is backend -> parent.parent is project root
    return Path(__file__).resolve().parent.parent / "dataset" / "politifact_evidence" / "processed"


class EvidenceRetriever:
    """
    In-memory TF-IDF retriever for historical claim-justification records.
    Supports split-based filtering to prevent evaluation data leakage.
    """

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        include_splits: Optional[Tuple[str, ...]] = ("train",),
        exclude_splits: Optional[Set[str]] = None,
        max_features: int = 30000,
        sublinear_tf: bool = True,
    ):
        """
        Initialize the EvidenceRetriever.

        :param processed_dir: Path to directory containing processed JSONL files.
        :param include_splits: Tuple of split names to include (default: ("train",) to prevent evaluation leakage).
                               Set to ("train", "val", "test") or None with exclude_splits=None to index everything.
        :param exclude_splits: Optional set of split names to explicitly exclude.
        :param max_features: Maximum TF-IDF vocabulary size.
        :param sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
        """
        self.processed_dir = Path(processed_dir) if processed_dir else get_default_processed_dir()
        self.include_splits = tuple(include_splits) if include_splits else None
        self.exclude_splits = set(exclude_splits) if exclude_splits else set()
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf

        self.records: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self.total_examined = 0
        self.total_indexed = 0
        self.total_excluded = 0
        self.exclusion_reasons: Dict[str, int] = {
            "excluded_split": 0,
            "missing_claim": 0,
            "missing_justification": 0,
            "duplicate_id": 0,
        }

        self._is_indexed = False

    def index_records(self, records: List[Dict[str, Any]]) -> "EvidenceRetriever":
        """
        Directly indexes an in-memory list of record dicts.
        Applies split filtering, non-empty claim and justification checks, and duplicate removal.
        Useful for unit testing with mock data or dynamic index creation.
        """
        self.records = []
        self.total_examined = 0
        self.total_indexed = 0
        self.total_excluded = 0
        for k in self.exclusion_reasons:
            self.exclusion_reasons[k] = 0

        seen_ids: Set[str] = set()

        for rec in records:
            self.total_examined += 1

            split_name = rec.get("split", "train")
            if self.include_splits and split_name not in self.include_splits:
                self.total_excluded += 1
                self.exclusion_reasons["excluded_split"] += 1
                continue

            if split_name in self.exclude_splits:
                self.total_excluded += 1
                self.exclusion_reasons["excluded_split"] += 1
                continue

            stmt_id = (rec.get("statement_id") or "").strip()
            claim = (rec.get("claim") or "").strip()
            justification = (rec.get("justification") or "").strip()

            if not claim:
                self.total_excluded += 1
                self.exclusion_reasons["missing_claim"] += 1
                continue

            if not justification or justification.lower() == "none":
                self.total_excluded += 1
                self.exclusion_reasons["missing_justification"] += 1
                continue

            if stmt_id and stmt_id in seen_ids:
                self.total_excluded += 1
                self.exclusion_reasons["duplicate_id"] += 1
                continue

            if stmt_id:
                seen_ids.add(stmt_id)

            self.records.append(rec)
            self.total_indexed += 1

        if self.records:
            corpus = [r["claim"] for r in self.records]
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.max_features,
                sublinear_tf=self.sublinear_tf,
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

        self._is_indexed = True
        return self

    def load_and_index(self) -> "EvidenceRetriever":
        """
        Loads valid records from processed JSONL files and fits the TF-IDF matrix.
        Only records with both a non-empty claim and justification are indexed.
        """
        self.records = []
        self.total_examined = 0
        self.total_indexed = 0
        self.total_excluded = 0
        for k in self.exclusion_reasons:
            self.exclusion_reasons[k] = 0

        split_files = [
            ("train", "train_evidence.jsonl"),
            ("val", "val_evidence.jsonl"),
            ("test", "test_evidence.jsonl"),
        ]

        seen_ids: Set[str] = set()

        for split_name, fname in split_files:
            if self.include_splits and split_name not in self.include_splits:
                fpath = self.processed_dir / fname
                if fpath.exists():
                    with open(fpath, "r", encoding="utf-8") as f:
                        for _ in f:
                            self.total_examined += 1
                            self.total_excluded += 1
                            self.exclusion_reasons["excluded_split"] += 1
                continue

            if split_name in self.exclude_splits:
                fpath = self.processed_dir / fname
                if fpath.exists():
                    with open(fpath, "r", encoding="utf-8") as f:
                        for _ in f:
                            self.total_examined += 1
                            self.total_excluded += 1
                            self.exclusion_reasons["excluded_split"] += 1
                continue

            fpath = self.processed_dir / fname
            if not fpath.exists():
                logger.warning(f"Evidence file not found: {fpath}")
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    self.total_examined += 1
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        rec = json.loads(line_str)
                    except json.JSONDecodeError:
                        self.total_excluded += 1
                        continue

                    stmt_id = (rec.get("statement_id") or "").strip()
                    claim = (rec.get("claim") or "").strip()
                    justification = (rec.get("justification") or "").strip()

                    if not claim:
                        self.total_excluded += 1
                        self.exclusion_reasons["missing_claim"] += 1
                        continue

                    if not justification or justification.lower() == "none":
                        self.total_excluded += 1
                        self.exclusion_reasons["missing_justification"] += 1
                        continue

                    if stmt_id and stmt_id in seen_ids:
                        self.total_excluded += 1
                        self.exclusion_reasons["duplicate_id"] += 1
                        continue

                    if stmt_id:
                        seen_ids.add(stmt_id)

                    self.records.append(rec)
                    self.total_indexed += 1

        if self.records:
            corpus = [r["claim"] for r in self.records]
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.max_features,
                sublinear_tf=self.sublinear_tf,
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

        self._is_indexed = True
        return self

    def search(
        self,
        claim: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search historical claims textually similar to the user-supplied claim.

        :param claim: User claim text.
        :param top_k: Maximum number of results to return (bounded between 1 and 20).
        :param min_score: Minimum cosine similarity score threshold (default: > 0.0).
        :return: List of ranked match dictionaries sorted by similarity_score descending.
        """
        if not self._is_indexed:
            self.load_and_index()

        text = claim.strip()
        if not text:
            raise ValueError("Claim cannot be empty or whitespace.")

        if top_k < 1:
            raise ValueError("top_k must be a positive integer >= 1.")

        top_k = min(top_k, 20)

        # Handle empty index
        if not self.records or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # Vectorize query
        q_vec = self.vectorizer.transform([text])

        # Cosine similarity (since TF-IDF vectors are L2-normalized, dot product == cosine similarity)
        sims = (self.tfidf_matrix * q_vec.T).toarray().ravel()

        # Filter indices where similarity > min_score
        non_zero_indices = [idx for idx, s in enumerate(sims) if s > min_score]
        if not non_zero_indices:
            return []

        # Sort indices by descending similarity, breaking ties deterministically by statement_id
        non_zero_indices.sort(
            key=lambda idx: (-sims[idx], str(self.records[idx].get("statement_id") or ""))
        )
        top_indices = non_zero_indices[:top_k]

        results = []
        for idx in top_indices:
            rec = self.records[idx]
            results.append({
                "statement_id": rec.get("statement_id"),
                "claim": rec.get("claim"),
                "verdict": rec.get("verdict"),
                "justification": rec.get("justification"),
                "speaker": rec.get("speaker"),
                "speaker_job": rec.get("speaker_job"),
                "state_info": rec.get("state_info"),
                "party_affiliation": rec.get("party_affiliation"),
                "subject": rec.get("subject"),
                "context": rec.get("context"),
                "split": rec.get("split"),
                "source_dataset": rec.get("source_dataset", "liar-plus"),
                "similarity_score": round(float(sims[idx]), 4),
                "has_context_alignment": check_claim_justification_alignment(
                    rec.get("claim") or "",
                    rec.get("speaker") or "",
                    rec.get("justification") or "",
                ),
            })

        return results

    def get_index_stats(self) -> Dict[str, Any]:
        """Returns statistics about the current index state."""
        return {
            "is_indexed": self._is_indexed,
            "total_examined": self.total_examined,
            "total_indexed": self.total_indexed,
            "total_excluded": self.total_excluded,
            "exclusion_reasons": dict(self.exclusion_reasons),
            "included_splits": list(self.include_splits) if self.include_splits else "all",
            "excluded_splits": list(self.exclude_splits),
            "vocab_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
        }


# Lazy singleton instance for API usage
_retriever_instance: Optional[EvidenceRetriever] = None


def get_evidence_retriever() -> EvidenceRetriever:
    """Returns the singleton EvidenceRetriever instance, initializing it if needed."""
    global _retriever_instance
    if _retriever_instance is None or not _retriever_instance._is_indexed:
        retriever = EvidenceRetriever(include_splits=("train",))
        retriever.load_and_index()
        _retriever_instance = retriever
    return _retriever_instance


def reset_evidence_retriever():
    """Resets the singleton instance (useful for testing)."""
    global _retriever_instance
    _retriever_instance = None
