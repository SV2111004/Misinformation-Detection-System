"""
RAG Engine for Historical Political Claim & Evidence Retrieval.

Combines:
1. Dense Semantic Retrieval via ChromaDB persistent vector index (using all-MiniLM-L6-v2 embeddings)
2. Lexical Retrieval via TF-IDF cosine similarity (reusing EvidenceRetriever)
3. Hybrid Score Fusion (weighted combination of semantic + lexical scores)
4. Structured Citation Preservation with upstream TSV row-shift integrity checks
5. Grounded Evidence Summary generation (LLM-based if configured, with rigorous deterministic fallback)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
from chromadb.config import Settings

try:
    from evidence_retriever import (
        DISCLAIMER_TEXT,
        EvidenceRetriever,
        check_claim_justification_alignment,
        get_evidence_retriever,
    )
except ImportError:
    from backend.evidence_retriever import (
        DISCLAIMER_TEXT,
        EvidenceRetriever,
        check_claim_justification_alignment,
        get_evidence_retriever,
    )

logger = logging.getLogger(__name__)

RAG_DISCLAIMER_TEXT = (
    "These historical citations and grounded summaries are retrieved from the LIAR-PLUS benchmark. "
    "Semantic similarity and lexical overlap reflect proximity to historical PolitiFact fact-checks, "
    "not independent real-time verification of the submitted query claim. "
    "Historical dataset labels are strictly benchmark attributes of past claims and do not replace "
    "the primary classifier prediction. Citations should be reviewed in their original context."
)

DEFAULT_PERSIST_DIR = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "politifact_evidence"
    / "chroma_db"
)
DEFAULT_COLLECTION_NAME = "liar_plus_evidence"
DEFAULT_PROCESSED_FILE = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "politifact_evidence"
    / "processed"
    / "train_evidence.jsonl"
)


class RAGEngine:
    """
    Persistent RAG Engine providing hybrid semantic + lexical retrieval over LIAR-PLUS.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        processed_file: Optional[Path] = None,
        lexical_retriever: Optional[EvidenceRetriever] = None,
        alpha: float = 0.5,
    ):
        """
        Initialize the RAG engine.

        :param persist_dir: Path to directory for persistent ChromaDB storage.
        :param collection_name: Name of ChromaDB collection.
        :param processed_file: Path to train_evidence.jsonl for ingestion.
        :param lexical_retriever: Optional pre-initialized EvidenceRetriever.
        :param alpha: Weight for semantic similarity in hybrid fusion (0.0 = pure lexical, 1.0 = pure semantic).
        """
        self.persist_dir = Path(persist_dir) if persist_dir else DEFAULT_PERSIST_DIR
        self.collection_name = collection_name
        self.processed_file = Path(processed_file) if processed_file else DEFAULT_PROCESSED_FILE
        self.alpha = max(0.0, min(1.0, alpha))

        self.lexical_retriever = lexical_retriever or get_evidence_retriever()
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self._is_ready = False

    def initialize_client(self) -> None:
        """Initializes the persistent ChromaDB client."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "LIAR-PLUS training evidence claims and justifications"},
        )
        self._is_ready = True

    def build_index(
        self,
        limit: Optional[int] = None,
        batch_size: int = 250,
        reset_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingests records from the processed JSONL file into the ChromaDB vector collection.

        :param limit: Optional maximum number of records to index (useful for testing).
        :param batch_size: Number of records per insertion batch.
        :param reset_existing: If True, deletes and recreates the collection before indexing.
        :return: Dict containing indexing statistics.
        """
        if self.client is None or self.collection is None:
            self.initialize_client()

        if reset_existing:
            logger.info(f"Resetting existing Chroma collection '{self.collection_name}'...")
            try:
                self.client.delete_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Error resetting collection: {e}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "LIAR-PLUS training evidence claims and justifications"},
            )

        current_count = self.collection.count()
        if current_count > 0 and not reset_existing and limit is None:
            logger.info(f"Collection '{self.collection_name}' already populated with {current_count} documents.")
            return {
                "status": "already_indexed",
                "total_documents": current_count,
                "collection_name": self.collection_name,
                "persist_dir": str(self.persist_dir),
            }

        if not self.processed_file.exists():
            raise FileNotFoundError(f"Processed evidence file not found: {self.processed_file}")

        logger.info(f"Starting ingestion from {self.processed_file} into ChromaDB...")

        documents: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        seen_ids: Set[str] = set()
        total_examined = 0
        total_added = 0
        skipped = 0

        with open(self.processed_file, "r", encoding="utf-8") as f:
            for line in f:
                total_examined += 1
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    rec = json.loads(line_str)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                stmt_id = (rec.get("statement_id") or "").strip()
                claim = (rec.get("claim") or "").strip()
                justification = (rec.get("justification") or "").strip()

                if not claim or not justification or justification.lower() == "none":
                    skipped += 1
                    continue

                if not stmt_id:
                    stmt_id = f"gen_id_{total_examined}"

                if stmt_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(stmt_id)

                # Format clean metadata (Chroma values must be str, int, float, or bool)
                meta = {
                    "statement_id": stmt_id,
                    "verdict": str(rec.get("verdict") or "unknown"),
                    "speaker": str(rec.get("speaker") or ""),
                    "speaker_job": str(rec.get("speaker_job") or ""),
                    "state_info": str(rec.get("state_info") or ""),
                    "party_affiliation": str(rec.get("party_affiliation") or ""),
                    "subject": str(rec.get("subject") or ""),
                    "context": str(rec.get("context") or ""),
                    "split": str(rec.get("split") or "train"),
                    "source_dataset": str(rec.get("source_dataset") or "liar-plus"),
                    "justification": justification[:1500],  # bounded length for metadata
                }

                documents.append(claim)
                ids.append(stmt_id)
                metadatas.append(meta)
                total_added += 1

                # Batch write to ChromaDB
                if len(documents) >= batch_size:
                    self.collection.add(
                        documents=documents,
                        ids=ids,
                        metadatas=metadatas,
                    )
                    documents, ids, metadatas = [], [], []

                if limit is not None and total_added >= limit:
                    break

        # Flush any remaining items in final batch
        if documents:
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )

        logger.info(f"Ingestion complete! Added {total_added} documents to ChromaDB.")
        return {
            "status": "indexed_successfully",
            "total_examined": total_examined,
            "total_added": total_added,
            "skipped": skipped,
            "collection_count": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
        }

    def load_or_build(self, limit: Optional[int] = None) -> "RAGEngine":
        """Ensures the vector index is initialized and populated."""
        if not self._is_ready:
            self.initialize_client()

        if self.collection.count() == 0:
            logger.info("ChromaDB collection is empty. Building index now...")
            self.build_index(limit=limit)

        return self

    def search(
        self,
        claim: str,
        top_k: int = 5,
        min_score: float = 0.05,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining dense semantic search (ChromaDB) with lexical TF-IDF.

        :param claim: Submitted claim query.
        :param top_k: Number of ranked evidence records to return (1-20).
        :param min_score: Minimum hybrid similarity score threshold.
        :param alpha: Weight for semantic vs lexical (defaults to instance alpha).
        :return: List of ranked match dicts sorted by hybrid similarity score descending.
        """
        text = claim.strip()
        if not text:
            raise ValueError("Claim text cannot be empty or whitespace.")

        top_k = max(1, min(20, top_k))
        cur_alpha = self.alpha if alpha is None else max(0.0, min(1.0, alpha))

        if not self._is_ready:
            self.load_or_build()

        # Candidate pool size for hybrid fusion
        candidate_k = min(max(top_k * 4, 20), 50)

        # 1. Dense Semantic Retrieval
        semantic_matches: Dict[str, Dict[str, Any]] = {}
        total_docs = self.collection.count()

        if total_docs > 0:
            query_k = min(candidate_k, total_docs)
            try:
                chroma_results = self.collection.query(
                    query_texts=[text],
                    n_results=query_k,
                    include=["documents", "metadatas", "distances"],
                )
                if chroma_results and chroma_results.get("ids") and chroma_results["ids"][0]:
                    result_ids = chroma_results["ids"][0]
                    result_metas = chroma_results["metadatas"][0] if chroma_results.get("metadatas") else []
                    result_dists = chroma_results["distances"][0] if chroma_results.get("distances") else []
                    result_docs = chroma_results["documents"][0] if chroma_results.get("documents") else []

                    for i, doc_id in enumerate(result_ids):
                        dist = result_dists[i] if i < len(result_dists) else 1.0
                        meta = result_metas[i] if i < len(result_metas) else {}
                        doc_claim = result_docs[i] if i < len(result_docs) else ""

                        # Convert cosine distance to similarity score in [0.0, 1.0]
                        # For cosine distance: dist = 1 - cos, so sim = max(0.0, 1.0 - dist)
                        sem_score = max(0.0, min(1.0, 1.0 - (dist if dist is not None else 1.0)))

                        semantic_matches[doc_id] = {
                            "statement_id": doc_id,
                            "claim": meta.get("claim") or doc_claim,
                            "verdict": meta.get("verdict", "unknown"),
                            "justification": meta.get("justification", ""),
                            "speaker": meta.get("speaker", ""),
                            "speaker_job": meta.get("speaker_job", ""),
                            "state_info": meta.get("state_info", ""),
                            "party_affiliation": meta.get("party_affiliation", ""),
                            "subject": meta.get("subject", ""),
                            "context": meta.get("context", ""),
                            "split": meta.get("split", "train"),
                            "source_dataset": meta.get("source_dataset", "liar-plus"),
                            "semantic_score": round(float(sem_score), 4),
                        }
            except Exception as e:
                logger.warning(f"ChromaDB semantic search failed: {e}")

        # 2. Lexical TF-IDF Retrieval
        lexical_matches: Dict[str, Dict[str, Any]] = {}
        try:
            tfidf_results = self.lexical_retriever.search(text, top_k=candidate_k, min_score=0.0)
            for res in tfidf_results:
                sid = str(res.get("statement_id") or "")
                if sid:
                    lexical_matches[sid] = res
        except Exception as e:
            logger.warning(f"Lexical TF-IDF retrieval failed: {e}")

        # 3. Hybrid Score Fusion
        all_candidate_ids: Set[str] = set(semantic_matches.keys()).union(set(lexical_matches.keys()))
        if not all_candidate_ids:
            return []

        ranked_results: List[Dict[str, Any]] = []

        for sid in all_candidate_ids:
            sem_item = semantic_matches.get(sid)
            lex_item = lexical_matches.get(sid)

            sem_score = sem_item["semantic_score"] if sem_item else 0.0
            lex_score = float(lex_item.get("similarity_score") or 0.0) if lex_item else 0.0

            # Linear hybrid fusion
            hybrid_score = round(cur_alpha * sem_score + (1.0 - cur_alpha) * lex_score, 4)

            if hybrid_score < min_score:
                continue

            # Merge record fields prioritizing richer metadata
            base_rec = lex_item or sem_item or {}
            claim_text = base_rec.get("claim") or ""
            speaker_text = base_rec.get("speaker") or ""
            justification_text = base_rec.get("justification") or ""

            # Check upstream TSV row-shift alignment
            has_context_alignment = check_claim_justification_alignment(
                claim=claim_text,
                speaker=speaker_text,
                justification=justification_text,
            )

            ranked_results.append({
                "statement_id": sid,
                "claim": claim_text,
                "verdict": base_rec.get("verdict", "unknown"),
                "justification": justification_text,
                "speaker": speaker_text,
                "speaker_job": base_rec.get("speaker_job", ""),
                "state_info": base_rec.get("state_info", ""),
                "party_affiliation": base_rec.get("party_affiliation", ""),
                "subject": base_rec.get("subject", ""),
                "context": base_rec.get("context", ""),
                "split": base_rec.get("split", "train"),
                "source_dataset": base_rec.get("source_dataset", "liar-plus"),
                "similarity_score": hybrid_score,
                "semantic_score": round(sem_score, 4),
                "lexical_score": round(lex_score, 4),
                "has_context_alignment": has_context_alignment,
            })

        # Sort descending by hybrid similarity, breaking ties deterministically by statement_id
        ranked_results.sort(
            key=lambda item: (-item["similarity_score"], str(item["statement_id"]))
        )

        return ranked_results[:top_k]

    def generate_grounded_summary(
        self,
        claim: str,
        evidence_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generates a grounded evidence summary based on retrieved historical records.

        If an LLM is configured via environment variables (e.g., GEMINI_API_KEY, OPENAI_API_KEY),
        the LLM synthesizes the summary. Otherwise, an objective deterministic grounded synthesis
        is produced that explicitly cites source IDs, flags historical label distributions,
        and reports insufficiency or alignment warnings.
        """
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if not evidence_records:
            return {
                "summary": (
                    "Insufficient historical evidence: No closely matching historical claims or "
                    "published PolitiFact justifications were found in the LIAR-PLUS training benchmark "
                    "meeting the minimum similarity threshold. The system cannot ground this claim in "
                    "past fact-checking records."
                ),
                "evidence_status": "insufficient",
                "citations_count": 0,
                "has_conflicts": False,
                "generation_mode": "deterministic_fallback",
            }

        # Analyze historical verdicts
        verdict_counts: Dict[str, int] = {}
        for r in evidence_records:
            v = r.get("verdict", "unknown")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        has_conflicts = len(verdict_counts) > 1
        misaligned_count = sum(1 for r in evidence_records if not r.get("has_context_alignment", True))

        # Check if an LLM is configured
        if gemini_key or openai_key:
            try:
                llm_summary = self._call_llm_for_summary(claim, evidence_records, gemini_key, openai_key)
                if llm_summary:
                    return {
                        "summary": llm_summary,
                        "evidence_status": "grounded",
                        "citations_count": len(evidence_records),
                        "has_conflicts": has_conflicts,
                        "verdict_distribution": verdict_counts,
                        "misaligned_citations_count": misaligned_count,
                        "generation_mode": "llm_grounded",
                    }
            except Exception as e:
                logger.warning(f"LLM synthesis invocation failed, falling back to deterministic: {e}")

        # Deterministic grounded synthesis
        top_rec = evidence_records[0]
        top_id = top_rec.get("statement_id")
        top_claim = top_rec.get("claim")
        top_verdict = top_rec.get("verdict")
        top_speaker = top_rec.get("speaker") or "an unidentified speaker"
        top_sim = top_rec.get("similarity_score", 0.0)

        summary_parts = [
            f"Retrieved {len(evidence_records)} relevant historical evidence record(s) from the LIAR-PLUS benchmark.",
            f"The closest historical match is Statement [{top_id}] by {top_speaker} "
            f"(hybrid similarity: {top_sim:.1%}), which was historically rated '{top_verdict}': \"{top_claim}\".",
        ]

        if has_conflicts:
            dist_str = ", ".join(f"{k}: {v}" for k, v in verdict_counts.items())
            summary_parts.append(
                f"Benchmark Divergence Detected: Retrieved historical claims show divergent historical verdicts ({dist_str}). "
                "Because past rulings addressed different nuances, historical precedent does not present a single consensus."
            )
        else:
            summary_parts.append(
                f"Historical Precedent Consensus: All {len(evidence_records)} retrieved benchmark records share the historical rating '{top_verdict}'."
            )

        if misaligned_count > 0:
            summary_parts.append(
                f"Data Quality Warning: {misaligned_count} of {len(evidence_records)} citation(s) exhibit potential upstream TSV row-shift anomalies "
                "between claim and justification text. Exercise caution when interpreting justification excerpts."
            )

        summary_parts.append(
            "Note: These citations provide historical context from past PolitiFact articles and do not constitute independent real-time verification."
        )

        return {
            "summary": " ".join(summary_parts),
            "evidence_status": "grounded" if top_sim >= 0.15 else "weak_precedent",
            "citations_count": len(evidence_records),
            "has_conflicts": has_conflicts,
            "verdict_distribution": verdict_counts,
            "misaligned_citations_count": misaligned_count,
            "generation_mode": "deterministic_fallback",
        }

    def _call_llm_for_summary(
        self,
        claim: str,
        evidence_records: List[Dict[str, Any]],
        gemini_key: Optional[str],
        openai_key: Optional[str],
    ) -> Optional[str]:
        """Helper to invoke configured LLM with strict grounding constraints."""
        import httpx

        citations_text = "\n\n".join(
            f"Citation [{i+1}] (ID: {r.get('statement_id')}, Historical Verdict: {r.get('verdict')}, Speaker: {r.get('speaker')}):\n"
            f"Claim: {r.get('claim')}\n"
            f"Justification: {r.get('justification')[:300]}"
            for i, r in enumerate(evidence_records[:5])
        )

        prompt = (
            f"You are a neutral fact-checking evidence synthesizer.\n"
            f"Submitted claim to analyze: \"{claim}\"\n\n"
            f"Historical Evidence Citations from LIAR-PLUS Benchmark:\n{citations_text}\n\n"
            f"Instructions:\n"
            f"1. Summarize the historical evidence strictly using ONLY the provided citations above.\n"
            f"2. Cite each claim using its Citation [number] or statement ID.\n"
            f"3. Explicitly report whether the historical verdicts agree or conflict.\n"
            f"4. Do not state whether the submitted claim is definitively true or false; state only what historical precedent shows.\n"
            f"5. Keep the summary under 120 words."
        )

        if gemini_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = httpx.post(url, json=payload, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        if openai_key:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are an objective academic evidence synthesizer."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.2,
            }
            resp = httpx.post(url, json=payload, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on the RAG collection."""
        if not self._is_ready:
            self.load_or_build()

        doc_count = self.collection.count() if self.collection else 0
        return {
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "document_count": doc_count,
            "is_persistent": True,
            "alpha": self.alpha,
            "vector_store": "ChromaDB (Local Persistent Client)",
            "embedding_model": "all-MiniLM-L6-v2 (ONNX Runtime)",
            "lexical_index_stats": self.lexical_retriever.get_index_stats(),
        }


# Lazy singleton instance
_rag_engine_instance: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Returns singleton RAGEngine instance."""
    global _rag_engine_instance
    if _rag_engine_instance is None or not _rag_engine_instance._is_ready:
        engine = RAGEngine()
        engine.load_or_build()
        _rag_engine_instance = engine
    return _rag_engine_instance


def reset_rag_engine():
    """Resets singleton instance."""
    global _rag_engine_instance
    _rag_engine_instance = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Building / verifying RAG persistent index...")
    engine = RAGEngine()
    stats = engine.load_or_build()
    print("RAG Engine ready! Stats:", engine.get_stats())

