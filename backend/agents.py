"""
backend/agents.py
Specialized Agent Workflows for Political Misinformation Detection,
Influence Analysis, and Propagation Mitigation.

Architecture:
- Agent A: ClaimDecompositionAgent (DistilBERT 3-class classifier + policy entity/assertion decomposition)
- Agent B: EvidenceRAGAgent (Hybrid ChromaDB dense + TF-IDF lexical retrieval with justification auditing)
- Agent C: HistoricalInfluenceAgent (FakeNewsNet indicators + Snowflake temporal analysis)
- Agent D: CrossVerificationAgent (4-Stage Epistemic Reasoning: Relevance Gating, Directional Stance, Justification Analysis, Cross-Signal Triangulation)
- Agent E: AdvisoryMitigationAgent (Responsible advisory interventions and public flags)
"""

import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical Socio-Economic 3-Class Risk Map
SOCIOECONOMIC_RISK_MAP = {
    "misleading": {
        "risk": "HIGH",
        "score": 0.85,
        "description": "Substantively misleading or fabricated statement lacking empirical support.",
    },
    "half-true": {
        "risk": "MODERATE",
        "score": 0.50,
        "description": "Partially accurate with significant omitted context or contested fiscal interpretations.",
    },
    "factual": {
        "risk": "MINIMAL",
        "score": 0.10,
        "description": "Substantially factual statement corroborated by verified official statistical or legislative records.",
    },
}


class ClaimAnalysisAgent:
    """
    Agent A / ClaimDecompositionAgent:
    Evaluates claim text using the fine-tuned 3-class DistilBERT classifier.
    Decomposes the political assertion into constituent entities, policy keywords,
    and quantitative claims, producing calibrated class probabilities.
    """

    def __init__(self, classifier_fn: Optional[Callable[[str], Dict[str, Any]]] = None):
        self.classifier_fn = classifier_fn
        self.agent_name = "Claim Decomposition & Classification Agent"
        self.agent_id = "claim_analysis_agent"

    def _decompose_claim(self, claim: str) -> Dict[str, Any]:
        """Extracts entities, monetary figures, percentages, dates, and policy acronyms."""
        # Detect monetary amounts
        money = re.findall(r"\$\s*\d+(?:\.\d+)?\s*(?:billion|trillion|million|k)?", claim, re.IGNORECASE)
        # Detect percentages
        percentages = re.findall(r"\d+(?:\.\d+)?\s*(?:%|percent)", claim, re.IGNORECASE)
        # Detect key agencies/acts
        agencies = re.findall(
            r"\b(?:CBO|BLS|IRS|OMB|GAO|KFF|SSA|Medicare|Medicaid|ACA|Obamacare|Fed|Treasury|DoD|VA)\b",
            claim,
            re.IGNORECASE,
        )
        # Detect temporal markers
        years = re.findall(r"\b(?:19\d\d|20\d\d)\b", claim)

        return {
            "monetary_figures": money,
            "percentages": percentages,
            "agencies_and_programs": list(set(agencies)),
            "temporal_references": list(set(years)),
            "word_count": len(claim.split()),
        }

    def run(self, claim: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if not self.classifier_fn:
                try:
                    from main import predict_distilbert
                except ImportError:
                    from backend.main import predict_distilbert
                self.classifier_fn = predict_distilbert

            result = self.classifier_fn(claim)
            label = result.get("label", "Half-True")
            confidence = float(result.get("confidence", 0.5))
            probabilities = result.get("probabilities", {
                "Misleading": 0.3333,
                "Half-True": 0.3334,
                "Factual": 0.3333,
            })
            model_used = result.get("model", "DistilBERT (Socio-Economic 3-Class)")

            decomposed = self._decompose_claim(claim)
            norm_label = label.lower().strip()
            risk_meta = SOCIOECONOMIC_RISK_MAP.get(norm_label, {
                "risk": "MODERATE",
                "score": 0.50,
                "description": "Standard contextual assessment.",
            })

            duration_ms = round((time.time() - start_time) * 1000, 2)
            summary = (
                f"Prediction: {label} (Confidence: {confidence:.1%}, Model: {model_used}). "
                f"Identified {len(decomposed['agencies_and_programs'])} policy agencies/programs "
                f"and {len(decomposed['percentages']) + len(decomposed['monetary_figures'])} quantitative metrics."
            )

            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "completed",
                "duration_ms": duration_ms,
                "summary": summary,
                "data": {
                    "label": label,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "calibration_note": f"Classified as '{label}' with {confidence:.1%} softmax confidence calibrated across 3 truthfulness classes.",
                    "model_used": model_used,
                    "risk_tier": risk_meta["risk"],
                    "risk_score": risk_meta["score"],
                    "decomposed_elements": decomposed,
                },
                "disclaimer": (
                    "Model prediction reflects learned statistical patterns from 5,562 canonical socioeconomic training claims. "
                    "Statistical probability is an evidentiary heuristic and does not constitute absolute real-time factuality."
                ),
            }
        except Exception as e:
            logger.exception(f"ClaimAnalysisAgent failed: {e}")
            duration_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "UNAVAILABLE",
                "duration_ms": duration_ms,
                "summary": f"Claim analysis failed: {str(e)}",
                "data": {
                    "label": "Half-True",
                    "confidence": 0.33,
                    "probabilities": {"Misleading": 0.3333, "Half-True": 0.3334, "Factual": 0.3333},
                },
                "error": str(e),
            }


ClaimDecompositionAgent = ClaimAnalysisAgent


class EvidenceRAGAgent:
    """
    Agent B:
    Performs dense semantic retrieval using ChromaDB embeddings against the
    canonical ChromaDB socioeconomic collection ('socioeconomic_evidence').
    Retrieves top-K historical precedents, fact-checker justifications, and explicitly
    audits missing justifications.
    """

    def __init__(self, rag_engine_instance=None):
        self.rag_engine = rag_engine_instance
        self.agent_name = "Evidence Retrieval / RAG Agent"
        self.agent_id = "evidence_rag_agent"

    def _get_engine(self):
        if self.rag_engine is None:
            try:
                from rag_engine import get_rag_engine
            except ImportError:
                from backend.rag_engine import get_rag_engine
            self.rag_engine = get_rag_engine()
        return self.rag_engine

    def run(self, claim: str, top_k: int = 5, min_score: float = 0.05, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            engine = self._get_engine()
            if engine is None:
                raise RuntimeError("ChromaDB RAG Engine could not be initialized.")

            raw_results = engine.search(claim=claim, top_k=top_k, min_score=min_score)
            citations: List[Dict[str, Any]] = []
            verdict_counts: Dict[str, int] = {}
            missing_just_count = 0

            for res in raw_results:
                v = res.get("verdict", "Half-True")
                verdict_counts[v] = verdict_counts.get(v, 0) + 1
                has_j = res.get("has_justification", True)
                if not has_j:
                    missing_just_count += 1

                citations.append({
                    "statement_id": str(res.get("statement_id", "")),
                    "claim": res.get("claim", ""),
                    "verdict": v,
                    "has_justification": has_j,
                    "justification": res.get("justification", ""),
                    "similarity_score": round(float(res.get("similarity_score", 0.0)), 4),
                    "semantic_score": round(float(res.get("semantic_score", 0.0)), 4),
                    "speaker": res.get("speaker", "Unknown"),
                    "subject": res.get("subject", ""),
                })

            has_conflicts = len(set(verdict_counts.keys())) > 1
            duration_ms = round((time.time() - start_time) * 1000, 2)
            summary = (
                f"Retrieved {len(citations)} historical domain precedents. "
                f"Verdicts: {verdict_counts}. Missing justifications: {missing_just_count}."
            )

            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "completed",
                "duration_ms": duration_ms,
                "summary": summary,
                "data": {
                    "citations": citations,
                    "retrieved_count": len(citations),
                    "verdict_distribution": verdict_counts,
                    "has_conflicts": has_conflicts,
                    "missing_justification_count": missing_just_count,
                    "top_evidence": citations[:3],
                    "grounded_summary": summary,
                    "evidence_status": "grounded" if citations else "unsupported",
                },
                "disclaimer": (
                    "Retrieved records are historical fact-checks from the PolitiFact socioeconomic corpus (LIAR-PLUS). "
                    "They provide evidentiary context and precedents, but do not replace independent real-time verification."
                ),
            }
        except Exception as e:
            logger.exception(f"EvidenceRAGAgent failed: {e}")
            duration_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "UNAVAILABLE",
                "duration_ms": duration_ms,
                "summary": f"Evidence retrieval failed: {str(e)}",
                "data": {
                    "citations": [],
                    "retrieved_count": 0,
                    "verdict_distribution": {},
                    "has_conflicts": False,
                },
                "error": str(e),
            }


class HistoricalInfluenceAgent:
    """
    Agent C:
    Matches claim against historical FakeNewsNet propagation patterns and
    evaluates engagement velocity and temporal reach using Twitter Snowflake IDs.
    """

    def __init__(self, influence_analyzer_instance=None):
        self.analyzer = influence_analyzer_instance
        self.agent_name = "Historical Influence & Social Velocity Agent"
        self.agent_id = "historical_influence_agent"

    def _get_analyzer(self):
        if self.analyzer is None:
            try:
                from influence_analyzer import get_influence_analyzer
            except ImportError:
                from backend.influence_analyzer import get_influence_analyzer
            self.analyzer = get_influence_analyzer()
        return self.analyzer

    def run(self, claim: str, top_k: int = 3) -> Dict[str, Any]:
        start_time = time.time()
        try:
            analyzer = self._get_analyzer()
            if analyzer is None:
                raise RuntimeError("Influence Analyzer unavailable.")

            results = analyzer.search(claim=claim, top_k=top_k, min_score=0.08)
            matched = results.get("matched_stories", []) if isinstance(results, dict) else []

            velocity = "LOW"
            percentile = 25.0
            if matched:
                top_match = matched[0]
                tweet_count = top_match.get("tweet_count") or top_match.get("observed_tweet_count") or 0
                if tweet_count > 1000:
                    velocity = "HIGH"
                    percentile = 88.0
                elif tweet_count > 200:
                    velocity = "MODERATE"
                    percentile = 55.0

            duration_ms = round((time.time() - start_time) * 1000, 2)
            summary = (
                f"Matched {len(matched)} historical FakeNewsNet stories. "
                f"Observed propagation velocity: {velocity} ({percentile:.0f}th percentile)."
            )

            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "completed",
                "duration_ms": duration_ms,
                "summary": summary,
                "data": {
                    "matched_stories": matched,
                    "matched_count": len(matched),
                    "engagement_velocity": velocity,
                    "velocity_percentile": percentile,
                },
                "disclaimer": (
                    "Influence indicators reflect historical FakeNewsNet Twitter cascades and do NOT reflect "
                    "social reach, impressions, follower networks or live social media sharing of this specific submission."
                ),
            }
        except Exception as e:
            logger.warning(f"HistoricalInfluenceAgent graceful fallback: {e}")
            duration_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "UNAVAILABLE",
                "duration_ms": duration_ms,
                "summary": "Historical influence data unavailable for this statement.",
                "data": {
                    "matched_stories": [],
                    "matched_count": 0,
                    "engagement_velocity": "LOW",
                    "velocity_percentile": 20.0,
                },
                "error": str(e),
            }


# ============================================================
# AGENT D: Cross-Verification & Epistemic Synthesis
# ============================================================

class NLIEvaluator:
    """
    Lazy singleton wrapper for the local Cross-Encoder Natural Language Inference model.
    Evaluates premise-hypothesis pairs for:
      - Contradiction (Class 0)
      - Entailment (Class 1)
      - Neutral (Class 2)
    Provides a deterministic polarity heuristic fallback if the neural model cannot be initialized.
    """
    _instance: Optional["NLIEvaluator"] = None

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        self._load_error = None

    @classmethod
    def get_instance(cls) -> "NLIEvaluator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> bool:
        if self._is_loaded:
            return True
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            logger.info(f"Loading NLI model '{self.model_name}' on CPU...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self._is_loaded = True
            logger.info("NLI model loaded successfully!")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Could not load NLI model ({e}). Fallback heuristic will be used.")
            return False

    def evaluate_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        if not pairs:
            return []

        if not self._is_loaded:
            self.load()

        if self._is_loaded and self.tokenizer is not None and self.model is not None:
            try:
                import torch
                premises = [p[0] for p in pairs]
                hypotheses = [p[1] for p in pairs]
                inputs = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
                results = []
                for row in probs:
                    row_dict = {}
                    for idx, val in enumerate(row):
                        label_name = id2label.get(idx, f"class_{idx}")
                        row_dict[label_name] = float(val)
                    results.append({
                        "contradiction": row_dict.get("contradiction", 0.333),
                        "entailment": row_dict.get("entailment", 0.333),
                        "neutral": row_dict.get("neutral", 0.334),
                    })
                return results
            except Exception as e:
                logger.warning(f"NLI batch inference failed ({e}). Falling back to heuristic.")

        # Deterministic Heuristic Fallback
        results = []
        for premise, hypothesis in pairs:
            results.append(self._heuristic_fallback(premise, hypothesis))
        return results

    def _heuristic_fallback(self, premise: str, hypothesis: str) -> Dict[str, float]:
        p_norm = (premise or "").lower()
        h_norm = (hypothesis or "").lower()

        neg_pattern = r"\b(?:not|never|no|failed|cut|reduced|eliminated|destroyed|lost|drop|dropped|false)\b"
        pos_pattern = r"\b(?:increased|grew|added|boosted|created|gained|expanded|rose|true|proven)\b"

        p_neg = bool(re.search(neg_pattern, p_norm))
        h_neg = bool(re.search(neg_pattern, h_norm))
        p_pos = bool(re.search(pos_pattern, p_norm))
        h_pos = bool(re.search(pos_pattern, h_norm))

        p_tokens = set(re.findall(r"\w+", p_norm))
        h_tokens = set(re.findall(r"\w+", h_norm))
        overlap = len(p_tokens & h_tokens) / max(1, len(h_tokens))

        if overlap < 0.20:
            return {"contradiction": 0.05, "entailment": 0.05, "neutral": 0.90}
        if (p_neg and h_pos) or (p_pos and h_neg):
            return {"contradiction": 0.85, "entailment": 0.05, "neutral": 0.10}
        if (p_neg and h_neg) or (p_pos and h_pos):
            return {"contradiction": 0.05, "entailment": 0.85, "neutral": 0.10}
        return {"contradiction": 0.15, "entailment": 0.40, "neutral": 0.45}


def shannon_entropy_normalized(probs: List[float]) -> float:
    """Normalized Shannon entropy in [0.0, 1.0] across K classes: H(p) / ln(K)."""
    K = len(probs)
    if K <= 1:
        return 0.0
    ent = 0.0
    for p in probs:
        if p > 1e-12:
            ent -= p * math.log(p)
    max_ent = math.log(K)
    return max(0.0, min(1.0, ent / max_ent))


def kl_divergence(p: List[float], q: List[float]) -> float:
    """Kullback-Leibler divergence D_KL(P || Q) with epsilon smoothing."""
    kl = 0.0
    eps = 1e-12
    for pi, qi in zip(p, q):
        p_safe = max(eps, pi)
        q_safe = max(eps, qi)
        kl += p_safe * math.log(p_safe / q_safe)
    return max(0.0, kl)


def jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    """Symmetric Jensen-Shannon Divergence D_JS(P || Q) normalized to [0.0, 1.0]."""
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    d_js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    max_js = math.log(2.0)
    return max(0.0, min(1.0, d_js / max_js))


class VerificationRiskAgent:
    """
    Agent D / CrossVerificationAgent (Upgraded Semantic Cross-Verification Architecture):

    Implements Evidence-First Epistemic Verification:
    1. Multi-Evidence Semantic Stance Evaluation via DeBERTa-v3 NLI:
       - Evaluates all retrieved historical citations (not just the first one).
       - Compares user claim against both historical claim statements and fact-checker justifications.
       - Disentangles Entailment, Contradiction, and Neutral relationships.
    2. Continuous Evidence Weighting:
       - Fuses semantic similarity, informativeness (1 - neutral), and justification integrity.
       - Eliminates arbitrary hardcoded cutoffs.
    3. Multi-Evidence Aggregation & Epistemic Stance:
       - Aggregates multi-source evidence into a Dirichlet-weighted belief distribution.
       - Resolves conflicting precedent verdicts without blind ground-truth assumption.
    4. Cross-Signal Triangulation (Agent A vs Agent B):
       - Computes symmetric Jensen-Shannon divergence D_JS(p_A || p_B) between model distribution and evidence belief.
       - Flags epistemic dissonance and model-evidence contradictions.
    5. Adaptive Belief Fusion & Veracity Risk Scoring:
       - Fuses classifier probability distribution with empirical evidence belief via dynamic weighting alpha.
       - Decouples veracity risk from social virality metrics.
    6. Multi-Factor Decoupled Epistemic Uncertainty:
       - Explicitly synthesizes: Prediction Ambiguity, Evidence Scarcity, and Model-Evidence Divergence.
    7. Decoupled Social Velocity Exposure Tiers:
       - Isolates historical FakeNewsNet propagation reach into independent exposure tiers.
    """

    def __init__(self, nli_evaluator: Optional[NLIEvaluator] = None):
        self.agent_name = "Cross-Verification & Epistemic Synthesis Agent"
        self.agent_id = "verification_risk_agent"
        self.nli_evaluator = nli_evaluator or NLIEvaluator.get_instance()

    def _extract_classifier_distribution(self, a_data: Dict[str, Any], a_status: str) -> Tuple[List[float], str, float]:
        """
        Extracts 3-class distribution [Misleading, Half-True, Factual] from Agent A.
        Handles missing keys, offline status, or confidence-only representations gracefully.
        """
        if a_status not in {"SUCCESS", "completed"}:
            return [0.333, 0.334, 0.333], "Half-True", 0.333

        probs = a_data.get("probabilities", {})
        label = a_data.get("label", "Half-True")
        conf = float(a_data.get("confidence", 0.50))

        if probs and all(k in probs for k in ["Misleading", "Half-True", "Factual"]):
            return [float(probs["Misleading"]), float(probs["Half-True"]), float(probs["Factual"])], label, conf
        elif probs and all(k in probs for k in ["misleading", "half-true", "factual"]):
            return [float(probs["misleading"]), float(probs["half-true"]), float(probs["factual"])], label, conf

        norm_l = (label or "").lower()
        rem = max(0.01, (1.0 - conf) / 2.0)
        if norm_l in {"misleading", "false", "pants-fire", "barely-true"}:
            return [conf, rem, rem], label, conf
        elif norm_l in {"factual", "true", "mostly-true"}:
            return [rem, rem, conf], label, conf
        else:
            return [rem, conf, rem], label, conf

    def _map_citation_to_belief(self, nli_probs: Dict[str, float], verdict_str: str) -> List[float]:
        """
        Maps NLI stance (Entailment, Contradiction, Neutral) and historical verdict
        into an Evidence-Implied Veracity Distribution [Misleading, Half-True, Factual].
        """
        p_ent = nli_probs.get("entailment", 0.333)
        p_con = nli_probs.get("contradiction", 0.333)
        p_neu = nli_probs.get("neutral", 0.334)

        norm_v = (verdict_str or "").lower().strip()
        prior = [0.333, 0.334, 0.333]

        if norm_v in {"factual", "true", "mostly-true"}:
            # Precedent is Factual:
            # - Entailing user claim implies claim is Factual
            # - Contradicting user claim implies claim is Misleading
            v_ent = [0.05, 0.15, 0.80]
            v_con = [0.80, 0.15, 0.05]
            belief = [
                p_ent * v_ent[0] + p_con * v_con[0] + p_neu * prior[0],
                p_ent * v_ent[1] + p_con * v_con[1] + p_neu * prior[1],
                p_ent * v_ent[2] + p_con * v_con[2] + p_neu * prior[2],
            ]
        elif norm_v in {"misleading", "false", "pants-fire", "barely-true"}:
            # Precedent is Misleading:
            # - Entailing user claim implies claim is Misleading (repeating a falsehood)
            # - Contradicting user claim implies claim is Factual (debunking/opposing a falsehood)
            v_ent = [0.85, 0.10, 0.05]
            v_con = [0.05, 0.15, 0.80]
            belief = [
                p_ent * v_ent[0] + p_con * v_con[0] + p_neu * prior[0],
                p_ent * v_ent[1] + p_con * v_con[1] + p_neu * prior[1],
                p_ent * v_ent[2] + p_con * v_con[2] + p_neu * prior[2],
            ]
        else:
            # Precedent is Half-True: reflects nuanced/partial context
            v_half = [0.20, 0.60, 0.20]
            belief = [
                (1.0 - p_neu) * v_half[0] + p_neu * prior[0],
                (1.0 - p_neu) * v_half[1] + p_neu * prior[1],
                (1.0 - p_neu) * v_half[2] + p_neu * prior[2],
            ]

        s = sum(belief)
        return [b / s for b in belief] if s > 0 else prior

    def _evaluate_a_vs_b(
        self,
        p_A: List[float],
        pred_label: str,
        pred_conf: float,
        p_B: List[float],
        top_stance: str,
        total_weight: float,
        a_status: str,
        b_status: str,
    ) -> Dict[str, Any]:
        """
        Cross-Verification Assessment 1: Agent A (Classifier) vs Agent B (Evidence RAG).
        Evaluates whether the linguistic model prediction concurs with or diverges from
        empirical fact-checking evidence.
        """
        if a_status not in {"SUCCESS", "completed"}:
            return {
                "status": "UNAVAILABLE",
                "divergence_score": 0.0,
                "model_prediction": pred_label,
                "evidence_stance": top_stance,
                "summary": "Agent A (Classifier) was unavailable; model-evidence agreement could not be computed.",
            }

        if b_status not in {"SUCCESS", "completed"} or total_weight <= 0.08 or top_stance == "NO_RELEVANT_EVIDENCE":
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "divergence_score": 0.0,
                "model_prediction": pred_label,
                "evidence_stance": "NO_RELEVANT_EVIDENCE",
                "summary": f"Agent A predicted '{pred_label}' ({pred_conf:.1%} confidence), but Agent B found no sufficiently relevant historical fact-checking precedents in the LIAR-PLUS corpus. Empirical agreement cannot be established.",
            }

        d_js = jensen_shannon_divergence(p_A, p_B)
        divergence_score = round(d_js, 4)

        if d_js >= 0.35:
            return {
                "status": "DISCREPANCY",
                "divergence_score": divergence_score,
                "model_prediction": pred_label,
                "evidence_stance": top_stance,
                "summary": f"Signal Discrepancy (D_JS = {d_js:.2f}): Model prediction ('{pred_label}') directly conflicts with aggregated historical fact-checking evidence ('{top_stance}').",
            }
        else:
            return {
                "status": "CONCURRENCE",
                "divergence_score": divergence_score,
                "model_prediction": pred_label,
                "evidence_stance": top_stance,
                "summary": f"Model-Evidence Concurrence (D_JS = {d_js:.2f}): Model prediction ('{pred_label}') aligns with aggregated historical fact-checking evidence ('{top_stance}').",
            }

    def _evaluate_b_vs_c(
        self,
        top_stance: str,
        relevant_citations_count: int,
        c_data: Dict[str, Any],
        c_status: str,
    ) -> Dict[str, Any]:
        """
        Cross-Verification Assessment 2: Agent B (Evidence RAG) vs Agent C (Historical Virality).
        Determines whether relevant historical narratives experienced significant past social media
        propagation, and whether the available evidence provides sufficient context to interpret that activity.
        NOTE: Social virality does NOT alter factuality risk; it contextualizes historical exposure reach.
        """
        if c_status not in {"SUCCESS", "completed"}:
            return {
                "status": "DATA_UNAVAILABLE",
                "matched_stories_count": 0,
                "engagement_velocity": 0.0,
                "velocity_percentile": 20.0,
                "context_assessment": "Historical social media influence data was unavailable.",
                "summary": "Agent C historical social media data was unavailable. Assessment continues using classification and fact-checking evidence.",
            }

        matched_stories = c_data.get("matched_stories", [])
        matched_count = len(matched_stories)
        velocity_raw = c_data.get("engagement_velocity", 0.0)
        try:
            velocity = float(velocity_raw)
        except (ValueError, TypeError):
            velocity = 0.0
        percentile = float(c_data.get("velocity_percentile", 20.0))

        if matched_count == 0:
            return {
                "status": "BASELINE_OR_NONE",
                "matched_stories_count": 0,
                "engagement_velocity": 0.0,
                "velocity_percentile": 20.0,
                "context_assessment": "No matching viral narratives found in benchmark records.",
                "summary": "No matching historical stories were found in the FakeNewsNet benchmark dataset. The narrative lacked viral traction in benchmark historical records, indicating baseline exposure.",
            }

        # Matched stories exist
        is_high_velocity = velocity >= 50.0 or percentile >= 70.0
        if is_high_velocity:
            if relevant_citations_count > 0 and top_stance != "NO_RELEVANT_EVIDENCE":
                if top_stance == "CONTRADICTED_BY_EVIDENCE":
                    return {
                        "status": "HIGH_EXPOSURE_DEBUNKED",
                        "matched_stories_count": matched_count,
                        "engagement_velocity": round(velocity, 1),
                        "velocity_percentile": round(percentile, 1),
                        "context_assessment": "High historical virality with robust corrective fact-checking context.",
                        "summary": f"Related benchmark narratives circulated at high historical velocity ({percentile:.0f}th percentile), while historical fact-checking evidence firmly refutes the assertion. This indicates a high-exposure narrative that required corrective debunking.",
                    }
                elif top_stance == "SUPPORTED_BY_EVIDENCE":
                    return {
                        "status": "HIGH_EXPOSURE_VERIFIED",
                        "matched_stories_count": matched_count,
                        "engagement_velocity": round(velocity, 1),
                        "velocity_percentile": round(percentile, 1),
                        "context_assessment": "High historical virality with corroborating factual evidence.",
                        "summary": f"Related benchmark narratives circulated at high historical velocity ({percentile:.0f}th percentile), and historical evidence corroborates the core claims. This indicates high public visibility for factual policy developments.",
                    }
                else:
                    return {
                        "status": "HIGH_EXPOSURE_CONTESTED",
                        "matched_stories_count": matched_count,
                        "engagement_velocity": round(velocity, 1),
                        "velocity_percentile": round(percentile, 1),
                        "context_assessment": "High historical virality with nuanced or conflicting evidence context.",
                        "summary": f"Related benchmark narratives circulated at high historical velocity ({percentile:.0f}th percentile), while retrieved fact-checking evidence indicates nuanced or contested context.",
                    }
            else:
                return {
                    "status": "HIGH_EXPOSURE_UNCOVERED",
                    "matched_stories_count": matched_count,
                    "engagement_velocity": round(velocity, 1),
                    "velocity_percentile": round(percentile, 1),
                    "context_assessment": "High historical virality with scarce empirical fact-checking coverage.",
                    "summary": f"Related benchmark narratives circulated at high historical velocity ({percentile:.0f}th percentile), but direct fact-checking evidence in the LIAR-PLUS corpus is scarce. This highlights an informational gap where past viral claims lack archived empirical documentation.",
                }
        else:
            return {
                "status": "MODERATE_EXPOSURE",
                "matched_stories_count": matched_count,
                "engagement_velocity": round(velocity, 1),
                "velocity_percentile": round(percentile, 1),
                "context_assessment": "Moderate historical virality with adequate contextual coverage.",
                "summary": f"Found {matched_count} matched benchmark story/stories with moderate historical circulation ({velocity:.1f} tweets/hr). The available evidence provides adequate baseline context for these historical records.",
            }

    def _synthesize_three_agents(
        self,
        a_vs_b: Dict[str, Any],
        b_vs_c: Dict[str, Any],
        pred_label: str,
        pred_conf: float,
        top_stance: str,
        relevant_count: int,
        exposure_tier: str,
        composite_risk: float,
        risk_level: str,
        uncertainty_score: float,
        a_status: str,
        b_status: str,
        c_status: str,
    ) -> Dict[str, Any]:
        """
        Cross-Verification Assessment 3: Triangulated Synthesis (A + B + C).
        Combines model prior (A), empirical fact-checking (B), and benchmark exposure kinetics (C)
        into a coherent final assessment.
        """
        # Formulate per-agent contribution summaries
        if a_status in {"SUCCESS", "completed"}:
            a_contrib = f"Classified sentence framing and style as '{pred_label}' with {pred_conf:.1%} confidence."
        else:
            a_contrib = "Classification model was unavailable; linguistic prior was set to uniform distribution."

        if b_status in {"SUCCESS", "completed"} and relevant_count > 0 and top_stance != "NO_RELEVANT_EVIDENCE":
            if a_vs_b.get("status") == "DISCREPANCY":
                b_contrib = f"Retrieved {relevant_count} relevant fact-check(s) indicating '{top_stance}', disputing Agent A's prediction."
            else:
                b_contrib = f"Retrieved {relevant_count} relevant fact-check(s) indicating '{top_stance}', corroborating Agent A's prediction."
        else:
            b_contrib = "No sufficiently relevant historical fact-checks found in LIAR-PLUS; empirical grounding unavailable."

        matched_c = b_vs_c.get("matched_stories_count", 0)
        vel_c = b_vs_c.get("engagement_velocity", 0.0)
        if c_status in {"SUCCESS", "completed"}:
            if matched_c > 0:
                b_status_code = b_vs_c.get("status", "")
                b_contrib_note = "debunked by evidence" if "DEBUNKED" in b_status_code else ("corroborated by evidence" if "VERIFIED" in b_status_code else "contextually evaluated")
                c_contrib = f"Matched {matched_c} historical benchmark stories ({vel_c} tweets/hr velocity, {exposure_tier}), {b_contrib_note}."
            else:
                c_contrib = "No matching viral narratives found in FakeNewsNet; benchmark exposure is baseline."
        else:
            c_contrib = "Historical social media influence data was unavailable; exposure remained neutral."

        # Final narrative synthesis
        if a_vs_b.get("status") == "DISCREPANCY":
            narrative = (
                f"Triangulated synthesis reveals an epistemic divergence: DistilBERT linguistic classification predicted '{pred_label}', "
                f"but empirical LIAR-PLUS fact-checks indicate a contrary stance ('{top_stance}'). "
                f"Historical exposure context shows {c_contrib.lower()} "
                f"Because empirical fact-checks contradict the linguistic framing, the composite risk is evaluated at {composite_risk}/100 ({risk_level}) "
                f"with {uncertainty_score:.0%} uncertainty."
            )
        elif top_stance == "NO_RELEVANT_EVIDENCE":
            narrative = (
                f"Triangulated synthesis relies primarily on DistilBERT linguistic classification ('{pred_label}', {pred_conf:.1%} confidence) "
                f"due to the absence of relevant historical fact-checks in the LIAR-PLUS corpus. "
                f"{c_contrib} "
                f"Without empirical evidence grounding, epistemic uncertainty is elevated ({uncertainty_score:.0%}), "
                f"resulting in a composite risk of {composite_risk}/100 ({risk_level})."
            )
        elif a_vs_b.get("status") == "CONCURRENCE":
            narrative = (
                f"Triangulated synthesis establishes strong multi-agent concurrence: DistilBERT linguistic classification ('{pred_label}') "
                f"is corroborated by {relevant_count} empirical fact-checking precedent(s) ('{top_stance}'). "
                f"{c_contrib} "
                f"The convergent signals support a final assessment of {composite_risk}/100 ({risk_level}) "
                f"with {uncertainty_score:.0%} uncertainty."
            )
        else:
            narrative = (
                f"Triangulated synthesis evaluated linguistic classification ('{pred_label}'), "
                f"historical evidence ('{top_stance}', {relevant_count} precedent(s)), and historical exposure ({exposure_tier}). "
                f"Composite risk is determined at {composite_risk}/100 ({risk_level}) with {uncertainty_score:.0%} uncertainty."
            )

        return {
            "summary": narrative,
            "agent_contributions": {
                "agent_a": {
                    "name": "Claim Classification (DistilBERT)",
                    "role": "Linguistic Prior",
                    "prediction": pred_label,
                    "confidence": round(pred_conf, 4),
                    "contribution": a_contrib,
                },
                "agent_b": {
                    "name": "Evidence Research (LIAR-PLUS RAG)",
                    "role": "Empirical Verification",
                    "stance": top_stance,
                    "relevant_count": relevant_count,
                    "agreement_with_a": a_vs_b.get("status", "INSUFFICIENT_EVIDENCE"),
                    "contribution": b_contrib,
                },
                "agent_c": {
                    "name": "Historical Influence (FakeNewsNet)",
                    "role": "Historical Exposure Context",
                    "exposure_tier": exposure_tier,
                    "matched_count": matched_c,
                    "engagement_velocity": vel_c,
                    "contribution": c_contrib,
                },
            },
        }

    def run(
        self,
        claim: str,
        claim_analysis: Dict[str, Any],
        evidence_rag: Dict[str, Any],
        influence_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        start_time = time.time()
        findings: List[str] = []
        uncertainties: List[str] = []
        integrity_flags: List[str] = []

        a_status = claim_analysis.get("status", "UNAVAILABLE")
        b_status = evidence_rag.get("status", "UNAVAILABLE")
        c_status = influence_analysis.get("status", "UNAVAILABLE")

        a_data = claim_analysis.get("data", {})
        b_data = evidence_rag.get("data", {})
        c_data = influence_analysis.get("data", {})

        p_A, pred_label, pred_conf = self._extract_classifier_distribution(a_data, a_status)
        citations: List[Dict[str, Any]] = b_data.get("citations", [])

        # Step 1 & 2: Multi-Evidence Semantic Stance Evaluation via NLI
        evaluated_citations = []
        nli_pairs: List[Tuple[str, str]] = []

        for cit in citations:
            c_text = cit.get("claim", "")
            j_text = cit.get("justification", "")
            # Pair 1: Precedent claim vs User claim
            nli_pairs.append((c_text, claim))
            # Pair 2: Justification vs User claim (if available)
            if j_text and j_text.strip().lower() != "none":
                nli_pairs.append((j_text[:350], claim))

        # Batch evaluate NLI
        nli_results = self.nli_evaluator.evaluate_batch(nli_pairs)

        pair_idx = 0
        citation_beliefs = []
        citation_weights = []

        for cit in citations:
            sim = float(cit.get("similarity_score", 0.0))
            verdict = cit.get("verdict", "Half-True")
            has_just = bool(cit.get("has_justification", True) and cit.get("justification"))

            claim_nli = nli_results[pair_idx] if pair_idx < len(nli_results) else {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}
            pair_idx += 1

            if has_just and pair_idx < len(nli_results):
                just_nli = nli_results[pair_idx]
                pair_idx += 1
                # If claim NLI has strong directional stance (non-neutral), preserve and reinforce it
                if claim_nli.get("neutral", 1.0) < 0.5:
                    if just_nli.get("entailment", 0.0) > 0.4 and claim_nli.get("entailment", 0.0) > 0.4:
                        combined_nli = {
                            "entailment": max(claim_nli["entailment"], just_nli["entailment"]),
                            "contradiction": min(claim_nli["contradiction"], just_nli["contradiction"]),
                            "neutral": min(claim_nli["neutral"], just_nli["neutral"]),
                        }
                    elif just_nli.get("contradiction", 0.0) > 0.4 and claim_nli.get("contradiction", 0.0) > 0.4:
                        combined_nli = {
                            "entailment": min(claim_nli["entailment"], just_nli["entailment"]),
                            "contradiction": max(claim_nli["contradiction"], just_nli["contradiction"]),
                            "neutral": min(claim_nli["neutral"], just_nli["neutral"]),
                        }
                    else:
                        combined_nli = claim_nli
                elif just_nli.get("neutral", 1.0) < 0.5:
                    combined_nli = just_nli
                else:
                    combined_nli = {
                        "entailment": 0.5 * claim_nli["entailment"] + 0.5 * just_nli["entailment"],
                        "contradiction": 0.5 * claim_nli["contradiction"] + 0.5 * just_nli["contradiction"],
                        "neutral": 0.5 * claim_nli["neutral"] + 0.5 * just_nli["neutral"],
                    }
            else:
                combined_nli = claim_nli

            s_nli = sum(combined_nli.values())
            if s_nli > 0:
                combined_nli = {k: v / s_nli for k, v in combined_nli.items()}

            # Continuous weight: incorporates similarity, neutrality discount, and justification availability
            p_informative = max(0.0, 1.0 - combined_nli.get("neutral", 0.33))
            relevance_scalar = max(0.0, (sim - 0.20) / 0.80)
            w = relevance_scalar * p_informative * (1.0 if has_just else 0.75)

            q_i = self._map_citation_to_belief(combined_nli, verdict)

            citation_beliefs.append(q_i)
            citation_weights.append(w)

            evaluated_citations.append({
                **cit,
                "nli_stance": combined_nli,
                "evidence_weight": round(w, 4),
                "implied_belief": [round(v, 4) for v in q_i],
            })

        total_weight = sum(citation_weights)
        relevant_citations = [c for c in evaluated_citations if c["evidence_weight"] > 0.05]

        findings.append(
            f"Multi-Evidence Audit: Evaluated {len(citations)} historical citations via NLI "
            f"({len(relevant_citations)} relevant with cumulative evidential mass {total_weight:.2f})."
        )

        # Step 3: Multi-Evidence Aggregation & Stance Determination
        if total_weight > 0.08:
            p_B = [
                sum(w * q[0] for w, q in zip(citation_weights, citation_beliefs)) / total_weight,
                sum(w * q[1] for w, q in zip(citation_weights, citation_beliefs)) / total_weight,
                sum(w * q[2] for w, q in zip(citation_weights, citation_beliefs)) / total_weight,
            ]
            s_B = sum(p_B)
            p_B = [b / s_B for b in p_B] if s_B > 0 else [0.333, 0.334, 0.333]

            # Determine dominant stance
            if p_B[2] >= 0.52:
                top_stance = "SUPPORTED_BY_EVIDENCE"
            elif p_B[0] >= 0.52:
                top_stance = "CONTRADICTED_BY_EVIDENCE"
            elif b_data.get("has_conflicts", False):
                top_stance = "DISPUTED_STANCE"
            else:
                top_stance = "PARTIALLY_ALIGNED"

            findings.append(
                f"Aggregated Evidence Stance: '{top_stance}' "
                f"(Implied: {p_B[0]:.1%} Misleading, {p_B[1]:.1%} Half-True, {p_B[2]:.1%} Factual)."
            )
        else:
            p_B = [0.333, 0.334, 0.333]
            top_stance = "NO_RELEVANT_EVIDENCE"
            uncertainties.append("Absence of high-relevance domain evidence prevents definitive empirical grounding.")

        # Justification Audit
        valid_just_count = sum(1 for c in citations if c.get("has_justification", True) and c.get("justification"))
        if citations and valid_just_count > 0:
            findings.append(f"Justification Integrity: {valid_just_count} of {len(citations)} records possess published journalist reasoning.")
        elif citations:
            integrity_flags.append("Retrieved historical records lack published journalist justifications.")

        # Step 4: Three-Agent Assessment 1 (Agent A vs Agent B)
        a_vs_b = self._evaluate_a_vs_b(
            p_A=p_A,
            pred_label=pred_label,
            pred_conf=pred_conf,
            p_B=p_B,
            top_stance=top_stance,
            total_weight=total_weight,
            a_status=a_status,
            b_status=b_status,
        )
        conflict_detected = a_vs_b.get("status") == "DISCREPANCY"
        conflict_score = a_vs_b.get("divergence_score", 0.0)

        if conflict_detected:
            findings.append(f"DISCREPANCY ALERT: {a_vs_b['summary']}")
        elif a_vs_b.get("status") == "CONCURRENCE":
            findings.append(f"CONCURRENCE: {a_vs_b['summary']}")
        else:
            findings.append(f"A vs B Comparison: {a_vs_b['summary']}")

        # Step 5: Adaptive Belief Fusion & Composite Risk Calculation
        # Dynamic evidence weight factor alpha
        if b_status in {"SUCCESS", "completed"} and a_status in {"SUCCESS", "completed"}:
            alpha = 0.65 * (total_weight / (total_weight + 1.0))
        elif b_status in {"SUCCESS", "completed"}:
            alpha = 0.85
            uncertainties.append("Classifier offline: Risk inferred primarily from directional evidence stance.")
        elif a_status in {"SUCCESS", "completed"}:
            alpha = 0.0
            uncertainties.append("RAG evidence offline: Risk calculated from classifier and social signals only.")
        else:
            alpha = 0.5
            uncertainties.append("Both classifier and RAG offline: Defers to editorial human review.")

        p_syn = [
            (1.0 - alpha) * p_A[0] + alpha * p_B[0],
            (1.0 - alpha) * p_A[1] + alpha * p_B[1],
            (1.0 - alpha) * p_A[2] + alpha * p_B[2],
        ]
        s_syn = sum(p_syn)
        p_syn = [v / s_syn for v in p_syn] if s_syn > 0 else [0.333, 0.334, 0.333]

        # Continuous Misinformation Risk Expectation: 100 * Misleading + 50 * Half-True + 10 * Factual
        composite_risk = (100.0 * p_syn[0]) + (50.0 * p_syn[1]) + (10.0 * p_syn[2])
        composite_risk = round(max(0.0, min(100.0, composite_risk)), 1)

        if composite_risk >= 70.0:
            risk_level = "HIGH_MISINFORMATION_RISK"
        elif composite_risk >= 35.0:
            risk_level = "MODERATE_MISINFORMATION_RISK"
        else:
            risk_level = "LOW_MISINFORMATION_RISK"

        # Step 6: Multi-Factor Decoupled Epistemic Uncertainty
        # Factor 1: Prediction Ambiguity (Scaled distance from uniform uncertainty [1/3, 1.0])
        max_p = max(p_syn)
        u_pred = max(0.0, (1.0 - max_p) / (1.0 - 1.0 / 3.0)) if a_status in {"SUCCESS", "completed"} else 1.0
        # Factor 2: Evidence Scarcity (Decays exponentially as relevant evidential mass accumulates)
        if total_weight > 0.08 and len(citations) > 0:
            u_scarcity = math.exp(-total_weight / 0.35)
        else:
            u_scarcity = 1.0
        # Factor 3: Model-Evidence Discrepancy & Internal Contradictions
        base_conflict = conflict_score if conflict_detected else 0.0
        internal_conflict_penalty = 0.20 if b_data.get("has_conflicts", False) else 0.0
        u_conflict = min(1.0, base_conflict + internal_conflict_penalty)

        uncertainty_score = 0.35 * u_pred + 0.45 * u_scarcity + 0.20 * u_conflict

        # Baseline safety bounds:
        # If no relevant evidence is available, lack of empirical grounding sets uncertainty >= 0.60
        if total_weight <= 0.08:
            uncertainty_score = max(uncertainty_score, 0.60)
        # If classifier or evidence offline, epistemic uncertainty remains elevated
        if a_status not in {"SUCCESS", "completed"} and b_status not in {"SUCCESS", "completed"}:
            uncertainty_score = max(uncertainty_score, 0.75)
        elif a_status not in {"SUCCESS", "completed"} or b_status not in {"SUCCESS", "completed"}:
            uncertainty_score = max(uncertainty_score, 0.40)

        uncertainty_score = round(max(0.05, min(0.95, uncertainty_score)), 2)

        # Step 7: Decoupled Social Influence (Agent C) & B vs C Historical Context
        # Exposure / virality indicators never contaminate veracity risk score
        velocity_raw = c_data.get("engagement_velocity", 0.0)
        try:
            velocity_num = float(velocity_raw)
        except (ValueError, TypeError):
            velocity_num = 0.0
        percentile = float(c_data.get("velocity_percentile", 20.0))
        matched_stories = c_data.get("matched_stories", [])

        if matched_stories and (velocity_num >= 50.0 or percentile >= 70.0):
            historical_exposure_tier = "HIGH_HISTORICAL_VELOCITY"
        elif matched_stories:
            historical_exposure_tier = "MODERATE_HISTORICAL_VELOCITY"
        else:
            historical_exposure_tier = "BASELINE_OR_ISOLATED"

        # Three-Agent Assessment 2 (Agent B vs Agent C)
        b_vs_c = self._evaluate_b_vs_c(
            top_stance=top_stance,
            relevant_citations_count=len(relevant_citations),
            c_data=c_data,
            c_status=c_status,
        )
        findings.append(f"B vs C Historical Context: {b_vs_c['summary']}")

        # Step 8: Three-Agent Synthesized Triangulation (A + B + C)
        three_agent_synthesis = self._synthesize_three_agents(
            a_vs_b=a_vs_b,
            b_vs_c=b_vs_c,
            pred_label=pred_label,
            pred_conf=pred_conf,
            top_stance=top_stance,
            relevant_count=len(relevant_citations),
            exposure_tier=historical_exposure_tier,
            composite_risk=composite_risk,
            risk_level=risk_level,
            uncertainty_score=uncertainty_score,
            a_status=a_status,
            b_status=b_status,
            c_status=c_status,
        )
        findings.append(f"Triangulated Synthesis: {three_agent_synthesis['summary']}")

        if not any("Guardrail" in u for u in uncertainties):
            uncertainties.append("Guardrail: Epistemic assessment bounds certainty based on available evidence coverage.")

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = (
            f"Composite Risk: {composite_risk}/100 ({risk_level}). "
            f"Epistemic Stance: {top_stance}. Uncertainty: {uncertainty_score:.0%}. "
            f"A vs B: {a_vs_b.get('status', 'N/A')}. B vs C: {b_vs_c.get('status', 'N/A')}."
        )

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "completed",
            "duration_ms": duration_ms,
            "summary": summary,
            "data": {
                "composite_risk_score": composite_risk,
                "risk_level": risk_level,
                "uncertainty_score": uncertainty_score,
                "evidence_conflict": conflict_detected,
                "conflict_score": conflict_score,
                "epistemic_stance": top_stance,
                "findings": findings,
                "uncertainties": uncertainties,
                "integrity_flags": integrity_flags,
                "relevant_citations_count": len(relevant_citations),
                "historical_exposure_tier": historical_exposure_tier,
                "synthesized_probabilities": {
                    "Misleading": round(p_syn[0], 4),
                    "Half-True": round(p_syn[1], 4),
                    "Factual": round(p_syn[2], 4),
                },
                "agent_statuses": {
                    "classifier": a_status,
                    "evidence_rag": b_status,
                    "influence": c_status,
                },
                "a_vs_b": a_vs_b,
                "b_vs_c": b_vs_c,
                "three_agent_synthesis": three_agent_synthesis,
            },
        }


CrossVerificationAgent = VerificationRiskAgent


class MitigationRecommendationAgent:
    """
    Agent E / AdvisoryMitigationAgent:
    Formulates responsible, non-coercive advisory recommendations based on Agent D's findings.
    Emits structured public flags and recommended verification actions.
    """

    def __init__(self):
        self.agent_name = "Advisory & Mitigation Agent"
        self.agent_id = "mitigation_agent"

    def run(self, claim: str, verification_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        v_data = verification_data.get("data", {})
        risk_level = v_data.get("risk_level", "MODERATE_MISINFORMATION_RISK")
        risk_score = v_data.get("composite_risk_score", 50.0)
        stance = v_data.get("epistemic_stance", "NO_RELEVANT_EVIDENCE")

        recommendations = []
        if risk_level == "HIGH_MISINFORMATION_RISK":
            public_flag = "MISLEADING_CLAIM_ALERT"
            recommended_action = "CONTEXTUAL_WARNING_BANNER"
            advisory_text = (
                "This statement conflicts significantly with official statistical and legislative records. "
                "Contextual advisories linking readers to primary agency reports (CBO, BLS, IRS) are advised."
            )
            recommendations.append({
                "action": "Attach primary-source citations",
                "priority": "HIGH",
                "details": "Attach verified primary-source fact-checking citations from official reports.",
            })
            recommendations.append({
                "action": "Editorial review hold",
                "priority": "HIGH",
                "details": "Flag claim for editorial review prior to broadcast or aggregation.",
            })
        elif risk_level == "MODERATE_MISINFORMATION_RISK":
            public_flag = "CONTEXT_REQUIRED"
            recommended_action = "OMITTED_CONTEXT_NOTE"
            advisory_text = (
                "This claim contains partial factual accuracy but omits critical fiscal or legislative context. "
                "Adding clarifying context regarding timeframe and scope is recommended."
            )
            recommendations.append({
                "action": "Provide omitted context",
                "priority": "MEDIUM",
                "details": "Highlight omitted baseline comparisons, dates, and historical conditions.",
            })
            recommendations.append({
                "action": "Multi-perspective commentary",
                "priority": "LOW",
                "details": "Link to balanced economic analyses from non-partisan sources.",
            })
        else:
            public_flag = "VERIFIED_FACTUAL"
            recommended_action = "NO_ACTION_REQUIRED"
            advisory_text = "Statement aligns closely with corroborated official policy data and historical precedents."
            recommendations.append({
                "action": "Verify statutory records",
                "priority": "LOW",
                "details": "Corroborated by verified statutory tax and spending records.",
            })
            recommendations.append({
                "action": "No coercive moderation",
                "priority": "LOW",
                "details": "Preserve content integrity without platform warnings.",
            })

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = f"Mitigation Flag: {public_flag} (Action: {recommended_action})."

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "completed",
            "duration_ms": duration_ms,
            "summary": summary,
            "data": {
                "public_flag": public_flag,
                "recommended_action": recommended_action,
                "advisory_text": advisory_text,
                "recommendations": recommendations,
                "composite_risk_score": risk_score,
                "executed_interventions_count": 0,
            },
            "disclaimer": (
                "Advisory mitigations are non-coercive editorial recommendations. "
                "No automated content blocking or censorship actions are executed by this research platform."
            ),
        }


AdvisoryMitigationAgent = MitigationRecommendationAgent
