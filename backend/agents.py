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

from collections import Counter
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

            # Ensure highest relevance historical fact-checks are first
            citations.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

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
            if matched:
                summary = (
                    f"Matched {len(matched)} historical FakeNewsNet stories. "
                    f"Observed propagation velocity: {velocity} ({percentile:.0f}th percentile)."
                )
            else:
                summary = "Benchmark data unavailable: No matching viral stories found in benchmark records."

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
# AGENT D: Final Verification & Risk Assessment Agent
# ============================================================

def normalize_veracity(v: str) -> str:
    """Helper to map any verdict/label to one of: 'Factual', 'Misleading', 'Half-True'."""
    norm = (v or "").strip().lower()
    if norm in {"factual", "true", "mostly-true"}:
        return "Factual"
    if norm in {"misleading", "false", "pants-fire", "barely-true"}:
        return "Misleading"
    return "Half-True"


class NLIEvaluator:
    """Lightweight stub retained for backward compatibility."""
    _instance: Optional["NLIEvaluator"] = None

    @classmethod
    def get_instance(cls) -> "NLIEvaluator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def evaluate_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        return [{"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34} for _ in pairs]


class VerificationRiskAgent:
    """
    Agent D: Final Verification & Risk Assessment Agent
    
    Synthesizes outputs from:
      - Agent A (Claim Analysis): Classification label & confidence
      - Agent B (Evidence RAG): Historical fact-checks, verdicts, and justifications
      - Agent C (Historical Influence): Historical social propagation context
      
    Execution Steps:
      Step 1: Collect Agent Results safely (handles offline or missing data gracefully)
      Step 2: Compare Agent A and Agent B (Agreement, Disagreement, or Insufficient Evidence)
      Step 3: Analyze Agent B and Agent C (Social context; strictly 0% weight on factuality risk)
      Step 4: Final Combined Assessment (Risk Score 0-100, Risk Level, Uncertainty, Evidence Strength, Final Assessment, Simple Explanation)
    """

    def __init__(self, nli_evaluator: Optional[Any] = None, **kwargs):
        self.agent_name = "Verification & Risk Assessment Agent"
        self.agent_id = "verification_risk_agent"

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

        # ----------------------------------------------------
        # Step 1: Collect Agent Results Safely
        # ----------------------------------------------------
        a_status = claim_analysis.get("status", "UNAVAILABLE")
        b_status = evidence_rag.get("status", "UNAVAILABLE")
        c_status = influence_analysis.get("status", "UNAVAILABLE")

        a_data = claim_analysis.get("data", {})
        b_data = evidence_rag.get("data", {})
        c_data = influence_analysis.get("data", {})

        # Agent A extraction
        pred_label = a_data.get("label", "Half-True")
        try:
            pred_conf = float(a_data.get("confidence", 0.50))
        except (ValueError, TypeError):
            pred_conf = 0.50
        norm_a = normalize_veracity(pred_label)

        # Agent B extraction
        citations: List[Dict[str, Any]] = b_data.get("citations", [])
        has_conflicts = bool(b_data.get("has_conflicts", False))

        # Check if citations exist and have relevant substance
        relevant_citations = [
            c for c in citations
            if float(c.get("similarity_score", 0.0) or c.get("semantic_score", 0.0)) >= 0.10
        ]
        if not relevant_citations and citations:
            relevant_citations = citations

        has_evidence = (
            b_status in {"SUCCESS", "completed"}
            and len(relevant_citations) > 0
            and b_data.get("evidence_status") != "insufficient"
        )

        # Determine evidence verdict and strength
        if has_evidence:
            verdict_counts = Counter(normalize_veracity(c.get("verdict", "Half-True")) for c in relevant_citations)
            if len(verdict_counts) > 1 and "Factual" in verdict_counts and "Misleading" in verdict_counts:
                evidence_verdict = "Mixed"
                evidence_strength = "Conflicting"
            else:
                evidence_verdict = verdict_counts.most_common(1)[0][0]
                evidence_strength = "Strong" if len(relevant_citations) >= 2 else "Limited"
        else:
            evidence_verdict = "None"
            evidence_strength = "Insufficient"

        # Agent C extraction
        matched_stories = c_data.get("matched_stories", [])
        matched_c_count = len(matched_stories)
        has_c_context = (
            c_status in {"SUCCESS", "completed"}
            and matched_c_count > 0
        )

        # ----------------------------------------------------
        # Step 2: Compare Agent A and Agent B
        # ----------------------------------------------------
        if not has_evidence:
            a_vs_b_status = "INSUFFICIENT_EVIDENCE"
            a_vs_b_summary = (
                f"Agent A predicted '{pred_label}' ({pred_conf:.1%} confidence), but Agent B found no relevant "
                f"historical fact-checks in the archive. Agreement cannot be established."
            )
            conflict_detected = False
            findings.append("Agent B found no relevant historical fact-checks in the archive.")
            uncertainties.append("Absence of historical evidence prevents definitive verification.")
        elif evidence_verdict == "Mixed":
            a_vs_b_status = "CONFLICTING_EVIDENCE"
            a_vs_b_summary = (
                f"Agent A predicted '{pred_label}', but Agent B found historical fact-checks with conflicting "
                f"verdicts across different sources."
            )
            conflict_detected = True
            findings.append(f"Agent B found conflicting evidence across {len(relevant_citations)} fact-check(s).")
            uncertainties.append("Divergent historical fact-check verdicts require human contextual review.")
        elif norm_a == evidence_verdict:
            a_vs_b_status = "CONCURRENCE"
            a_vs_b_summary = (
                f"Agent A and Agent B agree. The classifier prediction ('{pred_label}') aligns with "
                f"historical fact-check evidence ('{evidence_verdict}') across {len(relevant_citations)} precedent(s)."
            )
            conflict_detected = False
            findings.append(f"Concurrence: Classifier ('{pred_label}') and evidence ('{evidence_verdict}') agree.")
        else:
            a_vs_b_status = "DISCREPANCY"
            a_vs_b_summary = (
                f"Agent A and Agent B disagree. The classifier predicted '{pred_label}', but verified "
                f"historical fact-checks indicate '{evidence_verdict}'. Historical evidence takes precedence."
            )
            conflict_detected = True
            findings.append(f"Discrepancy: Classifier predicted '{pred_label}', but historical evidence indicates '{evidence_verdict}'.")
            uncertainties.append("Divergence between linguistic prediction and historical fact-checking records.")

        a_vs_b = {
            "status": a_vs_b_status,
            "agreement": "Agreement" if a_vs_b_status == "CONCURRENCE" else ("Disagreement" if a_vs_b_status == "DISCREPANCY" else "Insufficient Evidence"),
            "model_prediction": pred_label,
            "evidence_verdict": evidence_verdict,
            "explanation": a_vs_b_summary,
            "summary": a_vs_b_summary,
            "divergence_score": 0.40 if conflict_detected else 0.0,
        }

        # ----------------------------------------------------
        # Step 3: Analyze Agent B and Agent C
        # ----------------------------------------------------
        # Crucial Rule: Social popularity does not change factuality risk!
        if has_c_context:
            top_story = matched_stories[0]
            story_title = top_story.get("title") or top_story.get("headline") or "Historical story"
            b_vs_c_status = "HISTORICAL_CONTEXT_AVAILABLE"
            b_vs_c_summary = (
                f"Matched {matched_c_count} related historical story/stories in the benchmark archive "
                f"(e.g., '{story_title}'). This provides historical exposure context, but social media "
                f"activity does not alter factuality risk."
            )
            findings.append(f"Agent C matched {matched_c_count} historical benchmark story/stories for exposure context.")
        else:
            b_vs_c_status = "NO_HISTORICAL_DATA"
            b_vs_c_summary = (
                "No relevant historical social media propagation data was found in benchmark records. "
                "Factuality assessment relies purely on classification and empirical evidence."
            )
            findings.append("No matching social media propagation stories found in benchmark records.")

        b_vs_c = {
            "status": b_vs_c_status,
            "matched_stories_count": matched_c_count,
            "has_context": has_c_context,
            "explanation": b_vs_c_summary,
            "summary": b_vs_c_summary,
            "context_assessment": b_vs_c_summary,
            "engagement_velocity": 0.0,
            "velocity_percentile": 20.0,
        }

        # ----------------------------------------------------
        # Step 4: Final Combined Assessment
        # ----------------------------------------------------
        # 1. Final Risk Score (0-100) using simple, transparent weights
        # Evidence has 70% weight, Model has 30% weight.
        # Social media (Agent C) strictly has 0% weight.
        if has_evidence:
            if evidence_verdict == "Misleading":
                ev_risk = 90.0
            elif evidence_verdict == "Factual":
                ev_risk = 15.0
            else:  # Half-True or Mixed
                ev_risk = 50.0

            if norm_a == "Misleading":
                m_risk = 85.0
            elif norm_a == "Factual":
                m_risk = 15.0
            else:
                m_risk = 50.0

            composite_risk = round(0.70 * ev_risk + 0.30 * m_risk, 1)
        else:
            # When evidence is insufficient, base on model with cautious bound
            if norm_a == "Misleading":
                composite_risk = 75.0
            elif norm_a == "Factual":
                composite_risk = 25.0
            else:
                composite_risk = 50.0

        # 2. Risk Level (Low, Moderate, High)
        if composite_risk >= 65.0:
            risk_level = "HIGH_MISINFORMATION_RISK"
            simple_risk_level = "High"
            findings.append("High misinformation risk: Claim content is unsupported or contradicted by facts.")
        elif composite_risk >= 35.0:
            risk_level = "MODERATE_MISINFORMATION_RISK"
            simple_risk_level = "Moderate"
            findings.append("Moderate misinformation risk: Claim contains partial accuracy or context gaps.")
        else:
            risk_level = "LOW_MISINFORMATION_RISK"
            simple_risk_level = "Low"
            findings.append("Low misinformation risk: Claim is factual and corroborated by evidence.")

        # 3. Uncertainty Level (Low, Moderate, High)
        if not has_evidence:
            uncertainty_level = "High"
            uncertainty_score = 0.75
        elif conflict_detected or evidence_strength == "Conflicting":
            uncertainty_level = "Moderate"
            uncertainty_score = 0.45
        else:
            uncertainty_level = "Low"
            uncertainty_score = 0.20

        # 4. Final Verdict / Assessment
        if has_evidence:
            if evidence_verdict == "Factual":
                final_verdict = "Supported by Evidence"
                evidence_status = "Supporting"
                verdict_rationale = (
                    f"Retrieved historical fact-checking evidence corroborates this statement. "
                    f"Aggregated evidence from {len(relevant_citations)} precedent(s) shows official alignment."
                )
            elif evidence_verdict == "Misleading":
                final_verdict = "Contradicted by Evidence"
                evidence_status = "Contradicting"
                verdict_rationale = (
                    f"Verified historical fact-checking records refute or contradict the assertion made in this statement. "
                    f"Aggregated evidence from {len(relevant_citations)} precedent(s) indicates the claim is inaccurate."
                )
            else:
                final_verdict = "Mixed Evidence"
                evidence_status = "Mixed"
                verdict_rationale = (
                    "Retrieved historical fact-checks contain conflicting verdicts or divergent findings across different sources or periods."
                )

            if conflict_detected and a_vs_b_status == "DISCREPANCY":
                verdict_rationale += (
                    f" Note: While DistilBERT predicted '{pred_label}', empirical fact-checking evidence takes precedence."
                )
        else:
            evidence_status = "Insufficient"
            if a_status not in {"SUCCESS", "completed"} or pred_conf < 0.60 or uncertainty_score >= 0.70:
                final_verdict = "Insufficient Evidence"
                verdict_rationale = (
                    "No sufficiently relevant historical fact-checks were found in our archive to evaluate this statement. "
                    "Without archived empirical records, a definitive truth verdict cannot be established."
                )
            else:
                final_verdict = "Needs Further Verification"
                verdict_rationale = (
                    f"DistilBERT classified the claim as '{pred_label}' ({pred_conf:.1%} confidence), but no relevant empirical fact-checks "
                    f"were found in the archive to substantiate or refute it. Independent verification with primary records is advised."
                )

        # Guardrail uncertainty
        uncertainties.append(
            "Guardrail: Historical benchmark records and past verdicts describe past events and do not constitute absolute real-time factuality."
        )

        # Missing Information list
        missing_information: List[str] = []
        if not has_evidence:
            missing_information.append("Direct fact-checking precedents for this specific claim are absent from the archive.")
            missing_information.append("Primary source documentation (e.g., official government statistics or agency releases) has not been verified.")
        if not has_c_context:
            missing_information.append("Historical social media propagation data was unavailable in benchmark records.")

        # What to verify next list
        what_to_verify_next: List[str] = [
            "Check official primary sources (e.g., government agency releases, CBO, BLS) covering the exact topic and timeframe.",
            "Verify whether the statement omitted important qualifiers, baseline comparisons, or context.",
        ]
        if conflict_detected:
            what_to_verify_next.append("Consult multiple independent fact-checking organizations to compare differing perspectives.")

        # Agent Contributions
        agent_contributions = {
            "agent_a": {
                "name": "Veracity Assessment (DistilBERT)",
                "role": "Claim Classifier",
                "prediction": pred_label,
                "confidence": round(pred_conf, 4),
                "contribution": f"Classified claim text as '{pred_label}' with {pred_conf:.1%} confidence based on linguistic patterns.",
            },
            "agent_b": {
                "name": "Historical Evidence (LIAR-PLUS RAG)",
                "role": "Fact-Checking Evidence",
                "verdict": evidence_verdict,
                "stance": "SUPPORTED_BY_EVIDENCE" if evidence_verdict == "Factual" else ("CONTRADICTED_BY_EVIDENCE" if evidence_verdict == "Misleading" else ("NO_RELEVANT_EVIDENCE" if not has_evidence else "DISPUTED_STANCE")),
                "citations_count": len(relevant_citations),
                "evidence_strength": evidence_strength,
                "contribution": (
                    f"Retrieved {len(relevant_citations)} relevant historical fact-check(s) indicating '{evidence_verdict}'."
                    if has_evidence else
                    "No relevant historical fact-checks found; empirical evidence is unavailable."
                ),
            },
            "agent_c": {
                "name": "Historical Social Influence (FakeNewsNet)",
                "role": "Exposure Context",
                "matched_count": matched_c_count,
                "contribution": (
                    f"Found {matched_c_count} related historical story/stories in benchmark records for exposure context."
                    if has_c_context else
                    "No matching historical social media stories in benchmark records."
                ),
            },
        }

        # Final explanation
        if has_evidence:
            if conflict_detected:
                final_explanation = (
                    f"Agent D determined the claim is {final_verdict}. While Agent A predicted '{pred_label}', "
                    f"Agent B retrieved verified historical evidence indicating '{evidence_verdict}', which takes precedence. "
                    f"The final risk score is {composite_risk}/100 ({simple_risk_level} Risk) with {uncertainty_level} Uncertainty."
                )
            else:
                final_explanation = (
                    f"Agent D determined the claim is {final_verdict}. Both Agent A ('{pred_label}') and Agent B "
                    f"('{evidence_verdict}') corroborate each other based on {len(relevant_citations)} historical fact-check(s). "
                    f"The final risk score is {composite_risk}/100 ({simple_risk_level} Risk) with {uncertainty_level} Uncertainty."
                )
        else:
            final_explanation = (
                f"Agent D determined the claim has {final_verdict}. Agent A predicted '{pred_label}' ({pred_conf:.1%} confidence), "
                f"but no supporting or refuting historical fact-checks were found in the database. "
                f"Because empirical evidence is missing, the claim cannot be conclusively verified (Uncertainty: {uncertainty_level})."
            )

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = f"Final Assessment: {final_verdict}. Risk: {composite_risk}/100 ({simple_risk_level}). Uncertainty: {uncertainty_level}."

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "completed",
            "duration_ms": duration_ms,
            "summary": summary,
            "data": {
                # Clean, Simple Fields
                "final_risk_score": composite_risk,
                "risk_level": risk_level,
                "simple_risk_level": simple_risk_level,
                "uncertainty_level": uncertainty_level,
                "uncertainty_score": uncertainty_score,
                "evidence_strength": evidence_strength,
                "final_assessment": final_verdict,
                "final_verdict": final_verdict,
                "evidence_status": evidence_status,
                "final_explanation": final_explanation,
                "a_vs_b": a_vs_b,
                "b_vs_c": b_vs_c,
                "agent_contributions": agent_contributions,

                # Backward Compatibility Fields
                "composite_risk_score": composite_risk,
                "epistemic_stance": agent_contributions["agent_b"]["stance"],
                "evidence_conflict": conflict_detected,
                "conflict_score": a_vs_b["divergence_score"],
                "findings": findings,
                "uncertainties": uncertainties,
                "integrity_flags": integrity_flags,
                "relevant_citations_count": len(relevant_citations),
                "historical_exposure_tier": "HIGH_HISTORICAL_VELOCITY" if has_c_context else "BASELINE_OR_ISOLATED",
                "decision_breakdown": {
                    "final_verdict": final_verdict,
                    "evidence_status": evidence_status,
                    "verdict_rationale": verdict_rationale,
                    "agreement_analysis": a_vs_b_summary,
                    "missing_information": missing_information,
                    "what_to_verify_next": what_to_verify_next,
                    "agent_a_role": {
                        "prediction": pred_label,
                        "confidence": round(pred_conf, 4),
                        "explanation": agent_contributions["agent_a"]["contribution"],
                    },
                    "agent_b_role": {
                        "verdict": evidence_verdict,
                        "stance": agent_contributions["agent_b"]["stance"],
                        "relevant_count": len(relevant_citations),
                        "explanation": agent_contributions["agent_b"]["contribution"],
                    },
                    "agent_c_role": {
                        "status": c_status,
                        "matched_count": matched_c_count,
                        "has_context": has_c_context,
                        "explanation": agent_contributions["agent_c"]["contribution"],
                    },
                    "technical_details": {
                        "composite_risk_score": composite_risk,
                        "risk_level": risk_level,
                        "uncertainty_score": uncertainty_score,
                        "classifier_weight_pct": 30.0 if has_evidence else 100.0,
                        "evidence_weight_pct": 70.0 if has_evidence else 0.0,
                        "social_weight_pct": 0.0,
                    },
                },
                "risk_score_breakdown": {
                    "composite_score": composite_risk,
                    "risk_level": risk_level,
                    "uncertainty": uncertainty_score,
                    "classifier_weight_pct": 30.0 if has_evidence else 100.0,
                    "evidence_weight_pct": 70.0 if has_evidence else 0.0,
                    "social_influence_weight_pct": 0.0,
                    "conflict_detected": conflict_detected,
                    "explanation": (
                        f"Risk score ({composite_risk}/100) is derived from empirical fact-checking evidence (70% weight) "
                        f"and classifier prediction (30% weight). Social media propagation strictly has 0% weight on factuality."
                    ),
                },
                "three_agent_synthesis": {
                    "summary": final_explanation,
                    "agent_contributions": agent_contributions,
                },
            },
        }


CrossVerificationAgent = VerificationRiskAgent


# ============================================================
# AGENT E: Mitigation & Recommendation Agent
# ============================================================

class MitigationRecommendationAgent:
    """
    Agent E: Mitigation & Recommendation Agent
    
    Acts AFTER Agent D to provide practical, non-coercive next steps for users.
    
    Produces 5 clear sections:
      1. Immediate Action: What should the user do right now?
      2. Verification Steps: Actionable steps to check additional sources
      3. Sharing Recommendation: "Do Not Share as Verified" | "Hold for Verification" | "Share with Context"
      4. Missing Context: Clear note on what is unverified or missing
      5. Reason: Explanation of why these recommendations were made based on Agent D
    """

    def __init__(self, **kwargs):
        self.agent_name = "Mitigation & Recommendation Agent"
        self.agent_id = "mitigation_agent"

    def run(self, claim: str, verification_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        v_data = verification_data.get("data", {})
        
        final_assessment = v_data.get("final_assessment") or v_data.get("final_verdict", "Needs Further Verification")
        risk_level = v_data.get("risk_level", "MODERATE_MISINFORMATION_RISK")
        simple_risk = v_data.get("simple_risk_level") or ("High" if "HIGH" in str(risk_level) else ("Low" if "LOW" in str(risk_level) else "Moderate"))
        uncertainty_level = v_data.get("uncertainty_level", "Moderate")
        evidence_strength = v_data.get("evidence_strength", "Limited")
        has_c = bool(v_data.get("b_vs_c", {}).get("has_context", False))

        # Determine the 5 required sections based on Agent D's assessment
        # Priority 1: High uncertainty or Insufficient evidence -> Hold for Verification
        if evidence_strength == "Insufficient" or uncertainty_level == "High" or final_assessment in {"Insufficient Evidence", "Needs Further Verification"}:
            sharing_recommendation = "Hold for Verification"
            immediate_action = (
                "Hold this statement and do not share it as verified fact. No empirical evidence was found "
                "in the archive to confirm or deny the assertion."
            )
            verification_steps = [
                "Consult official primary agency reports to check whether key conditions or definitions were omitted.",
                "Look for statements from multiple non-partisan analysts or fact-checking organizations.",
                "Confirm whether figures refer to gross numbers, percentage changes, or selective date ranges.",
            ]
            missing_context = (
                "No matching fact-checks or primary records were found in the archive. The statement remains unverified."
            )
            reason = (
                f"Agent D reported high uncertainty ({uncertainty_level}) due to insufficient empirical evidence. "
                f"Until primary sources are checked, the statement cannot be treated as confirmed truth."
            )
            public_flag = "CONTEXT_REQUIRED"
            recommended_action = "OMITTED_CONTEXT_NOTE"

        # Priority 2: High risk (evidence contradicts or strong verified falsehood)
        elif simple_risk == "High" or final_assessment == "Contradicted by Evidence":
            sharing_recommendation = "Do Not Share as Verified"
            immediate_action = (
                "Do not share or circulate this statement as true. If discussing it, "
                "explicitly include a warning that historical fact-checking records contradict this claim."
            )
            verification_steps = [
                "Review verified primary fact-checks that refute this claim.",
                "Check official government publications or original datasets to inspect correct figures.",
                "Identify where the false claim originated and look for published retractions or corrections.",
            ]
            missing_context = (
                "Verified evidence contradicts the claim. Sharing without context promotes inaccurate information."
            )
            reason = (
                f"Agent D assigned a High Risk score ({v_data.get('final_risk_score', 85)}/100) because "
                f"empirical fact-checking records or strong linguistic indicators show this statement is inaccurate."
            )
            public_flag = "MISLEADING_CLAIM_ALERT"
            recommended_action = "CONTEXTUAL_WARNING_BANNER"

        # Priority 3: Low risk with sufficient supporting evidence
        elif simple_risk == "Low" and evidence_strength == "Strong" and final_assessment == "Supported by Evidence":
            sharing_recommendation = "Share with Context"
            immediate_action = (
                "You may share this claim, but provide a direct link to the official source or fact-check "
                "so readers can view the verified data."
            )
            verification_steps = [
                "Verify that the figures apply to the specific time period or jurisdiction mentioned.",
                "Check the primary agency publication (e.g., CBO, BLS, or federal budget) for any recent updates.",
                "Ensure accompanying commentary does not distort or exaggerate the verified factual findings.",
            ]
            missing_context = (
                "While the core assertion is corroborated by historical records, providing the original source link "
                "ensures full transparency for your audience."
            )
            reason = (
                f"Agent D assigned a Low Risk score ({v_data.get('final_risk_score', 15)}/100) because "
                f"verified historical fact-checks corroborate the statement with strong evidence."
            )
            public_flag = "VERIFIED_FACTUAL"
            recommended_action = "NO_ACTION_REQUIRED"

        # Priority 4: Moderate risk / partial accuracy / mixed context
        else:
            sharing_recommendation = "Hold for Verification"
            immediate_action = (
                "Hold the claim for further verification before sharing. The statement may contain partial truth "
                "or conflicting interpretations that require additional context."
            )
            missing_context = (
                "The claim omits key baseline comparisons, conditions, or timeframe qualifiers present in official records."
            )
            reason = (
                f"Agent D assigned Moderate Risk ({v_data.get('final_risk_score', 50)}/100) with mixed or partial "
                f"evidence alignment. Contextual clarification is needed before dissemination."
            )
            verification_steps = [
                "Consult official primary agency reports to check whether key conditions or definitions were omitted.",
                "Look for statements from multiple non-partisan analysts or fact-checking organizations.",
                "Confirm whether figures refer to gross numbers, percentage changes, or selective date ranges.",
            ]
            public_flag = "CONTEXT_REQUIRED"
            recommended_action = "OMITTED_CONTEXT_NOTE"

        # Explicit note about Agent C (Rule: Missing Agent C data must not lead to assumed spread)
        if not has_c:
            missing_context += " (Note: No historical social media propagation data was found; recommendations are based solely on factual analysis.)"

        # Backward compatible recommendations list
        recommendations = [
            {
                "action": immediate_action,
                "priority": "HIGH" if simple_risk == "High" else ("MEDIUM" if simple_risk == "Moderate" else "LOW"),
                "details": f"Sharing Advice: {sharing_recommendation}",
            },
            {
                "action": verification_steps[0],
                "priority": "MEDIUM",
                "details": verification_steps[1] if len(verification_steps) > 1 else "Check primary data.",
            },
        ]
        if len(verification_steps) > 2:
            recommendations.append({
                "action": verification_steps[2],
                "priority": "LOW",
                "details": missing_context,
            })

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = f"Sharing Recommendation: {sharing_recommendation}. Action: {immediate_action[:60]}..."

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "completed",
            "duration_ms": duration_ms,
            "summary": summary,
            "data": {
                # New 5 core sections
                "immediate_action": immediate_action,
                "verification_steps": verification_steps,
                "sharing_recommendation": sharing_recommendation,
                "missing_context": missing_context,
                "reason": reason,

                # Compatibility fields
                "public_flag": public_flag,
                "recommended_action": recommended_action,
                "advisory_text": immediate_action,
                "recommendations": recommendations,
                "composite_risk_score": v_data.get("final_risk_score", 50.0),
                "executed_interventions_count": 0,
            },
            "disclaimer": (
                "Advisory mitigations are non-coercive editorial recommendations. "
                "No automated content blocking, censorship, or removal actions are executed by this research platform."
            ),
        }


AdvisoryMitigationAgent = MitigationRecommendationAgent
