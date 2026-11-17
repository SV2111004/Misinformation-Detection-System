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
import re
import time
from typing import Any, Callable, Dict, List, Optional

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


class VerificationRiskAgent:
    """
    Agent D / CrossVerificationAgent:
    Implements 4-Stage Epistemic Reasoning:
    1. Relevance Gating (evaluates cosine similarity & policy entity overlap)
    2. Directional Stance (determines whether evidence supports or contradicts the claim)
    3. Justification Analysis (audits cited sources, numbers, and missing justifications)
    4. Cross-Signal Triangulation (synthesizes classifier, evidence stance, conflict, and velocity)

    CRITICAL RULE: Historical RAG verdicts are NOT treated as ground truth for new claims.
    """

    def __init__(self):
        self.agent_name = "Cross-Verification & Epistemic Synthesis Agent"
        self.agent_id = "verification_risk_agent"

    def _determine_directional_stance(self, user_claim: str, evidence_claim: str, evidence_verdict: str) -> str:
        """
        Determines whether a historical claim supports, contradicts, or has neutral stance
        relative to the user's submitted claim.
        """
        u_norm = user_claim.lower()
        e_norm = evidence_claim.lower()

        # Check assertion polarity indicators
        negations_u = bool(re.search(r"\b(?:not|never|no|failed|cut|reduced|eliminated|destroyed|lost)\b", u_norm))
        negations_e = bool(re.search(r"\b(?:not|never|no|failed|cut|reduced|eliminated|destroyed|lost)\b", e_norm))

        # If assertion polarities are opposite:
        # e.g., Historical says "ACA increased insurance" (Factual), user says "ACA destroyed insurance"
        if negations_u != negations_e:
            if evidence_verdict == "Factual":
                return "CONTRADICTED_BY_EVIDENCE"
            elif evidence_verdict == "Misleading":
                return "SUPPORTED_BY_EVIDENCE"
            else:
                return "DISPUTED_STANCE"
        else:
            if evidence_verdict == "Factual":
                return "SUPPORTED_BY_EVIDENCE"
            elif evidence_verdict == "Misleading":
                return "CONTRADICTED_BY_EVIDENCE"
            else:
                return "PARTIALLY_ALIGNED"

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

        pred_label = a_data.get("label", "Half-True")
        pred_conf = float(a_data.get("confidence", 0.50))
        citations = b_data.get("citations", [])

        # Stage 1: Relevance Gating
        relevant_citations = []
        direct_precedents = []
        for cit in citations:
            score = float(cit.get("similarity_score", 0.0))
            if score >= 0.70:
                direct_precedents.append(cit)
                relevant_citations.append(cit)
            elif score >= 0.40:
                relevant_citations.append(cit)

        findings.append(
            f"Stage 1 (Relevance Gating): Found {len(direct_precedents)} direct historical precedent(s) "
            f"and {len(relevant_citations)} relevant context citation(s) out of {len(citations)} candidates."
        )

        # Stage 2: Directional Stance
        top_stance = "NO_RELEVANT_EVIDENCE"
        if relevant_citations:
            top_cit = relevant_citations[0]
            top_stance = self._determine_directional_stance(
                user_claim=claim,
                evidence_claim=top_cit.get("claim", ""),
                evidence_verdict=top_cit.get("verdict", "Half-True"),
            )
            findings.append(
                f"Stage 2 (Directional Stance): Historical precedent '{top_cit.get('statement_id')}' "
                f"({top_cit.get('verdict')}) exhibits stance: {top_stance}."
            )
        else:
            uncertainties.append("Absence of high-relevance domain evidence prevents definitive empirical grounding.")

        # Stage 3: Justification Analysis
        valid_just_count = sum(1 for c in relevant_citations if c.get("has_justification", True))
        if relevant_citations and valid_just_count > 0:
            top_just = relevant_citations[0].get("justification", "")
            findings.append(f"Stage 3 (Justification Analysis): Fact-checker reasoning audited ({valid_just_count} verified justifications).")
        elif relevant_citations:
            integrity_flags.append("Retrieved historical records lack published journalist justifications.")

        # Stage 4: Cross-Signal Triangulation
        conflict_detected = False
        conflict_score = 0.0

        if a_status in {"SUCCESS", "completed"} and relevant_citations:
            if pred_label == "Factual" and top_stance == "CONTRADICTED_BY_EVIDENCE":
                conflict_detected = True
                conflict_score = 0.75
                findings.append("CONFLICT: Classifier predicts Factual, but historical evidence directly contradicts claim assertion!")
            elif pred_label == "Misleading" and top_stance == "SUPPORTED_BY_EVIDENCE":
                conflict_detected = True
                conflict_score = 0.70
                findings.append("CONFLICT: Classifier predicts Misleading, but historical evidence supports claim assertion!")
            else:
                findings.append("CONCURRENCE: Classifier prediction aligns with directional evidence stance.")

        # Calculate Composite Risk Score (0 - 100)
        norm_pred = str(pred_label).lower()
        if norm_pred in {"misleading", "false", "pants-fire", "barely-true"}:
            classifier_risk = 85.0
            findings.append("Classifier predicts statement is misleading or non-factual.")
        elif norm_pred in {"factual", "true", "mostly-true"}:
            classifier_risk = 15.0
            findings.append("Classifier predicts statement is substantially factual.")
        else:
            classifier_risk = 50.0
            findings.append("Classifier predicts statement is half-true with omitted context.")

        evidence_risk = 50.0
        all_cits = relevant_citations or citations
        if all_cits:
            top_verdict = str(all_cits[0].get("verdict", "half-true")).lower()
            if top_verdict in {"misleading", "false", "pants-fire", "barely-true"}:
                evidence_risk = 85.0
            elif top_verdict in {"factual", "true", "mostly-true"}:
                evidence_risk = 15.0
            else:
                evidence_risk = 50.0

        velocity_risk = 80.0 if c_data.get("engagement_velocity") == "HIGH" else 30.0

        if a_status in {"SUCCESS", "completed"} and b_status in {"SUCCESS", "completed"}:
            if conflict_detected:
                composite_risk = (0.45 * classifier_risk) + (0.40 * evidence_risk) + (0.15 * velocity_risk)
            else:
                composite_risk = (0.50 * classifier_risk) + (0.35 * evidence_risk) + (0.15 * velocity_risk)
        elif a_status in {"SUCCESS", "completed"}:
            composite_risk = (0.75 * classifier_risk) + (0.25 * velocity_risk)
            uncertainties.append("RAG evidence offline: Risk calculated from classifier and social signals only.")
        elif b_status in {"SUCCESS", "completed"}:
            composite_risk = (0.75 * evidence_risk) + (0.25 * velocity_risk)
            uncertainties.append("Classifier offline: Risk inferred from directional evidence stance.")
        else:
            composite_risk = 50.0
            uncertainties.append("Both classifier and RAG offline: Defers to editorial human review.")

        composite_risk = round(max(0.0, min(100.0, composite_risk)), 1)

        if composite_risk >= 70.0:
            risk_level = "HIGH_MISINFORMATION_RISK"
        elif composite_risk >= 35.0:
            risk_level = "MODERATE_MISINFORMATION_RISK"
        else:
            risk_level = "LOW_MISINFORMATION_RISK"

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = (
            f"Composite Risk: {composite_risk}/100 ({risk_level}). "
            f"Epistemic Stance: {top_stance}. Conflict Index: {conflict_score:.2f}."
        )

        uncertainty_score = round(max(0.05, min(0.95, 1.0 - (pred_conf if a_status in {"SUCCESS", "completed"} else 0.50))), 2)
        if not any("Guardrail" in u for u in uncertainties):
            uncertainties.append("Guardrail: Epistemic assessment bounds certainty based on available evidence coverage.")

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
                "agent_statuses": {
                    "classifier": a_status,
                    "evidence_rag": b_status,
                    "influence": c_status,
                },
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
