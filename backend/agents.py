"""
Specialized Agent Workflows for Political Misinformation Detection,
Influence Analysis, and Propagation Mitigation.

Defines:
- Agent A: Claim Analysis Agent (DistilBERT six-class classifier + confidence calibration)
- Agent B: Evidence Research / RAG Agent (Hybrid RAG retrieval + grounded synthesis)
- Agent C: Historical Influence Analysis Agent (FakeNewsNet metrics + snowflake temporal propagation)
- Agent D: Verification & Risk Assessment Agent (Cross-synthesis, uncertainty modeling, gap detection)
- Agent E: Mitigation Recommendation Agent (Responsible advisory interventions + non-execution disclaimer)
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# LIAR 6-class semantic risk levels
LIAR_RISK_MAP = {
    "pants-fire": {"risk": "EXTREME", "score": 1.0, "description": "Blatantly false conspiracy or fabricated claim."},
    "false": {"risk": "HIGH", "score": 0.8, "description": "Substantively false statement lacking factual basis."},
    "barely-true": {"risk": "MODERATE", "score": 0.6, "description": "Contains a shred of truth but critical facts omitted."},
    "half-true": {"risk": "MODERATE", "score": 0.4, "description": "Partially accurate with significant misleading context."},
    "mostly-true": {"risk": "LOW", "score": 0.2, "description": "Accurate overall with minor omissions or technicalities."},
    "true": {"risk": "MINIMAL", "score": 0.0, "description": "Factual statement supported by empirical records."},
}


class ClaimAnalysisAgent:
    """
    Agent A: Evaluates claim text using the fine-tuned DistilBERT classifier.
    Produces 6-class LIAR prediction, calibrated confidence, and explains
    that statistical confidence does not equate to verified truth.
    """

    def __init__(self, classifier_fn: Optional[Callable[[str], Dict[str, Any]]] = None):
        self.classifier_fn = classifier_fn
        self.agent_name = "Claim Analysis Agent"
        self.agent_id = "claim_analysis_agent"

    def run(self, claim: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if not self.classifier_fn:
                # Lazy import from main to prevent circular imports
                try:
                    from main import predict_distilbert
                except ImportError:
                    from backend.main import predict_distilbert
                self.classifier_fn = predict_distilbert

            result = self.classifier_fn(claim)
            label = result.get("label", "unknown")
            confidence = float(result.get("confidence", 0.0))
            risk_meta = LIAR_RISK_MAP.get(label, {"risk": "UNKNOWN", "score": 0.5, "description": "Unrecognized class"})

            duration_ms = round((time.time() - start_time) * 1000, 2)
            summary = (
                f"DistilBERT classified claim as '{label}' with {confidence:.1%} confidence. "
                f"Benchmark risk rating: {risk_meta['risk']}."
            )

            calibration_note = (
                "Model confidence measures statistical alignment with linguistic and stylistic patterns "
                "learned from the LIAR benchmark. High model confidence is not an independent proof of "
                "factual veracity and must be cross-referenced with empirical evidence."
            )

            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "completed",
                "duration_ms": duration_ms,
                "summary": summary,
                "data": {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "confidence_percentage": f"{confidence * 100:.1f}%",
                    "risk_category": risk_meta["risk"],
                    "class_description": risk_meta["description"],
                    "all_probabilities": result.get("all_probabilities", {}),
                    "model_name": "DistilBERT (LIAR 6-Class)",
                    "calibration_note": calibration_note,
                },
                "disclaimer": calibration_note,
            }
        except Exception as e:
            logger.exception(f"Error in {self.agent_name}: {e}")
            duration_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "failed",
                "duration_ms": duration_ms,
                "summary": f"Classifier execution failed: {str(e)}",
                "data": {},
                "error": str(e),
            }


class EvidenceRAGAgent:
    """
    Agent B: Retrieves historical evidence from the LIAR-PLUS benchmark using
    the hybrid RAG pipeline (ChromaDB semantic search + TF-IDF lexical search).
    Produces structured citations, detects conflicts, and generates a grounded summary.
    """

    def __init__(self, rag_engine=None):
        self._rag_engine = rag_engine
        self.agent_name = "Evidence Research / RAG Agent"
        self.agent_id = "evidence_rag_agent"

    def _get_engine(self):
        if self._rag_engine is None:
            try:
                try:
                    from rag_engine import get_rag_engine
                except ImportError:
                    from backend.rag_engine import get_rag_engine
                self._rag_engine = get_rag_engine()
            except Exception as e:
                logger.warning(f"Could not load RAG engine: {e}")
                self._rag_engine = None
        return self._rag_engine

    def run(self, claim: str, top_k: int = 5, min_score: float = 0.05) -> Dict[str, Any]:
        start_time = time.time()
        try:
            engine = self._get_engine()
            if engine is None:
                duration_ms = round((time.time() - start_time) * 1000, 2)
                return {
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "status": "failed",
                    "duration_ms": duration_ms,
                    "summary": "RAG evidence retrieval failed: Vector store dependency (ChromaDB/gRPC) is unavailable.",
                    "data": {"citations": [], "citations_count": 0},
                    "error": "ChromaDB/gRPC dependency unavailable",
                }
            records = engine.search(claim=claim, top_k=top_k, min_score=min_score)
            grounded_report = engine.generate_grounded_summary(claim=claim, evidence_records=records)

            duration_ms = round((time.time() - start_time) * 1000, 2)
            rec_count = len(records)

            if rec_count == 0:
                summary = "No relevant historical evidence found in LIAR-PLUS meeting the relevance threshold."
            else:
                top_item = records[0]
                synthesis_note = ""
                if grounded_report.get("synthesis_mode") == "llm":
                    provider_tag = str(grounded_report.get("llm_provider") or "LLM").capitalize()
                    model_tag = grounded_report.get("llm_model") or ""
                    synthesis_note = f" Grounded synthesis generated via {provider_tag} ({model_tag})."

                summary = (
                    f"Retrieved {rec_count} historical evidence record(s). "
                    f"Top match [{top_item['statement_id']}] by {top_item.get('speaker', 'N/A')} "
                    f"(verdict: '{top_item.get('verdict')}', hybrid similarity: {top_item['similarity_score']:.1%})."
                    f"{synthesis_note}"
                )

            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "completed",
                "duration_ms": duration_ms,
                "summary": summary,
                "data": {
                    "citations": records,
                    "citations_count": rec_count,
                    "grounded_summary": grounded_report.get("summary", ""),
                    "evidence_status": grounded_report.get("evidence_status", "unknown"),
                    "has_conflicts": grounded_report.get("has_conflicts", False),
                    "verdict_distribution": grounded_report.get("verdict_distribution", {}),
                    "misaligned_citations_count": grounded_report.get("misaligned_citations_count", 0),
                    "generation_mode": grounded_report.get("generation_mode", "deterministic_fallback"),
                    "synthesis_mode": grounded_report.get("synthesis_mode", "template"),
                    "llm_provider": grounded_report.get("llm_provider"),
                    "llm_model": grounded_report.get("llm_model"),
                    "llm_status": grounded_report.get("llm_status", "skipped_no_key"),
                    "fallback_reason": grounded_report.get("fallback_reason"),
                },
                "disclaimer": (
                    "Retrieved citations reflect historical PolitiFact rulings from the LIAR-PLUS dataset. "
                    "They do not constitute real-time live web searches or independent primary verifications."
                ),
            }
        except Exception as e:
            logger.exception(f"Error in {self.agent_name}: {e}")
            duration_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "failed",
                "duration_ms": duration_ms,
                "summary": f"RAG retrieval failed: {str(e)}",
                "data": {"citations": [], "citations_count": 0},
                "error": str(e),
            }


class HistoricalInfluenceAgent:
    """
    Agent C: Analyzes historical social engagement indicators from FakeNewsNet.
    Extracts tweet counts, benchmark percentiles, and Twitter Snowflake timestamps.
    Explicitly bounds metrics to historical benchmark stories, not the submitted claim.
    """

    def __init__(self, influence_analyzer=None):
        self._analyzer = influence_analyzer
        self.agent_name = "Historical Influence Agent"
        self.agent_id = "historical_influence_agent"

    def _get_analyzer(self):
        if self._analyzer is None:
            try:
                from influence_analyzer import get_influence_analyzer
            except ImportError:
                from backend.influence_analyzer import get_influence_analyzer
            self._analyzer = get_influence_analyzer()
        return self._analyzer

    def run(self, claim: str, top_k: int = 3, min_score: float = 0.08) -> Dict[str, Any]:
        start_time = time.time()
        try:
            analyzer = self._get_analyzer()
            matched_stories = analyzer.search(claim=claim, top_k=top_k, min_score=min_score)
            benchmark_stats = analyzer.get_benchmark_stats()

            duration_ms = round((time.time() - start_time) * 1000, 2)
            matched_count = len(matched_stories)

            if matched_count == 0:
                summary = "No matching historical stories found in the FakeNewsNet benchmark (N=1,056)."
            else:
                top_story = matched_stories[0]
                summary = (
                    f"Matched {matched_count} benchmark story/stories. "
                    f"Top match [{top_story['news_id']}] ('{top_story['dataset_label']}') recorded "
                    f"{top_story['observed_tweet_count']} historical tweets "
                    f"(percentile within label: {top_story.get('percentile_among_fake') if top_story['dataset_label'] == 'fake' else top_story.get('percentile_among_real')}%)."
                )

            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "completed",
                "duration_ms": duration_ms,
                "summary": summary,
                "data": {
                    "matched_count": matched_count,
                    "matched_stories": matched_stories,
                    "benchmark_baselines": benchmark_stats,
                },
                "disclaimer": (
                    "Influence indicators represent tweet IDs historically cataloged for matched stories "
                    "in the FakeNewsNet benchmark. They do NOT reflect social reach, impressions, follower "
                    "networks, or live propagation of the query claim."
                ),
            }
        except Exception as e:
            logger.exception(f"Error in {self.agent_name}: {e}")
            duration_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "failed",
                "duration_ms": duration_ms,
                "summary": f"Influence analysis failed: {str(e)}",
                "data": {"matched_count": 0, "matched_stories": []},
                "error": str(e),
            }


class VerificationRiskAgent:
    """
    Agent D: Reviews and cross-synthesizes outputs from Agents A, B, and C.
    Identifies conflicting signals, weak evidence, missing data, and dataset integrity issues.
    Reports an overall synthesized risk level and uncertainty assessment.
    """

    def __init__(self):
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

        # 1. Inspect Classifier Output (Agent A)
        a_data = claim_analysis.get("data", {})
        predicted_label = a_data.get("label", "unknown")
        confidence = float(a_data.get("confidence", 0.0))
        classifier_failed = claim_analysis.get("status") != "completed"

        if classifier_failed:
            findings.append("Primary classifier (DistilBERT) was unavailable or encountered an error.")
            uncertainties.append("Linguistic pattern classification is missing.")

        # 2. Inspect Evidence Output (Agent B)
        b_data = evidence_rag.get("data", {})
        citations = b_data.get("citations", [])
        evidence_status = b_data.get("evidence_status", "insufficient")
        has_conflicts = b_data.get("has_conflicts", False)
        misaligned_citations = b_data.get("misaligned_citations_count", 0)
        rag_failed = evidence_rag.get("status") != "completed"

        if rag_failed:
            findings.append("RAG evidence retrieval was unavailable.")
            uncertainties.append("Historical evidence cannot be validated.")
        elif not citations or evidence_status == "insufficient":
            findings.append("No historical fact-check precedent found in LIAR-PLUS benchmark for this claim.")
            uncertainties.append("The absence of historical evidence prevents definitive empirical grounding.")
        else:
            findings.append(f"Retrieved {len(citations)} historical precedent records.")
            if has_conflicts:
                findings.append("Historical PolitiFact precedents present conflicting verdicts on related aspects.")
                uncertainties.append("Benchmark conflict: Divergent historical rulings require human contextual review.")

        if misaligned_citations > 0:
            integrity_flags.append(
                f"{misaligned_citations} evidence citation(s) have potential upstream TSV row-shift discrepancies."
            )

        # 3. Inspect Influence Output (Agent C)
        c_data = influence_analysis.get("data", {})
        matched_stories = c_data.get("matched_stories", [])
        c_failed = influence_analysis.get("status") != "completed"

        if c_failed:
            findings.append("Historical influence benchmark matching was unavailable.")
        elif not matched_stories:
            findings.append("No historical FakeNewsNet stories matched this claim.")
            uncertainties.append("Historical benchmark engagement metrics are unobserved for this headline.")
        else:
            top_story = matched_stories[0]
            story_title = top_story.get("title") or top_story.get("headline") or "Historical story"
            story_tweets = top_story.get("observed_tweet_count") or top_story.get("recorded_tweet_count") or 0
            findings.append(
                f"Matched benchmark headline '{story_title}' ({story_tweets} historical tweets)."
            )

        # 4. Synthesize Cross-Agent Consistency & Risk Matrix
        # Determine overall synthesized risk level
        risk_level = "MODERATE"
        uncertainty_score = 0.5

        if classifier_failed and (not citations) and (not matched_stories):
            risk_level = "UNVERIFIED_INSUFFICIENT_EVIDENCE"
            uncertainty_score = 0.95
        elif not citations:
            # Model prediction exists but no evidence to back it up
            risk_level = "UNVERIFIED_MODEL_ONLY"
            uncertainty_score = 0.75
            uncertainties.append(
                f"DistilBERT predicts '{predicted_label}' ({confidence:.1%}), but no supporting benchmark citations exist."
            )
        else:
            top_verdict = citations[0].get("verdict", "")
            # Check agreement between classifier prediction and top historical evidence verdict
            is_prediction_false = predicted_label in {"pants-fire", "false", "barely-true"}
            is_evidence_false = top_verdict in {"pants-fire", "false", "barely-true"}

            if is_prediction_false and is_evidence_false:
                risk_level = "HIGH_MISINFORMATION_RISK"
                uncertainty_score = 0.25
                findings.append("Concurrence: Classifier prediction and historical evidence agree on non-factual classification.")
            elif not is_prediction_false and not is_evidence_false:
                risk_level = "LOW_MISINFORMATION_RISK"
                uncertainty_score = 0.25
                findings.append("Concurrence: Classifier prediction and historical evidence agree on substantially accurate classification.")
            else:
                risk_level = "DISCREPANCY_BETWEEN_MODEL_AND_EVIDENCE"
                uncertainty_score = 0.65
                findings.append(
                    f"Discrepancy: Classifier predicts '{predicted_label}' but top historical evidence was labeled '{top_verdict}'."
                )
                uncertainties.append("Divergence between linguistic pattern model and historical PolitiFact evidence.")

        # Guardrail: Never treat historical tweet counts or labels as proof of current claim truth
        uncertainties.append(
            "Guardrail applied: Historical benchmark tweet counts and past PolitiFact labels describe past events "
            "and do not constitute definitive proof of truth for the submitted statement."
        )

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = (
            f"Risk Level: {risk_level} (Uncertainty: {uncertainty_score:.0%}). "
            f"Identified {len(findings)} finding(s) and {len(uncertainties)} uncertainty factor(s)."
        )

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "completed",
            "duration_ms": duration_ms,
            "summary": summary,
            "data": {
                "risk_level": risk_level,
                "uncertainty_score": round(uncertainty_score, 2),
                "findings": findings,
                "uncertainties": uncertainties,
                "integrity_flags": integrity_flags,
                "classifier_status": claim_analysis.get("status"),
                "rag_status": evidence_rag.get("status"),
                "influence_status": influence_analysis.get("status"),
            },
        }


class MitigationRecommendationAgent:
    """
    Agent E: Formulates responsible, non-coercive mitigation recommendations
    based on the findings of Agent D.
    Explicitly distinguishes advisory recommendations from executed actions,
    clarifying that no automated content removal or platform interventions took place.
    """

    def __init__(self):
        self.agent_name = "Mitigation Recommendation Agent"
        self.agent_id = "mitigation_agent"

    def run(
        self,
        claim: str,
        verification_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        start_time = time.time()
        v_data = verification_data.get("data", {})
        risk_level = v_data.get("risk_level", "MODERATE")
        uncertainty = v_data.get("uncertainty_score", 0.5)

        recommendations: List[Dict[str, str]] = []

        if risk_level in {"HIGH_MISINFORMATION_RISK", "EXTREME"}:
            recommendations.append({
                "action": "Contextual Fact-Check Banner",
                "priority": "High",
                "target": "User / Reader Interface",
                "rationale": "Claim exhibits linguistic and evidence markers consistent with known false narratives.",
                "details": "Display an informational overlay linking readers to verified primary sources and fact-checks.",
            })
            recommendations.append({
                "action": "Human Editorial Escalation",
                "priority": "High",
                "target": "Verification Queue",
                "rationale": "High-risk political statements require certified human fact-checker validation before indexing.",
                "details": "Flag statement for manual review by credentialed fact-checkers.",
            })
            recommendations.append({
                "action": "Propagation Friction Advisory",
                "priority": "Medium",
                "target": "Distribution Channels",
                "rationale": "Unverified high-velocity claims pose rapid misinformation cascade risk.",
                "details": "Prompt users to read authoritative background articles prior to re-sharing.",
            })

        elif risk_level == "DISCREPANCY_BETWEEN_MODEL_AND_EVIDENCE":
            recommendations.append({
                "action": "Discrepancy Investigation",
                "priority": "High",
                "target": "Research Analyst",
                "rationale": "Classifier prediction contradicts historical PolitiFact precedent for similar statements.",
                "details": "Conduct a manual inspection to determine whether recent real-world events altered the claim's factual basis.",
            })
            recommendations.append({
                "action": "Dual-Context Attribution",
                "priority": "Medium",
                "target": "User Interface",
                "rationale": "Present both perspectives transparently to avoid algorithmic bias.",
                "details": "Show both the model assessment and historical precedent side-by-side with confidence bounds.",
            })

        elif risk_level in {"UNVERIFIED_INSUFFICIENT_EVIDENCE", "UNVERIFIED_MODEL_ONLY"}:
            recommendations.append({
                "action": "Primary Source Ingestion Request",
                "priority": "High",
                "target": "Evidence Database",
                "rationale": "Benchmark does not contain sufficient verified historical records for this claim.",
                "details": "Do not classify as definitively false without acquiring primary documentation (statutes, budget reports, agency releases).",
            })
            recommendations.append({
                "action": "Unverified Claim Label",
                "priority": "Medium",
                "target": "User Interface",
                "rationale": "Prevent false certainty when empirical grounding is missing.",
                "details": "Display clear notice: 'Unverified: Insufficient historical fact-check records found in benchmark.'",
            })

        else:  # LOW_MISINFORMATION_RISK or MINIMAL
            recommendations.append({
                "action": "Standard Archival Logging",
                "priority": "Low",
                "target": "Audit Log",
                "rationale": "Claim aligns with historically accurate or benign statements.",
                "details": "Maintain audit log of verification request without active intervention.",
            })
            recommendations.append({
                "action": "Periodic Grounding Refresh",
                "priority": "Low",
                "target": "Maintenance System",
                "rationale": "Ensure evidentiary standards remain current as political context shifts.",
                "details": "Re-run retrieval if updated political records are ingested into the database.",
            })

        disclaimer = (
            "IMPORTANT: All mitigation items are advisory decision-support recommendations. "
            "The system has NOT removed, censored, restricted, or intervened in live social media "
            "or real-world publication workflows."
        )

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = f"Generated {len(recommendations)} advisory recommendation(s) for risk profile '{risk_level}'."

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "completed",
            "duration_ms": duration_ms,
            "summary": summary,
            "data": {
                "risk_profile": risk_level,
                "uncertainty_level": f"{uncertainty:.0%}",
                "recommendations": recommendations,
                "advisory_notice": disclaimer,
                "executed_interventions_count": 0,
            },
            "disclaimer": disclaimer,
        }

