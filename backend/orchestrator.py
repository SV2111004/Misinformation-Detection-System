"""
Central Orchestrator Module for Political Misinformation Detection System.

Coordinates the multi-agent analysis workflow:
1. Validates submitted claim text.
2. Executes Agent A (Claim Analysis), Agent B (Evidence Research / RAG),
   and Agent C (Historical Influence) concurrently.
3. Feeds structured outputs to Agent D (Verification & Risk Assessment).
4. Feeds risk findings to Agent E (Mitigation Recommendations).
5. Tracks granular execution traces, status flags, and deterministic fallback notices.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import datetime
import logging
import time
from typing import Any, Dict, List, Optional

try:
    from agents import (
        ClaimAnalysisAgent,
        EvidenceRAGAgent,
        HistoricalInfluenceAgent,
        MitigationRecommendationAgent,
        VerificationRiskAgent,
    )
except ImportError:
    from backend.agents import (
        ClaimAnalysisAgent,
        EvidenceRAGAgent,
        HistoricalInfluenceAgent,
        MitigationRecommendationAgent,
        VerificationRiskAgent,
    )

logger = logging.getLogger(__name__)


class OrchestrationWorkflow:
    """
    Coordinates execution of specialized agents with parallel processing,
    error isolation, and traceable execution history.
    """

    def __init__(
        self,
        claim_agent: Optional[ClaimAnalysisAgent] = None,
        rag_agent: Optional[EvidenceRAGAgent] = None,
        influence_agent: Optional[HistoricalInfluenceAgent] = None,
        verification_agent: Optional[VerificationRiskAgent] = None,
        mitigation_agent: Optional[MitigationRecommendationAgent] = None,
        max_workers: int = 3,
    ):
        self.claim_agent = claim_agent or ClaimAnalysisAgent()
        self.rag_agent = rag_agent or EvidenceRAGAgent()
        self.influence_agent = influence_agent or HistoricalInfluenceAgent()
        self.verification_agent = verification_agent or VerificationRiskAgent()
        self.mitigation_agent = mitigation_agent or MitigationRecommendationAgent()
        self.max_workers = max_workers

    def _validate_claim(self, claim: str) -> str:
        """Validates and cleans input claim text."""
        if not isinstance(claim, str):
            raise ValueError("Claim must be a string.")
        cleaned = claim.strip()
        if len(cleaned) < 3:
            raise ValueError("Claim is too short. Minimum length is 3 characters.")
        if len(cleaned) > 5000:
            raise ValueError("Claim exceeds maximum length of 5000 characters.")
        return cleaned

    def _safe_resolve(
        self,
        future,
        fallback_id: str,
        fallback_name: str,
    ) -> Dict[str, Any]:
        """Safely extracts result from a thread pool future, catching uncaught agent exceptions."""
        try:
            res = future.result()
            if not isinstance(res, dict) or "status" not in res:
                return {
                    "agent_id": fallback_id,
                    "agent_name": fallback_name,
                    "status": "failed",
                    "duration_ms": 0.0,
                    "summary": "Agent returned malformed output.",
                    "data": {},
                    "error": "Invalid output format",
                }
            return res
        except Exception as e:
            logger.exception(f"Unhandled exception in {fallback_name}: {e}")
            return {
                "agent_id": fallback_id,
                "agent_name": fallback_name,
                "status": "failed",
                "duration_ms": 0.0,
                "summary": f"Agent crashed: {str(e)}",
                "data": {},
                "error": str(e),
            }

    def analyze_sync(
        self,
        claim: str,
        top_k_evidence: int = 5,
        top_k_influence: int = 3,
    ) -> Dict[str, Any]:
        """
        Synchronous execution entrypoint coordinating parallel phase 1 and sequential phase 2.
        """
        cleaned_claim = self._validate_claim(claim)
        start_overall = time.time()
        execution_trace: List[Dict[str, Any]] = []

        # Phase 1: Parallel execution of Agents A, B, and C
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fut_a = executor.submit(self.claim_agent.run, cleaned_claim)
            fut_b = executor.submit(self.rag_agent.run, cleaned_claim, top_k_evidence)
            fut_c = executor.submit(self.influence_agent.run, cleaned_claim, top_k_influence)

            res_a = self._safe_resolve(fut_a, "claim_analysis_agent", "Claim Analysis Agent")
            res_b = self._safe_resolve(fut_b, "evidence_rag_agent", "Evidence Research / RAG Agent")
            res_c = self._safe_resolve(fut_c, "historical_influence_agent", "Historical Influence Agent")

        execution_trace.append({
            "agent_id": res_a.get("agent_id", "claim_analysis_agent"),
            "agent_name": res_a.get("agent_name", "Claim Analysis Agent"),
            "status": res_a.get("status", "failed"),
            "duration_ms": res_a.get("duration_ms", 0.0),
            "summary": res_a.get("summary", "No summary available."),
            "error": res_a.get("error"),
        })
        execution_trace.append({
            "agent_id": res_b.get("agent_id", "evidence_rag_agent"),
            "agent_name": res_b.get("agent_name", "Evidence Research / RAG Agent"),
            "status": res_b.get("status", "failed"),
            "duration_ms": res_b.get("duration_ms", 0.0),
            "summary": res_b.get("summary", "No summary available."),
            "error": res_b.get("error"),
        })
        execution_trace.append({
            "agent_id": res_c.get("agent_id", "historical_influence_agent"),
            "agent_name": res_c.get("agent_name", "Historical Influence Agent"),
            "status": res_c.get("status", "failed"),
            "duration_ms": res_c.get("duration_ms", 0.0),
            "summary": res_c.get("summary", "No summary available."),
            "error": res_c.get("error"),
        })

        # Phase 2: Sequential execution of Agent D (Verification & Risk Assessment)
        try:
            res_d = self.verification_agent.run(
                claim=cleaned_claim,
                claim_analysis=res_a,
                evidence_rag=res_b,
                influence_analysis=res_c,
            )
        except Exception as e:
            logger.exception(f"Verification agent crashed: {e}")
            res_d = {
                "agent_id": "verification_risk_agent",
                "agent_name": "Verification & Risk Assessment Agent",
                "status": "failed",
                "duration_ms": 0.0,
                "summary": f"Verification failed: {str(e)}",
                "data": {"risk_level": "UNKNOWN", "uncertainty_score": 1.0, "findings": [], "uncertainties": [str(e)]},
                "error": str(e),
            }

        execution_trace.append({
            "agent_id": res_d.get("agent_id", "verification_risk_agent"),
            "agent_name": res_d.get("agent_name", "Verification & Risk Assessment Agent"),
            "status": res_d.get("status", "failed"),
            "duration_ms": res_d.get("duration_ms", 0.0),
            "summary": res_d.get("summary", "No summary available."),
            "error": res_d.get("error"),
        })

        # Phase 3: Sequential execution of Agent E (Mitigation Recommendations)
        try:
            res_e = self.mitigation_agent.run(
                claim=cleaned_claim,
                verification_data=res_d,
            )
        except Exception as e:
            logger.exception(f"Mitigation agent crashed: {e}")
            res_e = {
                "agent_id": "mitigation_agent",
                "agent_name": "Mitigation Recommendation Agent",
                "status": "failed",
                "duration_ms": 0.0,
                "summary": f"Mitigation recommendations failed: {str(e)}",
                "data": {"recommendations": []},
                "error": str(e),
            }

        execution_trace.append({
            "agent_id": res_e.get("agent_id", "mitigation_agent"),
            "agent_name": res_e.get("agent_name", "Mitigation Recommendation Agent"),
            "status": res_e.get("status", "failed"),
            "duration_ms": res_e.get("duration_ms", 0.0),
            "summary": res_e.get("summary", "No summary available."),
            "error": res_e.get("error"),
        })

        total_duration_ms = round((time.time() - start_overall) * 1000, 2)
        has_failures = any(t["status"] == "failed" for t in execution_trace)
        workflow_status = "completed_with_errors" if has_failures else "completed"

        return {
            "claim": cleaned_claim,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "workflow_status": workflow_status,
            "total_duration_ms": total_duration_ms,
            "claim_analysis": res_a,
            "evidence_rag": res_b,
            "historical_influence": res_c,
            "verification_risk": res_d,
            "mitigation_recommendations": res_e,
            "execution_trace": execution_trace,
            "limitations_and_disclaimers": [
                res_a.get("disclaimer"),
                res_b.get("disclaimer"),
                res_c.get("disclaimer"),
                res_e.get("disclaimer"),
            ],
        }

    async def analyze(
        self,
        claim: str,
        top_k_evidence: int = 5,
        top_k_influence: int = 3,
    ) -> Dict[str, Any]:
        """
        Async wrapper executing the parallel and sequential pipeline on an event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.analyze_sync,
            claim,
            top_k_evidence,
            top_k_influence,
        )


# Singleton instance
_orchestrator_instance: Optional[OrchestrationWorkflow] = None


def get_orchestrator() -> OrchestrationWorkflow:
    """Returns singleton orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestrationWorkflow()
    return _orchestrator_instance

