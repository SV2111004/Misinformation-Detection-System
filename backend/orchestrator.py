"""
Central Orchestrator Module for Socio-Economic Political Misinformation Detection System.

Coordinates the multi-agent analysis workflow:
1. Validates submitted claim text.
2. Supports Parallel (default) and Sequential execution modes with explicit timeout isolation.
3. Phase 1: Executes Agent A (Claim Decomposition), Agent B (Evidence RAG),
   and Agent C (Historical Influence) concurrently with per-agent 3.0s timeout.
4. Phase 2: Feeds structured outputs to Agent D (Cross-Verification & Epistemic Reasoning)
   which explicitly distinguishes SUCCESS, TIMEOUT, and UNAVAILABLE signals.
5. Phase 3: Feeds verified risk findings to Agent E (Advisory Mitigation Recommendations).
6. Tracks granular execution traces, duration, and status flags.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import datetime
import logging
import os
import time
from typing import Any, Dict, List, Optional

try:
    from agents import (
        ClaimAnalysisAgent,
        EvidenceRAGAgent,
        HistoricalInfluenceAgent,
        MitigationRecommendationAgent,
        VerificationRiskAgent,
        ClaimDecompositionAgent,
        CrossVerificationAgent,
        AdvisoryMitigationAgent,
    )
except ImportError:
    from backend.agents import (
        ClaimAnalysisAgent,
        EvidenceRAGAgent,
        HistoricalInfluenceAgent,
        MitigationRecommendationAgent,
        VerificationRiskAgent,
        ClaimDecompositionAgent,
        CrossVerificationAgent,
        AdvisoryMitigationAgent,
    )

logger = logging.getLogger(__name__)


class OrchestrationWorkflow:
    """
    Coordinates execution of specialized agents with parallel/sequential processing,
    error isolation, timeout enforcement, and traceable execution history.
    """

    def __init__(
        self,
        claim_agent: Optional[ClaimAnalysisAgent] = None,
        rag_agent: Optional[EvidenceRAGAgent] = None,
        influence_agent: Optional[HistoricalInfluenceAgent] = None,
        verification_agent: Optional[VerificationRiskAgent] = None,
        mitigation_agent: Optional[MitigationRecommendationAgent] = None,
        max_workers: int = 3,
        agent_timeout: float = 3.0,
        global_timeout: float = 5.0,
        mode: Optional[str] = None,
    ):
        self.claim_agent = claim_agent or ClaimAnalysisAgent()
        self.rag_agent = rag_agent or EvidenceRAGAgent()
        self.influence_agent = influence_agent or HistoricalInfluenceAgent()
        self.verification_agent = verification_agent or VerificationRiskAgent()
        self.mitigation_agent = mitigation_agent or MitigationRecommendationAgent()
        self.max_workers = max_workers
        self.agent_timeout = agent_timeout
        self.global_timeout = global_timeout
        self.mode = (mode or os.environ.get("ORCHESTRATION_MODE", "parallel")).lower().strip()

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
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Safely extracts result from a thread pool future with explicit timeout and error isolation."""
        try:
            res = future.result(timeout=timeout)
            if not isinstance(res, dict) or "status" not in res:
                return {
                    "agent_id": fallback_id,
                    "agent_name": fallback_name,
                    "status": "UNAVAILABLE",
                    "duration_ms": 0.0,
                    "summary": "Agent returned malformed output.",
                    "data": {},
                    "error": "Invalid output format",
                }
            return res
        except (TimeoutError, FuturesTimeoutError):
            logger.warning(f"Timeout ({timeout}s) exceeded for {fallback_name}")
            return {
                "agent_id": fallback_id,
                "agent_name": fallback_name,
                "status": "TIMEOUT",
                "duration_ms": round((timeout or self.agent_timeout) * 1000.0, 2),
                "summary": f"Agent exceeded {timeout or self.agent_timeout}s execution deadline.",
                "data": {},
                "error": f"Execution deadline ({timeout or self.agent_timeout}s) exceeded",
            }
        except Exception as e:
            logger.exception(f"Unhandled exception in {fallback_name}: {e}")
            return {
                "agent_id": fallback_id,
                "agent_name": fallback_name,
                "status": "UNAVAILABLE",
                "duration_ms": 0.0,
                "summary": f"Agent crashed: {str(e)}",
                "data": {},
                "error": str(e),
            }

    def analyze_parallel(
        self,
        claim: str,
        top_k_evidence: int = 5,
        top_k_influence: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes Phase 1 (Agents A, B, C) concurrently in ThreadPoolExecutor with per-agent timeout,
        followed by sequential Phase 2 (Agent D) and Phase 3 (Agent E).
        """
        cleaned_claim = self._validate_claim(claim)
        start_overall = time.time()
        execution_trace: List[Dict[str, Any]] = []

        # Phase 1: Concurrent execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fut_a = executor.submit(self.claim_agent.run, cleaned_claim)
            fut_b = executor.submit(self.rag_agent.run, cleaned_claim, top_k_evidence)
            fut_c = executor.submit(self.influence_agent.run, cleaned_claim, top_k_influence)

            res_a = self._safe_resolve(
                fut_a,
                getattr(self.claim_agent, "agent_id", "claim_analysis_agent"),
                getattr(self.claim_agent, "agent_name", "Claim Analysis Agent"),
                timeout=self.agent_timeout,
            )
            res_b = self._safe_resolve(
                fut_b,
                getattr(self.rag_agent, "agent_id", "evidence_rag_agent"),
                getattr(self.rag_agent, "agent_name", "Evidence Research / RAG Agent"),
                timeout=self.agent_timeout,
            )
            res_c = self._safe_resolve(
                fut_c,
                getattr(self.influence_agent, "agent_id", "historical_influence_agent"),
                getattr(self.influence_agent, "agent_name", "Historical Influence Agent"),
                timeout=self.agent_timeout,
            )

        execution_trace.append({
            "agent_id": res_a.get("agent_id", "claim_analysis_agent"),
            "agent_name": res_a.get("agent_name", "Claim Analysis Agent"),
            "status": res_a.get("status", "UNAVAILABLE"),
            "duration_ms": res_a.get("duration_ms", 0.0),
            "summary": res_a.get("summary", "No summary available."),
            "error": res_a.get("error"),
        })
        execution_trace.append({
            "agent_id": res_b.get("agent_id", "evidence_rag_agent"),
            "agent_name": res_b.get("agent_name", "Evidence Research / RAG Agent"),
            "status": res_b.get("status", "UNAVAILABLE"),
            "duration_ms": res_b.get("duration_ms", 0.0),
            "summary": res_b.get("summary", "No summary available."),
            "error": res_b.get("error"),
        })
        execution_trace.append({
            "agent_id": res_c.get("agent_id", "historical_influence_agent"),
            "agent_name": res_c.get("agent_name", "Historical Influence Agent"),
            "status": res_c.get("status", "UNAVAILABLE"),
            "duration_ms": res_c.get("duration_ms", 0.0),
            "summary": res_c.get("summary", "No summary available."),
            "error": res_c.get("error"),
        })

        # Phase 2: Sequential execution of Agent D (Cross-Verification)
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
                "agent_id": getattr(self.verification_agent, "agent_id", "verification_risk_agent"),
                "agent_name": getattr(self.verification_agent, "agent_name", "Verification & Risk Assessment Agent"),
                "status": "UNAVAILABLE",
                "duration_ms": 0.0,
                "summary": f"Verification failed: {str(e)}",
                "data": {
                    "composite_risk_score": 50.0,
                    "risk_level": "MODERATE_MISINFORMATION_RISK",
                    "findings": [],
                    "uncertainties": [str(e)],
                },
                "error": str(e),
            }

        execution_trace.append({
            "agent_id": res_d.get("agent_id", "verification_risk_agent"),
            "agent_name": res_d.get("agent_name", "Verification & Risk Assessment Agent"),
            "status": res_d.get("status", "UNAVAILABLE"),
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
                "agent_id": getattr(self.mitigation_agent, "agent_id", "mitigation_agent"),
                "agent_name": getattr(self.mitigation_agent, "agent_name", "Mitigation Recommendation Agent"),
                "status": "UNAVAILABLE",
                "duration_ms": 0.0,
                "summary": f"Mitigation recommendations failed: {str(e)}",
                "data": {"recommendations": []},
                "error": str(e),
            }

        execution_trace.append({
            "agent_id": res_e.get("agent_id", "mitigation_agent"),
            "agent_name": res_e.get("agent_name", "Mitigation Recommendation Agent"),
            "status": res_e.get("status", "UNAVAILABLE"),
            "duration_ms": res_e.get("duration_ms", 0.0),
            "summary": res_e.get("summary", "No summary available."),
            "error": res_e.get("error"),
        })

        total_duration_ms = round((time.time() - start_overall) * 1000, 2)
        has_errors = any(t["status"] in {"UNAVAILABLE", "TIMEOUT", "failed"} for t in execution_trace)
        workflow_status = "completed_with_errors" if has_errors else "completed"

        return {
            "claim": cleaned_claim,
            "mode": "parallel",
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

    def analyze_sequential(
        self,
        claim: str,
        top_k_evidence: int = 5,
        top_k_influence: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes all agents sequentially (A -> B -> C -> D -> E) for single-threaded benchmarking
        and deterministic fallback.
        """
        cleaned_claim = self._validate_claim(claim)
        start_overall = time.time()
        execution_trace: List[Dict[str, Any]] = []

        # Agent A
        try:
            res_a = self.claim_agent.run(cleaned_claim)
        except Exception as e:
            res_a = {
                "agent_id": getattr(self.claim_agent, "agent_id", "claim_analysis_agent"),
                "agent_name": getattr(self.claim_agent, "agent_name", "Claim Analysis Agent"),
                "status": "UNAVAILABLE",
                "duration_ms": 0.0,
                "summary": f"Agent crashed: {str(e)}",
                "data": {},
                "error": str(e),
            }

        execution_trace.append({
            "agent_id": res_a.get("agent_id", "claim_analysis_agent"),
            "agent_name": res_a.get("agent_name", "Claim Analysis Agent"),
            "status": res_a.get("status", "UNAVAILABLE"),
            "duration_ms": res_a.get("duration_ms", 0.0),
            "summary": res_a.get("summary", "No summary available."),
            "error": res_a.get("error"),
        })

        # Agent B
        try:
            res_b = self.rag_agent.run(cleaned_claim, top_k=top_k_evidence)
        except Exception as e:
            res_b = {
                "agent_id": getattr(self.rag_agent, "agent_id", "evidence_rag_agent"),
                "agent_name": getattr(self.rag_agent, "agent_name", "Evidence Research / RAG Agent"),
                "status": "UNAVAILABLE",
                "duration_ms": 0.0,
                "summary": f"Agent crashed: {str(e)}",
                "data": {},
                "error": str(e),
            }

        execution_trace.append({
            "agent_id": res_b.get("agent_id", "evidence_rag_agent"),
            "agent_name": res_b.get("agent_name", "Evidence Research / RAG Agent"),
            "status": res_b.get("status", "UNAVAILABLE"),
            "duration_ms": res_b.get("duration_ms", 0.0),
            "summary": res_b.get("summary", "No summary available."),
            "error": res_b.get("error"),
        })

        # Agent C
        try:
            res_c = self.influence_agent.run(cleaned_claim, top_k=top_k_influence)
        except Exception as e:
            res_c = {
                "agent_id": getattr(self.influence_agent, "agent_id", "historical_influence_agent"),
                "agent_name": getattr(self.influence_agent, "agent_name", "Historical Influence Agent"),
                "status": "UNAVAILABLE",
                "duration_ms": 0.0,
                "summary": f"Agent crashed: {str(e)}",
                "data": {},
                "error": str(e),
            }

        execution_trace.append({
            "agent_id": res_c.get("agent_id", "historical_influence_agent"),
            "agent_name": res_c.get("agent_name", "Historical Influence Agent"),
            "status": res_c.get("status", "UNAVAILABLE"),
            "duration_ms": res_c.get("duration_ms", 0.0),
            "summary": res_c.get("summary", "No summary available."),
            "error": res_c.get("error"),
        })

        # Agent D
        try:
            res_d = self.verification_agent.run(
                claim=cleaned_claim,
                claim_analysis=res_a,
                evidence_rag=res_b,
                influence_analysis=res_c,
            )
        except Exception as e:
            res_d = {
                "agent_id": getattr(self.verification_agent, "agent_id", "verification_risk_agent"),
                "agent_name": getattr(self.verification_agent, "agent_name", "Verification & Risk Assessment Agent"),
                "status": "UNAVAILABLE",
                "duration_ms": 0.0,
                "summary": f"Verification failed: {str(e)}",
                "data": {
                    "composite_risk_score": 50.0,
                    "risk_level": "MODERATE_MISINFORMATION_RISK",
                    "findings": [],
                    "uncertainties": [str(e)],
                },
                "error": str(e),
            }

        execution_trace.append({
            "agent_id": res_d.get("agent_id", "verification_risk_agent"),
            "agent_name": res_d.get("agent_name", "Verification & Risk Assessment Agent"),
            "status": res_d.get("status", "UNAVAILABLE"),
            "duration_ms": res_d.get("duration_ms", 0.0),
            "summary": res_d.get("summary", "No summary available."),
            "error": res_d.get("error"),
        })

        # Agent E
        try:
            res_e = self.mitigation_agent.run(
                claim=cleaned_claim,
                verification_data=res_d,
            )
        except Exception as e:
            res_e = {
                "agent_id": getattr(self.mitigation_agent, "agent_id", "mitigation_agent"),
                "agent_name": getattr(self.mitigation_agent, "agent_name", "Mitigation Recommendation Agent"),
                "status": "UNAVAILABLE",
                "duration_ms": 0.0,
                "summary": f"Mitigation recommendations failed: {str(e)}",
                "data": {"recommendations": []},
                "error": str(e),
            }

        execution_trace.append({
            "agent_id": res_e.get("agent_id", "mitigation_agent"),
            "agent_name": res_e.get("agent_name", "Mitigation Recommendation Agent"),
            "status": res_e.get("status", "UNAVAILABLE"),
            "duration_ms": res_e.get("duration_ms", 0.0),
            "summary": res_e.get("summary", "No summary available."),
            "error": res_e.get("error"),
        })

        total_duration_ms = round((time.time() - start_overall) * 1000, 2)
        has_errors = any(t["status"] in {"UNAVAILABLE", "TIMEOUT", "failed"} for t in execution_trace)
        workflow_status = "completed_with_errors" if has_errors else "completed"

        return {
            "claim": cleaned_claim,
            "mode": "sequential",
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

    def analyze_sync(
        self,
        claim: str,
        top_k_evidence: int = 5,
        top_k_influence: int = 3,
    ) -> Dict[str, Any]:
        """
        Synchronous execution entrypoint routing to parallel or sequential execution based on self.mode.
        """
        if self.mode == "sequential":
            return self.analyze_sequential(claim, top_k_evidence=top_k_evidence, top_k_influence=top_k_influence)
        return self.analyze_parallel(claim, top_k_evidence=top_k_evidence, top_k_influence=top_k_influence)

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
