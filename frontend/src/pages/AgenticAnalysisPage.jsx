import { useState } from "react";
import ClaimInputCard from "../components/ClaimInputCard";
import { API_URL, getLabelInfo } from "../utils/constants";

export default function AgenticAnalysisPage({
  claim,
  setClaim,
  onSaveHistory,
  onNavigate,
}) {
  const [agentData, setAgentData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isTraceExpanded, setIsTraceExpanded] = useState(true);

  const handleRunAgentWorkflow = async (e) => {
    if (e) e.preventDefault();
    const targetClaim = claim.trim();

    if (!targetClaim) {
      setError("Please enter a claim for multi-agent synthesis.");
      return;
    }
    if (targetClaim.length < 3) {
      setError("Claim must contain at least 3 characters.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/agent/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          claim: targetClaim,
          top_k_evidence: 5,
          top_k_influence: 3,
        }),
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(data?.detail || data?.error || `Multi-agent server returned HTTP ${response.status}`);
      }

      setAgentData(data);
      if (onSaveHistory) {
        onSaveHistory({
          claim: targetClaim,
          distilbert: {
            label: data?.claim_analysis?.data?.label,
            confidence: data?.claim_analysis?.data?.confidence,
          },
          baseline: {
            label: "agent-workflow",
            confidence: data?.claim_analysis?.data?.confidence || 0,
          },
        });
      }
    } catch (err) {
      console.error("Multi-agent workflow execution failed:", err);
      setError(
        err?.message?.includes("Failed to fetch")
          ? "Unable to reach the multi-agent orchestrator. Ensure FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const trace = agentData?.execution_trace || [];
  const verification = agentData?.verification_risk?.data;
  const mitigation = agentData?.mitigation_recommendations?.data;
  const recommendations = mitigation?.recommendations || [];

  const riskClassMap = {
    HIGH_MISINFORMATION_RISK: { title: "High Misinformation Risk", class: "risk-high", color: "#b91c1c" },
    DISCREPANCY_BETWEEN_MODEL_AND_EVIDENCE: { title: "Model vs Evidence Discrepancy", class: "risk-discrepancy", color: "#d97706" },
    LOW_MISINFORMATION_RISK: { title: "Low Misinformation Risk", class: "risk-low", color: "#16a34a" },
    UNVERIFIED_INSUFFICIENT_EVIDENCE: { title: "Unverified / Insufficient Evidence", class: "risk-unverified", color: "#64748b" },
    UNVERIFIED_MODEL_ONLY: { title: "Unverified (Model Only, No Evidence)", class: "risk-unverified", color: "#64748b" },
    MODERATE: { title: "Moderate Risk / Contextual Nuance", class: "risk-moderate", color: "#f59e0b" },
  };

  const riskMeta = riskClassMap[verification?.risk_level] || {
    title: verification?.risk_level || "MODERATE RISK",
    class: "risk-moderate",
    color: "#f59e0b",
  };

  const uncertaintyPct = verification?.uncertainty_score !== undefined
    ? Math.round(verification.uncertainty_score * 100)
    : 50;

  return (
    <div className="page-container agentic-page">
      {/* Header */}
      <div className="page-header">
        <div className="header-breadcrumbs">
          <button type="button" className="crumb-link" onClick={() => onNavigate("/")}>
            Dashboard
          </button>
          <span className="crumb-sep">/</span>
          <span className="crumb-current">Multi-Agent Workflow</span>
        </div>
        <h1>Multi-Agent Orchestrator Pipeline</h1>
        <p className="page-description">
          Orchestrates 5 specialized working agents to classify claims, search RAG evidence, analyze social signals, assess uncertainty, and synthesize advisory mitigations.
        </p>
      </div>

      {/* Claim Input Form */}
      <ClaimInputCard
        claim={claim}
        onClaimChange={setClaim}
        onSubmit={handleRunAgentWorkflow}
        loading={loading}
        error={error}
        onClear={() => {
          setClaim("");
          setAgentData(null);
          setError("");
        }}
        buttonLabel="Run Multi-Agent Pipeline"
        placeholder="Enter a political claim to orchestrate all 5 specialized agents across classification, RAG, and risk synthesis..."
      />

      {/* Agent Workflow Execution Output */}
      {agentData && (
        <div className="agent-results-container">
          {/* Workflow Status Strip */}
          <div className="workflow-summary-strip">
            <div className="summary-left">
              <span className="summary-title">Orchestrator Execution Status:</span>
              <span className={`workflow-status-badge badge-${agentData.workflow_status}`}>
                {agentData.workflow_status === "completed" ? "✓ All Agents Completed" : "Completed with Fallbacks"}
              </span>
            </div>
            <div className="summary-right">
              <span className="latency-text">
                Total Execution Latency: <strong>{agentData.total_duration_ms?.toFixed(0)} ms</strong>
              </span>
              <button
                type="button"
                className="btn-toggle-trace"
                onClick={() => setIsTraceExpanded((prev) => !prev)}
              >
                {isTraceExpanded ? "▲ Hide Trace" : "▼ View Execution Trace"}
              </button>
            </div>
          </div>

          {/* 1. Execution Trace Across 5 Agents */}
          {isTraceExpanded && trace.length > 0 && (
            <section className="agent-trace-section">
              <div className="section-subheading">
                <h3>Live Agent Execution Trace (5 Working Agents)</h3>
                <span className="trace-subtext">Step-by-step telemetry from the multi-agent graph orchestrator</span>
              </div>

              <div className="agent-trace-grid">
                {trace.map((agent, idx) => {
                  const isCompleted = agent.status === "completed";
                  return (
                    <div key={agent.agent_id || idx} className={`agent-trace-box ${isCompleted ? "trace-ok" : "trace-err"}`}>
                      <div className="agent-box-header">
                        <div className="agent-identity">
                          <span className="agent-number-chip">Agent {idx + 1}</span>
                          <div>
                            <strong className="agent-title">{agent.agent_name || agent.agent_id}</strong>
                            <span className="agent-duration-text">{agent.duration_ms ? `${agent.duration_ms.toFixed(0)} ms` : ""}</span>
                          </div>
                        </div>
                        <span className={`status-pill ${isCompleted ? "online" : "offline"}`}>
                          {isCompleted ? "✓ Completed" : "⚠ Error / Fallback"}
                        </span>
                      </div>
                      <p className="agent-summary-p">{agent.summary}</p>
                      {agent.error && <span className="agent-err-detail">Error: {agent.error}</span>}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* 2. Agent D: Verification Findings & Risk Uncertainty Assessment */}
          {verification && (
            <section className="verification-assessment-section">
              <div className="section-subheading">
                <h3>Agent D · Verification Findings &amp; Uncertainty Assessment</h3>
                <span className="trace-subtext">Synthesized from classifier softmax, RAG citations, and benchmark tweet signals</span>
              </div>

              <div className="verification-card-box">
                <div className="verification-header-row">
                  <div className="risk-badge-col">
                    <span className="box-eyebrow">SYNTHESIZED RISK LEVEL</span>
                    <div className={`risk-level-banner ${riskMeta.class}`}>
                      <span className="risk-indicator-dot" />
                      <strong>{riskMeta.title}</strong>
                    </div>
                  </div>

                  <div className="uncertainty-gauge-col">
                    <div className="uncertainty-title-flex">
                      <span>Quantified Uncertainty:</span>
                      <strong>{uncertaintyPct}%</strong>
                    </div>
                    <div className="uncertainty-track">
                      <div className="uncertainty-fill" style={{ width: `${uncertaintyPct}%` }} />
                    </div>
                    <span className="uncertainty-explanation">
                      {uncertaintyPct >= 70
                        ? "High Uncertainty: Conflicting evidence or lack of verified PolitiFact precedent."
                        : uncertaintyPct >= 40
                        ? "Moderate Uncertainty: Partial evidence overlap; contextual nuances present."
                        : "Low Uncertainty: Strong alignment observed between classifier and historical record."}
                    </span>
                  </div>
                </div>

                <div className="findings-uncertainties-grid">
                  <div className="panel-box">
                    <h4>📋 Key Verification Findings</h4>
                    {verification.findings && verification.findings.length > 0 ? (
                      <ul className="findings-bullet-list">
                        {verification.findings.map((finding, idx) => (
                          <li key={idx}>✓ {finding}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="empty-subtext">No conclusive empirical findings generated.</p>
                    )}
                  </div>

                  <div className="panel-box">
                    <h4>⚠️ Evidence Limitations &amp; Knowledge Gaps</h4>
                    {verification.uncertainties && verification.uncertainties.length > 0 ? (
                      <ul className="uncertainties-bullet-list">
                        {verification.uncertainties.map((unc, idx) => (
                          <li key={idx}>! {unc}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="empty-subtext">No specific uncertainty flags reported.</p>
                    )}
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* 3. Agent E: Mitigation Recommendations */}
          {mitigation && (
            <section className="mitigation-advisories-section">
              <div className="section-subheading">
                <h3>Agent E · Actionable Mitigation Recommendations</h3>
                <span className="trace-subtext">Responsible next-step advisory actions for verification, editorial review, and platform labeling</span>
              </div>

              {/* Responsible Advisory Guardrail Notice */}
              <div className="mitigation-notice-box">
                <strong>Advisory Guardrail Boundary:</strong>
                <p>
                  All recommendations below are decision-support advisories for human reviewers. The system operates in analytical mode only and does not execute automated censorship, account bans, or live external platform actions.
                </p>
              </div>

              {recommendations.length > 0 ? (
                <div className="recommendations-list-grid">
                  {recommendations.map((rec, idx) => (
                    <article key={idx} className="recommendation-item-card">
                      <div className="rec-card-top">
                        <span className={`priority-chip priority-${rec.priority?.toLowerCase()}`}>
                          {rec.priority?.toUpperCase()} PRIORITY
                        </span>
                        <span className="target-chip">🎯 Target: {rec.target || "Reader / System"}</span>
                      </div>
                      <h4 className="rec-title">{rec.action}</h4>
                      <div className="rec-section-block">
                        <span className="rec-field-heading">RATIONALE:</span>
                        <p>{rec.rationale}</p>
                      </div>
                      <div className="rec-section-block">
                        <span className="rec-field-heading">RECOMMENDED IMPLEMENTATION:</span>
                        <p>{rec.details}</p>
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <div className="empty-state-box">
                  <p>No immediate mitigation actions required. Statement risk profile is within standard informational bounds.</p>
                </div>
              )}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
