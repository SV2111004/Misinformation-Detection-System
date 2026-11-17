import { useState } from "react";
import ClaimInputCard from "../components/ClaimInputCard";
import { API_URL, getLabelInfo, formatSpeaker, formatAffiliation } from "../utils/constants";

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
  const [activeTab, setActiveTab] = useState("synthesis");

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
  const claimAnalysis = agentData?.claim_analysis?.data;
  const evidenceRAG = agentData?.evidence_rag?.data;
  const historicalInfluence = agentData?.historical_influence?.data;
  const verification = agentData?.verification_risk?.data;
  const mitigation = agentData?.mitigation_recommendations?.data;
  const recommendations = mitigation?.recommendations || [];
  const citations = evidenceRAG?.citations || [];
  const matchedStories = historicalInfluence?.matched_stories || [];

  const riskClassMap = {
    HIGH_MISINFORMATION_RISK: { title: "High Misinformation Risk", class: "risk-high", color: "#b91c1c" },
    DISCREPANCY_BETWEEN_MODEL_AND_EVIDENCE: { title: "Model vs Evidence Discrepancy", class: "risk-discrepancy", color: "#d97706" },
    LOW_MISINFORMATION_RISK: { title: "Low Misinformation Risk", class: "risk-low", color: "#16a34a" },
    UNVERIFIED_INSUFFICIENT_EVIDENCE: { title: "Unverified / Insufficient Evidence", class: "risk-unverified", color: "#64748b" },
    UNVERIFIED_MODEL_ONLY: { title: "Unverified (Model Only, No Evidence)", class: "risk-unverified", color: "#64748b" },
    MODERATE: { title: "Moderate Risk / Contextual Nuance", class: "risk-moderate", color: "#f59e0b" },
    MODERATE_MISINFORMATION_RISK: { title: "Moderate Risk / Contextual Omission", class: "risk-moderate", color: "#d97706" },
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

          {/* Agent Module View Selector */}
          <div className="agent-tabs-header-strip">
            <span className="toggle-strip-label">Inspect Agent Outputs:</span>
            <div className="toggle-pill-group">
              <button
                type="button"
                className={`toggle-pill-btn ${activeTab === "synthesis" ? "active" : ""}`}
                onClick={() => setActiveTab("synthesis")}
              >
                🛡️ Epistemic Synthesis &amp; Mitigations (Agents D &amp; E)
              </button>
              <button
                type="button"
                className={`toggle-pill-btn ${activeTab === "evidence" ? "active" : ""}`}
                onClick={() => setActiveTab("evidence")}
              >
                📚 Semantic Evidence RAG (Agent B) {citations.length > 0 && `(${citations.length})`}
              </button>
              <button
                type="button"
                className={`toggle-pill-btn ${activeTab === "decomposition" ? "active" : ""}`}
                onClick={() => setActiveTab("decomposition")}
              >
                🤖 Classifier &amp; Decomposition (Agent A)
              </button>
              <button
                type="button"
                className={`toggle-pill-btn ${activeTab === "influence" ? "active" : ""}`}
                onClick={() => setActiveTab("influence")}
              >
                📊 Social Propagation Signals (Agent C)
              </button>
              <button
                type="button"
                className={`toggle-pill-btn ${activeTab === "all" ? "active" : ""}`}
                onClick={() => setActiveTab("all")}
              >
                🔍 View All Agents
              </button>
            </div>
          </div>

          {/* SECTION: Agent A - Classification & Policy Decomposition */}
          {(activeTab === "decomposition" || activeTab === "all") && claimAnalysis && (
            <section className="agent-section-card agent-decomposition-section">
              <div className="section-subheading">
                <h3>Agent A · Claim Classification &amp; Policy Decomposition</h3>
                <span className="trace-subtext">3-Class DistilBERT sequence evaluation and structural policy entity extraction</span>
              </div>

              <div className="agent-summary-strip-row">
                <div>
                  Primary Prediction: <strong>{claimAnalysis.label || "N/A"}</strong> (Confidence: {((claimAnalysis.confidence || 0) * 100).toFixed(1)}%)
                </div>
                <div className="stat-pills-row">
                  <span className={`stat-pill-sm ${claimAnalysis.risk_tier === "HIGH" ? "pill-amber" : "pill-green"}`}>
                    Risk Tier: {claimAnalysis.risk_tier || "MODERATE"}
                  </span>
                </div>
              </div>

              {/* 3-Class Probabilities */}
              {claimAnalysis.probabilities && (
                <div className="probabilities-breakdown-card" style={{ marginBottom: "16px" }}>
                  <h4>3-Class Tertiary Probability Distribution</h4>
                  <div className="probability-bars-grid">
                    {[
                      { key: "Misleading", label: "Misleading (Substantively False / Fabricated)", level: 2 },
                      { key: "Half-True", label: "Half-True (Partially Accurate / Context Omitted)", level: 4 },
                      { key: "Factual", label: "Factual (Substantially Accurate & Corroborated)", level: 6 },
                    ].map((cls) => {
                      const prob = claimAnalysis.probabilities[cls.key] !== undefined
                        ? claimAnalysis.probabilities[cls.key]
                        : claimAnalysis.probabilities[cls.key.toLowerCase()] || 0;
                      const pct = (prob * 100).toFixed(1);
                      const isWinner = String(claimAnalysis.label).toLowerCase() === cls.key.toLowerCase();
                      return (
                        <div key={cls.key} className={`probability-bar-row ${isWinner ? "row-winner" : ""}`}>
                          <span className="prob-label">{cls.label} {isWinner && "★"}</span>
                          <div className="prob-track">
                            <div
                              className={`prob-fill level-fill-${cls.level}`}
                              style={{ width: `${Math.max(2, prob * 100)}%` }}
                            />
                          </div>
                          <span className="prob-percent">{pct}%</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Decomposed Policy Entities */}
              {claimAnalysis.decomposed_elements && (
                <div>
                  <h4 style={{ fontSize: "13px", fontWeight: "700", marginBottom: "8px", color: "var(--text-secondary)" }}>
                    Decomposed Policy Elements &amp; Verifiable Quantifiers:
                  </h4>
                  <div className="decomposed-entities-grid">
                    <div className="entity-chip-group">
                      <h5>💵 Monetary Amounts</h5>
                      <div className="entity-chips-flex">
                        {claimAnalysis.decomposed_elements.monetary_amounts?.length > 0 ? (
                          claimAnalysis.decomposed_elements.monetary_amounts.map((m, i) => (
                            <span key={i} className="entity-pill pill-money">{m}</span>
                          ))
                        ) : (
                          <span className="empty-subtext">None extracted</span>
                        )}
                      </div>
                    </div>

                    <div className="entity-chip-group">
                      <h5>📊 Percentages &amp; Rates</h5>
                      <div className="entity-chips-flex">
                        {claimAnalysis.decomposed_elements.percentages?.length > 0 ? (
                          claimAnalysis.decomposed_elements.percentages.map((p, i) => (
                            <span key={i} className="entity-pill pill-pct">{p}</span>
                          ))
                        ) : (
                          <span className="empty-subtext">None extracted</span>
                        )}
                      </div>
                    </div>

                    <div className="entity-chip-group">
                      <h5>🏛️ Policy Agencies &amp; Programs</h5>
                      <div className="entity-chips-flex">
                        {claimAnalysis.decomposed_elements.policy_agencies?.length > 0 ? (
                          claimAnalysis.decomposed_elements.policy_agencies.map((a, i) => (
                            <span key={i} className="entity-pill pill-agency">{a}</span>
                          ))
                        ) : (
                          <span className="empty-subtext">None extracted</span>
                        )}
                      </div>
                    </div>

                    <div className="entity-chip-group">
                      <h5>⚖️ Comparative / Fiscal Terms</h5>
                      <div className="entity-chips-flex">
                        {claimAnalysis.decomposed_elements.comparative_terms?.length > 0 ? (
                          claimAnalysis.decomposed_elements.comparative_terms.map((t, i) => (
                            <span key={i} className="entity-pill pill-term">{t}</span>
                          ))
                        ) : (
                          <span className="empty-subtext">None extracted</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* SECTION: Agent B - Semantic Evidence RAG Citations */}
          {(activeTab === "evidence" || activeTab === "all") && evidenceRAG && (
            <section className="agent-section-card agent-evidence-section">
              <div className="section-subheading">
                <h3>Agent B · Semantic Evidence RAG (ChromaDB Vector Store)</h3>
                <span className="trace-subtext">
                  Historical precedent rulings retrieved exclusively via dense semantic embeddings (all-MiniLM-L6-v2)
                </span>
              </div>

              <div className="agent-summary-strip-row">
                <div>
                  Retrieved <strong>{citations.length} Precedent Rulings</strong> from 5,562 canonical training claims
                </div>
                <div className="stat-pills-row">
                  <span className="stat-pill-sm pill-green">
                    Retrieval Mode: Semantic Only
                  </span>
                  {evidenceRAG.missing_justification_count === 0 ? (
                    <span className="stat-pill-sm pill-green">✓ 100% Verified Justifications</span>
                  ) : (
                    <span className="stat-pill-sm pill-amber">
                      ⚠️ {evidenceRAG.missing_justification_count} Missing Justifications
                    </span>
                  )}
                  {evidenceRAG.has_conflicts && (
                    <span className="stat-pill-sm pill-amber">⚠️ Divergent Historical Verdicts</span>
                  )}
                </div>
              </div>

              {citations.length === 0 ? (
                <div className="empty-results-box">
                  <p>No historical statements in the PolitiFact corpus met the semantic similarity threshold for this claim.</p>
                </div>
              ) : (
                <div className="evidence-cards-list">
                  {citations.map((item, idx) => {
                    const verdictInfo = getLabelInfo(item.verdict);
                    const speakerName = formatSpeaker(item.speaker);
                    const speakerMeta = [
                      item.speaker_job,
                      formatAffiliation(item.party_affiliation),
                      item.state_info,
                    ].filter(Boolean).join(" · ");

                    const similarityPct = item.similarity_score !== undefined
                      ? (item.similarity_score * 100).toFixed(1)
                      : null;

                    return (
                      <article key={item.statement_id || idx} className="evidence-card-item">
                        <div className="card-top-bar">
                          <div className="card-tags-group">
                            <span className="dataset-source-tag">
                              LIAR-PLUS · Train Split
                            </span>
                            {item.has_justification ? (
                              <span className="context-badge badge-aligned">
                                ✓ Justification Available
                              </span>
                            ) : (
                              <span className="context-badge badge-anomaly">
                                ⚠️ No Justification Text
                              </span>
                            )}
                          </div>

                          {similarityPct && (
                            <div className="similarity-badge">
                              <span className="sim-label">Semantic Similarity</span>
                              <strong>{similarityPct}%</strong>
                            </div>
                          )}
                        </div>

                        {/* Historical Verdict Banner */}
                        <div className="historical-verdict-row">
                          <span className="historical-caption">POLITIFACT HISTORICAL VERDICT:</span>
                          <span className={`veracity-chip chip-${item.verdict}`}>
                            {verdictInfo.title}
                          </span>
                          <span className="verdict-note-text">
                            *Historical ruling in verified training corpus.*
                          </span>
                        </div>

                        {/* Speaker Header */}
                        <div className="speaker-header-row">
                          <div className="speaker-avatar-circle">
                            {speakerName ? speakerName.charAt(0).toUpperCase() : "P"}
                          </div>
                          <div>
                            <strong className="speaker-name">{speakerName}</strong>
                            <span className="speaker-subtitle">{speakerMeta || "Political Public Figure"}</span>
                          </div>
                        </div>

                        {/* Historical Statement */}
                        <div className="claim-quote-box">
                          <span className="quote-label">Historical Statement:</span>
                          <p>&ldquo;{item.claim}&rdquo;</p>
                        </div>

                        {/* Published Justification */}
                        <div className="justification-quote-box">
                          <span className="quote-label">Published PolitiFact Justification:</span>
                          <p>{item.justification || "No justification text available for this record."}</p>
                        </div>

                        {/* Footer Tags */}
                        {item.statement_id && (
                          <div className="card-footer-meta">
                            <span className="statement-id-tag">Statement ID: {item.statement_id}</span>
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>
              )}
            </section>
          )}

          {/* SECTION: Agent C - Historical Social Influence */}
          {(activeTab === "influence" || activeTab === "all") && historicalInfluence && (
            <section className="agent-section-card agent-influence-section">
              <div className="section-subheading">
                <h3>Agent C · Historical Social Influence &amp; Viral Velocity</h3>
                <span className="trace-subtext">
                  FakeNewsNet social volume indicators and 64-bit Twitter Snowflake timestamp kinetics
                </span>
              </div>

              <div className="agent-summary-strip-row">
                <div>
                  Matched <strong>{matchedStories.length} Benchmark Stories</strong> in FakeNewsNet
                </div>
                <div className="stat-pills-row">
                  <span className="stat-pill-sm">
                    Engagement Velocity: <strong>{historicalInfluence.engagement_velocity?.toFixed(1) || 0} tweets/hr</strong>
                  </span>
                  <span className="stat-pill-sm pill-amber">
                    Velocity Percentile: <strong>{historicalInfluence.velocity_percentile?.toFixed(0) || 0}%</strong>
                  </span>
                  <span className="stat-pill-sm">
                    Max First 24h Tweets: <strong>{historicalInfluence.max_first_24h_tweets || 0}</strong>
                  </span>
                </div>
              </div>

              {matchedStories.length === 0 ? (
                <div className="empty-results-box">
                  <p>No historical FakeNewsNet stories showed headline similarity to this claim.</p>
                </div>
              ) : (
                <div className="matched-stories-list">
                  {matchedStories.map((story, idx) => (
                    <div key={idx} className="story-mini-card">
                      <div className="story-mini-header">
                        <h4 className="story-mini-title">{story.title}</h4>
                        <span className={`status-pill ${story.label === "fake" ? "offline" : "online"}`}>
                          {story.label === "fake" ? "Fake News Record" : "Real News Record"}
                        </span>
                      </div>
                      <div className="story-mini-stats">
                        <span>Total Tweets: <strong>{story.total_tweets?.toLocaleString() || 0}</strong></span>
                        <span>First 24h: <strong>{story.first_24h_tweets?.toLocaleString() || 0}</strong></span>
                        <span>Velocity: <strong>{story.engagement_velocity?.toFixed(1) || 0} tw/hr</strong></span>
                        <span>Headline Similarity: <strong>{((story.similarity_score || 0) * 100).toFixed(1)}%</strong></span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* SECTION: Agent D - Verification Findings & Risk Uncertainty Assessment */}
          {(activeTab === "synthesis" || activeTab === "all") && verification && (
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

          {/* SECTION: Agent E - Mitigation Recommendations */}
          {(activeTab === "synthesis" || activeTab === "all") && mitigation && (
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
