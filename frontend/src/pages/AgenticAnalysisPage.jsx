import { useState } from "react";
import ClaimInputCard from "../components/ClaimInputCard";
import { API_URL, getLabelInfo, formatSpeaker, formatAffiliation, formatDateTime } from "../utils/constants";

// Helper to strip accidental duplicate quotes from claims
function cleanClaimText(text) {
  if (!text) return "";
  let clean = String(text).trim();
  while (
    (clean.startsWith('"') && clean.endsWith('"')) ||
    (clean.startsWith("'") && clean.endsWith("'")) ||
    (clean.startsWith("“") && clean.endsWith("”")) ||
    (clean.startsWith("‘") && clean.endsWith("’"))
  ) {
    clean = clean.slice(1, -1).trim();
  }
  return clean;
}

// Helper to extract a short, readable takeaway from fact-checker justification
function getShortJustification(text) {
  if (!text) return "No published fact-checker justification available for this record.";
  const trimmed = text.trim();
  const firstPeriod = trimmed.indexOf(". ");
  if (firstPeriod > 25 && firstPeriod < 160) {
    return trimmed.slice(0, firstPeriod + 1);
  }
  if (trimmed.length > 140) {
    return trimmed.slice(0, 137) + "...";
  }
  return trimmed;
}

// Helpers for Sharing Recommendations (Agent E)
const getSharingRecClass = (rec) => {
  const norm = String(rec || "").toLowerCase();
  if (norm.includes("not share")) return "danger";
  if (norm.includes("share with context")) return "success";
  return "warning";
};

const getSharingRecIcon = (rec) => {
  const norm = String(rec || "").toLowerCase();
  if (norm.includes("not share")) return "🚫";
  if (norm.includes("share with context")) return "✅";
  return "⏳";
};

// Helper for Stance metadata and plain English explanation
const getStanceMeta = (stance) => {
  switch (stance) {
    case "SUPPORTED_BY_EVIDENCE":
      return {
        title: "Supported by Evidence",
        className: "stance-pill-supported",
        icon: "✓",
        color: "#15803d",
        bg: "#f0fdf4",
        border: "#bbf7d0",
        desc: "Historical fact-checking records corroborate the factual claims in this statement.",
      };
    case "CONTRADICTED_BY_EVIDENCE":
      return {
        title: "Contradicted by Evidence",
        className: "stance-pill-contradicted",
        icon: "✗",
        color: "#b91c1c",
        bg: "#fef2f2",
        border: "#fecaca",
        desc: "Historical fact-checking records refute or contradict the assertion made in this statement.",
      };
    case "DISPUTED_STANCE":
      return {
        title: "Disputed / Mixed Evidence",
        className: "stance-pill-disputed",
        icon: "⚠️",
        color: "#b45309",
        bg: "#fffbeb",
        border: "#fde68a",
        desc: "Retrieved historical fact-checks contain conflicting verdicts or divergent findings.",
      };
    case "PARTIALLY_ALIGNED":
      return {
        title: "Partially Aligned Context",
        className: "stance-pill-partial",
        icon: "ℹ️",
        color: "#0369a1",
        bg: "#f0f9ff",
        border: "#bae6fd",
        desc: "Historical records partially match the topic, but specific figures or qualifiers differ.",
      };
    case "NO_RELEVANT_EVIDENCE":
    default:
      return {
        title: "Insufficient Evidence",
        className: "stance-pill-insufficient",
        icon: "⚪",
        color: "#475569",
        bg: "#f8fafc",
        border: "#e2e8f0",
        desc: "No sufficiently relevant historical fact-checking records were found in the dataset archive.",
      };
  }
};

// Helper for Final Verification Result badge and colors
const getFinalVerdictBadge = (finalVerdict) => {
  const norm = String(finalVerdict || "").toLowerCase().trim();
  if (norm.includes("supported by evidence")) {
    return {
      title: "Supported by Evidence",
      color: "#15803d",
      bg: "#f0fdf4",
      border: "#bbf7d0",
      badgeBg: "#dcfce7",
      badgeColor: "#166534",
      badgeBorder: "#86efac",
      icon: "✓",
      desc: "Corroborated by verified historical fact-checks and official records.",
    };
  }
  if (norm.includes("contradicted by evidence")) {
    return {
      title: "Contradicted by Evidence",
      color: "#b91c1c",
      bg: "#fef2f2",
      border: "#fecaca",
      badgeBg: "#fee2e2",
      badgeColor: "#991b1b",
      badgeBorder: "#fca5a5",
      icon: "✗",
      desc: "Refuted or contradicted by empirical historical fact-checking records.",
    };
  }
  if (norm.includes("mixed evidence") || norm.includes("disputed")) {
    return {
      title: "Mixed Evidence",
      color: "#b45309",
      bg: "#fffbeb",
      border: "#fde68a",
      badgeBg: "#fef3c7",
      badgeColor: "#92400e",
      badgeBorder: "#fcd34d",
      icon: "⚠️",
      desc: "Historical records show conflicting verdicts or partially accurate figures with omitted context.",
    };
  }
  if (norm.includes("insufficient evidence")) {
    return {
      title: "Insufficient Evidence",
      color: "#475569",
      bg: "#f8fafc",
      border: "#e2e8f0",
      badgeBg: "#f1f5f9",
      badgeColor: "#334155",
      badgeBorder: "#cbd5e1",
      icon: "⚪",
      desc: "No sufficiently relevant historical fact-checks were found in our records to make a definitive determination.",
    };
  }
  return {
    title: "Needs Further Verification",
    color: "#c2410c",
    bg: "#fff7ed",
    border: "#ffedd5",
    badgeBg: "#ffedd5",
    badgeColor: "#9a3412",
    badgeBorder: "#fdba74",
    icon: "🔍",
    desc: "Classifier prediction lacks empirical grounding in our fact-checking archive. Independent primary verification is advised.",
  };
};

export default function AgenticAnalysisPage({
  claim,
  setClaim,
  onSaveHistory,
  onNavigate,
}) {
  const [agentData, setAgentData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [collapsedCitations, setCollapsedCitations] = useState({});
  const [showAllCitations, setShowAllCitations] = useState(false);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  const handleRunAgentWorkflow = async (e) => {
    if (e) e.preventDefault();
    const targetClaim = claim.trim();

    if (!targetClaim) {
      setError("Please enter a claim to analyze.");
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
      setShowAllCitations(false);
      setShowTechnicalDetails(false);

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
          ? "Unable to reach the multi-agent orchestrator. Ensure FastAPI backend is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  // Safe Extraction of Agent Outputs
  const claimAnalysis = agentData?.claim_analysis?.data;
  const evidenceRAG = agentData?.evidence_rag?.data;
  const historicalInfluence = agentData?.historical_influence?.data;
  const verification = agentData?.verification_risk?.data;
  const mitigation = agentData?.mitigation_recommendations?.data;

  const citations = evidenceRAG?.citations || [];
  const matchedStories = historicalInfluence?.matched_stories || [];
  const recommendations = mitigation?.recommendations || [];

  // Citation Expand/Collapse Controls
  const toggleCitation = (id) => {
    setCollapsedCitations((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const expandAllCitations = () => {
    setCollapsedCitations({});
  };

  const collapseAllCitations = () => {
    const allClosed = {};
    citations.forEach((c, idx) => {
      allClosed[c.statement_id || idx] = true;
    });
    setCollapsedCitations(allClosed);
  };

  // Safe Metric Derivations
  const stanceKey = verification?.epistemic_stance || "NO_RELEVANT_EVIDENCE";
  const stanceMeta = getStanceMeta(stanceKey);

  const totalRetrieved = citations.length || evidenceRAG?.retrieved_count || 0;
  const relevantCount = verification?.relevant_citations_count !== undefined
    ? verification.relevant_citations_count
    : citations.length;

  const hasConflicts = Boolean(evidenceRAG?.has_conflicts || verification?.evidence_conflict);
  const isNoEvidence = stanceKey === "NO_RELEVANT_EVIDENCE" || totalRetrieved === 0 || relevantCount === 0;

  // Cross-Analysis fields from Agent D
  const aVsB = verification?.a_vs_b;
  const threeAgentSynth = verification?.three_agent_synthesis;
  const agentContributions = threeAgentSynth?.agent_contributions;
  const decisionBreakdown = verification?.decision_breakdown;

  // Derive Final Verdict Label
  const finalVerdictText =
    decisionBreakdown?.final_verdict ||
    verification?.final_verdict ||
    (stanceKey === "SUPPORTED_BY_EVIDENCE"
      ? "Supported by Evidence"
      : stanceKey === "CONTRADICTED_BY_EVIDENCE"
      ? "Contradicted by Evidence"
      : stanceKey === "DISPUTED_STANCE"
      ? "Mixed Evidence"
      : "Needs Further Verification");

  const finalVerdictBadge = getFinalVerdictBadge(finalVerdictText);

  // Confidence for Agent A
  const confidencePercent = Math.round((Number(claimAnalysis?.confidence) || 0.5) * 1000) / 10;
  const normalizedLabel = String(claimAnalysis?.label || "Half-True").toLowerCase();

  return (
    <div className="page-container agentic-page">
      {/* Page Header */}
      <div className="page-header">
        <div className="header-breadcrumbs">
          <button type="button" className="crumb-link" onClick={() => onNavigate("/")}>
            Dashboard
          </button>
          <span className="crumb-sep">/</span>
          <span className="crumb-current">Multi-Agent Workflow</span>
        </div>
        <h1>Multi-Agent Workflow</h1>
        <p className="page-description">
          Cross-verify political claims through sequential agent reasoning: classification, empirical evidence retrieval, and social context.
        </p>
      </div>

      {/* Claim Input Card */}
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
        buttonLabel="Analyze Claim"
        placeholder="Enter a political statement (e.g., 'The federal corporate tax rate was lowered to 21 percent')..."
      />

      {/* Results Flow in Strict Sequential Order */}
      {agentData && (
        <div className="agent-results-container">
          {/* =========================================================
              1. SUBMITTED STATEMENT (TOP)
          ========================================================= */}
          <section className="agent-result-card claim-summary-card">
            <span className="card-kicker">Submitted Claim</span>
            <p className="clean-statement-text">&ldquo;{cleanClaimText(agentData.claim)}&rdquo;</p>
            {agentData.timestamp && (
              <span className="statement-timestamp">Analyzed on {formatDateTime(agentData.timestamp)}</span>
            )}
          </section>

          {/* =========================================================
              2. AGENT A: VERACITY ASSESSMENT (DistilBERT Classification)
          ========================================================= */}
          {claimAnalysis && (
            <section className="agent-result-card agent-a-card">
              <div className="agent-card-header">
                <div className="agent-header-titles">
                  <span className="agent-role-tag">Agent A</span>
                  <div>
                    <h3 className="agent-title">Veracity Assessment</h3>
                    <span className="agent-sub-title">DistilBERT Claim Classification</span>
                  </div>
                </div>
              </div>

              <div className="agent-card-body">
                <div className="veracity-assessment-hero">
                  <div className="predicted-label-container">
                    <span className="metric-label">Predicted Classification</span>
                    <div className="prominent-classification-display">
                      <span className={`veracity-chip-large chip-large-${normalizedLabel}`}>
                        {claimAnalysis.label || "Half-True"}
                      </span>
                    </div>
                  </div>

                  <div className="confidence-meter-container">
                    <div className="confidence-label-row">
                      <span className="confidence-label-text">Classification Confidence</span>
                      <strong className="confidence-value-text">{confidencePercent}%</strong>
                    </div>
                    <div className="confidence-progress-track">
                      <div
                        className={`confidence-progress-fill fill-${normalizedLabel}`}
                        style={{ width: `${Math.min(100, Math.max(8, confidencePercent))}%` }}
                      />
                    </div>
                    <span className="confidence-sub-caption">
                      Model confidence indicates certainty in the predicted label across canonical training claims, not proof of absolute truth.
                    </span>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* =========================================================
              3. AGENT B: HISTORICAL FACT-CHECKING EVIDENCE
          ========================================================= */}
          {evidenceRAG && (
            <section className="agent-result-card agent-b-card">
              <div className="agent-card-header">
                <div className="agent-header-titles">
                  <span className="agent-role-tag">Agent B</span>
                  <div>
                    <h3 className="agent-title">Historical Fact-Checking Evidence</h3>
                    <span className="agent-sub-title">ChromaDB Semantic Evidence Retrieval</span>
                  </div>
                </div>
                <span className="evidence-count-badge">
                  {relevantCount} Relevant Record{relevantCount === 1 ? "" : "s"}
                </span>
              </div>

              <div className="agent-card-body">
                {/* Evidence Stance Banner */}
                <div className={`evidence-summary-banner banner-stance-${stanceKey.toLowerCase()}`}>
                  {isNoEvidence ? (
                    <p>
                      <strong>Insufficient Evidence:</strong> No sufficiently relevant historical fact-checks were found in our archive for this specific claim.
                    </p>
                  ) : hasConflicts ? (
                    <p>
                      Retrieved {relevantCount} historical fact-check{relevantCount === 1 ? "" : "s"}.{" "}
                      <strong>The evidence conflicts:</strong> past fact-checkers reached differing verdicts on related statements.
                    </p>
                  ) : (
                    <p>
                      Retrieved {relevantCount} relevant historical fact-check{relevantCount === 1 ? "" : "s"}.{" "}
                      The records collectively <strong>{stanceKey === "SUPPORTED_BY_EVIDENCE" ? "support" : (stanceKey === "CONTRADICTED_BY_EVIDENCE" ? "contradict" : "partially align with")}</strong> this claim.
                    </p>
                  )}
                </div>

                {/* Historical Evidence Records (Top 2 Initially, Expandable to All) */}
                {citations.length > 0 && (
                  <div className="evidence-citations-wrapper">
                    <div className="citations-controls-bar">
                      <span className="citations-count-label">
                        Showing {showAllCitations ? citations.length : Math.min(2, citations.length)} of {citations.length} Historical Records
                      </span>
                      <div className="citations-toggle-links">
                        <button type="button" className="btn-link" onClick={expandAllCitations}>Expand All</button>
                        <span className="divider">•</span>
                        <button type="button" className="btn-link" onClick={collapseAllCitations}>Collapse All</button>
                      </div>
                    </div>

                    <div className="evidence-cards-stack">
                      {(showAllCitations ? citations : citations.slice(0, 2)).map((item, idx) => {
                        const idKey = item.statement_id || idx;
                        const isExpanded = !collapsedCitations[idKey];
                        const verdictInfo = getLabelInfo(item.verdict);
                        const speakerName = formatSpeaker(item.speaker);
                        const speakerDetails = [
                          item.speaker_job,
                          formatAffiliation(item.party_affiliation),
                          item.state_info,
                        ]
                          .filter(Boolean)
                          .join(" · ");
                        const shortExplanation = getShortJustification(item.justification);

                        return (
                          <article key={idKey} className="expandable-evidence-card">
                            <header
                              className="evidence-card-trigger"
                              onClick={() => toggleCitation(idKey)}
                              role="button"
                              tabIndex={0}
                              onKeyDown={(e) => {
                                if (e.key === "Enter" || e.key === " ") toggleCitation(idKey);
                              }}
                            >
                              <div className="evidence-trigger-left">
                                <span className={`veracity-chip chip-${item.verdict}`}>
                                  Past Verdict: {verdictInfo.title}
                                </span>
                                <span className="evidence-claim-preview">
                                  &ldquo;{cleanClaimText(item.claim)}&rdquo;
                                </span>
                              </div>
                              <span className="expand-chevron">
                                {isExpanded ? "▲ Hide Details" : "▼ View Justification"}
                              </span>
                            </header>

                            {/* Short takeaway visible when collapsed */}
                            {!isExpanded && (
                              <div className="evidence-short-takeaway">
                                <span className="takeaway-label">Fact-Checker Summary:</span>
                                <p>{shortExplanation}</p>
                              </div>
                            )}

                            {/* Expanded Full Justification & Metadata */}
                            {isExpanded && (
                              <div className="evidence-card-expanded-body">
                                <div className="evidence-field">
                                  <span className="field-label">Original Statement:</span>
                                  <blockquote className="historical-statement-quote">
                                    &ldquo;{cleanClaimText(item.claim)}&rdquo;
                                  </blockquote>
                                </div>

                                <div className="evidence-field">
                                  <span className="field-label">Fact-Checker Finding &amp; Justification:</span>
                                  <p className="justification-full-text">
                                    {item.justification || "No published justification text available for this record."}
                                  </p>
                                </div>

                                <footer className="evidence-meta-footer">
                                  <span><strong>Speaker:</strong> {speakerName} {speakerDetails && `(${speakerDetails})`}</span>
                                  {item.context && <span><strong>Context:</strong> {item.context}</span>}
                                </footer>
                              </div>
                            )}
                          </article>
                        );
                      })}
                    </div>

                    {/* View More / Show Less Toggle for records beyond top 2 */}
                    {citations.length > 2 && (
                      <div className="view-more-container">
                        <button
                          type="button"
                          className="btn-view-more"
                          onClick={() => setShowAllCitations(!showAllCitations)}
                        >
                          {showAllCitations ? "Show Less (Top 2 Only)" : `View More (${citations.length - 2} More Records)`}
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </section>
          )}

          {/* =========================================================
              4. AGENT C: HISTORICAL SOCIAL INFLUENCE
          ========================================================= */}
          <section className="agent-result-card agent-c-card">
            <div className="agent-card-header">
              <div className="agent-header-titles">
                <span className="agent-role-tag">Agent C</span>
                <div>
                  <h3 className="agent-title">Historical Social Influence</h3>
                  <span className="agent-sub-title">FakeNewsNet Benchmark Cascade Matching</span>
                </div>
              </div>
            </div>

            <div className="agent-card-body">
              {matchedStories.length === 0 ? (
                <div className="neutral-notice-box">
                  <span className="notice-icon">ℹ️</span>
                  <div>
                    <strong>Benchmark data unavailable:</strong> No matching viral stories were found in our benchmark archive.
                    <p className="notice-clarification">
                      This means benchmark records are unavailable for this specific statement, not that the claim did not circulate online.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="social-activity-content">
                  <p className="social-summary-p">
                    Matched {matchedStories.length} related news narrative{matchedStories.length === 1 ? "" : "s"} in historical benchmark records:
                  </p>
                  <div className="social-stories-list">
                    {matchedStories.map((story, idx) => (
                      <div key={idx} className="social-story-row">
                        <div className="story-info">
                          <strong className="story-headline">&ldquo;{cleanClaimText(story.title || story.headline)}&rdquo;</strong>
                        </div>
                        <span className={`story-type-badge ${story.label === "fake" ? "badge-false" : "badge-real"}`}>
                          {story.label === "fake" ? "Past False Story" : "Past Verified Story"}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* =========================================================
              SECTION A — AGENT D: FINAL VERIFICATION & RISK ASSESSMENT
          ========================================================= */}
          {verification && (
            <section className="agent-result-card section-a-agent-d-card">
              <div className="agent-card-header">
                <div className="agent-header-titles">
                  <span className="agent-role-tag agent-role-d">Section A · Agent D</span>
                  <div>
                    <h3 className="agent-title">Final Verification &amp; Risk Assessment</h3>
                    <span className="agent-sub-title">Multi-Agent Verification &amp; Decision Summary</span>
                  </div>
                </div>
                <span className={`final-verdict-chip chip-verdict-${(verification.simple_risk_level || "moderate").toLowerCase()}`}>
                  {verification.final_assessment || verification.final_verdict || finalVerdictText}
                </span>
              </div>

              <div className="agent-card-body">
                {/* 4 Core Decision Metrics: Score, Risk Level, Uncertainty, Evidence Strength */}
                <div className="d-metrics-summary-grid">
                  <div className="d-metric-tile">
                    <span className="d-tile-label">Final Risk Score</span>
                    <div className="d-tile-value-row">
                      <strong className="d-tile-number">
                        {verification.final_risk_score ?? verification.composite_risk_score ?? 50}
                      </strong>
                      <span className="d-tile-denom">/ 100</span>
                    </div>
                    <span className={`d-tile-badge badge-${(verification.simple_risk_level || "moderate").toLowerCase()}`}>
                      {verification.simple_risk_level || "Moderate"} Risk
                    </span>
                  </div>

                  <div className="d-metric-tile">
                    <span className="d-tile-label">Risk Level</span>
                    <strong className="d-tile-main-text">{verification.simple_risk_level || "Moderate"}</strong>
                    <span className="d-tile-subtext">Factuality &amp; Misinformation Risk</span>
                  </div>

                  <div className="d-metric-tile">
                    <span className="d-tile-label">Uncertainty Level</span>
                    <strong className="d-tile-main-text">{verification.uncertainty_level || "Moderate"}</strong>
                    <span className="d-tile-subtext">Evidence Coverage &amp; Grounding</span>
                  </div>

                  <div className="d-metric-tile">
                    <span className="d-tile-label">Evidence Strength</span>
                    <strong className="d-tile-main-text">{verification.evidence_strength || "Limited"}</strong>
                    <span className="d-tile-subtext">{relevantCount} Precedent Record{relevantCount === 1 ? "" : "s"}</span>
                  </div>
                </div>

                {/* Final Explanation Box */}
                <div className="d-final-explanation-card">
                  <span className="d-box-title">Final Assessment Explanation</span>
                  <p className="d-final-explanation-text">
                    {verification.final_explanation || verification.summary || "No assessment explanation available."}
                  </p>
                </div>

                {/* Step 2 & 3: Comparisons Grid (A vs B and B vs C) */}
                <div className="d-comparisons-grid">
                  {/* Agent A vs Agent B Comparison */}
                  <div className="d-comparison-box">
                    <div className="d-comp-header">
                      <span className="d-comp-tag">Step 2: Model vs Evidence</span>
                      <span className={`d-comp-status-pill status-${(verification.a_vs_b?.agreement || "info").toLowerCase().replace(/\s+/g, "-")}`}>
                        {verification.a_vs_b?.agreement || "Comparison"}
                      </span>
                    </div>
                    <h4 className="d-comp-title">Agent A (Prediction) vs Agent B (Evidence)</h4>
                    <p className="d-comp-desc">
                      {verification.a_vs_b?.explanation || "Comparison between classifier prediction and empirical evidence."}
                    </p>
                  </div>

                  {/* Agent B vs Agent C Context */}
                  <div className="d-comparison-box">
                    <div className="d-comp-header">
                      <span className="d-comp-tag">Step 3: Historical Social Context</span>
                      <span className="d-comp-status-pill status-context">
                        {verification.b_vs_c?.has_context ? "Context Matched" : "No Match"}
                      </span>
                    </div>
                    <h4 className="d-comp-title">Agent B (Evidence) vs Agent C (Social Spread)</h4>
                    <p className="d-comp-desc">
                      {verification.b_vs_c?.explanation || "Historical social media spread context. (Social activity does not alter factuality risk.)"}
                    </p>
                  </div>
                </div>

                {/* Contribution of All Three Agents */}
                <div className="d-contributions-wrapper">
                  <h4 className="d-contributions-heading">Overall Contribution of All Three Agents</h4>
                  <div className="d-contributions-grid">
                    <div className="d-contrib-card">
                      <div className="d-contrib-top">
                        <span className="d-contrib-chip chip-a">Agent A</span>
                        <strong className="d-contrib-role">Claim Classifier</strong>
                      </div>
                      <p className="d-contrib-text">
                        {verification.agent_contributions?.agent_a?.contribution ||
                          `Classified statement as '${claimAnalysis?.label || "Half-True"}' with ${confidencePercent}% confidence.`}
                      </p>
                    </div>

                    <div className="d-contrib-card">
                      <div className="d-contrib-top">
                        <span className="d-contrib-chip chip-b">Agent B</span>
                        <strong className="d-contrib-role">Fact-Checking Evidence</strong>
                      </div>
                      <p className="d-contrib-text">
                        {verification.agent_contributions?.agent_b?.contribution ||
                          (isNoEvidence
                            ? "No relevant historical fact-checks found in the archive."
                            : `Retrieved ${relevantCount} historical fact-check(s).`)}
                      </p>
                    </div>

                    <div className="d-contrib-card">
                      <div className="d-contrib-top">
                        <span className="d-contrib-chip chip-c">Agent C</span>
                        <strong className="d-contrib-role">Social Exposure Context</strong>
                      </div>
                      <p className="d-contrib-text">
                        {verification.agent_contributions?.agent_c?.contribution ||
                          (matchedStories.length > 0
                            ? `Matched ${matchedStories.length} related historical story/stories in benchmark records.`
                            : "No relevant historical social media data was found in benchmark records.")}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* =========================================================
              SECTION B — AGENT E: MITIGATION & RECOMMENDATIONS
          ========================================================= */}
          {mitigation && (
            <section className="agent-result-card section-b-agent-e-card">
              <div className="agent-card-header">
                <div className="agent-header-titles">
                  <span className="agent-role-tag agent-role-e">Section B · Agent E</span>
                  <div>
                    <h3 className="agent-title">Mitigation &amp; Practical Recommendations</h3>
                    <span className="agent-sub-title">Actionable Guidance for Users and Reviewers</span>
                  </div>
                </div>
              </div>

              <div className="agent-card-body">
                {/* 3. Sharing Recommendation Banner */}
                <div className={`sharing-rec-banner rec-banner-${getSharingRecClass(mitigation.sharing_recommendation)}`}>
                  <div className="sharing-rec-left">
                    <span className="sharing-rec-kicker">Recommended Sharing Decision:</span>
                    <h3 className="sharing-rec-verdict">
                      {mitigation.sharing_recommendation || "Hold for Verification"}
                    </h3>
                  </div>
                  <span className="sharing-rec-icon">
                    {getSharingRecIcon(mitigation.sharing_recommendation)}
                  </span>
                </div>

                <div className="e-recommendations-grid">
                  {/* 1. Immediate Action */}
                  <div className="e-card e-immediate-card">
                    <div className="e-card-header">
                      <span className="e-card-icon">⚡</span>
                      <h4>1. Immediate Action</h4>
                    </div>
                    <p className="e-card-body-text">
                      {mitigation.immediate_action || "Pause and verify this statement before sharing or citing it."}
                    </p>
                  </div>

                  {/* 2. Verification Steps */}
                  <div className="e-card e-steps-card">
                    <div className="e-card-header">
                      <span className="e-card-icon">📋</span>
                      <h4>2. Verification Steps</h4>
                    </div>
                    {Array.isArray(mitigation.verification_steps) && mitigation.verification_steps.length > 0 ? (
                      <ol className="e-steps-list">
                        {mitigation.verification_steps.map((step, idx) => (
                          <li key={idx}>{step}</li>
                        ))}
                      </ol>
                    ) : (
                      <p className="e-card-body-text">
                        Consult official primary agency reports and independent fact-checkers before citing this claim.
                      </p>
                    )}
                  </div>

                  {/* 4. Missing Context */}
                  <div className="e-card e-context-card">
                    <div className="e-card-header">
                      <span className="e-card-icon">🔍</span>
                      <h4>3. Missing Context &amp; Information</h4>
                    </div>
                    <p className="e-card-body-text">
                      {mitigation.missing_context || "No major informational gaps identified in benchmark records."}
                    </p>
                  </div>

                  {/* 5. Reason for Recommendation */}
                  <div className="e-card e-reason-card">
                    <div className="e-card-header">
                      <span className="e-card-icon">💡</span>
                      <h4>4. Reason for Recommendation</h4>
                    </div>
                    <p className="e-card-body-text">
                      {mitigation.reason || "Recommendations are derived directly from Agent D's risk assessment."}
                    </p>
                  </div>
                </div>

                {/* Advisory Guardrail Note */}
                <div className="e-advisory-guardrail">
                  <span className="guardrail-icon">🛡️</span>
                  <p>
                    <strong>Advisory Notice:</strong> Recommendations are non-coercive editorial suggestions to help users make informed decisions. No automated censorship or content removal is executed.
                  </p>
                </div>
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
