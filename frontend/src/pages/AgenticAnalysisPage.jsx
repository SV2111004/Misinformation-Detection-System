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
              5. AGENT D: CROSS-VERIFICATION & FINAL ASSESSMENT
          ========================================================= */}
          {verification && (
            <section className="agent-result-card conclusion-card agent-d-card">
              <div className="agent-card-header">
                <div className="agent-header-titles">
                  <span className="agent-role-tag">Agent D</span>
                  <div>
                    <h3 className="agent-title">Cross-Verification &amp; Final Assessment</h3>
                    <span className="agent-sub-title">Triangulated Multi-Agent Reasoning</span>
                  </div>
                </div>
              </div>

              <div className="agent-card-body">
                <p className="comparison-intro">
                  Agent D genuinely compares the outputs of Agents A, B, and C to identify corroboration, conflicts, or data gaps:
                </p>

                {/* 3 Comparison Columns */}
                <div className="findings-comparison-grid">
                  <div className="comparison-column">
                    <span className="col-agent-label">Agent A: Veracity Assessment</span>
                    <strong className="col-agent-verdict">{claimAnalysis?.label || "Analyzed"}</strong>
                    <p>
                      {decisionBreakdown?.agent_a_role?.explanation ||
                        `DistilBERT classified claim as '${claimAnalysis?.label || "Half-True"}' with ${confidencePercent}% confidence.`}
                    </p>
                  </div>

                  <div className="comparison-column">
                    <span className="col-agent-label">Agent B: Historical Evidence</span>
                    <strong className="col-agent-verdict">{stanceMeta.title}</strong>
                    <p>
                      {decisionBreakdown?.agent_b_role?.explanation ||
                        (isNoEvidence
                          ? "No relevant historical fact-checks were available to corroborate or dispute."
                          : `Retrieved ${relevantCount} historical fact-check(s).`)}
                    </p>
                  </div>

                  <div className="comparison-column">
                    <span className="col-agent-label">Agent C: Social Influence</span>
                    <strong className="col-agent-verdict">
                      {matchedStories.length > 0 ? `${matchedStories.length} Matched Stories` : "Unavailable"}
                    </strong>
                    <p>
                      {decisionBreakdown?.agent_c_role?.explanation ||
                        (matchedStories.length > 0
                          ? `Identified ${matchedStories.length} matched benchmark narrative(s) in historical records.`
                          : "Benchmark data unavailable: No matching viral stories in FakeNewsNet.")}
                    </p>
                  </div>
                </div>

                {/* Agreement / Discrepancy Alert */}
                {decisionBreakdown?.agreement_analysis ? (
                  <div className={`comparison-alert-box ${aVsB?.status === "DISCREPANCY" ? "alert-discrepancy" : "alert-concurrence"}`}>
                    <strong>Multi-Agent Agreement Analysis:</strong> {decisionBreakdown.agreement_analysis}
                  </div>
                ) : aVsB?.status === "DISCREPANCY" ? (
                  <div className="comparison-alert-box alert-discrepancy">
                    <strong>Comparison Note:</strong> The classifier prediction and historical fact-checking evidence disagree. Verified historical evidence takes precedence over statistical classification patterns.
                  </div>
                ) : null}
              </div>
            </section>
          )}

          {/* =========================================================
              6. AGENT E: ADVISORY RECOMMENDATIONS
          ========================================================= */}
          {recommendations && recommendations.length > 0 && (
            <section className="agent-result-card next-steps-card agent-e-card">
              <div className="agent-card-header">
                <div className="agent-header-titles">
                  <span className="agent-role-tag">Agent E</span>
                  <div>
                    <h3 className="agent-title">Advisory Recommendations</h3>
                    <span className="agent-sub-title">Actionable Verification Guidance</span>
                  </div>
                </div>
              </div>

              <div className="agent-card-body">
                <p className="next-steps-intro">
                  Recommended verification actions for readers and reviewers before sharing or relying on this claim:
                </p>

                <div className="recommendations-stack">
                  {recommendations.map((rec, idx) => {
                    const hasRationale = Boolean(rec.rationale && String(rec.rationale).trim());
                    const hasDetails = Boolean(rec.details && String(rec.details).trim());

                    return (
                      <div key={idx} className="clean-rec-card">
                        <div className="rec-header">
                          {rec.priority && (
                            <span className={`rec-priority-pill priority-${String(rec.priority).toLowerCase()}`}>
                              {String(rec.priority).toUpperCase()} PRIORITY
                            </span>
                          )}
                          {rec.target && (
                            <span className="rec-target-label">For {rec.target}</span>
                          )}
                        </div>

                        <h4 className="rec-action-heading">{rec.action}</h4>

                        {hasRationale && (
                          <div className="rec-field">
                            <strong>Why:</strong>
                            <p>{rec.rationale}</p>
                          </div>
                        )}

                        {hasDetails && (
                          <div className="rec-field">
                            <strong>Suggested Action:</strong>
                            <p>{rec.details}</p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>
          )}

          {/* =========================================================
              7. FINAL VERIFICATION RESULT (AT THE VERY END)
          ========================================================= */}
          {verification && (
            <section
              className="agent-result-card final-verification-result-card"
              style={{
                borderLeft: `5px solid ${finalVerdictBadge.color}`,
                backgroundColor: finalVerdictBadge.bg,
              }}
            >
              <div className="final-result-top-banner">
                <span
                  className="final-result-pill"
                  style={{
                    backgroundColor: finalVerdictBadge.badgeBg,
                    color: finalVerdictBadge.badgeColor,
                    borderColor: finalVerdictBadge.badgeBorder,
                  }}
                >
                  {finalVerdictBadge.icon} Final Verification Result
                </span>
                <h2 className="final-verdict-heading" style={{ color: finalVerdictBadge.color }}>
                  {finalVerdictText}
                </h2>
              </div>

              <p className="final-verdict-rationale-text">
                {decisionBreakdown?.verdict_rationale ||
                  threeAgentSynth?.summary ||
                  verification?.summary ||
                  finalVerdictBadge.desc}
              </p>

              {/* 3 Structured Details Cards */}
              <div className="final-breakdown-trio-grid">
                <div className="trio-card">
                  <span className="trio-kicker">Evidence Status</span>
                  <strong className="trio-title">{decisionBreakdown?.evidence_status || stanceMeta.title}</strong>
                  <p className="trio-text">
                    {stanceMeta.desc}
                  </p>
                </div>

                <div className="trio-card">
                  <span className="trio-kicker">Missing Information</span>
                  <strong className="trio-title">Unarchived Elements</strong>
                  <ul className="trio-list">
                    {decisionBreakdown?.missing_information && decisionBreakdown.missing_information.length > 0 ? (
                      decisionBreakdown.missing_information.map((item, i) => <li key={i}>{item}</li>)
                    ) : (
                      <li>No major informational gaps identified in benchmark records.</li>
                    )}
                  </ul>
                </div>

                <div className="trio-card">
                  <span className="trio-kicker">What You Should Verify Next</span>
                  <strong className="trio-title">Actionable Checks</strong>
                  <ul className="trio-list">
                    {decisionBreakdown?.what_to_verify_next && decisionBreakdown.what_to_verify_next.length > 0 ? (
                      decisionBreakdown.what_to_verify_next.map((item, i) => <li key={i}>{item}</li>)
                    ) : (
                      <li>Check official primary agency sources (CBO, BLS, IRS).</li>
                    )}
                  </ul>
                </div>
              </div>

              {/* Optional Expandable Technical Details Section */}
              <div className="technical-details-toggle-container">
                <button
                  type="button"
                  className="btn-technical-toggle"
                  onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                >
                  {showTechnicalDetails ? "▲ Hide Technical Details" : "▼ Show Technical Details (Risk Weights & Metrics)"}
                </button>

                {showTechnicalDetails && (
                  <div className="technical-details-expanded-panel">
                    <div className="tech-metrics-grid">
                      <div className="tech-metric-box">
                        <span className="tech-metric-name">Composite Risk Score</span>
                        <strong className="tech-metric-val">{verification.composite_risk_score} / 100</strong>
                        <span className="tech-metric-sub">{verification.risk_level?.replace(/_/g, " ")}</span>
                      </div>

                      <div className="tech-metric-box">
                        <span className="tech-metric-name">Epistemic Uncertainty</span>
                        <strong className="tech-metric-val">{((Number(verification.uncertainty_score) || 0) * 100).toFixed(0)}%</strong>
                        <span className="tech-metric-sub">Distance from uniform + scarcity</span>
                      </div>

                      <div className="tech-metric-box">
                        <span className="tech-metric-name">Evidence Mass &amp; Weight</span>
                        <strong className="tech-metric-val">
                          {decisionBreakdown?.technical_details?.evidence_weight_pct ?? 65}% Evidence
                        </strong>
                        <span className="tech-metric-sub">
                          Classifier Prior: {decisionBreakdown?.technical_details?.classifier_weight_pct ?? 35}%
                        </span>
                      </div>

                      <div className="tech-metric-box">
                        <span className="tech-metric-name">Jensen-Shannon Divergence</span>
                        <strong className="tech-metric-val">
                          {verification.conflict_score !== undefined ? verification.conflict_score.toFixed(3) : "0.000"}
                        </strong>
                        <span className="tech-metric-sub">Model vs Evidence distance</span>
                      </div>
                    </div>

                    {verification.synthesized_probabilities && (
                      <div className="tech-probabilities-row">
                        <span className="tech-prob-label">Synthesized Belief Distribution:</span>
                        <div className="tech-prob-pills">
                          <span className="prob-pill pill-misleading">
                            Misleading: {(verification.synthesized_probabilities.Misleading * 100).toFixed(1)}%
                          </span>
                          <span className="prob-pill pill-half-true">
                            Half-True: {(verification.synthesized_probabilities["Half-True"] * 100).toFixed(1)}%
                          </span>
                          <span className="prob-pill pill-factual">
                            Factual: {(verification.synthesized_probabilities.Factual * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )}

                    {agentData.risk_score_breakdown?.explanation && (
                      <p className="tech-explanation-note">
                        {agentData.risk_score_breakdown.explanation}
                      </p>
                    )}
                  </div>
                )}
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
