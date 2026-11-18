import { useState } from "react";
import ClaimInputCard from "../components/ClaimInputCard";
import { API_URL, getLabelInfo, formatSpeaker, formatAffiliation, formatDateTime } from "../utils/constants";

// Helper for Stance metadata and styling
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
        desc: "Historical fact-checking records corroborate the factual claims made in this statement.",
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
        title: "Disputed / Conflicting Evidence",
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
        desc: "Historical precedent partially matches the claim, but specific numbers, qualifiers, or context differ.",
      };
    case "NO_RELEVANT_EVIDENCE":
    default:
      return {
        title: "Insufficient Empirical Evidence",
        className: "stance-pill-insufficient",
        icon: "⚪",
        color: "#475569",
        bg: "#f8fafc",
        border: "#e2e8f0",
        desc: "No sufficiently relevant historical fact-checking records were found in the dataset archive.",
      };
  }
};

// Helper for Risk Level metadata and styling
const getRiskMeta = (riskLevel, riskScore) => {
  const score = Number(riskScore) || 50.0;
  if (riskLevel === "HIGH_MISINFORMATION_RISK" || score >= 70.0) {
    return {
      title: "High Misinformation Risk",
      category: "High Risk",
      className: "risk-tag-high",
      color: "#b91c1c",
      bg: "#fef2f2",
      border: "#fecaca",
      desc: "Statement conflicts with verified evidence or exhibits strong linguistic markers of misinformation.",
    };
  }
  if (riskLevel === "LOW_MISINFORMATION_RISK" || score < 35.0) {
    return {
      title: "Low Misinformation Risk",
      category: "Low Risk",
      className: "risk-tag-low",
      color: "#15803d",
      bg: "#f0fdf4",
      border: "#bbf7d0",
      desc: "Statement aligns with verified official figures or corroborated factual precedents.",
    };
  }
  return {
    title: "Moderate Misinformation Risk",
    category: "Moderate Risk",
    className: "risk-tag-moderate",
    color: "#d97706",
    bg: "#fffbeb",
    border: "#fde68a",
    desc: "Statement contains nuanced, partially accurate, or contextually contested claims.",
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
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [showRawJson, setShowRawJson] = useState(false);

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
  const trace = agentData?.execution_trace || [];

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
  const riskScore = verification?.composite_risk_score !== undefined
    ? Number(verification.composite_risk_score).toFixed(1)
    : "50.0";
  const riskMeta = getRiskMeta(verification?.risk_level, riskScore);

  const stanceKey = verification?.epistemic_stance || "NO_RELEVANT_EVIDENCE";
  const stanceMeta = getStanceMeta(stanceKey);

  const uncertaintyScore = verification?.uncertainty_score !== undefined
    ? Number(verification.uncertainty_score)
    : 0.50;
  const uncertaintyPct = Math.round(uncertaintyScore * 100);

  const confidencePct = claimAnalysis?.confidence !== undefined
    ? (Number(claimAnalysis.confidence) * 100).toFixed(1)
    : "0.0";

  const totalRetrieved = citations.length || evidenceRAG?.retrieved_count || 0;
  const relevantCount = verification?.relevant_citations_count !== undefined
    ? verification.relevant_citations_count
    : (citations.filter((c) => (c.similarity_score || 0) >= 0.45).length);

  const hasConflicts = Boolean(evidenceRAG?.has_conflicts || verification?.evidence_conflict);
  const isNoEvidence = stanceKey === "NO_RELEVANT_EVIDENCE" || totalRetrieved === 0 || relevantCount === 0;

  // Three-Agent Cross-Analysis fields from Agent D
  const aVsB = verification?.a_vs_b;
  const bVsC = verification?.b_vs_c;
  const threeAgentSynth = verification?.three_agent_synthesis;
  const agentContributions = threeAgentSynth?.agent_contributions;

  // Plain-English Explanation for Agent A
  const getAgentAExplanation = (label) => {
    switch (String(label).toLowerCase()) {
      case "misleading":
        return "DistilBERT classified this claim as Misleading because its wording and patterns match political claims previously determined to be inaccurate, exaggerated, or factually unsubstantiated.";
      case "half-true":
        return "DistilBERT classified this claim as Half-True because it shares characteristics with statements that contain a factual core but omit important context, nuances, or qualifiers.";
      case "factual":
        return "DistilBERT classified this claim as Factual because its phrasing aligns with verified statements corroborated by reliable economic or policy data.";
      default:
        return claimAnalysis?.calibration_note || "Linguistic classification completed based on stylistic patterns in the training corpus.";
    }
  };

  return (
    <div className="page-container agentic-page">
      {/* Page Header */}
      <div className="page-header">
        <div className="header-breadcrumbs">
          <button type="button" className="crumb-link" onClick={() => onNavigate("/")}>
            Dashboard
          </button>
          <span className="crumb-sep">/</span>
          <span className="crumb-current">Claim Verification</span>
        </div>
        <h1>Political Misinformation Detection System</h1>
        <p className="page-description">
          A multi-agent system combining DistilBERT linguistic analysis, LIAR-PLUS historical fact-checking evidence, and FakeNewsNet benchmark data to evaluate political claims.
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

      {/* Verification Results Flow */}
      {agentData && (
        <div className="simple-results-flow">
          {/* Status Bar */}
          <div className="simple-status-strip">
            <div className="status-left">
              <span className="status-indicator-dot online" />
              <span>Analysis Complete &bull; 5 Working Agents Finished</span>
            </div>
            {agentData.total_duration_ms && (
              <span className="status-right-text">
                Response Time: {(agentData.total_duration_ms / 1000).toFixed(2)}s
              </span>
            )}
          </div>

          {/* =========================================================
              SECTION 1: SUBMITTED CLAIM
          ========================================================= */}
          <section className="simple-card claim-hero-card">
            <div className="simple-card-header">
              <span className="card-number-tag">1</span>
              <h2 className="card-title">Submitted Claim</h2>
            </div>
            <div className="claim-quote-content">
              <p className="submitted-claim-quote">&ldquo;{agentData.claim}&rdquo;</p>
              <div className="claim-metadata-row">
                <span className="meta-item">
                  <strong>Domain:</strong> Socio-Economic &amp; Domestic Welfare Policy
                </span>
                {agentData.timestamp && (
                  <span className="meta-item">
                    <strong>Analyzed:</strong> {formatDateTime(agentData.timestamp)}
                  </span>
                )}
              </div>
            </div>
          </section>

          {/* =========================================================
              SECTION 2: AGENT A - CLAIM CLASSIFICATION (DistilBERT)
          ========================================================= */}
          {claimAnalysis && (
            <section className="simple-card agent-a-card">
              <div className="simple-card-header">
                <span className="card-number-tag">2</span>
                <div>
                  <h2 className="card-title">Agent A &ndash; Claim Classification (DistilBERT)</h2>
                  <span className="card-subtitle">Linguistic and stylistic analysis of the statement</span>
                </div>
              </div>

              <div className="simple-card-body">
                <div className="result-highlight-box">
                  <div className="highlight-item">
                    <span className="highlight-label">Predicted Label:</span>
                    <span className={`label-badge badge-${String(claimAnalysis.label).toLowerCase()}`}>
                      {claimAnalysis.label || "Unknown"}
                    </span>
                  </div>
                  <div className="highlight-item">
                    <span className="highlight-label">Model Confidence:</span>
                    <span className="highlight-value">{confidencePct}%</span>
                  </div>
                  <div className="highlight-item">
                    <span className="highlight-label">Risk Category:</span>
                    <span className="highlight-value highlight-muted">{claimAnalysis.risk_tier || "Standard"}</span>
                  </div>
                </div>

                <div className="explanation-paragraph">
                  <strong>What this means:</strong>
                  <p>{getAgentAExplanation(claimAnalysis.label)}</p>
                </div>
              </div>
            </section>
          )}

          {/* =========================================================
              SECTION 3: AGENT B - EVIDENCE ANALYSIS (LIAR-PLUS RAG)
          ========================================================= */}
          {evidenceRAG && (
            <section className="simple-card agent-b-card">
              <div className="simple-card-header">
                <span className="card-number-tag">3</span>
                <div>
                  <h2 className="card-title">Agent B &ndash; Historical Fact-Checking Evidence (LIAR-PLUS)</h2>
                  <span className="card-subtitle">
                    Retrieves published PolitiFact records and journalist justifications using semantic search
                  </span>
                </div>
              </div>

              <div className="simple-card-body">
                {/* 3A: What does the evidence tell us? */}
                <div className="evidence-summary-container">
                  <h3 className="subheading-title">What does the evidence tell us?</h3>

                  <div className="retrieval-stats-banner">
                    <div className="retrieval-stat">
                      <span className="retrieval-stat-num">{totalRetrieved}</span>
                      <span className="retrieval-stat-desc">Historical Records Retrieved</span>
                    </div>
                    <div className="retrieval-stat">
                      <span className="retrieval-stat-num">{relevantCount}</span>
                      <span className="retrieval-stat-desc">Verified Relevant Precedents</span>
                    </div>
                    <div className="retrieval-stat">
                      <span className="retrieval-stat-num">
                        {evidenceRAG.missing_justification_count !== undefined
                          ? `${totalRetrieved - evidenceRAG.missing_justification_count}/${totalRetrieved}`
                          : `${totalRetrieved}/${totalRetrieved}`}
                      </span>
                      <span className="retrieval-stat-desc">Records with Journalist Justifications</span>
                    </div>
                  </div>

                  {/* Collective Indication derived from Grounded Summary & Epistemic Stance */}
                  <div className="evidence-collective-verdict">
                    <strong>Collective Evidence Indication:</strong>
                    {isNoEvidence ? (
                      <p className="neutral-warning-text">
                        No sufficiently relevant historical fact-checking records were found in the dataset archive for this specific statement. Agent B cannot confirm or refute this claim based on empirical historical precedents.
                      </p>
                    ) : (
                      <p>
                        {evidenceRAG.grounded_summary || (
                          <>
                            Retrieved <strong>{relevantCount} relevant precedent(s)</strong> from the LIAR-PLUS archive.
                            The collective evidence stance is evaluated as <strong>{stanceMeta.title}</strong>.
                          </>
                        )}
                      </p>
                    )}

                    {/* Limitations & Flags */}
                    {hasConflicts && (
                      <div className="evidence-limitation-pill warning-pill">
                        <span>⚠️</span>
                        <span>
                          <strong>Conflicting Historical Rulings:</strong> Retrieved historical records contain divergent verdicts, reflecting past factual debate or varying context in past statements.
                        </span>
                      </div>
                    )}

                    {evidenceRAG.missing_justification_count > 0 && (
                      <div className="evidence-limitation-pill notice-pill">
                        <span>ℹ️</span>
                        <span>
                          <strong>Note:</strong> {evidenceRAG.missing_justification_count} of {totalRetrieved} retrieved records lack written journalist justifications in the dataset.
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* 3B: Retrieved Historical Claims (Expandable Cards) */}
                <div className="citations-block">
                  <div className="citations-header-bar">
                    <h4 className="citations-section-title">
                      Retrieved Historical Precedents ({citations.length})
                    </h4>
                    {citations.length > 0 && (
                      <div className="citation-btn-group">
                        <button type="button" className="simple-text-btn" onClick={expandAllCitations}>
                          Expand All
                        </button>
                        <span className="btn-sep">|</span>
                        <button type="button" className="simple-text-btn" onClick={collapseAllCitations}>
                          Collapse All
                        </button>
                      </div>
                    )}
                  </div>

                  <div className="important-disclaimer-box">
                    <strong>Important Note on Evidence:</strong>
                    <span>
                      The original PolitiFact verdict displayed on each card evaluates the <em>historical statement</em> made in the past by that speaker, NOT the user&rsquo;s newly submitted claim. High semantic similarity indicates topical relevance, not automatic agreement or disagreement.
                    </span>
                  </div>

                  {citations.length === 0 ? (
                    <div className="empty-message-box">
                      <p>No historical claims in the PolitiFact dataset met the relevance threshold for this query.</p>
                    </div>
                  ) : (
                    <div className="citations-list">
                      {citations.map((item, idx) => {
                        const idKey = item.statement_id || idx;
                        const isExpanded = !collapsedCitations[idKey];
                        const verdictInfo = getLabelInfo(item.verdict);
                        const speakerName = formatSpeaker(item.speaker);
                        const speakerJob = item.speaker_job ? `${item.speaker_job}` : "";
                        const party = formatAffiliation(item.party_affiliation);
                        const state = item.state_info ? `${item.state_info}` : "";
                        const speakerContext = [speakerJob, party, state].filter(Boolean).join(" &bull; ");
                        const similarityPercent = item.similarity_score !== undefined
                          ? (Number(item.similarity_score) * 100).toFixed(1)
                          : null;

                        return (
                          <div key={idKey} className="simple-citation-card">
                            {/* Card Header */}
                            <div
                              className="citation-card-header"
                              onClick={() => toggleCitation(idKey)}
                              role="button"
                              tabIndex={0}
                              onKeyDown={(e) => {
                                if (e.key === "Enter" || e.key === " ") toggleCitation(idKey);
                              }}
                            >
                              <div className="header-left">
                                <span className="citation-idx">#{idx + 1}</span>
                                <span
                                  className="citation-verdict-chip"
                                  style={{
                                    backgroundColor: verdictInfo.bg,
                                    color: verdictInfo.color,
                                    borderColor: verdictInfo.border,
                                  }}
                                >
                                  Past Verdict: {verdictInfo.title}
                                </span>
                                {similarityPercent && (
                                  <span className="citation-sim-chip">
                                    {similarityPercent}% Topical Match
                                  </span>
                                )}
                                {!isExpanded && (
                                  <span className="citation-preview-quote">
                                    &ldquo;{item.claim}&rdquo;
                                  </span>
                                )}
                              </div>
                              <span className="citation-toggle-icon">
                                {isExpanded ? "▲" : "▼"}
                              </span>
                            </div>

                            {/* Card Body (Collapsible) */}
                            {isExpanded && (
                              <div className="citation-card-body">
                                {/* Neutral Relationship Indicator */}
                                <div className="neutral-relationship-tag">
                                  <span>📌</span>
                                  <span>
                                    Semantically related historical precedent; direct relevance to the submitted claim is evaluated in Agent D&rsquo;s combined synthesis.
                                  </span>
                                </div>

                                <div className="citation-field">
                                  <span className="field-title">Original Historical Statement:</span>
                                  <blockquote className="field-quote">
                                    &ldquo;{item.claim}&rdquo;
                                  </blockquote>
                                </div>

                                <div className="citation-field">
                                  <span className="field-title">Original Fact-Checker Justification:</span>
                                  <div className="justification-text-box">
                                    {item.justification ? (
                                      <p>{item.justification}</p>
                                    ) : (
                                      <p className="missing-text-note">
                                        No published journalist justification available for this record in the LIAR-PLUS dataset.
                                      </p>
                                    )}
                                  </div>
                                </div>

                                <div className="citation-meta-footer">
                                  <span><strong>Speaker:</strong> {speakerName} {speakerContext && `(${speakerContext})`}</span>
                                  {item.context && <span><strong>Context:</strong> {item.context}</span>}
                                  <span><strong>Source:</strong> LIAR-PLUS verified record ({item.statement_id || `rec-${idx}`})</span>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}

          {/* =========================================================
              SECTION 4: AGENT C - HISTORICAL SOCIAL MEDIA ANALYSIS
          ========================================================= */}
          <section className="simple-card agent-c-card">
            <div className="simple-card-header">
              <span className="card-number-tag">4</span>
              <div>
                <h2 className="card-title">Agent C &ndash; Historical Social Media Analysis (FakeNewsNet)</h2>
                <span className="card-subtitle">
                  Historical propagation kinetics and viral velocity in benchmark records
                </span>
              </div>
            </div>

            <div className="simple-card-body">
              {/* Disclaimer: Popularity != Factuality */}
              <div className="social-disclaimer-banner">
                <span className="banner-icon">ℹ️</span>
                <div>
                  <strong>Important Distinction:</strong> Virality does not establish factuality. Historical tweet counts and engagement velocity measure past public attention and circulation speed in benchmark records, NOT whether a statement is factually true.
                </div>
              </div>

              {matchedStories.length === 0 ? (
                <div className="empty-social-box">
                  <p>
                    <strong>Historical social media data unavailable:</strong> No matching viral narratives found in the FakeNewsNet benchmark dataset.
                  </p>
                </div>
              ) : (
                <div className="social-data-content">
                  <div className="social-metrics-row">
                    <div className="social-metric-box">
                      <span className="sm-label">Engagement Velocity:</span>
                      <span className="sm-value">
                        {Number(historicalInfluence?.engagement_velocity || 0).toFixed(1)} tweets/hr
                      </span>
                    </div>
                    <div className="social-metric-box">
                      <span className="sm-label">Velocity Percentile:</span>
                      <span className="sm-value">
                        {Math.round(Number(historicalInfluence?.velocity_percentile || 0))}%
                      </span>
                    </div>
                    <div className="social-metric-box">
                      <span className="sm-label">Matched Benchmark Stories:</span>
                      <span className="sm-value">{matchedStories.length}</span>
                    </div>
                  </div>

                  <h4 className="matched-stories-heading">Matched Benchmark Stories in FakeNewsNet:</h4>
                  <div className="matched-stories-grid">
                    {matchedStories.map((story, idx) => (
                      <div key={idx} className="story-card">
                        <div className="story-card-top">
                          <strong className="story-title">{story.title}</strong>
                          <span className={`story-label-pill ${story.label === "fake" ? "label-fake" : "label-real"}`}>
                            {story.label === "fake" ? "Benchmark Fake" : "Benchmark Real"}
                          </span>
                        </div>
                        <div className="story-meta-stats">
                          <span>Observed Tweets: <strong>{Number(story.total_tweets || 0).toLocaleString()}</strong></span>
                          <span>First 24h: <strong>{Number(story.first_24h_tweets || 0).toLocaleString()}</strong></span>
                          <span>Velocity: <strong>{Number(story.engagement_velocity || 0).toFixed(1)} tw/hr</strong></span>
                          <span>Headline Similarity: <strong>{((story.similarity_score || 0) * 100).toFixed(1)}%</strong></span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* =========================================================
              SECTION 5: AGENT D - THREE-AGENT CROSS-VERIFICATION & FINAL CONCLUSION
          ========================================================= */}
          {verification && (
            <section className="simple-card agent-d-card">
              <div className="simple-card-header">
                <span className="card-number-tag final-tag">5</span>
                <div>
                  <h2 className="card-title">Agent D &ndash; Three-Agent Cross-Verification &amp; Final Conclusion</h2>
                  <span className="card-subtitle">
                    Cross-verification combining linguistic analysis (A), historical fact-checking (B), and benchmark social kinetics (C)
                  </span>
                </div>
              </div>

              <div className="simple-card-body">
                {/* 5A: How did the three agents contribute to the final decision? */}
                <div className="agents-contribution-box">
                  <h3 className="subheading-title">How did the three agents contribute to the final decision?</h3>
                  <div className="three-agent-entries-grid">
                    {/* Agent A Entry */}
                    <div className="agent-contrib-card">
                      <div className="contrib-header">
                        <span className="step-badge">Agent A</span>
                        <strong className="contrib-agent-name">Claim Classification (DistilBERT)</strong>
                      </div>
                      <p className="contrib-text">
                        {agentContributions?.agent_a?.contribution || (
                          <>
                            Predicted <strong>{claimAnalysis?.label || "Unknown"}</strong> with {confidencePct}% confidence based on stylistic phrasing and sentence framing.
                          </>
                        )}
                      </p>
                    </div>

                    {/* Agent B Entry */}
                    <div className="agent-contrib-card">
                      <div className="contrib-header">
                        <span className="step-badge">Agent B</span>
                        <strong className="contrib-agent-name">Evidence RAG (LIAR-PLUS)</strong>
                      </div>
                      <p className="contrib-text">
                        {agentContributions?.agent_b?.contribution || (
                          isNoEvidence ? (
                            "Found no relevant historical fact-checks in LIAR-PLUS; could not empirically confirm or dispute Agent A."
                          ) : (
                            <>
                              Retrieved {relevantCount} verified relevant fact-check(s) with an aggregated stance of <strong>{stanceMeta.title}</strong>
                              {aVsB?.status === "DISCREPANCY" ? " (disagreed with Agent A)." : " (agreed with Agent A)."}
                            </>
                          )
                        )}
                      </p>
                    </div>

                    {/* Agent C Entry */}
                    <div className="agent-contrib-card">
                      <div className="contrib-header">
                        <span className="step-badge">Agent C</span>
                        <strong className="contrib-agent-name">Historical Social Influence (FakeNewsNet)</strong>
                      </div>
                      <p className="contrib-text">
                        {agentContributions?.agent_c?.contribution || (
                          matchedStories.length > 0 ? (
                            `Matched ${matchedStories.length} benchmark narrative(s) with ${Number(historicalInfluence?.engagement_velocity || 0).toFixed(1)} tweets/hr velocity, providing historical exposure context without altering factuality risk.`
                          ) : (
                            "No matching viral narratives found in FakeNewsNet; benchmark exposure is baseline and does not indicate historical viral contagion."
                          )
                        )}
                      </p>
                    </div>
                  </div>
                </div>

                {/* 5B: Final Combined Assessment */}
                <div className="final-combined-section">
                  <h3 className="subheading-title">Final Combined Assessment</h3>
                  <div
                    className="final-verdict-hero-box"
                    style={{
                      backgroundColor: riskMeta.bg,
                      borderColor: riskMeta.border,
                    }}
                  >
                    <div className="hero-top-row">
                      <div>
                        <span className="hero-label">Final Risk Assessment:</span>
                        <h3 className="hero-risk-title" style={{ color: riskMeta.color }}>
                          {riskMeta.title} ({riskScore} / 100)
                        </h3>
                      </div>
                      <div className="hero-badges">
                        <span
                          className="hero-stance-pill"
                          style={{
                            backgroundColor: stanceMeta.bg,
                            color: stanceMeta.color,
                            borderColor: stanceMeta.border,
                          }}
                        >
                          {stanceMeta.icon} {stanceMeta.title}
                        </span>
                        <span className="hero-uncertainty-pill">
                          Uncertainty: {uncertaintyPct}% ({uncertaintyPct >= 65 ? "High" : uncertaintyPct >= 35 ? "Moderate" : "Low"})
                        </span>
                      </div>
                    </div>

                    {/* Short synthesized explanation combining all three outputs */}
                    <p className="hero-explanation-p">
                      {threeAgentSynth?.summary || verification?.summary || stanceMeta.desc}
                    </p>
                  </div>
                </div>

                {/* Cross-Verification Pair Summaries (A vs B and B vs C) */}
                <div className="cross-pairs-grid">
                  {aVsB && (
                    <div className={`cross-pair-box ${aVsB.status === "DISCREPANCY" ? "pair-alert" : aVsB.status === "CONCURRENCE" ? "pair-success" : "pair-neutral"}`}>
                      <strong>A vs B Assessment (Model vs Evidence):</strong>
                      <p>{aVsB.summary}</p>
                    </div>
                  )}
                  {bVsC && (
                    <div className="cross-pair-box pair-neutral">
                      <strong>B vs C Assessment (Evidence vs Historical Exposure Context):</strong>
                      <p>{bVsC.summary}</p>
                    </div>
                  )}
                </div>

                {/* Key Verification Findings from Backend */}
                {verification.findings && verification.findings.length > 0 && (
                  <div className="findings-section">
                    <h3 className="subheading-title">Key Verification Findings:</h3>
                    <ul className="simple-findings-list">
                      {verification.findings.map((finding, idx) => (
                        <li key={idx}>
                          <span className="bullet-icon">&bull;</span>
                          <span>{finding}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Limitations and Knowledge Gaps */}
                {verification.uncertainties && verification.uncertainties.length > 0 && (
                  <div className="uncertainties-section">
                    <h3 className="subheading-title">Evidence Limitations &amp; Knowledge Gaps:</h3>
                    <ul className="simple-uncertainties-list">
                      {verification.uncertainties.map((item, idx) => (
                        <li key={idx}>
                          <span className="uncertainty-icon">⚠️</span>
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* =========================================================
              SECTION 6: AGENT E - ACTIONABLE RECOMMENDATIONS
          ========================================================= */}
          {mitigation && (
            <section className="simple-card agent-e-card">
              <div className="simple-card-header">
                <span className="card-number-tag">6</span>
                <div>
                  <h2 className="card-title">Agent E &ndash; Actionable Recommendations</h2>
                  <span className="card-subtitle">
                    Guidance for readers, journalists, and reviewers based on the assessment
                  </span>
                </div>
              </div>

              <div className="simple-card-body">
                <p className="recommendations-intro">
                  These recommendations provide practical guidance for readers and evaluators when reviewing or sharing this claim.
                </p>

                {recommendations.length === 0 ? (
                  <div className="empty-message-box">
                    <p>No immediate intervention required. The statement is within standard informational bounds.</p>
                  </div>
                ) : (
                  <div className="recommendations-list">
                    {recommendations.map((rec, idx) => (
                      <div key={idx} className="simple-rec-card">
                        <div className="rec-top-row">
                          <span className={`priority-badge priority-${rec.priority?.toLowerCase()}`}>
                            {rec.priority?.toUpperCase()} PRIORITY
                          </span>
                          <span className="rec-target-badge">
                            Target: {rec.target || "Reader / Fact-Checker"}
                          </span>
                        </div>
                        <h4 className="rec-action-title">{rec.action}</h4>
                        <div className="rec-detail-row">
                          <strong>Why:</strong>
                          <p>{rec.rationale}</p>
                        </div>
                        <div className="rec-detail-row">
                          <strong>Suggested Action:</strong>
                          <p>{rec.details}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </section>
          )}

          {/* =========================================================
              SECTION 7: COLLAPSED ADVANCED TECHNICAL DETAILS
          ========================================================= */}
          <section className="technical-details-container">
            <button
              type="button"
              className="technical-toggle-btn"
              onClick={() => setShowTechnicalDetails((prev) => !prev)}
            >
              <span>{showTechnicalDetails ? "▲ Hide Advanced Technical Details" : "▼ View Advanced Technical Details (For Examiners & Technical Review)"}</span>
              <span className="toggle-hint">Execution trace, probability distributions, entity decomposition &amp; JSON</span>
            </button>

            {showTechnicalDetails && (
              <div className="technical-content-panel">
                {/* 1. Agent Execution Trace */}
                <div className="tech-subsection">
                  <h3 className="tech-heading">1. Multi-Agent Execution Telemetry (5 Agents)</h3>
                  <div className="trace-table-container">
                    <table className="trace-table">
                      <thead>
                        <tr>
                          <th>Agent ID</th>
                          <th>Agent Name</th>
                          <th>Status</th>
                          <th>Latency (ms)</th>
                          <th>Summary</th>
                        </tr>
                      </thead>
                      <tbody>
                        {trace.map((t, idx) => (
                          <tr key={t.agent_id || idx}>
                            <td><strong>{t.agent_id}</strong></td>
                            <td>{t.agent_name || t.agent_id}</td>
                            <td>
                              <span className={`status-pill ${t.status === "completed" ? "online" : "offline"}`}>
                                {t.status === "completed" ? "✓ Completed" : "⚠ Error / Fallback"}
                              </span>
                            </td>
                            <td>{t.duration_ms ? `${t.duration_ms.toFixed(0)} ms` : "N/A"}</td>
                            <td>{t.summary}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* 2. Technical Cross-Verification Metrics */}
                {verification && (
                  <div className="tech-subsection">
                    <h3 className="tech-heading">2. Cross-Verification Mathematical Metrics</h3>
                    <div className="tech-two-col">
                      <div className="tech-box">
                        <h4>A vs B Model-Evidence Divergence</h4>
                        <p style={{ margin: "4px 0", fontSize: "12px", color: "var(--text-secondary)" }}>
                          Jensen-Shannon Divergence <em>D</em><sub>JS</sub>(p<sub>A</sub> &parallel; p<sub>B</sub>):{" "}
                          <strong>{verification.conflict_score !== undefined ? Number(verification.conflict_score).toFixed(4) : "0.0000"}</strong>
                        </p>
                        <p style={{ margin: "4px 0", fontSize: "11px", color: "var(--text-muted)" }}>
                          Threshold: &ge; 0.35 indicates statistically significant discrepancy between linguistic classifier prior and empirical fact-checks.
                        </p>
                      </div>

                      <div className="tech-box">
                        <h4>Historical Social Exposure Kinetics</h4>
                        <p style={{ margin: "4px 0", fontSize: "12px", color: "var(--text-secondary)" }}>
                          Exposure Tier: <strong>{verification.historical_exposure_tier || "BASELINE_OR_ISOLATED"}</strong>
                        </p>
                        <p style={{ margin: "4px 0", fontSize: "11px", color: "var(--text-muted)" }}>
                          Virality Decoupling: Social kinetics reflect past FakeNewsNet benchmark spread and are strictly decoupled from factuality risk calculation.
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* 3. Probability Distributions */}
                <div className="tech-subsection">
                  <h3 className="tech-heading">3. Probability Distributions</h3>
                  <div className="tech-two-col">
                    {/* Agent A Probabilities */}
                    {claimAnalysis?.probabilities && (
                      <div className="tech-box">
                        <h4>Agent A (DistilBERT Prior)</h4>
                        <div className="prob-rows">
                          {Object.entries(claimAnalysis.probabilities).map(([k, v]) => (
                            <div key={k} className="prob-metric-row">
                              <span className="prob-k">{k}:</span>
                              <div className="prob-track-mini">
                                <div className="prob-fill-mini" style={{ width: `${Math.max(3, Number(v) * 100)}%` }} />
                              </div>
                              <span className="prob-v">{(Number(v) * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Agent D Synthesized Probabilities */}
                    {verification?.synthesized_probabilities && (
                      <div className="tech-box">
                        <h4>Agent D (Fused Synthesis)</h4>
                        <div className="prob-rows">
                          {Object.entries(verification.synthesized_probabilities).map(([k, v]) => (
                            <div key={k} className="prob-metric-row">
                              <span className="prob-k">{k}:</span>
                              <div className="prob-track-mini">
                                <div className="prob-fill-mini" style={{ width: `${Math.max(3, Number(v) * 100)}%` }} />
                              </div>
                              <span className="prob-v">{(Number(v) * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* 4. Extracted Policy Elements */}
                {claimAnalysis?.decomposed_elements && (
                  <div className="tech-subsection">
                    <h3 className="tech-heading">4. Extracted Policy Elements (Agent A Decomposition)</h3>
                    <div className="entity-tags-grid">
                      <div className="entity-group">
                        <strong>Monetary Figures:</strong>
                        <div className="tags-flex">
                          {claimAnalysis.decomposed_elements.monetary_amounts?.length > 0 ? (
                            claimAnalysis.decomposed_elements.monetary_amounts.map((m, i) => (
                              <span key={i} className="tech-pill">{m}</span>
                            ))
                          ) : (
                            <span className="tech-empty">None detected</span>
                          )}
                        </div>
                      </div>

                      <div className="entity-group">
                        <strong>Percentages &amp; Rates:</strong>
                        <div className="tags-flex">
                          {claimAnalysis.decomposed_elements.percentages?.length > 0 ? (
                            claimAnalysis.decomposed_elements.percentages.map((p, i) => (
                              <span key={i} className="tech-pill">{p}</span>
                            ))
                          ) : (
                            <span className="tech-empty">None detected</span>
                          )}
                        </div>
                      </div>

                      <div className="entity-group">
                        <strong>Agencies &amp; Programs:</strong>
                        <div className="tags-flex">
                          {claimAnalysis.decomposed_elements.policy_agencies?.length > 0 ? (
                            claimAnalysis.decomposed_elements.policy_agencies.map((a, i) => (
                              <span key={i} className="tech-pill">{a}</span>
                            ))
                          ) : (
                            <span className="tech-empty">None detected</span>
                          )}
                        </div>
                      </div>

                      <div className="entity-group">
                        <strong>Comparative Terms:</strong>
                        <div className="tags-flex">
                          {claimAnalysis.decomposed_elements.comparative_terms?.length > 0 ? (
                            claimAnalysis.decomposed_elements.comparative_terms.map((t, i) => (
                              <span key={i} className="tech-pill">{t}</span>
                            ))
                          ) : (
                            <span className="tech-empty">None detected</span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 5. Raw JSON Inspection */}
                <div className="tech-subsection">
                  <div className="json-header-row">
                    <h3 className="tech-heading">5. Full API Response Payload (JSON)</h3>
                    <button
                      type="button"
                      className="simple-text-btn"
                      onClick={() => setShowRawJson((prev) => !prev)}
                    >
                      {showRawJson ? "Hide Raw JSON" : "Show Raw JSON"}
                    </button>
                  </div>
                  {showRawJson && (
                    <pre className="raw-json-block">
                      {JSON.stringify(agentData, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
