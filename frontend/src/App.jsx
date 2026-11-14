import { useState, useEffect, useCallback } from "react";
import "./App.css";
import "./sna/sna.css";
import SnaNavCards from "./sna/SnaNavCards";
import SocialNetworkPage from "./sna/SocialNetworkPage";
import PropagationAnalysisPage from "./sna/PropagationAnalysisPage";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const LABEL_INFO = {
  "pants-fire": {
    title: "Pants on Fire",
    description: "LIAR dataset classification (Extremely False)",
    level: 1,
    color: "#b91c1c",
  },
  false: {
    title: "False",
    description: "LIAR dataset classification (False Statement)",
    level: 2,
    color: "#dc2626",
  },
  "barely-true": {
    title: "Barely True",
    description: "LIAR dataset classification (Mostly Untrue)",
    level: 3,
    color: "#d97706",
  },
  "half-true": {
    title: "Half True",
    description: "LIAR dataset classification (Partially Accurate)",
    level: 4,
    color: "#f59e0b",
  },
  "mostly-true": {
    title: "Mostly True",
    description: "LIAR dataset classification (Substantially Accurate)",
    level: 5,
    color: "#2563eb",
  },
  true: {
    title: "True",
    description: "LIAR dataset classification (Factually Accurate)",
    level: 6,
    color: "#16a34a",
  },
};

// Illustrative political domain examples
const EXAMPLES = [
  "The state unemployment rate dropped below 5 percent following new manufacturing investments.",
  "Federal healthcare spending increased by 20 percent under the recent budget bill.",
  "The proposed tax reform will eliminate deductions for middle-income households.",
  "BREAKING: First NFL Team Declares Bankruptcy Over Kneeling Thugs",
  "Clean energy subsidies created thousands of domestic solar and wind manufacturing jobs.",
];

function formatLabel(label) {
  if (!label) return "Unknown";
  if (label === "pants-fire") return "Pants on Fire";
  return label
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatSpeaker(speaker) {
  if (!speaker) return "Not specified";
  return speaker
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatAffiliation(party) {
  if (!party || party.toLowerCase() === "none") return "";
  return party.charAt(0).toUpperCase() + party.slice(1);
}

function formatDateTime(isoString) {
  if (!isoString) return "N/A";
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZoneName: "short",
    });
  } catch {
    return isoString;
  }
}

function getLabelInfo(label) {
  return (
    LABEL_INFO[label] || {
      title: formatLabel(label),
      description: "LIAR six-class classification",
      level: 0,
      color: "#64748b",
    }
  );
}

function ConfidenceBar({ value, type }) {
  const percentage = Math.max(0, Math.min(100, (value || 0) * 100));

  return (
    <div className="confidence-wrapper">
      <div className="confidence-track">
        <div
          className={`confidence-fill ${type}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="confidence-value">{percentage.toFixed(1)}%</span>
    </div>
  );
}

function ModelCard({ model, result, type }) {
  const info = getLabelInfo(result?.label);

  return (
    <article className={`model-card ${type}`}>
      <div className="model-card-top">
        <div>
          <span className="model-eyebrow">MODEL</span>
          <h3>{model}</h3>
        </div>
        <div className="model-badge">
          {type === "baseline" ? "ML" : "AI"}
        </div>
      </div>

      <div className="model-result">
        <span className="result-eyebrow">PREDICTED CLASS</span>
        <h2 className={`label-level-${info.level}`}>{info.title}</h2>
        <p>{info.description}</p>
      </div>

      <div className="confidence-section">
        <div className="confidence-header">
          <span>Model Confidence</span>
          <strong>{((result?.confidence || 0) * 100).toFixed(1)}%</strong>
        </div>
        <ConfidenceBar value={result?.confidence} type={type} />
      </div>
    </article>
  );
}

/* ============================================================
   SECTION 2: PRIMARY ASSESSMENT COMPONENT (DISTILBERT)
============================================================ */

function PrimaryAssessmentSection({ comparison }) {
  const distilbertResult = comparison?.distilbert;
  const distilbertLabel = distilbertResult?.label;
  const distilbertInfo = getLabelInfo(distilbertLabel);
  const confidencePercent = ((distilbertResult?.confidence || 0) * 100).toFixed(1);

  return (
    <section className="primary-assessment-section" id="primary-assessment">
      <div className="section-heading">
        <div>
          <span className="eyebrow">PRIMARY CLASSIFICATION ASSESSMENT</span>
          <h2>Primary Assessment</h2>
          <p className="section-subtitle">
            Fine-tuned DistilBERT transformer classification across the LIAR 6-class political veracity benchmark.
          </p>
        </div>

        <div className="primary-model-badge">
          <span className="model-chip">AI TRANSFORMER</span>
          <span className="architecture-chip">DistilBERT</span>
        </div>
      </div>

      <div className="primary-card">
        <div className="primary-card-header">
          <div className="primary-verdict-display">
            <span className="result-eyebrow">DISTILBERT PREDICTED CLASS</span>
            <div className="verdict-banner-row">
              <span className={`primary-verdict-title level-${distilbertInfo.level}`}>
                {distilbertInfo.title}
              </span>
              <span className="benchmark-tag">LIAR Benchmark Standard</span>
            </div>
            <p className="primary-verdict-explanation">
              The fine-tuned DistilBERT neural model assigned its highest softmax probability to the{" "}
              <strong>{distilbertInfo.title}</strong> class for this statement.
            </p>
          </div>

          <div className="primary-confidence-box">
            <div className="confidence-title-row">
              <span className="confidence-label">Model Confidence</span>
              <strong className="confidence-score-text">{confidencePercent}%</strong>
            </div>

            <div className="primary-confidence-track">
              <div
                className={`primary-confidence-fill level-${distilbertInfo.level}`}
                style={{ width: `${Math.min(100, Math.max(0, confidencePercent))}%` }}
              />
            </div>

            <div className="confidence-explanation-note">
              <span>
                💡 <strong>Statistical Metric:</strong> Model confidence reflects softmax class probability under the fine-tuned DistilBERT weights, not a guarantee or proof of factual truth.
              </span>
            </div>
          </div>
        </div>

        {/* LIAR 6-Class Spectrum specific to DistilBERT */}
        <div className="truth-scale-card primary-scale-card">
          <div className="truth-scale-header">
            <div>
              <span className="eyebrow">BENCHMARK SPECTRUM</span>
              <h4>LIAR Six-Class Veracity Spectrum</h4>
            </div>
            <span className="scale-indicator-label">
              Primary Model Predicted: <strong>{distilbertInfo.title}</strong>
            </span>
          </div>

          <div className="scale">
            {[
              { label: "Pants on Fire", key: "pants-fire", level: 1 },
              { label: "False", key: "false", level: 2 },
              { label: "Barely True", key: "barely-true", level: 3 },
              { label: "Half True", key: "half-true", level: 4 },
              { label: "Mostly True", key: "mostly-true", level: 5 },
              { label: "True", key: "true", level: 6 },
            ].map((item) => {
              const isSelected = distilbertLabel === item.key;
              return (
                <div
                  className={`scale-item ${isSelected ? "primary-active" : ""}`}
                  key={item.key}
                >
                  <div
                    className={`scale-dot ${
                      isSelected ? `level-${item.level}-active active-pulse` : ""
                    }`}
                  />
                  <span className={isSelected ? "scale-text-active" : ""}>
                    {item.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ============================================================
   SECTION: AGENT EXECUTION TRACE COMPONENT
============================================================ */

function AgentExecutionTraceSection({ trace, totalDurationMs, workflowStatus, isOpen, onToggle }) {
  if (!trace || !Array.isArray(trace) || trace.length === 0) return null;

  const agentIcons = {
    claim_analysis_agent: "🤖",
    evidence_rag_agent: "📚",
    historical_influence_agent: "📊",
    verification_risk_agent: "⚖️",
    mitigation_agent: "🛡️",
  };

  const agentLabels = {
    claim_analysis_agent: "Agent A: Claim Analysis",
    evidence_rag_agent: "Agent B: Evidence Research / RAG",
    historical_influence_agent: "Agent C: Historical Influence",
    verification_risk_agent: "Agent D: Verification & Risk",
    mitigation_agent: "Agent E: Mitigation Recommendations",
  };

  const agentTech = {
    claim_analysis_agent: "DistilBERT 6-Class Neural Classifier",
    evidence_rag_agent: "Hybrid RAG (ChromaDB Vector + TF-IDF)",
    historical_influence_agent: "FakeNewsNet + Snowflake Timeline Engine",
    verification_risk_agent: "Cross-Synthesis & Uncertainty Quantification",
    mitigation_agent: "Actionable Advisory Mitigation Engine",
  };

  return (
    <section className="agent-trace-section" id="agent-trace-section">
      <div className="agent-trace-header" onClick={onToggle} role="button" tabIndex={0}>
        <div className="agent-trace-title-group">
          <div className="agent-trace-badge-row">
            <span className="eyebrow">AGENT WORKFLOW</span>
            <span className={`workflow-status-pill ${workflowStatus}`}>
              {workflowStatus === "completed" ? "All Agents Completed" : "Completed with Fallbacks"}
            </span>
          </div>
          <h2>Agent Execution Trace (5 Specialized Working Agents)</h2>
          <p className="section-subtitle">
            Parallel and sequential orchestration of specialized neural, evidentiary, and risk assessment agents.
          </p>
        </div>

        <div className="agent-trace-controls">
          <span className="total-latency-chip" title="Total end-to-end multi-agent execution latency">
            ⚡ Total Latency: <strong>{totalDurationMs ? `${totalDurationMs.toFixed(0)} ms` : "N/A"}</strong>
          </span>
          <button type="button" className="trace-toggle-btn" aria-label="Toggle agent trace">
            {isOpen ? "▲ Collapse Trace" : "▼ Expand Trace"}
          </button>
        </div>
      </div>

      {isOpen && (
        <div className="agent-trace-body">
          <div className="agent-cards-grid">
            {trace.map((agent, idx) => {
              const icon = agentIcons[agent.agent_id] || "⚙️";
              const label = agentLabels[agent.agent_id] || agent.agent_name || `Agent ${idx + 1}`;
              const tech = agentTech[agent.agent_id] || "Specialized Module";
              const isCompleted = agent.status === "completed";

              return (
                <div className={`agent-trace-card status-${agent.status}`} key={agent.agent_id || idx}>
                  <div className="agent-card-header">
                    <div className="agent-title-col">
                      <span className="agent-avatar-icon">{icon}</span>
                      <div>
                        <h4 className="agent-name-text">{label}</h4>
                        <span className="agent-tech-text">{tech}</span>
                      </div>
                    </div>

                    <div className="agent-status-col">
                      <span className={`agent-status-pill ${isCompleted ? "pill-completed" : "pill-failed"}`}>
                        {isCompleted ? "✓ Completed" : "⚠ Error / Fallback"}
                      </span>
                      <span className="agent-duration-chip">
                        {agent.duration_ms !== undefined ? `${agent.duration_ms.toFixed(0)} ms` : ""}
                      </span>
                    </div>
                  </div>

                  <div className="agent-summary-box">
                    <p className="agent-summary-text">{agent.summary}</p>
                    {agent.error && (
                      <p className="agent-error-text">Error Detail: {agent.error}</p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}

/* ============================================================
   SECTION: VERIFICATION & RISK ASSESSMENT COMPONENT (AGENT D)
============================================================ */

function VerificationRiskSection({ verificationData }) {
  if (!verificationData || !verificationData.data) return null;
  const data = verificationData.data;
  const riskLevel = data.risk_level || "MODERATE";
  const uncertainty = data.uncertainty_score !== undefined ? data.uncertainty_score : 0.5;
  const uncertaintyPercent = Math.round(uncertainty * 100);

  const riskClassMap = {
    HIGH_MISINFORMATION_RISK: { title: "High Misinformation Risk", class: "risk-high", color: "#b91c1c" },
    DISCREPANCY_BETWEEN_MODEL_AND_EVIDENCE: { title: "Model vs Evidence Discrepancy", class: "risk-discrepancy", color: "#d97706" },
    LOW_MISINFORMATION_RISK: { title: "Low Misinformation Risk", class: "risk-low", color: "#16a34a" },
    UNVERIFIED_INSUFFICIENT_EVIDENCE: { title: "Unverified / Insufficient Evidence", class: "risk-unverified", color: "#64748b" },
    UNVERIFIED_MODEL_ONLY: { title: "Unverified (Model Only, No Evidence)", class: "risk-unverified", color: "#64748b" },
    MODERATE: { title: "Moderate Risk / Contextual Nuance", class: "risk-moderate", color: "#f59e0b" },
  };

  const riskInfo = riskClassMap[riskLevel] || { title: riskLevel, class: "risk-moderate", color: "#f59e0b" };

  return (
    <section className="verification-risk-section" id="verification-risk">
      <div className="section-heading">
        <div>
          <span className="eyebrow">AGENT D · CROSS-SYNTHESIS</span>
          <h2>Verification Findings &amp; Uncertainty Assessment</h2>
          <p className="section-subtitle">
            Cross-synthesizes classifier predictions, historical PolitiFact evidence, and social volume metrics to quantify uncertainty.
          </p>
        </div>
      </div>

      <div className="verification-dashboard-card">
        <div className="verification-header-banner">
          <div className="risk-level-col">
            <span className="risk-eyebrow">SYNTHESIZED RISK PROFILE</span>
            <div className={`risk-banner-badge ${riskInfo.class}`}>
              <span className="risk-indicator-dot" />
              <span className="risk-banner-title">{riskInfo.title}</span>
            </div>
            <p className="risk-banner-subtext">
              Synthesized from DistilBERT semantic classification, LIAR-PLUS precedent records, and FakeNewsNet signals.
            </p>
          </div>

          <div className="uncertainty-gauge-col">
            <div className="uncertainty-title-row">
              <span className="uncertainty-label">Assessment Uncertainty</span>
              <strong className="uncertainty-value-text">{uncertaintyPercent}%</strong>
            </div>
            <div className="uncertainty-bar-track">
              <div
                className="uncertainty-bar-fill"
                style={{ width: `${uncertaintyPercent}%` }}
              />
            </div>
            <span className="uncertainty-caption">
              {uncertaintyPercent >= 70
                ? "High uncertainty: Significant evidence gaps or conflicting signals require caution."
                : uncertaintyPercent >= 40
                ? "Moderate uncertainty: Some corroboration found; context-dependent factors exist."
                : "Low uncertainty: Strong corroboration observed between classifier and historical benchmark."}
            </span>
          </div>
        </div>

        <div className="verification-content-grid">
          {/* Key Evidentiary Findings */}
          <div className="findings-panel">
            <div className="panel-heading">
              <span className="panel-icon">📋</span>
              <h4>Key Findings</h4>
            </div>
            {data.findings && data.findings.length > 0 ? (
              <ul className="findings-list">
                {data.findings.map((finding, idx) => (
                  <li key={idx} className="finding-item">
                    <span className="finding-bullet">✓</span>
                    <span>{finding}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="panel-empty-text">No conclusive empirical findings generated.</p>
            )}
          </div>

          {/* Uncertainty Factors & Evidence Limitations */}
          <div className="uncertainties-panel">
            <div className="panel-heading">
              <span className="panel-icon">⚠️</span>
              <h4>Evidence Limitations &amp; What Cannot Be Established</h4>
            </div>
            {data.uncertainties && data.uncertainties.length > 0 ? (
              <ul className="uncertainties-list">
                {data.uncertainties.map((unc, idx) => (
                  <li key={idx} className="uncertainty-item">
                    <span className="uncertainty-bullet">!</span>
                    <span>{unc}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="panel-empty-text">No specific uncertainty flags reported.</p>
            )}
          </div>
        </div>

        {/* Dataset Integrity Flags (Row-shift warning) */}
        {data.integrity_flags && data.integrity_flags.length > 0 && (
          <div className="integrity-alert-banner">
            <span className="integrity-icon">⚠️</span>
            <div className="integrity-content">
              <strong>Dataset Integrity &amp; Anomaly Warning:</strong>
              {data.integrity_flags.map((flag, idx) => (
                <p key={idx}>{flag}</p>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

/* ============================================================
   SECTION: MITIGATION RECOMMENDATIONS COMPONENT (AGENT E)
============================================================ */

function MitigationRecommendationsSection({ mitigationData }) {
  if (!mitigationData || !mitigationData.data) return null;
  const data = mitigationData.data;
  const recommendations = data.recommendations || [];

  return (
    <section className="mitigation-section" id="mitigation-recommendations">
      <div className="section-heading">
        <div>
          <span className="eyebrow">AGENT E · PROPAGATION MITIGATION</span>
          <h2>Mitigation Recommendations</h2>
          <p className="section-subtitle">
            Responsible, non-coercive next-step advisories generated by Agent E to inform user verification, editorial review, and platform labeling.
          </p>
        </div>
      </div>

      <div className="mitigation-container">
        {/* Prominent Non-Execution Disclaimer */}
        <div className="mitigation-disclaimer-card">
          <div className="disclaimer-header-row">
            <span className="disclaimer-badge advisory-badge">ADVISORY GUARDRAIL</span>
            <h4>Responsible Advisory Boundary Notice</h4>
          </div>
          <p className="disclaimer-body-text">
            {mitigationData.disclaimer || (
              "All mitigation items below are decision-support recommendations. " +
              "The system provides analytical advisories only and has NOT automatically removed, restricted, " +
              "or intervened in live external social media or publication workflows."
            )}
          </p>
        </div>

        {recommendations.length > 0 ? (
          <div className="recommendations-grid">
            {recommendations.map((rec, idx) => {
              const priorityClass =
                rec.priority?.toLowerCase() === "high"
                  ? "priority-high"
                  : rec.priority?.toLowerCase() === "medium"
                  ? "priority-medium"
                  : "priority-low";

              return (
                <article className="recommendation-card" key={idx}>
                  <div className="rec-top-row">
                    <span className={`priority-tag ${priorityClass}`}>
                      {rec.priority?.toUpperCase()} PRIORITY
                    </span>
                    <span className="target-stakeholder-tag">
                      🎯 {rec.target || "Reader / System"}
                    </span>
                  </div>

                  <h3 className="rec-action-title">{rec.action}</h3>

                  <div className="rec-rationale-box">
                    <span className="rec-field-label">RATIONALE</span>
                    <p className="rec-rationale-text">{rec.rationale}</p>
                  </div>

                  <div className="rec-details-box">
                    <span className="rec-field-label">RECOMMENDED ACTION</span>
                    <p className="rec-details-text">{rec.details}</p>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          <div className="evidence-status-card empty-card">
            <span className="status-icon empty-icon">✓</span>
            <div>
              <strong>No immediate mitigation actions required.</strong>
              <p>Statement risk profile is within standard informational bounds.</p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

/* ============================================================
   SECTION 3: EVIDENCE CARD COMPONENT
============================================================ */

function EvidenceCard({ item }) {
  const verdictInfo = getLabelInfo(item.verdict);
  const similarityScore = item.similarity_score;
  const similarityPercent =
    similarityScore !== undefined && similarityScore !== null
      ? (similarityScore * 100).toFixed(1)
      : null;

  const speakerName = formatSpeaker(item.speaker);
  const speakerDetails = [
    item.speaker_job,
    formatAffiliation(item.party_affiliation),
    item.state_info,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <article className="evidence-card">
      <div className="evidence-card-top">
        <div className="evidence-meta-badges">
          <span className="evidence-split-tag">
            {item.source_dataset || "liar-plus"} · {item.split || "train"}
          </span>

          {item.has_context_alignment ? (
            <span
              className="alignment-badge aligned"
              title="Speaker or claim keywords appear in justification text"
            >
              Context: Aligned
            </span>
          ) : (
            <span
              className="alignment-badge unverified"
              title="Upstream TSV row-shift anomaly: justification keywords do not match speaker/claim"
            >
              Context: Upstream Unverified
            </span>
          )}
        </div>

        {similarityPercent !== null && (
          <div
            className="evidence-score-pill"
            title="Hybrid similarity combining dense semantic vector matching (ChromaDB MiniLM) with TF-IDF lexical overlap."
          >
            <span className="score-label">
              {item.semantic_score !== undefined ? "Hybrid Relevance" : "Textual Similarity"}
            </span>
            <strong>{similarityPercent}%</strong>
            {item.semantic_score !== undefined && (
              <span className="score-subtext">
                Sem: {(item.semantic_score * 100).toFixed(0)}% · Lex: {((item.lexical_score || 0) * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )}
      </div>

      {/* Historical Benchmark Verdict Banner */}
      <div className="historical-verdict-box">
        <span className="historical-verdict-eyebrow">
          HISTORICAL RECORD VERDICT (POLITIFACT ARCHIVE)
        </span>
        <div className="historical-verdict-value">
          <span className={`evidence-verdict-badge level-${verdictInfo.level}`}>
            <span className="verdict-dot" />
            {verdictInfo.title}
          </span>
          <span className="historical-verdict-disclaimer">
            *Original PolitiFact label for the historical claim below — not a verdict on your submitted claim.*
          </span>
        </div>
      </div>

      <div className="evidence-speaker-row">
        <div className="speaker-avatar">
          {speakerName ? speakerName.charAt(0).toUpperCase() : "?"}
        </div>

        <div className="speaker-details">
          <span className="speaker-name">{speakerName}</span>
          <span className="speaker-meta">
            {speakerDetails || "Political public figure"}
          </span>
        </div>
      </div>

      <div className="evidence-claim-block">
        <span className="evidence-block-label">HISTORICAL CLAIM</span>
        <p className="evidence-claim-text">&ldquo;{item.claim}&rdquo;</p>
      </div>

      <div className="evidence-justification-block">
        <span className="evidence-block-label">PUBLISHED JUSTIFICATION</span>
        <p className="evidence-justification-text">
          {item.justification || "No justification text available for this record."}
        </p>

        {!item.has_context_alignment && (
          <div className="alignment-warning-box">
            <span>
              ⚠️ <strong>Upstream Data Anomaly:</strong> This benchmark record exhibits potential row-shift misalignment from the raw LIAR-PLUS dataset. The justification excerpt may correspond to an adjacent benchmark statement.
            </span>
          </div>
        )}
      </div>

      <div className="evidence-card-footer">
        <div className="evidence-tags">
          {item.subject && (
            <span className="evidence-tag subject-tag">
              {item.subject.split(",").slice(0, 3).join(", ")}
            </span>
          )}

          {item.context && (
            <span
              className="evidence-tag context-tag"
              title={item.context}
            >
              {item.context.length > 38
                ? `${item.context.slice(0, 38)}...`
                : item.context}
            </span>
          )}
        </div>

        {item.statement_id && (
          <span className="evidence-id-tag">
            ID: {item.statement_id}
          </span>
        )}
      </div>
    </article>
  );
}

/* ============================================================
   SECTION 3: HISTORICAL EVIDENCE SECTION COMPONENT
============================================================ */

function HistoricalEvidenceSection({
  evidence,
  loading,
  error,
  onRetry,
  evidenceStats,
}) {
  return (
    <section className="evidence-section" id="evidence-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">POLITIFACT / LIAR-PLUS ARCHIVE</span>
          <h2>Retrieved Historical Evidence</h2>
          <p className="section-subtitle">
            Historical claims and published PolitiFact justifications retrieved via TF-IDF textual similarity matching.
          </p>
        </div>

        {evidenceStats && evidenceStats.total_indexed && (
          <div
            className="corpus-badge"
            title="Index status from GET /evidence/stats"
          >
            <span className="corpus-dot" />
            <span>
              Corpus: {evidenceStats.total_indexed.toLocaleString()} training claims ({evidenceStats.vocab_size?.toLocaleString()} features)
            </span>
          </div>
        )}
      </div>

      {/* Prominent Evidence Disclaimer & Provenance Notice */}
      <div className="evidence-disclaimer-box">
        <div className="disclaimer-title-row">
          <span className="disclaimer-badge">DISCLAIMER</span>
          <h4>Evidence Retrieval &amp; Semantic Scope</h4>
        </div>

        <p className="disclaimer-primary-text">
          These results represent historical political claims and their published justifications retrieved from the LIAR-PLUS dataset. High textual similarity does not establish factual correctness. Retrieved justifications have not been independently verified by this system and should not be treated as proof of the submitted claim.
        </p>

        <div className="disclaimer-subnotes">
          <div className="disclaimer-note-item">
            <strong>Relevance Metric:</strong> Textual similarity measures lexical vocabulary overlap (TF-IDF cosine similarity), not the probability that your claim is true.
          </div>

          <div className="disclaimer-note-item">
            <strong>Data Integrity Notice:</strong> Justifications reflect published excerpts from the LIAR-PLUS benchmark. An empirical audit revealed upstream extraction noise and row shifts in portions of the raw training dataset; unaligned records are flagged explicitly.
          </div>
        </div>
      </div>

      {/* State 1: Loading */}
      {loading && (
        <div className="evidence-status-card loading-card">
          <span className="spinner" />
          <div>
            <strong>Searching historical evidence corpus...</strong>
            <p>Scanning 10,000+ indexed political claims and published justifications.</p>
          </div>
        </div>
      )}

      {/* State 2: API Error */}
      {!loading && error && (
        <div className="evidence-status-card error-card">
          <div className="status-icon error-icon">!</div>
          <div className="status-content">
            <strong>Evidence Retrieval Encountered an Error</strong>
            <p>{error}</p>
            {onRetry && (
              <button
                type="button"
                className="evidence-retry-btn"
                onClick={onRetry}
              >
                Retry Evidence Search
              </button>
            )}
          </div>
        </div>
      )}

      {/* State 3: Empty Results */}
      {!loading && !error && evidence && evidence.results?.length === 0 && (
        <div className="evidence-status-card empty-card">
          <div className="status-icon empty-icon">⌕</div>
          <div className="status-content">
            <strong>No sufficiently relevant historical evidence was found in the current dataset.</strong>
            <p>
              The submitted claim did not have sufficient vocabulary overlap with indexed historical claims (threshold &ge; 0.08).
              Try analyzing claims with broader political policy phrasing or related keywords.
            </p>
          </div>
        </div>
      )}

      {/* State 4: Results Display */}
      {!loading && !error && evidence && evidence.results?.length > 0 && (
        <>
          {/* RAG Grounded Evidence Synthesis Banner */}
          {evidence.grounded_summary && (
            <div className={`grounded-summary-banner status-${evidence.evidence_status || "grounded"}`}>
              <div className="grounded-summary-header">
                <div className="grounded-summary-title">
                  <span className="grounded-icon">📚</span>
                  <h4>Grounded Evidence Synthesis (RAG Agent)</h4>
                </div>
                <div className="grounded-badges">
                  {evidence.has_conflicts && (
                    <span className="conflict-badge">⚠ Precedent Divergence</span>
                  )}
                  <span className="generation-mode-chip">
                    {evidence.generation_mode === "llm_grounded" ? "LLM Synthesized" : "Deterministic Fallback"}
                  </span>
                </div>
              </div>
              <p className="grounded-summary-text">{evidence.grounded_summary}</p>
            </div>
          )}

          <div className="evidence-results-meta">
            <span>
              Showing <strong>{evidence.results.length}</strong> top matching historical records for query:
              <em> &ldquo;{evidence.query}&rdquo;</em>
            </span>
            <span className="meta-split-notice">Split: train (evaluation-isolated)</span>
          </div>

          <div className="evidence-grid">
            {evidence.results.map((item, index) => (
              <EvidenceCard
                key={item.statement_id ? `${item.statement_id}-${index}` : index}
                item={item}
              />
            ))}
          </div>
        </>
      )}
    </section>
  );
}

/* ============================================================
   SECTION 4: INFLUENCE CARD COMPONENT (FAKENEWSNET)
============================================================ */

function InfluenceCard({ story }) {
  const isFake = story.dataset_label === "fake";
  const temporal = story.temporal_metrics;

  return (
    <article className="influence-card">
      <div className="influence-card-top">
        <div className="influence-top-badges">
          <span className={`story-label-pill ${isFake ? "label-fake" : "label-real"}`}>
            HISTORICAL BENCHMARK: {story.dataset_label?.toUpperCase()}
          </span>

          {story.domain && (
            <span className="story-domain-tag">
              🌐 {story.domain}
            </span>
          )}
        </div>

        <div className="evidence-score-pill" title="TF-IDF Cosine Similarity with historical headline">
          <span className="score-label">Headline Relevance</span>
          <strong>{(story.similarity_score * 100).toFixed(1)}%</strong>
        </div>
      </div>

      <div className="historical-story-box">
        <span className="historical-verdict-eyebrow">MATCHED HISTORICAL STORY HEADLINE</span>
        <h3 className="story-title">&ldquo;{story.title}&rdquo;</h3>
        <span className="story-label-disclaimer">
          *Original FakeNewsNet ground-truth label for the matched story above — not a rating on your claim.*
        </span>
      </div>

      {/* Influence Metrics Grid */}
      <div className="influence-metrics-grid">
        <div className="influence-metric-box">
          <span className="metric-eyebrow">OBSERVED TWEET VOLUME</span>
          <strong className="metric-number">{story.observed_tweet_count.toLocaleString()}</strong>
          <span className="metric-caption">Captured public tweet IDs citing story</span>
        </div>

        <div className="influence-metric-box">
          <span className="metric-eyebrow">AMONG FAKE STORIES</span>
          <strong className="metric-number">{story.percentile_among_fake}%</strong>
          <span className="metric-caption">Higher volume than {story.percentile_among_fake}% of 432 fake stories (median: 79)</span>
        </div>

        <div className="influence-metric-box">
          <span className="metric-eyebrow">AMONG REAL STORIES</span>
          <strong className="metric-number">{story.percentile_among_real}%</strong>
          <span className="metric-caption">Higher volume than {story.percentile_among_real}% of 624 real stories (median: 8)</span>
        </div>
      </div>

      {/* Temporal Dynamics Section (When valid Snowflake IDs exist) */}
      {temporal && temporal.has_temporal_data ? (
        <div className="temporal-dynamics-card">
          <div className="temporal-header">
            <span className="temporal-title">Decoded Snowflake Temporal Profile</span>
            <span className="temporal-valid-badge">
              {temporal.valid_snowflake_ids.toLocaleString()} of {temporal.total_recorded_ids.toLocaleString()} IDs Decoded
            </span>
          </div>

          <div className="temporal-stats-row">
            <div className="temporal-stat">
              <span>Earliest Recorded Tweet:</span>
              <strong>{formatDateTime(temporal.earliest_recorded_tweet_utc)}</strong>
            </div>

            <div className="temporal-stat">
              <span>Latest Recorded Tweet:</span>
              <strong>{formatDateTime(temporal.latest_recorded_tweet_utc)}</strong>
            </div>

            <div className="temporal-stat">
              <span>Observed Lifespan:</span>
              <strong>{temporal.observed_span_days} days ({temporal.observed_span_hours} hrs)</strong>
            </div>

            <div className="temporal-stat">
              <span>First 24h Share:</span>
              <strong>{temporal.first_24h_percentage}% ({temporal.first_24h_tweets_count} tweets)</strong>
            </div>
          </div>

          <p className="temporal-disclaimer-note">
            *Earliest and latest timestamps among the recorded IDs. Decoded using Twitter Snowflake format (Nov 2010 epoch). Represents an observational sample, not a complete timeline of all online activity.*
          </p>
        </div>
      ) : (
        <div className="temporal-unavailable-box">
          <span>ℹ️ {temporal?.message || "Temporal metrics unavailable (no valid Snowflake IDs found in record)."}</span>
        </div>
      )}
    </article>
  );
}

/* ============================================================
   SECTION 4: HISTORICAL INFLUENCE SECTION COMPONENT (FAKENEWSNET)
============================================================ */

function HistoricalInfluenceSection({
  influence,
  loading,
  error,
  onRetry,
  influenceStats,
}) {
  return (
    <section className="influence-section" id="influence-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">FAKENEWSNET / TWITTER BENCHMARK</span>
          <h2>Historical Influence Indicators — FakeNewsNet Benchmark</h2>
          <p className="section-subtitle">
            Historical public Twitter engagement indicators and tweet volume recorded in the FakeNewsNet-PolitiFact benchmark archive.
          </p>
        </div>

        {influenceStats && influenceStats.total_stories && (
          <div
            className="corpus-badge"
            title="Index status from GET /influence/stats"
          >
            <span className="corpus-dot" />
            <span>
              Benchmark: {influenceStats.total_stories.toLocaleString()} stories ({influenceStats.total_tweet_ids?.toLocaleString()} referenced tweet IDs)
            </span>
          </div>
        )}
      </div>

      {/* Prominent Influence Disclaimer Box */}
      <div className="influence-disclaimer-box">
        <div className="disclaimer-title-row">
          <span className="disclaimer-badge influence-badge-title">BENCHMARK DISCLAIMER</span>
          <h4>Historical Engagement Scope &amp; Measurement Boundaries</h4>
        </div>

        <p className="disclaimer-primary-text">
          These indicators describe historical tweet IDs recorded for matching benchmark stories. They do not measure the reach or influence of the newly submitted claim. Tweet-ID timestamps are decoded only where valid. The dataset does not contain follower counts, impressions, or retweet relationship graphs.
        </p>

        <div className="disclaimer-subnotes">
          <div className="disclaimer-note-item">
            <strong>Volume Metric:</strong> Observed tweet count reflects public tweets captured citing the historical story URL or headline in FakeNewsNet.
          </div>

          <div className="disclaimer-note-item">
            <strong>Temporal Metric:</strong> Timestamps are derived via mathematical decoding of 64-bit Twitter Snowflake IDs; they represent an observational sample rather than an exhaustive social media log.
          </div>
        </div>
      </div>

      {/* State 1: Loading */}
      {loading && (
        <div className="evidence-status-card loading-card">
          <span className="spinner" />
          <div>
            <strong>Analyzing historical social influence...</strong>
            <p>Scanning 1,056 FakeNewsNet political news stories and 583,000+ referenced tweet IDs.</p>
          </div>
        </div>
      )}

      {/* State 2: API Error */}
      {!loading && error && (
        <div className="evidence-status-card error-card">
          <div className="status-icon error-icon">!</div>
          <div className="status-content">
            <strong>Influence Analysis Encountered an Error</strong>
            <p>{error}</p>
            {onRetry && (
              <button
                type="button"
                className="evidence-retry-btn"
                onClick={onRetry}
              >
                Retry Influence Search
              </button>
            )}
          </div>
        </div>
      )}

      {/* State 3: Empty Results */}
      {!loading && !error && influence && influence.results?.length === 0 && (
        <div className="evidence-status-card empty-card">
          <div className="status-icon empty-icon">⌕</div>
          <div className="status-content">
            <strong>No matching historical social propagation record found in the FakeNewsNet benchmark.</strong>
            <p>
              The submitted claim did not have sufficient vocabulary overlap with indexed FakeNewsNet headlines (threshold &ge; 0.08).
              Social sharing metrics and temporal velocity cannot be inferred for unindexed claims.
            </p>
          </div>
        </div>
      )}

      {/* State 4: Matched Results */}
      {!loading && !error && influence && influence.results?.length > 0 && (
        <div className="influence-results-container">
          <div className="evidence-results-meta">
            <span>
              Showing <strong>{influence.results.length}</strong> matching historical story record(s) for query:
              <em> &ldquo;{influence.query}&rdquo;</em>
            </span>
            <span className="meta-split-notice">Dataset: FakeNewsNet-PolitiFact</span>
          </div>

          <div className="influence-cards-grid">
            {influence.results.map((story) => (
              <InfluenceCard key={story.news_id} story={story} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

/* ============================================================
   SECTION 5: EVIDENCE QUALITY AND LIMITATIONS COMPONENT
============================================================ */

function EvidenceLimitationsSection() {
  return (
    <section className="limitations-section" id="limitations-section">
      <div className="limitations-card">
        <div className="limitations-header">
          <div className="limitations-icon">⚠️</div>
          <div>
            <span className="eyebrow">INTEGRITY &amp; METHODOLOGY</span>
            <h3>Evidence Quality and Limitations</h3>
          </div>
        </div>

        <div className="limitations-grid">
          <div className="limitation-item">
            <h4>1. Textual Similarity ≠ Factual Proof</h4>
            <p>
              Retrieval scores measure TF-IDF n-gram and vocabulary overlap between claims. A high similarity score indicates topical resemblance, not that the submitted claim is factually verified or true.
            </p>
          </div>

          <div className="limitation-item">
            <h4>2. Strict Verdict Separation</h4>
            <p>
              The veracity label shown on each historical card is PolitiFact&apos;s original benchmark rating for that specific historical claim. It is never applied to your submitted claim.
            </p>
          </div>

          <div className="limitation-item">
            <h4>3. Upstream Benchmark Row Shifts</h4>
            <p>
              The LIAR-PLUS benchmark was assembled via automated scraping. While validation and test splits maintain high alignment (&gt;95%), portions of the raw training split suffer from documented row shifts. Records without matching keywords are explicitly flagged as <em>Upstream Unverified</em>.
            </p>
          </div>

          <div className="limitation-item">
            <h4>4. Static Corpus vs. Live Verification</h4>
            <p>
              This system searches an indexed archive of 10,238 historical training claims and 1,056 FakeNewsNet stories. It does not perform live web scraping, browse real-time news feeds, or claim autonomous real-world verification.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ============================================================
   SECTION 6: ANALYSIS SUMMARY COMPONENT
============================================================ */

function AnalysisSummarySection({ comparison, evidence, influence }) {
  const distilbertLabel = comparison?.distilbert?.label;
  const baselineLabel = comparison?.baseline?.label;
  const distilbertInfo = getLabelInfo(distilbertLabel);
  const baselineInfo = getLabelInfo(baselineLabel);
  const modelsAgree = distilbertLabel === baselineLabel;

  const resultCount = evidence?.results?.length || 0;
  const alignedCount =
    evidence?.results?.filter((r) => r.has_context_alignment).length || 0;
  const topScore =
    resultCount > 0 ? (evidence.results[0].similarity_score * 100).toFixed(1) : null;

  const influenceMatches = influence?.results?.length || 0;
  const topInfluenceStory = influenceMatches > 0 ? influence.results[0] : null;

  return (
    <section className="analysis-summary-section" id="analysis-summary">
      <div className="summary-header-row">
        <div>
          <span className="eyebrow">SYSTEM SYNTHESIS</span>
          <h2>Analysis Summary</h2>
          <p className="section-subtitle">
            Synthesized system assessment combining transformer classification, archival retrieval context, and social influence indicators.
          </p>
        </div>
      </div>

      <div className="analysis-summary-card">
        <div className="summary-facts-grid four-col-grid">
          {/* Fact 1: Primary Model Assessment */}
          <div className="summary-fact-box primary-fact">
            <span className="fact-label">PRIMARY CLASSIFICATION</span>
            <div className="fact-verdict">
              <span className={`verdict-pill level-${distilbertInfo.level}`}>
                {distilbertInfo.title}
              </span>
            </div>
            <p className="fact-description">
              Fine-tuned DistilBERT predicted <strong>{distilbertInfo.title}</strong> with{" "}
              {((comparison?.distilbert?.confidence || 0) * 100).toFixed(1)}% model confidence across the 6 LIAR benchmark categories.
            </p>
          </div>

          {/* Fact 2: Model Architecture Consensus */}
          <div className="summary-fact-box consensus-fact">
            <span className="fact-label">MODEL CONSENSUS</span>
            <div className="fact-verdict">
              <span className={`consensus-pill ${modelsAgree ? "agree" : "differ"}`}>
                {modelsAgree ? "✓ Models Agree" : "≠ Models Differ"}
              </span>
            </div>
            <p className="fact-description">
              {modelsAgree ? (
                <>
                  Baseline TF-IDF + LR agrees with DistilBERT (both predicted{" "}
                  <strong>{baselineInfo.title}</strong>), confirming statistical convergence across linear and transformer architectures.
                </>
              ) : (
                <>
                  Baseline TF-IDF + LR predicted <strong>{baselineInfo.title}</strong>, while DistilBERT predicted{" "}
                  <strong>{distilbertInfo.title}</strong>. Contextual transformer embeddings captured nuanced linguistic cues differing from the n-gram baseline.
                </>
              )}
            </p>
          </div>

          {/* Fact 3: Historical Evidence Grounding */}
          <div className="summary-fact-box evidence-fact">
            <span className="fact-label">ARCHIVE EVIDENCE CONTEXT</span>
            <div className="fact-verdict">
              <span className="evidence-count-pill">
                {resultCount > 0 ? `${resultCount} Historical Matches` : "0 Matches Found"}
              </span>
            </div>
            <p className="fact-description">
              {resultCount > 0 ? (
                <>
                  Retrieved <strong>{resultCount}</strong> textually related historical claim(s) from PolitiFact (top textual similarity:{" "}
                  <strong>{topScore}%</strong>). {alignedCount} of {resultCount} record(s) have verified context alignment.
                </>
              ) : (
                <>
                  No historical benchmark claims met the minimum textual similarity threshold (0.08) in the LIAR-PLUS archive.
                </>
              )}
            </p>
          </div>

          {/* Fact 4: Historical Social Engagement Context */}
          <div className="summary-fact-box influence-fact">
            <span className="fact-label">SOCIAL INFLUENCE CONTEXT</span>
            <div className="fact-verdict">
              <span className="influence-count-pill">
                {influenceMatches > 0
                  ? `${topInfluenceStory.observed_tweet_count.toLocaleString()} Tweets Recorded`
                  : "0 Matches Found"}
              </span>
            </div>
            <p className="fact-description">
              {influenceMatches > 0 ? (
                <>
                  Matched historical story in FakeNewsNet with{" "}
                  <strong>{topInfluenceStory.observed_tweet_count.toLocaleString()}</strong> recorded tweet IDs (
                  <strong>{topInfluenceStory.percentile_among_fake}%</strong> percentile among fake stories). Observed sharing span:{" "}
                  <strong>{topInfluenceStory.temporal_metrics?.observed_span_days || 0} days</strong>.
                </>
              ) : (
                <>
                  No matching historical news headlines found in the FakeNewsNet benchmark archive (threshold &ge; 0.08).
                </>
              )}
            </p>
          </div>
        </div>

        <div className="summary-grounding-disclaimer">
          <span className="disclaimer-icon">ℹ️</span>
          <p>
            <strong>Grounded Synthesis Notice:</strong> This summary is based strictly on local ML model inference and benchmark archive retrieval. No external web fact-checking or definitive real-world proof is generated.
          </p>
        </div>
      </div>
    </section>
  );
}

/* ============================================================
   SECTION 7: PHASE 1 MODEL COMPARISON (COLLAPSIBLE)
============================================================ */

function Phase1ComparisonSection({ comparison, isOpen, onToggle }) {
  const baselineLabel = comparison?.baseline?.label;
  const distilbertLabel = comparison?.distilbert?.label;
  const modelsAgree = Boolean(
    baselineLabel && distilbertLabel && baselineLabel === distilbertLabel
  );
  const baselineInfo = getLabelInfo(baselineLabel);
  const distilbertInfo = getLabelInfo(distilbertLabel);

  return (
    <section className="collapsible-comparison-section" id="model-comparison">
      <div
        className="collapsible-header-card"
        onClick={onToggle}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onToggle();
          }
        }}
      >
        <div className="collapsible-left">
          <span className="eyebrow">PHASE 1 COMPARATIVE EVALUATION</span>
          <h3>Phase 1 Model Comparison (TF-IDF + LR vs DistilBERT)</h3>
          <p className="collapsible-subtitle">
            Inspect agreement and divergent predictions between traditional n-gram baseline and neural transformer models.
          </p>
        </div>

        <div className="collapsible-right">
          <div className={`agreement-badge ${modelsAgree ? "agree" : "different"}`}>
            <span>{modelsAgree ? "✓" : "!"}</span>
            {modelsAgree ? "Models Agree" : "Models Differ"}
          </div>

          <button
            type="button"
            className="collapsible-toggle-btn"
            onClick={(e) => {
              e.stopPropagation();
              onToggle();
            }}
            aria-expanded={isOpen}
          >
            {isOpen ? "Hide Model Comparison ▲" : "View Model Comparison ▼"}
          </button>
        </div>
      </div>

      {isOpen && (
        <div className="collapsible-content">
          <p className="comparison-intro-note">
            In Phase 1, both models evaluate the claim across the 6-class LIAR benchmark. Compare the traditional TF-IDF + Logistic Regression baseline against the fine-tuned DistilBERT neural model below.
          </p>

          <div className="model-grid">
            <ModelCard
              model="TF-IDF + Logistic Regression"
              result={comparison.baseline}
              type="baseline"
            />
            <ModelCard
              model="DistilBERT"
              result={comparison.distilbert}
              type="transformer"
            />
          </div>

          {/* Model Consensus block */}
          <div
            className={`comparison-summary ${
              modelsAgree ? "summary-agree" : "summary-different"
            }`}
          >
            <div className="summary-content">
              <span className="eyebrow">MODEL CONSENSUS</span>
              {modelsAgree ? (
                <>
                  <h2>{formatLabel(baselineLabel)}</h2>
                  <p>
                    Both the baseline classifier and the transformer model converged on the same 6-class category.
                  </p>
                </>
              ) : (
                <>
                  <h2>Models Disagree</h2>
                  <p>
                    The two models produced different classifications. DistilBERT uses deep self-attention context, whereas TF-IDF relies strictly on token frequencies.
                  </p>
                </>
              )}
            </div>

            <div className="summary-right">
              <div className="summary-model">
                <span>TF-IDF + LR</span>
                <strong>{baselineInfo.title}</strong>
              </div>
              <div className="summary-divider" />
              <div className="summary-model">
                <span>DistilBERT</span>
                <strong>{distilbertInfo.title}</strong>
              </div>
            </div>
          </div>

          {/* Comparative Truthfulness Scale */}
          <div className="truth-scale-card">
            <div className="truth-scale-header">
              <div>
                <span className="eyebrow">LIAR LABEL SCALE</span>
                <h3>Comparative Veracity Positions</h3>
              </div>
              <span>Dual Model Distribution</span>
            </div>

            <div className="scale">
              {[
                { label: "Pants on Fire", key: "pants-fire" },
                { label: "False", key: "false" },
                { label: "Barely True", key: "barely-true" },
                { label: "Half True", key: "half-true" },
                { label: "Mostly True", key: "mostly-true" },
                { label: "True", key: "true" },
              ].map((item) => {
                const isBaseline = baselineLabel === item.key;
                const isDistilbert = distilbertLabel === item.key;

                return (
                  <div className="scale-item" key={item.key}>
                    <div
                      className={`scale-dot ${isBaseline ? "baseline-dot" : ""} ${
                        isDistilbert ? "transformer-dot" : ""
                      } ${isBaseline && isDistilbert ? "both-dot" : ""}`}
                    />
                    <span>{item.label}</span>
                  </div>
                );
              })}
            </div>

            <div className="scale-legend">
              <span>
                <i className="legend-baseline" />
                TF-IDF + LR (Baseline)
              </span>
              <span>
                <i className="legend-transformer" />
                DistilBERT (Transformer)
              </span>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

/* ============================================================
   MAIN APPLICATION
============================================================ */

function App() {
  const [currentPath, setCurrentPath] = useState(() => {
    return window.location.pathname || "/";
  });

  const navigate = (path) => {
    window.history.pushState({}, "", path);
    setCurrentPath(path);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(window.location.pathname || "/");
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const [claim, setClaim] = useState("");
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem("misinformation-history");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Evidence Retrieval State
  const [evidence, setEvidence] = useState(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState("");
  const [evidenceStats, setEvidenceStats] = useState(null);
  const [lastSearchedEvidenceClaim, setLastSearchedEvidenceClaim] = useState("");

  // Influence Analysis State (Phase 3)
  const [influence, setInfluence] = useState(null);
  const [influenceLoading, setInfluenceLoading] = useState(false);
  const [influenceError, setInfluenceError] = useState("");
  const [influenceStats, setInfluenceStats] = useState(null);
  const [lastSearchedInfluenceClaim, setLastSearchedInfluenceClaim] = useState("");

  // Multi-Agent Analysis State (Phase 4)
  const [agentAnalysis, setAgentAnalysis] = useState(null);
  const [isTraceOpen, setIsTraceOpen] = useState(true);

  // Phase 1 Comparison Collapsible State
  const [isComparisonOpen, setIsComparisonOpen] = useState(false);

  /* =========================
     FETCH STATS ON MOUNT
  ========================= */

  useEffect(() => {
    let isMounted = true;

    // Fetch Evidence stats
    fetch(`${API_URL}/evidence/stats`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted && data && data.total_indexed) {
          setEvidenceStats(data);
        }
      })
      .catch((err) => {
        console.warn("Evidence stats could not be loaded:", err);
      });

    // Fetch Influence stats
    fetch(`${API_URL}/influence/stats`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted && data && data.total_stories) {
          setInfluenceStats(data);
        }
      })
      .catch((err) => {
        console.warn("Influence stats could not be loaded:", err);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  /* =========================
     SAVE HISTORY
  ========================= */

  const saveHistory = (data) => {
    if (!data?.baseline?.label || !data?.distilbert?.label) {
      return;
    }

    const newItem = {
      id: Date.now(),
      claim: data.claim,
      baseline: data.baseline?.label,
      distilbert: data.distilbert?.label,
      time: new Date().toLocaleTimeString(),
    };

    setHistory((prevHistory) => {
      const updated = [newItem, ...prevHistory].slice(0, 8);
      try {
        localStorage.setItem(
          "misinformation-history",
          JSON.stringify(updated)
        );
      } catch (err) {
        console.error("Failed to save history to localStorage:", err);
      }
      return updated;
    });
  };

  /* =========================
     FETCH EVIDENCE
  ========================= */

  const fetchEvidence = useCallback(
    async (text) => {
      const targetClaim = (text || claim).trim();
      if (!targetClaim || targetClaim.length < 3) return;

      setEvidenceLoading(true);
      setEvidenceError("");

      try {
        const response = await fetch(`${API_URL}/evidence/search`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            claim: targetClaim,
            top_k: 5,
            min_score: 0.08,
          }),
        });

        const data = await response.json().catch(() => null);

        if (!response.ok) {
          let errorMessage = `Evidence search error (${response.status})`;
          if (data) {
            if (typeof data.detail === "string") {
              errorMessage = data.detail;
            } else if (Array.isArray(data.detail) && data.detail[0]?.msg) {
              errorMessage = data.detail[0].msg;
            } else if (data.error) {
              errorMessage = data.error;
            }
          }
          throw new Error(errorMessage);
        }

        if (!data || !Array.isArray(data.results)) {
          throw new Error("Invalid response format received from evidence retrieval server.");
        }

        setEvidence(data);
        setLastSearchedEvidenceClaim(targetClaim);
      } catch (err) {
        console.error("Evidence retrieval failed:", err);
        const message =
          err?.message && !err.message.includes("Failed to fetch")
            ? err.message
            : "Unable to retrieve evidence from the server. Ensure the backend is running on port 8000.";
        setEvidenceError(message);
      } finally {
        setEvidenceLoading(false);
      }
    },
    [claim]
  );

  /* =========================
     FETCH INFLUENCE (PHASE 3)
  ========================= */

  const fetchInfluence = useCallback(
    async (text) => {
      const targetClaim = (text || claim).trim();
      if (!targetClaim || targetClaim.length < 3) return;

      setInfluenceLoading(true);
      setInfluenceError("");

      try {
        const response = await fetch(`${API_URL}/influence/analyze`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            claim: targetClaim,
            top_k: 3,
            min_score: 0.08,
          }),
        });

        const data = await response.json().catch(() => null);

        if (!response.ok) {
          let errorMessage = `Influence search error (${response.status})`;
          if (data) {
            if (typeof data.detail === "string") {
              errorMessage = data.detail;
            } else if (Array.isArray(data.detail) && data.detail[0]?.msg) {
              errorMessage = data.detail[0].msg;
            } else if (data.error) {
              errorMessage = data.error;
            }
          }
          throw new Error(errorMessage);
        }

        if (!data || !Array.isArray(data.results)) {
          throw new Error("Invalid response format received from influence analysis server.");
        }

        setInfluence(data);
        setLastSearchedInfluenceClaim(targetClaim);
      } catch (err) {
        console.error("Influence analysis failed:", err);
        const message =
          err?.message && !err.message.includes("Failed to fetch")
            ? err.message
            : "Unable to retrieve influence indicators from the server. Ensure the backend is running on port 8000.";
        setInfluenceError(message);
      } finally {
        setInfluenceLoading(false);
      }
    },
    [claim]
  );

  /* =========================
     ANALYZE CLAIM (MULTI-AGENT ORCHESTRATOR)
  ========================= */

  const analyzeClaim = async (text = claim) => {
    const finalClaim = text.trim();

    if (!finalClaim) {
      setError("Please enter a claim first.");
      return;
    }

    if (finalClaim.length < 3) {
      setError("Claim must contain at least 3 characters.");
      return;
    }

    if (finalClaim.length > 5000) {
      setError("Claim cannot exceed 5000 characters.");
      return;
    }

    // Reset previous states so stale results from previous claims are never confused
    setLoading(true);
    setError("");
    setAgentAnalysis(null);
    setComparison(null);
    setEvidence(null);
    setEvidenceError("");
    setLastSearchedEvidenceClaim(finalClaim);
    setInfluence(null);
    setInfluenceError("");
    setLastSearchedInfluenceClaim(finalClaim);
    setIsComparisonOpen(false);

    try {
      // 1. Run Multi-Agent Orchestrator (POST /agent/analyze) and Phase 1 comparison (POST /compare) concurrently
      const [agentRes, compareRes] = await Promise.allSettled([
        fetch(`${API_URL}/agent/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ claim: finalClaim, top_k_evidence: 5, top_k_influence: 3 }),
        }),
        fetch(`${API_URL}/compare`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ claim: finalClaim }),
        }),
      ]);

      let compareData = null;
      if (compareRes.status === "fulfilled" && compareRes.value.ok) {
        compareData = await compareRes.value.json().catch(() => null);
      }

      if (agentRes.status === "fulfilled" && agentRes.value.ok) {
        const agentData = await agentRes.value.json().catch(() => null);
        if (agentData) {
          setAgentAnalysis(agentData);

          // Populate Primary Assessment from Agent A
          const distilbertPayload = {
            label: agentData.claim_analysis?.data?.label || compareData?.distilbert?.label,
            confidence: agentData.claim_analysis?.data?.confidence || compareData?.distilbert?.confidence || 0.0,
            probabilities: agentData.claim_analysis?.data?.all_probabilities || {},
          };

          const syntheticComparison = {
            claim: finalClaim,
            distilbert: distilbertPayload,
            baseline: compareData?.baseline || { label: "unknown", confidence: 0.0 },
          };
          setComparison(syntheticComparison);
          saveHistory(syntheticComparison);

          // Populate RAG Evidence from Agent B
          const ragData = agentData.evidence_rag?.data || {};
          setEvidence({
            query: finalClaim,
            results: ragData.citations || [],
            grounded_summary: ragData.grounded_summary,
            evidence_status: ragData.evidence_status,
            has_conflicts: ragData.has_conflicts,
            verdict_distribution: ragData.verdict_distribution,
            generation_mode: ragData.generation_mode,
          });

          // Populate Historical Influence from Agent C
          const infData = agentData.historical_influence?.data || {};
          setInfluence({
            query: finalClaim,
            results: infData.matched_stories || [],
            benchmark_baselines: infData.benchmark_baselines || {},
          });

          return;
        }
      }

      // Fallback if /agent/analyze was unavailable
      if (compareData && compareData.baseline && compareData.distilbert) {
        setComparison(compareData);
        saveHistory(compareData);
        fetchEvidence(finalClaim);
        fetchInfluence(finalClaim);
      } else {
        throw new Error("Unable to obtain analysis from agent orchestrator or comparison server.");
      }
    } catch (err) {
      console.error("Analysis execution failed:", err);
      const message =
        err?.message && !err.message.includes("Failed to fetch")
          ? err.message
          : "Unable to connect to the detection server. Make sure FastAPI is running on port 8000.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    analyzeClaim();
  };

  const handleSelectExample = (example) => {
    setClaim(example);
    setError("");
  };

  const clearAll = () => {
    setClaim("");
    setAgentAnalysis(null);
    setComparison(null);
    setError("");
    setEvidence(null);
    setEvidenceError("");
    setLastSearchedEvidenceClaim("");
    setInfluence(null);
    setInfluenceError("");
    setLastSearchedInfluenceClaim("");
    setIsComparisonOpen(false);
  };

  return (
    <div className="app">
      {/* =========================
          NAVBAR
      ========================= */}

      <header className="navbar">
        <div
          className="brand"
          onClick={() => navigate("/")}
          role="button"
          tabIndex={0}
          style={{ cursor: "pointer" }}
        >
          <div className="brand-mark">M</div>
          <div className="brand-text">
            <h2>MisinformationAI</h2>
            <span>Detection, Influence &amp; Analysis System</span>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          <span>AI SYSTEM ONLINE</span>
        </div>
      </header>

      {/* =========================
          SEPARATE SNA PAGES ROUTING
      ========================= */}
      {currentPath === "/social-network" ? (
        <SocialNetworkPage onBack={() => navigate("/")} />
      ) : currentPath === "/propagation-analysis" ? (
        <PropagationAnalysisPage onBack={() => navigate("/")} />
      ) : (
        <main>
        {/* =========================
            HERO
        ========================= */}

        <section className="hero">
          <div className="hero-badges">
            <span>PHASE 04</span>
            <span>MULTI-AGENT AI · PROPER RAG · INFLUENCE</span>
          </div>

          <h1>
            Detect misinformation
            <br />
            <span>and analyze social spread.</span>
          </h1>

          <p>
            Analyze political statements using fine-tuned transformer NLP,
            retrieve historical justifications from PolitiFact, and evaluate public social influence benchmarks from FakeNewsNet.
          </p>
        </section>

        {/* =========================
            CLAIM ANALYZER
        ========================= */}

        <section className="analyzer-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">CLAIM ANALYZER</span>
              <h2>What do you want to analyze?</h2>
            </div>

            <div className="heading-tags">
              <span className="dataset-tag">LIAR · 6-Class Classification</span>
              {evidenceStats?.total_indexed && (
                <span className="corpus-tag" title="Indexed LIAR-PLUS training corpus">
                  Corpus: {evidenceStats.total_indexed.toLocaleString()} Claims
                </span>
              )}
              {influenceStats?.total_stories && (
                <span className="corpus-tag influence-tag" title="Indexed FakeNewsNet benchmark stories">
                  Influence: {influenceStats.total_stories.toLocaleString()} Stories
                </span>
              )}
            </div>
          </div>

          <form className="analyzer-card" onSubmit={handleSubmit}>
            <textarea
              value={claim}
              onChange={(event) => setClaim(event.target.value)}
              placeholder="Enter a political statement, headline, or policy claim here..."
              rows={6}
            />

            <div className="analyzer-footer">
              <span className="character-count">{claim.length} characters</span>

              <div className="button-group">
                <button
                  type="button"
                  className="clear-button"
                  onClick={clearAll}
                >
                  Clear
                </button>

                <button
                  type="submit"
                  className="analyze-button"
                  disabled={loading && evidenceLoading && influenceLoading}
                >
                  {loading || evidenceLoading || influenceLoading ? (
                    <>
                      <span className="spinner" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      Analyze Claim
                      <span className="arrow">→</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>

          {/* EXAMPLES */}

          <div className="examples">
            <span>Try an illustrative political example:</span>

            {EXAMPLES.map((example, index) => (
              <button
                key={index}
                type="button"
                onClick={() => handleSelectExample(example)}
              >
                {example}
              </button>
            ))}
          </div>

          {/* ERROR */}

          {error && (
            <div className="error-box">
              <div className="error-icon">!</div>
              <div>
                <strong>Analysis failed</strong>
                <p>{error}</p>
              </div>
            </div>
          )}
        </section>

        {/* =========================
            SOCIAL NETWORK ANALYSIS (SNA) 4 NAVIGATION CARDS
        ========================= */}
        <SnaNavCards onNavigate={navigate} />

        {/* ============================================================
            RESULT SCREEN HIERARCHY
            1. Submitted Political Claim
            2. Primary Assessment — DistilBERT prediction and model confidence
            3. Retrieved Historical Evidence — LIAR-PLUS records
            4. Historical Influence Indicators — FakeNewsNet Benchmark
            5. Evidence Quality and Limitations
            6. Analysis Summary
            7. Phase 1 Model Comparison — Collapsible section
        ============================================================ */}

        {comparison && !loading && (
          <div className="results-hierarchy-wrapper">
            {/* 1. Submitted Political Claim */}
            <div className="claim-preview">
              <span className="claim-eyebrow">SUBMITTED POLITICAL CLAIM</span>
              <p>&ldquo;{comparison.claim}&rdquo;</p>
            </div>

            {/* 2. Primary Assessment — DistilBERT prediction and model confidence (Agent A) */}
            <PrimaryAssessmentSection comparison={comparison} />

            {/* 3. Agent Execution Trace (5 Specialized Agents) */}
            {agentAnalysis?.execution_trace && (
              <AgentExecutionTraceSection
                trace={agentAnalysis.execution_trace}
                totalDurationMs={agentAnalysis.total_duration_ms}
                workflowStatus={agentAnalysis.workflow_status}
                isOpen={isTraceOpen}
                onToggle={() => setIsTraceOpen((prev) => !prev)}
              />
            )}

            {/* 4. Verification Findings & Risk Uncertainty Assessment (Agent D) */}
            {agentAnalysis?.verification_risk && (
              <VerificationRiskSection verificationData={agentAnalysis.verification_risk} />
            )}

            {/* 5. Mitigation Recommendations (Agent E) */}
            {agentAnalysis?.mitigation_recommendations && (
              <MitigationRecommendationsSection mitigationData={agentAnalysis.mitigation_recommendations} />
            )}

            {/* 6. Retrieved Historical Evidence — LIAR-PLUS RAG records (Agent B) */}
            <HistoricalEvidenceSection
              evidence={evidence}
              loading={evidenceLoading}
              error={evidenceError}
              onRetry={() => fetchEvidence(lastSearchedEvidenceClaim || claim)}
              evidenceStats={evidenceStats}
            />

            {/* 7. Historical Influence Indicators — FakeNewsNet Benchmark (Agent C) */}
            <HistoricalInfluenceSection
              influence={influence}
              loading={influenceLoading}
              error={influenceError}
              onRetry={() => fetchInfluence(lastSearchedInfluenceClaim || claim)}
              influenceStats={influenceStats}
            />

            {/* 8. Evidence Quality and Limitations */}
            <EvidenceLimitationsSection />

            {/* 9. Analysis Summary */}
            <AnalysisSummarySection
              comparison={comparison}
              evidence={evidence}
              influence={influence}
            />

            {/* 10. Phase 1 Model Comparison — Collapsible section */}
            <Phase1ComparisonSection
              comparison={comparison}
              isOpen={isComparisonOpen}
              onToggle={() => setIsComparisonOpen((prev) => !prev)}
            />
          </div>
        )}

        {/* =========================
            PIPELINE
        ========================= */}

        <section className="pipeline-section">
          <div className="section-heading centered">
            <span className="eyebrow">HOW IT WORKS</span>
            <h2>From claim to evidence and social influence analysis</h2>
            <p>
              Phase 3 combines fine-tuned DistilBERT classification, lexical evidence retrieval, and FakeNewsNet social engagement indicators.
            </p>
          </div>

          <div className="pipeline">
            <div className="pipeline-step">
              <div className="step-number">01</div>
              <div className="step-icon">⌕</div>
              <h3>Input Claim</h3>
              <p>User enters a political statement or news story for analysis.</p>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <div className="step-number">02</div>
              <div className="step-icon">AI</div>
              <h3>Primary Assessment</h3>
              <p>Fine-tuned DistilBERT classifies veracity across 6 LIAR levels.</p>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <div className="step-number">03</div>
              <div className="step-icon">🗂</div>
              <h3>Evidence Retrieval</h3>
              <p>LIAR-PLUS archive is searched for similar historical claims &amp; rulings.</p>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <div className="step-number">04</div>
              <div className="step-icon">📊</div>
              <h3>Influence Analysis</h3>
              <p>FakeNewsNet archive provides observed Twitter volume and temporal metrics.</p>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <div className="step-number">05</div>
              <div className="step-icon">✓</div>
              <h3>Synthesis</h3>
              <p>Users inspect synthesized assessment, data integrity flags, and comparative metrics.</p>
            </div>
          </div>
        </section>

        {/* =========================
            HISTORY
        ========================= */}

        <section className="history-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">RECENT ACTIVITY</span>
              <h2>Prediction History</h2>
            </div>

            <span className="history-count">
              {history.length} saved
            </span>
          </div>

          {history.length === 0 ? (
            <div className="empty-history">
              <div className="empty-icon">◷</div>
              <p>No predictions yet.</p>
              <span>Your recent claim analyses will appear here.</span>
            </div>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <div className="history-item" key={item.id}>
                  <div className="history-main">
                    <p>{item.claim}</p>
                    <span>{item.time}</span>
                  </div>

                  <div className="history-models">
                    <span>
                      Primary (DistilBERT):
                      <strong>{formatLabel(item.distilbert)}</strong>
                    </span>

                    <span>
                      Baseline (TF-IDF):
                      <strong>{formatLabel(item.baseline)}</strong>
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* =========================
            MODEL INFORMATION
        ========================= */}

        <section className="info-section">
          <div className="info-card">
            <div className="info-icon">01</div>
            <div>
              <span className="eyebrow">DATASETS</span>
              <h3>LIAR &amp; LIAR-PLUS</h3>
              <p>
                12,800+ political statements from PolitiFact with six-class veracity ratings and extracted justifications.
              </p>
            </div>
          </div>

          <div className="info-card">
            <div className="info-icon">02</div>
            <div>
              <span className="eyebrow">SOCIAL INFLUENCE</span>
              <h3>FakeNewsNet (PolitiFact)</h3>
              <p>
                1,056 political news stories linked to 583,000+ public tweet IDs with decoded Snowflake temporal metrics.
              </p>
            </div>
          </div>

          <div className="info-card">
            <div className="info-icon">03</div>
            <div>
              <span className="eyebrow">PRIMARY CLASSIFIER</span>
              <h3>DistilBERT</h3>
              <p>
                A compact transformer architecture fine-tuned for six-class political misinformation classification.
              </p>
            </div>
          </div>

          <div className="info-card">
            <div className="info-icon">04</div>
            <div>
              <span className="eyebrow">BASELINE MODEL</span>
              <h3>TF-IDF + Logistic Regression</h3>
              <p>
                A traditional n-gram NLP baseline used for comparative Phase 1 benchmarking.
              </p>
            </div>
          </div>
        </section>
      </main>
      )}

      {/* =========================
          FOOTER
      ========================= */}

      <footer>
        <div>
          <strong>Misinformation Detection System</strong>
          <span>Phase 03 · ML/NLP + Evidence Retrieval + Social Influence</span>
        </div>
        <span>Built with React + FastAPI + LIAR-PLUS + FakeNewsNet</span>
      </footer>
    </div>
  );
}

export default App;
