import { useState } from "react";
import ClaimInputCard from "../components/ClaimInputCard";
import TruthScale from "../components/TruthScale";
import ConfidenceBar from "../components/ConfidenceBar";
import { API_URL, getLabelInfo, TARGET_DOMAIN } from "../utils/constants";

export default function ClaimAnalysisPage({
  claim,
  setClaim,
  onSaveHistory,
  onNavigate,
}) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async (e) => {
    if (e) e.preventDefault();
    const targetClaim = claim.trim();

    if (!targetClaim) {
      setError("Please enter a political statement or claim to analyze.");
      return;
    }
    if (targetClaim.length < 3) {
      setError("Claim must contain at least 3 characters.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: targetClaim }),
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(data?.detail || data?.error || `Server responded with HTTP ${response.status}`);
      }

      setResult(data);
      if (onSaveHistory) {
        onSaveHistory({
          claim: targetClaim,
          distilbert: {
            label: data.label,
            confidence: data.confidence,
          },
          baseline: {
            label: data.model || "Socio-Economic 3-Class",
            confidence: data.confidence,
          },
        });
      }
    } catch (err) {
      console.error("Claim analysis failed:", err);
      setError(
        err?.message?.includes("Failed to fetch")
          ? "Unable to connect to the backend server (http://127.0.0.1:8000). Ensure FastAPI is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const activeLabel = result?.label;
  const labelInfo = getLabelInfo(activeLabel);
  const probabilities = result?.probabilities || {};

  return (
    <div className="page-container claim-analysis-page">
      {/* Header */}
      <div className="page-header">
        <div className="header-breadcrumbs">
          <button type="button" className="crumb-link" onClick={() => onNavigate("/")}>
            Dashboard
          </button>
          <span className="crumb-sep">/</span>
          <span className="crumb-current">Claim Classifier</span>
        </div>
        <h1>Claim Classifier</h1>
        <p className="page-description">
          Evaluate political statements and policy claims across verified veracity levels.
        </p>
      </div>

      {/* Claim Input Form */}
      <ClaimInputCard
        claim={claim}
        onClaimChange={setClaim}
        onSubmit={handleAnalyze}
        loading={loading}
        error={error}
        onClear={() => {
          setClaim("");
          setResult(null);
          setError("");
        }}
        buttonLabel="Classify Claim"
      />

      {/* Results View */}
      {result && (
        <div className="analysis-results-section">
          {/* Primary Assessment Card */}
          <div className="primary-verdict-card">
            <div className="verdict-card-header">
              <div className="verdict-title-col">
                <span className="verdict-eyebrow">VERACITY ASSESSMENT</span>
                <div className="verdict-banner-row">
                  <h2 className={`verdict-class-name level-text-${labelInfo.level}`}>
                    {labelInfo.title}
                  </h2>
                </div>
                <p className="verdict-description-text">
                  Classified as <strong>{labelInfo.title}</strong> based on alignment with verified policy records and published fact-checks.
                </p>
              </div>

              <div className="verdict-confidence-box">
                <ConfidenceBar
                  value={result?.confidence || 0.5}
                  label="Classification Confidence"
                  level={labelInfo.level}
                />
              </div>
            </div>

            {/* Veracity Spectrum */}
            <div className="verdict-spectrum-wrapper">
              <TruthScale
                activeLabel={activeLabel}
                mode="3class"
              />
            </div>

            {/* Class Probability Distribution Breakdown */}
            {Object.keys(probabilities).length > 0 && (
              <div className="probabilities-breakdown-card">
                <h4>Class Probability Distribution</h4>
                <div className="probability-bars-grid">
                  {[
                    { key: "Misleading", label: "Misleading", level: 2 },
                    { key: "Half-True", label: "Half-True", level: 4 },
                    { key: "Factual", label: "Factual", level: 6 },
                  ].map((cls) => {
                    const prob =
                      probabilities[cls.key] !== undefined
                        ? probabilities[cls.key]
                        : probabilities[cls.key.toLowerCase()] || 0;
                    const pct = (prob * 100).toFixed(1);
                    const isWinner =
                      String(activeLabel).toLowerCase() === cls.key.toLowerCase();
                    return (
                      <div
                        key={cls.key}
                        className={`probability-bar-row ${isWinner ? "row-winner" : ""}`}
                      >
                        <span className="prob-label">
                          {cls.label} {isWinner && <span className="winner-indicator">✓</span>}
                        </span>
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
          </div>

          {/* Quick Nav Suggestions to Other Modules */}
          <div className="post-analysis-actions-card">
            <h4>Recommended Next Steps</h4>
            <div className="quick-actions-row">
              <button
                type="button"
                className="btn-action-outline"
                onClick={() => onNavigate("/evidence")}
              >
                Search Historical Evidence →
              </button>
              <button
                type="button"
                className="btn-action-outline"
                onClick={() => onNavigate("/agentic-analysis")}
              >
                Run Multi-Agent Synthesis →
              </button>
              <button
                type="button"
                className="btn-action-outline"
                onClick={() => onNavigate("/influence-analysis")}
              >
                Inspect Social Signals →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
