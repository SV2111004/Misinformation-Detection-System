import { useState } from "react";
import ClaimInputCard from "../components/ClaimInputCard";
import TruthScale from "../components/TruthScale";
import ConfidenceBar from "../components/ConfidenceBar";
import { API_URL, getLabelInfo, formatLabel } from "../utils/constants";

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
      const response = await fetch(`${API_URL}/compare`, {
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
        onSaveHistory(data);
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

  const distilbertData = result?.distilbert;
  const labelInfo = getLabelInfo(distilbertData?.label);
  const probabilities = distilbertData?.probabilities || {};

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
        <h1>Neural Misinformation Classifier (DistilBERT)</h1>
        <p className="page-description">
          Evaluates submitted political claims against the 6-class LIAR veracity benchmark using our fine-tuned DistilBERT transformer.
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
      {result && distilbertData && (
        <div className="analysis-results-section">
          {/* Primary Assessment Card */}
          <div className="primary-verdict-card">
            <div className="verdict-card-header">
              <div className="verdict-title-col">
                <span className="verdict-eyebrow">DISTILBERT PREDICTED VERACITY CLASS</span>
                <div className="verdict-banner-row">
                  <h2 className={`verdict-class-name level-text-${labelInfo.level}`}>
                    {labelInfo.title}
                  </h2>
                  <span className="verdict-standard-tag">LIAR Benchmark Standard</span>
                </div>
                <p className="verdict-description-text">
                  The fine-tuned DistilBERT neural model assigned its highest softmax probability to the{" "}
                  <strong>{labelInfo.title}</strong> class for this statement.
                </p>
              </div>

              <div className="verdict-confidence-box">
                <ConfidenceBar
                  value={distilbertData.confidence}
                  label="Model Confidence"
                  level={labelInfo.level}
                />
                <span className="confidence-caveat-note">
                  *Softmax class alignment under fine-tuned weights. Confidence reflects stylistic and linguistic match with benchmark patterns, not empirical proof.*
                </span>
              </div>
            </div>

            {/* 6-Class Spectrum */}
            <div className="verdict-spectrum-wrapper">
              <TruthScale activeLabel={distilbertData.label} />
            </div>

            {/* Class Probability Distribution Breakdown (if available) */}
            {Object.keys(probabilities).length > 0 && (
              <div className="probabilities-breakdown-card">
                <h4>Full 6-Class Softmax Probability Distribution</h4>
                <div className="probability-bars-grid">
                  {[
                    { key: "pants-fire", label: "Pants on Fire", level: 1 },
                    { key: "false", label: "False", level: 2 },
                    { key: "barely-true", label: "Barely True", level: 3 },
                    { key: "half-true", label: "Half True", level: 4 },
                    { key: "mostly-true", label: "Mostly True", level: 5 },
                    { key: "true", label: "True", level: 6 },
                  ].map((cls) => {
                    const prob = probabilities[cls.key] || 0;
                    const pct = (prob * 100).toFixed(1);
                    const isWinner = distilbertData.label === cls.key;
                    return (
                      <div
                        key={cls.key}
                        className={`probability-bar-row ${isWinner ? "row-winner" : ""}`}
                      >
                        <span className="prob-label">
                          {cls.label} {isWinner && "★"}
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
            <h4>Next Analysis Steps for this Claim:</h4>
            <div className="quick-actions-row">
              <button
                type="button"
                className="btn-action-outline"
                onClick={() => onNavigate("/model-comparison")}
              >
                ⚖️ Compare with TF-IDF Baseline →
              </button>
              <button
                type="button"
                className="btn-action-outline"
                onClick={() => onNavigate("/evidence")}
              >
                📚 Search PolitiFact Evidence (RAG) →
              </button>
              <button
                type="button"
                className="btn-action-outline"
                onClick={() => onNavigate("/agentic-analysis")}
              >
                🤖 Run Full Multi-Agent Pipeline →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
