import { useState } from "react";
import ClaimInputCard from "../components/ClaimInputCard";
import TruthScale from "../components/TruthScale";
import ConfidenceBar from "../components/ConfidenceBar";
import { API_URL, getLabelInfo, formatLabel } from "../utils/constants";

export default function ModelComparisonPage({
  claim,
  setClaim,
  onSaveHistory,
  onNavigate,
}) {
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleCompare = async (e) => {
    if (e) e.preventDefault();
    const targetClaim = claim.trim();

    if (!targetClaim) {
      setError("Please enter a claim to perform comparative evaluation.");
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

      setComparison(data);
      if (onSaveHistory) {
        onSaveHistory(data);
      }
    } catch (err) {
      console.error("Comparison failed:", err);
      setError(
        err?.message?.includes("Failed to fetch")
          ? "Unable to connect to the backend server (http://127.0.0.1:8000). Ensure FastAPI is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const baselineData = comparison?.baseline;
  const distilbertData = comparison?.distilbert;
  const modelsAgree =
    Boolean(baselineData?.label && distilbertData?.label && baselineData.label === distilbertData.label);

  const baselineInfo = getLabelInfo(baselineData?.label);
  const distilbertInfo = getLabelInfo(distilbertData?.label);

  return (
    <div className="page-container model-comparison-page">
      {/* Header */}
      <div className="page-header">
        <div className="header-breadcrumbs">
          <button type="button" className="crumb-link" onClick={() => onNavigate("/")}>
            Dashboard
          </button>
          <span className="crumb-sep">/</span>
          <span className="crumb-current">Model Comparison</span>
        </div>
        <h1>Model Comparison: Baseline vs. Deep Transformer</h1>
        <p className="page-description">
          Evaluates agreement, divergence, and confidence differentials between our n-gram TF-IDF + Logistic Regression baseline and the fine-tuned DistilBERT transformer.
        </p>
      </div>

      {/* Claim Input Form */}
      <ClaimInputCard
        claim={claim}
        onClaimChange={setClaim}
        onSubmit={handleCompare}
        loading={loading}
        error={error}
        onClear={() => {
          setClaim("");
          setComparison(null);
          setError("");
        }}
        buttonLabel="Compare Models"
      />

      {/* Comparison Results */}
      {comparison && baselineData && distilbertData && (
        <div className="comparison-results-section">
          {/* Agreement / Disagreement Banner */}
          <div className={`consensus-banner ${modelsAgree ? "consensus-agree" : "consensus-diverge"}`}>
            <div className="consensus-icon">{modelsAgree ? "✓" : "!"}</div>
            <div className="consensus-text">
              <h3>
                {modelsAgree
                  ? `Models Agree — Concurrence on '${distilbertInfo.title}'`
                  : `Models Diverge — Prediction Conflict Detected`}
              </h3>
              <p>
                {modelsAgree
                  ? `Both the TF-IDF n-gram classifier and the DistilBERT neural model converged on the same veracity category across the LIAR 6-class spectrum.`
                  : `DistilBERT classified this claim as '${distilbertInfo.title}', whereas the TF-IDF baseline classified it as '${baselineInfo.title}'. Deep contextual attention captured nuances that bag-of-words token matching missed.`}
              </p>
            </div>
            <div className="consensus-chips">
              <span className="chip-label">TF-IDF: <strong>{baselineInfo.title}</strong></span>
              <span className="chip-label">DistilBERT: <strong>{distilbertInfo.title}</strong></span>
            </div>
          </div>

          {/* Side-by-Side Model Cards */}
          <div className="side-by-side-grid">
            {/* Card 1: Baseline */}
            <div className="model-eval-card card-baseline">
              <div className="eval-card-header">
                <div>
                  <span className="eval-eyebrow">PHASE 1 BASELINE MODEL</span>
                  <h3>TF-IDF + Logistic Regression</h3>
                </div>
                <span className="eval-tech-tag">N-Gram Classifier</span>
              </div>

              <div className="eval-prediction-block">
                <span className="eval-label-caption">Predicted Class</span>
                <h2 className={`eval-class-name level-text-${baselineInfo.level}`}>
                  {baselineInfo.title}
                </h2>
                <p className="eval-subtext">{baselineInfo.description}</p>
              </div>

              <div className="eval-confidence-box">
                <ConfidenceBar
                  value={baselineData.confidence}
                  label="Baseline Confidence"
                  level={baselineInfo.level}
                />
              </div>

              <div className="eval-architecture-note">
                <strong>Methodology:</strong> Extracts 1-gram and 2-gram token frequencies over the LIAR corpus and applies a multinomial logistic regression hyperplane. Relies on keyword cues without sentence syntax.
              </div>
            </div>

            {/* Card 2: DistilBERT */}
            <div className="model-eval-card card-transformer">
              <div className="eval-card-header">
                <div>
                  <span className="eval-eyebrow">PRIMARY CLASSIFIER</span>
                  <h3>Fine-Tuned DistilBERT</h3>
                </div>
                <span className="eval-tech-tag tag-primary">Transformer AI</span>
              </div>

              <div className="eval-prediction-block">
                <span className="eval-label-caption">Predicted Class</span>
                <h2 className={`eval-class-name level-text-${distilbertInfo.level}`}>
                  {distilbertInfo.title}
                </h2>
                <p className="eval-subtext">{distilbertInfo.description}</p>
              </div>

              <div className="eval-confidence-box">
                <ConfidenceBar
                  value={distilbertData.confidence}
                  label="Transformer Confidence"
                  level={distilbertInfo.level}
                />
              </div>

              <div className="eval-architecture-note">
                <strong>Methodology:</strong> 6-layer bidirectional transformer with multi-head self-attention, fine-tuned on LIAR political statements. Evaluates negation, semantic dependency, and contextual framing.
              </div>
            </div>
          </div>

          {/* Dual Veracity Spectrum */}
          <div className="comparison-scale-container">
            <TruthScale
              activeLabel={distilbertData.label}
              secondaryLabel={baselineData.label}
              title="Dual Model Placement on LIAR Veracity Spectrum"
              showSecondary={true}
            />
          </div>
        </div>
      )}
    </div>
  );
}
