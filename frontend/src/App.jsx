import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const LABEL_INFO = {
  "pants-fire": {
    title: "Pants on Fire",
    description: "LIAR dataset classification",
    level: 1,
  },
  false: {
    title: "False",
    description: "LIAR dataset classification",
    level: 2,
  },
  "barely-true": {
    title: "Barely True",
    description: "LIAR dataset classification",
    level: 3,
  },
  "half-true": {
    title: "Half True",
    description: "LIAR dataset classification",
    level: 4,
  },
  "mostly-true": {
    title: "Mostly True",
    description: "LIAR dataset classification",
    level: 5,
  },
  true: {
    title: "True",
    description: "LIAR dataset classification",
    level: 6,
  },
};

const EXAMPLES = [
  "The government has announced a new policy.",
  "Scientists have discovered a new treatment for cancer.",
  "The unemployment rate has reached a record low.",
];

function formatLabel(label) {
  if (!label) return "Unknown";

  return label
    .split("-")
    .map(
      (word) =>
        word.charAt(0).toUpperCase() + word.slice(1)
    )
    .join(" ");
}

function getLabelInfo(label) {
  return (
    LABEL_INFO[label] || {
      title: formatLabel(label),
      description: "Model classification",
      level: 0,
    }
  );
}

function ConfidenceBar({ value, type }) {
  const percentage = Math.max(
    0,
    Math.min(100, (value || 0) * 100)
  );

  return (
    <div className="confidence-wrapper">
      <div className="confidence-track">
        <div
          className={`confidence-fill ${type}`}
          style={{ width: `${percentage}%` }}
        />
      </div>

      <span className="confidence-value">
        {percentage.toFixed(1)}%
      </span>
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
        <span className="result-eyebrow">
          PREDICTED CLASS
        </span>

        <h2>{info.title}</h2>

        <p>{info.description}</p>
      </div>

      <div className="confidence-section">
        <div className="confidence-header">
          <span>Model Confidence</span>

          <strong>
            {((result?.confidence || 0) * 100).toFixed(1)}%
          </strong>
        </div>

        <ConfidenceBar
          value={result?.confidence}
          type={type}
        />
      </div>
    </article>
  );
}

function App() {
  const [claim, setClaim] = useState("");
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  /* =========================
     LOAD HISTORY
  ========================= */

  useEffect(() => {
    const saved =
      localStorage.getItem("misinformation-history");

    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch {
        setHistory([]);
      }
    }
  }, []);

  /* =========================
     SAVE HISTORY
  ========================= */

  const saveHistory = (data) => {
    const newItem = {
      id: Date.now(),
      claim: data.claim,
      baseline: data.baseline?.label,
      distilbert: data.distilbert?.label,
      time: new Date().toLocaleTimeString(),
    };

    const updated = [newItem, ...history].slice(0, 8);

    setHistory(updated);

    localStorage.setItem(
      "misinformation-history",
      JSON.stringify(updated)
    );
  };

  /* =========================
     ANALYZE CLAIM
  ========================= */

  const analyzeClaim = async (text = claim) => {
    const finalClaim = text.trim();

    if (!finalClaim) {
      setError("Please enter a claim first.");
      return;
    }

    setLoading(true);
    setError("");
    setComparison(null);

    try {
      const response = await fetch(
        `${API_URL}/compare`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            claim: finalClaim,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}`
        );
      }

      const data = await response.json();

      setComparison(data);
      saveHistory(data);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to connect to the detection server. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    analyzeClaim();
  };

  const useExample = (example) => {
    setClaim(example);
    setError("");
  };

  const clearAll = () => {
    setClaim("");
    setComparison(null);
    setError("");
  };

  /* =========================
     COMPARISON DATA
  ========================= */

  const baselineLabel =
    comparison?.baseline?.label;

  const distilbertLabel =
    comparison?.distilbert?.label;

  const modelsAgree =
    Boolean(
      baselineLabel &&
        distilbertLabel &&
        baselineLabel === distilbertLabel
    );

  const baselineInfo =
    getLabelInfo(baselineLabel);

  const distilbertInfo =
    getLabelInfo(distilbertLabel);

  return (
    <div className="app">
      {/* =========================
          NAVBAR
      ========================= */}

      <header className="navbar">
        <div className="brand">
          <div className="brand-mark">M</div>

          <div className="brand-text">
            <h2>MisinformationAI</h2>

            <span>
              Detection &amp; Analysis System
            </span>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          <span>AI SYSTEM ONLINE</span>
        </div>
      </header>

      <main>
        {/* =========================
            HERO
        ========================= */}

        <section className="hero">
          <div className="hero-badges">
            <span>PHASE 01</span>
            <span>ML + NLP</span>
          </div>

          <h1>
            Detect misinformation
            <br />
            <span>before it spreads.</span>
          </h1>

          <p>
            Analyze a claim using traditional machine
            learning and transformer-based NLP, then
            compare their predictions side by side.
          </p>
        </section>

        {/* =========================
            CLAIM ANALYZER
        ========================= */}

        <section className="analyzer-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">
                CLAIM ANALYZER
              </span>

              <h2>
                What do you want to analyze?
              </h2>
            </div>

            <span className="dataset-tag">
              LIAR · 6-class classification
            </span>
          </div>

          <form
            className="analyzer-card"
            onSubmit={handleSubmit}
          >
            <textarea
              value={claim}
              onChange={(event) =>
                setClaim(event.target.value)
              }
              placeholder="Enter a statement or claim here..."
              rows={6}
            />

            <div className="analyzer-footer">
              <span className="character-count">
                {claim.length} characters
              </span>

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
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      Analyze Claim
                      <span className="arrow">
                        →
                      </span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>

          {/* EXAMPLES */}

          <div className="examples">
            <span>Try an example:</span>

            {EXAMPLES.map((example, index) => (
              <button
                key={index}
                type="button"
                onClick={() => useExample(example)}
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
            RESULTS
        ========================= */}

        {comparison && !loading && (
          <section className="results-section">
            <div className="result-header">
              <div>
                <span className="eyebrow">
                  ANALYSIS COMPLETE
                </span>

                <h2>Model Comparison</h2>
              </div>

              <div
                className={`agreement-badge ${
                  modelsAgree
                    ? "agree"
                    : "different"
                }`}
              >
                <span>
                  {modelsAgree ? "✓" : "!"}
                </span>

                {modelsAgree
                  ? "Models Agree"
                  : "Models Differ"}
              </div>
            </div>

            {/* CLAIM */}

            <div className="claim-preview">
              <span>ANALYZED CLAIM</span>

              <p>
                "{comparison.claim}"
              </p>
            </div>

            {/* MODEL CARDS */}

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

            {/* =========================
                CONSENSUS
            ========================= */}

            <div
              className={`comparison-summary ${
                modelsAgree
                  ? "summary-agree"
                  : "summary-different"
              }`}
            >
              <div className="summary-content">
                <span className="eyebrow">
                  MODEL CONSENSUS
                </span>

                {modelsAgree ? (
                  <>
                    <h2>
                      {formatLabel(
                        baselineLabel
                      )}
                    </h2>

                    <p>
                      Both models produced the same
                      classification for this claim.
                    </p>
                  </>
                ) : (
                  <>
                    <h2>Models Disagree</h2>

                    <p>
                      The two models produced
                      different classifications.
                      Further evidence-based
                      verification is recommended.
                    </p>
                  </>
                )}
              </div>

              <div className="summary-right">
                <div className="summary-model">
                  <span>TF-IDF + LR</span>

                  <strong>
                    {baselineInfo.title}
                  </strong>
                </div>

                <div className="summary-divider" />

                <div className="summary-model">
                  <span>DistilBERT</span>

                  <strong>
                    {distilbertInfo.title}
                  </strong>
                </div>
              </div>
            </div>

            {/* =========================
                TRUTHFULNESS SCALE
            ========================= */}

            <div className="truth-scale-card">
              <div className="truth-scale-header">
                <div>
                  <span className="eyebrow">
                    LIAR LABEL SCALE
                  </span>

                  <h3>
                    Six-class classification
                  </h3>
                </div>

                <span>
                  Based on predicted labels
                </span>
              </div>

              <div className="scale">
                {[
                  {
                    label: "Pants Fire",
                    key: "pants-fire",
                  },
                  {
                    label: "False",
                    key: "false",
                  },
                  {
                    label: "Barely True",
                    key: "barely-true",
                  },
                  {
                    label: "Half True",
                    key: "half-true",
                  },
                  {
                    label: "Mostly True",
                    key: "mostly-true",
                  },
                  {
                    label: "True",
                    key: "true",
                  },
                ].map((item) => {
                  const isBaseline =
                    baselineLabel === item.key;

                  const isDistilbert =
                    distilbertLabel === item.key;

                  return (
                    <div
                      className="scale-item"
                      key={item.key}
                    >
                      <div
                        className={`scale-dot ${
                          isBaseline
                            ? "baseline-dot"
                            : ""
                        } ${
                          isDistilbert
                            ? "transformer-dot"
                            : ""
                        } ${
                          isBaseline &&
                          isDistilbert
                            ? "both-dot"
                            : ""
                        }`}
                      />

                      <span>
                        {item.label}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="scale-legend">
                <span>
                  <i className="legend-baseline" />
                  TF-IDF + LR
                </span>

                <span>
                  <i className="legend-transformer" />
                  DistilBERT
                </span>
              </div>
            </div>

            {/* DISCLAIMER */}

            <div className="disclaimer">
              <span className="info-symbol">
                i
              </span>

              <p>
                This is a machine-learning
                classification based on the LIAR
                dataset. Model confidence represents
                confidence in the predicted class,
                not proof that the claim is factually
                true or false.
              </p>
            </div>
          </section>
        )}

        {/* =========================
            PIPELINE
        ========================= */}

        <section className="pipeline-section">
          <div className="section-heading centered">
            <span className="eyebrow">
              HOW IT WORKS
            </span>

            <h2>
              From claim to classification
            </h2>

            <p>
              Phase 1 combines traditional NLP with
              transformer-based language understanding.
            </p>
          </div>

          <div className="pipeline">
            <div className="pipeline-step">
              <div className="step-number">
                01
              </div>

              <div className="step-icon">
                ⌕
              </div>

              <h3>Input Claim</h3>

              <p>
                User provides a statement for analysis.
              </p>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <div className="step-number">
                02
              </div>

              <div className="step-icon">
                TF
              </div>

              <h3>TF-IDF</h3>

              <p>
                Text is converted into numerical
                features.
              </p>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <div className="step-number">
                03
              </div>

              <div className="step-icon">
                AI
              </div>

              <h3>DistilBERT</h3>

              <p>
                Transformer model analyzes language
                context.
              </p>
            </div>

            <div className="pipeline-line" />

            <div className="pipeline-step">
              <div className="step-number">
                04
              </div>

              <div className="step-icon">
                ✓
              </div>

              <h3>Compare</h3>

              <p>
                Predictions and confidence are
                compared.
              </p>
            </div>
          </div>
        </section>

        {/* =========================
            HISTORY
        ========================= */}

        <section className="history-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">
                RECENT ACTIVITY
              </span>

              <h2>Prediction History</h2>
            </div>

            <span className="history-count">
              {history.length} saved
            </span>
          </div>

          {history.length === 0 ? (
            <div className="empty-history">
              <div className="empty-icon">
                ◷
              </div>

              <p>No predictions yet.</p>

              <span>
                Your recent claim analyses will
                appear here.
              </span>
            </div>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <div
                  className="history-item"
                  key={item.id}
                >
                  <div className="history-main">
                    <p>{item.claim}</p>

                    <span>{item.time}</span>
                  </div>

                  <div className="history-models">
                    <span>
                      ML:
                      <strong>
                        {formatLabel(
                          item.baseline
                        )}
                      </strong>
                    </span>

                    <span>
                      AI:
                      <strong>
                        {formatLabel(
                          item.distilbert
                        )}
                      </strong>
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
            <div className="info-icon">
              01
            </div>

            <div>
              <span className="eyebrow">
                DATASET
              </span>

              <h3>LIAR Dataset</h3>

              <p>
                Six-way classification of statements
                ranging from Pants-on-Fire to True.
              </p>
            </div>
          </div>

          <div className="info-card">
            <div className="info-icon">
              02
            </div>

            <div>
              <span className="eyebrow">
                BASELINE
              </span>

              <h3>
                TF-IDF + Logistic Regression
              </h3>

              <p>
                A traditional NLP baseline used to
                compare performance against the
                transformer model.
              </p>
            </div>
          </div>

          <div className="info-card">
            <div className="info-icon">
              03
            </div>

            <div>
              <span className="eyebrow">
                TRANSFORMER
              </span>

              <h3>DistilBERT</h3>

              <p>
                A compact transformer architecture
                fine-tuned for six-class classification.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* =========================
          FOOTER
      ========================= */}

      <footer>
        <div>
          <strong>
            Misinformation Detection System
          </strong>

          <span>
            Phase 01 · ML/NLP Foundation
          </span>
        </div>

        <span>
          Built with React + FastAPI
        </span>
      </footer>
    </div>
  );
}

export default App;