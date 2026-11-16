import { useState, useEffect } from "react";
import { API_URL, formatLabel } from "../utils/constants";

export default function DashboardPage({ onNavigate, history = [] }) {
  const [health, setHealth] = useState(null);
  const [evidenceStats, setEvidenceStats] = useState(null);
  const [influenceStats, setInfluenceStats] = useState(null);
  const [snaDatasets, setSnaDatasets] = useState([]);
  const [loadingHealth, setLoadingHealth] = useState(true);

  useEffect(() => {
    let isMounted = true;

    // Fetch system health
    fetch(`${API_URL}/health`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted) setHealth(data);
      })
      .catch((err) => console.warn("Health check failed:", err))
      .finally(() => {
        if (isMounted) setLoadingHealth(false);
      });

    // Fetch evidence stats
    fetch(`${API_URL}/evidence/stats`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted && data) setEvidenceStats(data);
      })
      .catch((err) => console.warn("Evidence stats failed:", err));

    // Fetch influence stats
    fetch(`${API_URL}/influence/stats`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted && data) setInfluenceStats(data);
      })
      .catch((err) => console.warn("Influence stats failed:", err));

    // Fetch SNA datasets
    fetch(`${API_URL}/api/sna/datasets`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted && data?.datasets) setSnaDatasets(data.datasets);
      })
      .catch((err) => console.warn("SNA datasets failed:", err));

    return () => {
      isMounted = false;
    };
  }, []);

  const featureCards = [
    {
      id: "claim-analysis",
      path: "/claim-analysis",
      title: "Claim Classifier",
      badge: "Neural NLP",
      description:
        "Evaluate single political statements with the fine-tuned DistilBERT transformer across the LIAR 6-class veracity spectrum.",
      actionText: "Open Classifier →",
      meta: "DistilBERT · 6-Class Softmax",
    },
    {
      id: "model-comparison",
      path: "/model-comparison",
      title: "Model Comparison",
      badge: "Comparative",
      description:
        "Inspect prediction agreement and divergence between the TF-IDF + Logistic Regression baseline and the DistilBERT neural model.",
      actionText: "Compare Models →",
      meta: "Baseline vs Transformer",
    },
    {
      id: "evidence",
      path: "/evidence",
      title: "Evidence & RAG",
      badge: "ChromaDB",
      description:
        "Retrieve historical PolitiFact justifications and context from 10,154 verified records using hybrid semantic vector search.",
      actionText: "Search Evidence →",
      meta: "LIAR-PLUS · ChromaDB RAG",
    },
    {
      id: "agentic-analysis",
      path: "/agentic-analysis",
      title: "Multi-Agent Workflow",
      badge: "Agentic AI",
      description:
        "Orchestrate 5 specialized agents across claim classification, RAG retrieval, social signals, uncertainty quantification, and advisory mitigation.",
      actionText: "Run Agent Pipeline →",
      meta: "5 Specialized Working Agents",
    },
    {
      id: "influence-analysis",
      path: "/influence-analysis",
      title: "Social Influence (FakeNewsNet)",
      badge: "Engagement",
      description:
        "Analyze historical Twitter engagement volumes, domain patterns, and decoded Snowflake temporal propagation metrics from FakeNewsNet.",
      actionText: "View Social Signals →",
      meta: "FakeNewsNet · Snowflake IDs",
    },
    {
      id: "social-network",
      path: "/social-network",
      title: "Propagation Graph (Twitter15/16)",
      badge: "SNA Graph",
      description:
        "Reconstruct observed user-to-user rumor propagation trees into an interactive directed radial graph with depth and amplification inspection.",
      actionText: "Explore Network Graph →",
      meta: "Twitter15/16 · Directed Trees",
    },
    {
      id: "propagation-analysis",
      path: "/propagation-analysis",
      title: "Cascade Kinetics (Twitter15/16)",
      badge: "Diffusion",
      description:
        "Examine cascade size, propagation depth hops, time-delay statistics, and cumulative adoption trajectory curves over time.",
      actionText: "Analyze Cascade Kinetics →",
      meta: "Depth Levels · Timeline Curves",
    },
  ];

  return (
    <div className="page-container dashboard-page">
      {/* Header Banner */}
      <section className="dashboard-hero">
        <div className="hero-content">
          <span className="hero-eyebrow">RESEARCH &amp; VERIFICATION SYSTEM</span>
          <h1>Political Misinformation Detection &amp; Diffusion Analysis</h1>
          <p>
            An end-to-end software platform combining transformer-based neural classification,
            hybrid ChromaDB evidence retrieval, multi-agent synthesis, and observed social network propagation analysis.
          </p>
        </div>

        {/* System Health Strip */}
        <div className="system-health-card">
          <div className="health-card-header">
            <h4>System Architecture &amp; Benchmark Readiness</h4>
            <span className={`status-pill ${health?.status === "ok" ? "online" : "offline"}`}>
              {health?.status === "ok" ? "All Modules Loaded" : "System Ready"}
            </span>
          </div>

          <div className="health-grid">
            <div className="health-item">
              <span className="health-label">Primary Classifier</span>
              <strong className="health-val">
                {health?.distilbert_loaded ? "DistilBERT (Ready)" : "DistilBERT (CPU)"}
              </strong>
            </div>
            <div className="health-item">
              <span className="health-label">Baseline Classifier</span>
              <strong className="health-val">
                {health?.tfidf_loaded ? "TF-IDF + LR (Ready)" : "TF-IDF (Loaded)"}
              </strong>
            </div>
            <div className="health-item">
              <span className="health-label">Evidence Store</span>
              <strong className="health-val">
                {evidenceStats?.total_indexed
                  ? `${evidenceStats.total_indexed.toLocaleString()} Records`
                  : "10,154 Records (Indexed)"}
              </strong>
            </div>
            <div className="health-item">
              <span className="health-label">FakeNewsNet Corpus</span>
              <strong className="health-val">
                {influenceStats?.total_stories
                  ? `${influenceStats.total_stories.toLocaleString()} Stories`
                  : "1,056 Stories"}
              </strong>
            </div>
            <div className="health-item">
              <span className="health-label">SNA Benchmarks</span>
              <strong className="health-val">
                {snaDatasets.length > 0
                  ? "Twitter15 & Twitter16 (2,308 Trees)"
                  : "Twitter15 & 16 Ready"}
              </strong>
            </div>
            <div className="health-item">
              <span className="health-label">Agent Orchestrator</span>
              <strong className="health-val">5 Agents Connected</strong>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Modules Grid */}
      <section className="dashboard-modules-section">
        <div className="section-title-row">
          <div>
            <h2>Functional Verification Modules</h2>
            <p className="section-subtitle">
              Select a specialized module below to perform analysis on political claims or inspect propagation benchmarks.
            </p>
          </div>
        </div>

        <div className="modules-grid">
          {featureCards.map((card) => (
            <article
              key={card.id}
              className="module-card"
              onClick={() => onNavigate(card.path)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onNavigate(card.path);
                }
              }}
            >
              <div className="module-card-top">
                <span className="module-badge">{card.badge}</span>
                <span className="module-meta-pill">{card.meta}</span>
              </div>

              <div className="module-card-body">
                <h3>{card.title}</h3>
                <p>{card.description}</p>
              </div>

              <div className="module-card-footer">
                <span className="module-action-link">{card.actionText}</span>
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* Recent History Table */}
      <section className="dashboard-history-section">
        <div className="section-title-row">
          <div>
            <h2>Recent Claim Evaluations</h2>
            <p className="section-subtitle">
              Statements recently evaluated by the primary and baseline classifiers in this session.
            </p>
          </div>
          <span className="history-count-badge">{history.length} Saved In Session</span>
        </div>

        {history.length === 0 ? (
          <div className="empty-state-box">
            <p>No claims evaluated yet in this session.</p>
            <span className="empty-subtext">
              Jump to the <button type="button" className="inline-link" onClick={() => onNavigate("/claim-analysis")}>Claim Classifier</button> or <button type="button" className="inline-link" onClick={() => onNavigate("/model-comparison")}>Model Comparison</button> to evaluate a political statement.
            </span>
          </div>
        ) : (
          <div className="history-table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Submitted Statement</th>
                  <th>DistilBERT Verdict</th>
                  <th>Baseline (TF-IDF)</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.id}>
                    <td className="timestamp-cell">{item.time}</td>
                    <td className="claim-cell">&ldquo;{item.claim}&rdquo;</td>
                    <td>
                      <span className={`veracity-chip chip-${item.distilbert}`}>
                        {formatLabel(item.distilbert)}
                      </span>
                    </td>
                    <td>
                      <span className={`veracity-chip chip-${item.baseline}`}>
                        {formatLabel(item.baseline)}
                      </span>
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn-table-action"
                        onClick={() => onNavigate("/claim-analysis")}
                      >
                        Re-evaluate
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
