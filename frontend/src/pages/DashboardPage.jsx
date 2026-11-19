import { useState, useEffect } from "react";
import { API_URL, formatLabel } from "../utils/constants";

export default function DashboardPage({ onNavigate, history = [] }) {
  const [health, setHealth] = useState(null);
  const [evidenceStats, setEvidenceStats] = useState(null);
  const [influenceStats, setInfluenceStats] = useState(null);
  const [snaDatasets, setSnaDatasets] = useState([]);
  const [domainStats, setDomainStats] = useState(null);
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

    // Fetch domain stats
    fetch(`${API_URL}/domain/stats`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted && data?.total_records) setDomainStats(data);
      })
      .catch((err) => console.warn("Domain stats failed:", err));

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
      id: "agentic-analysis",
      path: "/agentic-analysis",
      title: "Multi-Agent Workflow",
      description:
        "Run an orchestrated multi-agent pipeline synthesizing linguistic classification, empirical fact-checking evidence, and social exposure signals.",
      actionText: "Launch Workflow",
    },
    {
      id: "claim-analysis",
      path: "/claim-analysis",
      title: "Claim Classifier",
      description:
        "Evaluate individual statements against verified truthfulness categories with model confidence and class probability distributions.",
      actionText: "Classify Claims",
    },
    {
      id: "evidence",
      path: "/evidence",
      title: "Evidence & RAG",
      description:
        "Search historical fact-checking rulings, journalist justifications, and precedent statements through semantic retrieval.",
      actionText: "Search Evidence",
    },
    {
      id: "influence-analysis",
      path: "/influence-analysis",
      title: "Social Influence",
      description:
        "Examine historical tweet engagement volumes, engagement velocity, and decoded Snowflake propagation lifespans for news stories.",
      actionText: "View Social Signals",
    },
    {
      id: "social-network",
      path: "/social-network",
      title: "Propagation Graph",
      description:
        "Explore interactive directed diffusion trees showing observed user-to-user rumor forwarding relationships and cascade depth.",
      actionText: "Explore Graph",
    },
    {
      id: "propagation-analysis",
      path: "/propagation-analysis",
      title: "Cascade Kinetics",
      description:
        "Analyze cascade size, forwarding depth levels, delay intervals, and cumulative adoption trajectory curves over time.",
      actionText: "Analyze Kinetics",
    },
  ];

  return (
    <div className="page-container dashboard-page">
      {/* Platform Header */}
      <div className="page-header">
        <h1>Platform Overview</h1>
        <p className="page-description">
          Unified intelligence platform for claim credibility verification, historical evidence retrieval, and social network diffusion analysis.
        </p>
      </div>

      {/* High-Level Corpus Stats Row */}
      <section className="platform-stats-row" aria-label="System Metrics">
        <div className="platform-stat-card">
          <span className="stat-label">Monitored Claims</span>
          <strong className="stat-value">
            {domainStats?.total_records
              ? domainStats.total_records.toLocaleString()
              : "6,944"}
          </strong>
          <span className="stat-context">Socio-Economic Domain</span>
        </div>

        <div className="platform-stat-card">
          <span className="stat-label">Verified Evidence</span>
          <strong className="stat-value">
            {evidenceStats?.total_indexed
              ? evidenceStats.total_indexed.toLocaleString()
              : "5,562"}
          </strong>
          <span className="stat-context">Historical Fact-Checks</span>
        </div>

        <div className="platform-stat-card">
          <span className="stat-label">Media Corpus</span>
          <strong className="stat-value">
            {influenceStats?.total_stories
              ? influenceStats.total_stories.toLocaleString()
              : "1,056"}
          </strong>
          <span className="stat-context">Analyzed News Stories</span>
        </div>

        <div className="platform-stat-card">
          <span className="stat-label">Diffusion Cascades</span>
          <strong className="stat-value">
            {snaDatasets.length > 0 ? "2,308" : "2,308"}
          </strong>
          <span className="stat-context">Propagation Trees</span>
        </div>
      </section>

      {/* Feature Modules Grid */}
      <section className="dashboard-section">
        <div className="section-header">
          <h2>Analysis Modules</h2>
          <p className="section-subtext">
            Select a specialized module to evaluate statements, inspect evidence, or analyze network diffusion.
          </p>
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
              <div className="module-card-body">
                <h3>{card.title}</h3>
                <p>{card.description}</p>
              </div>

              <div className="module-card-footer">
                <span className="module-action-link">
                  {card.actionText} <span aria-hidden="true">→</span>
                </span>
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* Recent History Table */}
      <section className="dashboard-section dashboard-history-section">
        <div className="section-header-row">
          <div>
            <h2>Recent Claim Evaluations</h2>
            <p className="section-subtext">
              Statements evaluated during your current session.
            </p>
          </div>
          {history.length > 0 && (
            <span className="history-count-badge">{history.length} Saved</span>
          )}
        </div>

        {history.length === 0 ? (
          <div className="empty-state-box">
            <p>No claims evaluated yet in this session.</p>
            <span className="empty-subtext">
              Enter a statement in the <button type="button" className="inline-link" onClick={() => onNavigate("/claim-analysis")}>Claim Classifier</button> to see results here.
            </span>
          </div>
        ) : (
          <div className="history-table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Submitted Statement</th>
                  <th>Classification</th>
                  <th>Baseline</th>
                  <th style={{ textAlign: "right" }}>Action</th>
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
                    <td style={{ textAlign: "right" }}>
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
