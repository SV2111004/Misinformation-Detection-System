import { useState, useEffect } from "react";
import ClaimInputCard from "../components/ClaimInputCard";
import {
  API_URL,
  getLabelInfo,
  formatSpeaker,
  formatAffiliation,
} from "../utils/constants";

export default function EvidenceResearchPage({
  claim,
  setClaim,
  onNavigate,
}) {
  const [evidenceData, setEvidenceData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [evidenceStats, setEvidenceStats] = useState(null);

  // Filters
  const [topK, setTopK] = useState(5);
  const [minScore, setMinScore] = useState(0.08);

  useEffect(() => {
    let isMounted = true;
    fetch(`${API_URL}/evidence/stats`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted && data) setEvidenceStats(data);
      })
      .catch((err) => console.warn("Could not load evidence stats:", err));

    return () => {
      isMounted = false;
    };
  }, []);

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    const targetClaim = claim.trim();

    if (!targetClaim) {
      setError("Please enter a claim or query to search evidence.");
      return;
    }
    if (targetClaim.length < 3) {
      setError("Query must be at least 3 characters long.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/evidence/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          claim: targetClaim,
          top_k: Number(topK),
          min_score: Number(minScore),
        }),
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(data?.detail || data?.error || `Evidence search returned HTTP ${response.status}`);
      }

      setEvidenceData(data);
    } catch (err) {
      console.error("Evidence search failed:", err);
      setError(
        err?.message?.includes("Failed to fetch")
          ? "Unable to reach the evidence retrieval server. Make sure FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const results = evidenceData?.results || [];

  return (
    <div className="page-container evidence-page">
      {/* Header */}
      <div className="page-header">
        <div className="header-breadcrumbs">
          <button type="button" className="crumb-link" onClick={() => onNavigate("/")}>
            Dashboard
          </button>
          <span className="crumb-sep">/</span>
          <span className="crumb-current">Evidence &amp; RAG Research</span>
        </div>
        <h1>PolitiFact Evidence Retrieval &amp; Semantic RAG</h1>
        <p className="page-description">
          Queries 5,562 verified PolitiFact rulings in the canonical socioeconomic training corpus to retrieve precedent claims, published journalistic justifications, and factual rulings.
        </p>
      </div>

      {/* Corpus Status Strip */}
      {evidenceStats && (
        <div className="corpus-info-strip">
          <div className="corpus-stat-item">
            <span className="corpus-stat-label">Indexed Corpus</span>
            <strong>{evidenceStats.total_indexed?.toLocaleString()} PolitiFact Records</strong>
          </div>
          <div className="corpus-stat-item">
            <span className="corpus-stat-label">Benchmark Split</span>
            <strong>LIAR-PLUS ({evidenceStats.split || "Train"})</strong>
          </div>
          <div className="corpus-stat-item">
            <span className="corpus-stat-label">Vector Store</span>
            <strong>ChromaDB + MiniLM Embeddings</strong>
          </div>
          <div className="corpus-stat-item">
            <span className="corpus-stat-label">Retrieval Mode</span>
            <strong>Semantic Similarity (ChromaDB Embeddings)</strong>
          </div>
        </div>
      )}

      {/* Claim / Query Input */}
      <ClaimInputCard
        claim={claim}
        onClaimChange={setClaim}
        onSubmit={handleSearch}
        loading={loading}
        error={error}
        onClear={() => {
          setClaim("");
          setEvidenceData(null);
          setError("");
        }}
        buttonLabel="Retrieve Precedent Evidence"
        placeholder="Enter a statement, political topic, or claim keywords to retrieve corroborating or debunking evidence..."
      />

      {/* Search Filter Controls Bar */}
      <div className="evidence-controls-card">
        <div className="control-field">
          <label htmlFor="top-k-select">Retrieval Limit (Top-K):</label>
          <select
            id="top-k-select"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="select-input"
          >
            <option value={3}>Top 3 Citations</option>
            <option value={5}>Top 5 Citations</option>
            <option value={10}>Top 10 Citations</option>
          </select>
        </div>

        <div className="control-field">
          <label htmlFor="min-score-slider">
            Minimum Relevance Threshold: <strong>{(minScore * 100).toFixed(0)}%</strong>
          </label>
          <input
            id="min-score-slider"
            type="range"
            min="0.02"
            max="0.40"
            step="0.02"
            value={minScore}
            onChange={(e) => setMinScore(Number(e.target.value))}
            className="range-input"
          />
        </div>
      </div>

      {/* Results Section */}
      {evidenceData && (
        <div className="evidence-results-section">
          <div className="results-header-row">
            <div>
              <h3>Retrieved Benchmark Records</h3>
              <span className="results-subtext">
                Found {results.length} historical statements matching query relevance &gt;= {(minScore * 100).toFixed(0)}%
              </span>
            </div>
            {evidenceData.evidence_status && (
              <span className={`grounded-status-tag status-${evidenceData.evidence_status}`}>
                Status: {evidenceData.evidence_status.toUpperCase()}
              </span>
            )}
          </div>

          {/* Grounded Summary if present */}
          {evidenceData.grounded_summary && (
            <div className="grounded-summary-card">
              <div className="summary-card-header">
                <span className="summary-icon">📄</span>
                <h4>Grounded RAG Evidence Synthesis</h4>
              </div>
              <p className="summary-body-text">{evidenceData.grounded_summary}</p>
            </div>
          )}

          {results.length === 0 ? (
            <div className="empty-results-box">
              <p>No historical statements in the PolitiFact corpus met the {Math.round(minScore * 100)}% relevance threshold.</p>
              <span className="empty-hint">Try lowering the minimum threshold slider above or expanding your claim keywords.</span>
            </div>
          ) : (
            <div className="evidence-cards-list">
              {results.map((item, idx) => {
                const verdictInfo = getLabelInfo(item.verdict);
                const speakerName = formatSpeaker(item.speaker);
                const speakerMeta = [
                  item.speaker_job,
                  formatAffiliation(item.party_affiliation),
                  item.state_info,
                ]
                  .filter(Boolean)
                  .join(" · ");

                const similarityPct =
                  item.similarity_score !== undefined
                    ? (item.similarity_score * 100).toFixed(1)
                    : null;

                return (
                  <article key={item.statement_id || idx} className="evidence-card-item">
                    <div className="card-top-bar">
                      <div className="card-tags-group">
                        <span className="dataset-source-tag">
                          {item.source_dataset || "LIAR-PLUS"} · {item.split || "train"}
                        </span>
                        {item.has_context_alignment ? (
                          <span className="context-badge badge-aligned" title="Keywords match justification context">
                            ✓ Context: Aligned
                          </span>
                        ) : (
                          <span className="context-badge badge-anomaly" title="Raw TSV benchmark anomaly: excerpt may correspond to adjacent row">
                            ⚠️ Context: Upstream Unverified
                          </span>
                        )}
                      </div>

                      {similarityPct && (
                        <div className="similarity-badge">
                          <span className="sim-label">Semantic Similarity</span>
                          <strong>{similarityPct}%</strong>
                        </div>
                      )}
                    </div>

                    {/* Historical Verdict Banner */}
                    <div className="historical-verdict-row">
                      <span className="historical-caption">POLITIFACT HISTORICAL VERDICT:</span>
                      <span className={`veracity-chip chip-${item.verdict}`}>
                        {verdictInfo.title}
                      </span>
                      <span className="verdict-note-text">
                        *PolitiFact's ruling for the historical statement below.*
                      </span>
                    </div>

                    {/* Speaker Header */}
                    <div className="speaker-header-row">
                      <div className="speaker-avatar-circle">
                        {speakerName ? speakerName.charAt(0).toUpperCase() : "P"}
                      </div>
                      <div>
                        <strong className="speaker-name">{speakerName}</strong>
                        <span className="speaker-subtitle">{speakerMeta || "Political Public Figure"}</span>
                      </div>
                    </div>

                    {/* Historical Statement */}
                    <div className="claim-quote-box">
                      <span className="quote-label">Historical Statement:</span>
                      <p>&ldquo;{item.claim}&rdquo;</p>
                    </div>

                    {/* Published Justification */}
                    <div className="justification-quote-box">
                      <span className="quote-label">Published PolitiFact Justification:</span>
                      <p>{item.justification || "No justification text available for this record."}</p>
                    </div>

                    {/* Footer Tags */}
                    <div className="card-footer-meta">
                      <div className="subject-tags-row">
                        {item.subject && (
                          <span className="meta-tag">
                            Topic: {item.subject.split(",").slice(0, 3).join(", ")}
                          </span>
                        )}
                        {item.context && (
                          <span className="meta-tag" title={item.context}>
                            Venue: {item.context.length > 35 ? `${item.context.slice(0, 35)}...` : item.context}
                          </span>
                        )}
                      </div>
                      {item.statement_id && (
                        <span className="statement-id-tag">ID: {item.statement_id}</span>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}

          {/* Academic Transparency Box */}
          <div className="academic-disclaimer-box">
            <h4>Academic Boundary &amp; Benchmark Limitations Notice</h4>
            <p>
              Evidence records reflect published fact-checks from the LIAR-PLUS dataset (Wang et al., ACL 2017 / Alhindi et al., EMNLP 2018).
              These historical findings describe prior political claims analyzed by PolitiFact journalists and provide precedent context.
              They serve as evidentiary references, not an automated ruling on newly submitted statements.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
