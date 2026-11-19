import { useState, useEffect } from "react";
import ClaimInputCard from "../components/ClaimInputCard";
import { API_URL, formatDateTime } from "../utils/constants";

export default function HistoricalInfluencePage({
  claim,
  setClaim,
  onNavigate,
}) {
  const [influenceData, setInfluenceData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [influenceStats, setInfluenceStats] = useState(null);

  useEffect(() => {
    let isMounted = true;
    fetch(`${API_URL}/influence/stats`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted && data) setInfluenceStats(data);
      })
      .catch((err) => console.warn("Could not load influence stats:", err));

    return () => {
      isMounted = false;
    };
  }, []);

  const handleAnalyzeInfluence = async (e) => {
    if (e) e.preventDefault();
    const targetClaim = claim.trim();

    if (!targetClaim) {
      setError("Please enter a claim or headline to evaluate historical influence.");
      return;
    }
    if (targetClaim.length < 3) {
      setError("Query must be at least 3 characters long.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/influence/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          claim: targetClaim,
          top_k: 3,
          min_score: 0.08,
        }),
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(data?.detail || data?.error || `Influence search returned HTTP ${response.status}`);
      }

      setInfluenceData(data);
    } catch (err) {
      console.error("Influence analysis failed:", err);
      setError(
        err?.message?.includes("Failed to fetch")
          ? "Unable to reach the influence analysis server. Ensure FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const results = influenceData?.results || [];

  return (
    <div className="page-container influence-page">
      {/* Header */}
      <div className="page-header">
        <div className="header-breadcrumbs">
          <button type="button" className="crumb-link" onClick={() => onNavigate("/")}>
            Dashboard
          </button>
          <span className="crumb-sep">/</span>
          <span className="crumb-current">Social Influence</span>
        </div>
        <h1>Social Influence</h1>
        <p className="page-description">
          Analyze historical engagement volume, viral velocity, and temporal propagation lifespans for news stories.
        </p>
      </div>

      {/* Corpus Status Strip */}
      {influenceStats && (
        <div className="corpus-info-strip">
          <div className="corpus-stat-item">
            <span className="corpus-stat-label">Verified Archive</span>
            <strong>{influenceStats.total_stories?.toLocaleString()} News Stories</strong>
          </div>
          <div className="corpus-stat-item">
            <span className="corpus-stat-label">Referenced Activity</span>
            <strong>{influenceStats.total_tweet_ids?.toLocaleString()} Captured Tweets</strong>
          </div>
          <div className="corpus-stat-item">
            <span className="corpus-stat-label">Temporal Decoding</span>
            <strong>64-Bit Timestamp Precision</strong>
          </div>
        </div>
      )}

      {/* Claim / Headline Input */}
      <ClaimInputCard
        claim={claim}
        onClaimChange={setClaim}
        onSubmit={handleAnalyzeInfluence}
        loading={loading}
        error={error}
        onClear={() => {
          setClaim("");
          setInfluenceData(null);
          setError("");
        }}
        buttonLabel="Analyze Social Engagement"
        placeholder="Enter a news headline, topic, or claim keywords to evaluate against historical social engagement..."
      />

      {/* Results Section */}
      {influenceData && (
        <div className="influence-results-section">
          <div className="results-header-row">
            <div>
              <h3>Matched News Narratives</h3>
              <span className="results-subtext">
                Found {results.length} stories matching headline relevance
              </span>
            </div>
          </div>

          {results.length === 0 ? (
            <div className="empty-results-box">
              <p>No matching stories found in the historical news archive.</p>
              <span className="empty-hint">Try testing another headline or broader policy keywords.</span>
            </div>
          ) : (
            <div className="influence-cards-list">
              {results.map((story, idx) => {
                const isFake = story.dataset_label === "fake";
                const temporal = story.temporal_metrics;

                return (
                  <article key={story.story_id || idx} className="influence-story-card">
                    <div className="story-top-row">
                      <div className="story-badge-group">
                        <span className={`story-veracity-badge ${isFake ? "badge-fake" : "badge-real"}`}>
                          {isFake ? "Misinformation Narrative" : "Verified Real Narrative"}
                        </span>
                        {story.domain && (
                          <span className="domain-pill-tag">{story.domain}</span>
                        )}
                      </div>

                      <div className="relevance-score-badge">
                        <span>Headline Match:</span>
                        <strong>{(story.similarity_score * 100).toFixed(1)}%</strong>
                      </div>
                    </div>

                    <div className="story-headline-box">
                      <span className="headline-caption">Matched Headline:</span>
                      <h3>&ldquo;{story.title}&rdquo;</h3>
                    </div>

                    {/* Engagement Metrics Grid */}
                    <div className="engagement-metrics-grid">
                      <div className="engagement-metric-item">
                        <span className="metric-title">Observed Tweet Volume</span>
                        <strong className="metric-value">{story.observed_tweet_count?.toLocaleString()}</strong>
                        <span className="metric-caption">Captured public citations</span>
                      </div>

                      <div className="engagement-metric-item">
                        <span className="metric-title">Relative to False Narratives</span>
                        <strong className="metric-value">{story.percentile_among_fake}% Percentile</strong>
                        <span className="metric-caption">Higher than {story.percentile_among_fake}% of false stories</span>
                      </div>

                      <div className="engagement-metric-item">
                        <span className="metric-title">Relative to Real Narratives</span>
                        <strong className="metric-value">{story.percentile_among_real}% Percentile</strong>
                        <span className="metric-caption">Higher than {story.percentile_among_real}% of real stories</span>
                      </div>
                    </div>

                    {/* Decoded Snowflake Temporal Profile */}
                    {temporal && temporal.has_temporal_data ? (
                      <div className="temporal-profile-card">
                        <div className="temporal-card-header">
                          <h4>Temporal Profile</h4>
                          <span className="temporal-badge">
                            {temporal.valid_snowflake_ids?.toLocaleString()} Timestamps Decoded
                          </span>
                        </div>

                        <div className="temporal-grid">
                          <div className="temporal-item">
                            <span className="temporal-label">Earliest Tweet:</span>
                            <strong>{formatDateTime(temporal.earliest_recorded_tweet_utc)}</strong>
                          </div>
                          <div className="temporal-item">
                            <span className="temporal-label">Latest Tweet:</span>
                            <strong>{formatDateTime(temporal.latest_recorded_tweet_utc)}</strong>
                          </div>
                          <div className="temporal-item">
                            <span className="temporal-label">Observed Lifespan:</span>
                            <strong>{temporal.observed_span_days} days ({temporal.observed_span_hours} hrs)</strong>
                          </div>
                          <div className="temporal-item">
                            <span className="temporal-label">First 24-Hour Activity:</span>
                            <strong>{temporal.first_24h_percentage}% ({temporal.first_24h_tweets_count} tweets)</strong>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="temporal-missing-box">
                        <p>Temporal profile unavailable for this record.</p>
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
