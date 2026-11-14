import { useState, useEffect } from "react";
import PostSelectorBar from "./PostSelectorBar";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function PropagationAnalysisPage({ onBack }) {
  const [dataset, setDataset] = useState("twitter15");
  const [postId, setPostId] = useState("");

  const [propData, setPropData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!postId) return;
    let isMounted = true;
    setLoading(true);
    setError("");

    fetch(`${API_URL}/api/sna/propagation/${dataset}/${postId}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!isMounted) return;
        setPropData(data);
      })
      .catch((err) => {
        if (isMounted) {
          console.error("Failed to fetch propagation analysis:", err);
          setError("Failed to calculate cascade propagation statistics.");
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [dataset, postId]);

  return (
    <div className="sna-page-container">
      {/* Page Header */}
      <div className="sna-page-header">
        <button type="button" className="sna-back-button" onClick={onBack}>
          ← Back to Dashboard
        </button>
        <div className="sna-header-titles">
          <h1>Propagation Analysis</h1>
          <p>
            Summarizes how the selected post propagated through the observed cascade across depth levels and time.
          </p>
        </div>
      </div>

      {/* Dataset & Post Selection */}
      <PostSelectorBar
        selectedDataset={dataset}
        selectedPostId={postId}
        onDatasetChange={(ds) => setDataset(ds)}
        onPostSelect={(id) => setPostId(id)}
      />

      {loading && (
        <div className="sna-loading-state">
          <span className="spinner" />
          <p>Loading propagation statistics...</p>
        </div>
      )}

      {error && <div className="sna-error-pill">{error}</div>}

      {!loading && !error && propData && (
        <>
          {/* Summary Metric Cards */}
          <div className="sna-kpi-grid">
            <div className="sna-kpi-card">
              <span className="sna-kpi-label">Cascade Size</span>
              <strong className="sna-kpi-number">{propData.total_nodes}</strong>
              <span className="sna-kpi-sub">Total recorded users</span>
            </div>

            <div className="sna-kpi-card">
              <span className="sna-kpi-label">Propagation Edges</span>
              <strong className="sna-kpi-number">{propData.total_edges}</strong>
              <span className="sna-kpi-sub">Forwarding links</span>
            </div>

            <div className="sna-kpi-card">
              <span className="sna-kpi-label">Maximum Depth</span>
              <strong className="sna-kpi-number">{propData.max_depth} Hops</strong>
              <span className="sna-kpi-sub">Longest forwarding chain</span>
            </div>

            <div className="sna-kpi-card">
              <span className="sna-kpi-label">Median Delay</span>
              <strong className="sna-kpi-number">
                {propData.delay_stats?.median_minutes} min
              </strong>
              <span className="sna-kpi-sub">
                ~{Math.round(((propData.delay_stats?.median_minutes || 0) / 60) * 10) / 10} hrs to 50% spread
              </span>
            </div>
          </div>

          {/* Two-Column Charts: Depth Hierarchy & Time Bins */}
          <div className="sna-two-col-grid">
            {/* Chart 1: Depth Levels */}
            <div className="sna-content-card">
              <div className="sna-card-header">
                <h3>Cascade Depth Levels</h3>
                <span className="sna-subtext">Participants reached at each forwarding hop</span>
              </div>

              <div className="sna-depth-chart">
                {propData.depth_levels?.map((lvl) => (
                  <div key={lvl.level} className="sna-depth-row">
                    <div className="sna-depth-meta">
                      <span className="sna-depth-name">{lvl.name}</span>
                      <span className="sna-depth-counts">
                        <strong>{lvl.count}</strong> ({lvl.percentage}%)
                      </span>
                    </div>
                    <div className="sna-bar-track">
                      <div
                        className="sna-bar-fill"
                        style={{
                          width: `${Math.max(4, lvl.percentage)}%`,
                          background:
                            lvl.level === 0
                              ? "#d97706"
                              : lvl.level === 1
                              ? "#2563eb"
                              : lvl.level === 2
                              ? "#7c3aed"
                              : "#db2777",
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Chart 2: Time Intervals */}
            <div className="sna-content-card">
              <div className="sna-card-header">
                <h3>Temporal Adoption Intervals</h3>
                <span className="sna-subtext">Distribution across time elapsed from publication</span>
              </div>

              <div className="sna-timebins-chart">
                {propData.time_bins?.map((bin) => (
                  <div key={bin.label} className="sna-bin-row">
                    <span className="sna-bin-label">{bin.label}</span>
                    <div className="sna-bin-track">
                      <div
                        className="sna-bin-fill"
                        style={{
                          width: `${Math.max(3, bin.percentage)}%`,
                          background: "#3b82f6",
                        }}
                      >
                        {bin.count > 0 && <span className="sna-bin-val">{bin.count}</span>}
                      </div>
                    </div>
                    <span className="sna-bin-pct">{bin.percentage}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Full-Width Chart: Cumulative Adoption Curve */}
          <div className="sna-content-card">
            <div className="sna-card-header-flex">
              <div>
                <h3>Cumulative Adoption Trajectory</h3>
                <span className="sna-subtext">Cumulative percentage of participants reached over time</span>
              </div>
              <div className="sna-velocity-summary">
                <span>
                  <strong>50% Reached:</strong> {propData.delay_stats?.time_to_50_pct_minutes} min
                </span>
                <span>
                  <strong>80% Reached:</strong> {propData.delay_stats?.time_to_80_pct_minutes} min
                </span>
              </div>
            </div>

            <div className="sna-timeline-svg-wrapper">
              <svg className="sna-timeline-svg" viewBox="0 0 800 240">
                <defs>
                  <linearGradient id="curveFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#2563eb" stopOpacity="0.25" />
                    <stop offset="100%" stopColor="#2563eb" stopOpacity="0.02" />
                  </linearGradient>
                </defs>

                {/* Grid lines */}
                <line x1="50" y1="30" x2="760" y2="30" stroke="#f1f5f9" strokeWidth="1" />
                <line x1="50" y1="80" x2="760" y2="80" stroke="#f1f5f9" strokeWidth="1" />
                <line x1="50" y1="130" x2="760" y2="130" stroke="#f1f5f9" strokeWidth="1" />
                <line x1="50" y1="180" x2="760" y2="180" stroke="#cbd5e1" strokeWidth="1" />

                {/* Y-axis labels */}
                <text x="40" y="34" textAnchor="end" fontSize="11" fill="#64748b">100%</text>
                <text x="40" y="109" textAnchor="end" fontSize="11" fill="#64748b">50%</text>
                <text x="40" y="184" textAnchor="end" fontSize="11" fill="#64748b">0%</text>

                {/* Area and Line plotting */}
                {(() => {
                  const pts = propData.cumulative_timeline || [];
                  if (pts.length < 2) return null;

                  const minX = 50;
                  const maxX = 760;
                  const minY = 30;
                  const maxY = 180;
                  const maxDelay = Math.max(1, pts[pts.length - 1].delay_minutes);

                  const coords = pts.map((pt) => ({
                    x: minX + (pt.delay_minutes / maxDelay) * (maxX - minX),
                    y: maxY - (pt.percentage / 100) * (maxY - minY),
                    pt,
                  }));

                  const lineD = coords.reduce(
                    (acc, c, idx) => (idx === 0 ? `M ${c.x} ${c.y}` : `${acc} L ${c.x} ${c.y}`),
                    ""
                  );
                  const areaD = `${lineD} L ${coords[coords.length - 1].x} ${maxY} L ${coords[0].x} ${maxY} Z`;

                  return (
                    <g>
                      <path d={areaD} fill="url(#curveFill)" />
                      <path d={lineD} fill="none" stroke="#2563eb" strokeWidth="2" />
                      {coords.map((c, i) => (
                        <circle
                          key={i}
                          cx={c.x}
                          cy={c.y}
                          r={i === 0 || i === coords.length - 1 ? 4 : 2}
                          fill="#ffffff"
                          stroke="#2563eb"
                          strokeWidth="2"
                        />
                      ))}
                    </g>
                  );
                })()}

                {/* X-axis labels */}
                <text x="50" y="206" fontSize="11" fill="#64748b">
                  t = 0 min (Source Post)
                </text>
                <text x="760" y="206" textAnchor="end" fontSize="11" fill="#64748b">
                  t = {propData.delay_stats?.max_minutes} min (Last Observed Forward)
                </text>
              </svg>
            </div>
          </div>
        </>
      )}

      {/* Academic Note Footer */}
      <div className="sna-footnote">
        <p>
          Note: Propagation statistics reflect recorded events in the benchmark dataset. They measure forwarding dynamics
          among participating users and do not represent total impressions or passive readership.
        </p>
      </div>
    </div>
  );
}
