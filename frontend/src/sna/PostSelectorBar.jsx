import { useState, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const RUMOR_LABELS = [
  { id: "all", label: "All Veracity" },
  { id: "false", label: "False Rumor" },
  { id: "true", label: "True Rumor" },
  { id: "unverified", label: "Unverified" },
  { id: "non-rumor", label: "Non-Rumor" },
];

export default function PostSelectorBar({
  selectedDataset,
  selectedPostId,
  onDatasetChange,
  onPostSelect,
}) {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeLabel, setActiveLabel] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [totalCount, setTotalCount] = useState(0);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError("");

    const params = new URLSearchParams({
      dataset: selectedDataset,
      label: activeLabel,
      limit: "100",
    });
    if (searchTerm.trim()) {
      params.set("query", searchTerm.trim());
    }

    fetch(`${API_URL}/api/sna/posts?${params.toString()}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!isMounted) return;
        setPosts(data.posts || []);
        setTotalCount(data.total || 0);

        if (data.posts && data.posts.length > 0) {
          const exists = data.posts.some((p) => p.post_id === selectedPostId);
          if (!exists) {
            onPostSelect(data.posts[0].post_id, data.posts[0]);
          }
        }
      })
      .catch((err) => {
        if (isMounted) {
          console.error("Failed to fetch SNA posts:", err);
          setError("Failed to load posts from backend API.");
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [selectedDataset, activeLabel, searchTerm]);

  const currentPost = posts.find((p) => p.post_id === selectedPostId) || posts[0];

  return (
    <div className="sna-selector-card">
      <div className="sna-selector-controls-row">
        {/* Dataset Selection */}
        <div className="sna-control-group">
          <label className="sna-control-label">Dataset Benchmark:</label>
          <div className="sna-pill-group">
            <button
              type="button"
              className={`sna-pill-btn ${selectedDataset === "twitter15" ? "active" : ""}`}
              onClick={() => onDatasetChange("twitter15")}
            >
              Twitter15
            </button>
            <button
              type="button"
              className={`sna-pill-btn ${selectedDataset === "twitter16" ? "active" : ""}`}
              onClick={() => onDatasetChange("twitter16")}
            >
              Twitter16
            </button>
          </div>
        </div>

        {/* Veracity Filter */}
        <div className="sna-control-group">
          <label className="sna-control-label">Filter by Label:</label>
          <div className="sna-filter-buttons">
            {RUMOR_LABELS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`sna-filter-btn ${activeLabel === item.id ? "active" : ""}`}
                onClick={() => setActiveLabel(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Post Dropdown and Search */}
      <div className="sna-cascade-picker">
        <label className="sna-control-label">
          Select Cascade ({totalCount} Available):
        </label>
        <div className="sna-picker-inputs">
          <select
            className="sna-post-select"
            value={selectedPostId || (currentPost?.post_id ?? "")}
            onChange={(e) => {
              const post = posts.find((p) => p.post_id === e.target.value);
              onPostSelect(e.target.value, post);
            }}
            disabled={loading || posts.length === 0}
          >
            {posts.map((p) => (
              <option key={p.post_id} value={p.post_id}>
                [{p.label.toUpperCase()}] #{p.post_id} — {p.text_preview}
              </option>
            ))}
          </select>

          <input
            type="text"
            className="sna-search-input"
            placeholder="Search keyword or post ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {/* Selected Post Summary Box */}
      {currentPost && (
        <div className="sna-selected-post-box">
          <div className="sna-post-meta-badges">
            <span className={`sna-veracity-badge badge-${currentPost.label}`}>
              {currentPost.label.toUpperCase()}
            </span>
            <span className="sna-meta-tag">Post ID: {currentPost.post_id}</span>
            <span className="sna-meta-tag">Dataset: {selectedDataset.toUpperCase()}</span>
          </div>
          <p className="sna-selected-post-text">&ldquo;{currentPost.text}&rdquo;</p>
        </div>
      )}

      {error && <div className="sna-error-pill">{error}</div>}
    </div>
  );
}
