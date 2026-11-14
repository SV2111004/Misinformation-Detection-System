import { useState, useEffect, useMemo } from "react";
import PostSelectorBar from "./PostSelectorBar";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const DEPTH_COLORS = [
  "#d97706", // Root (Hop 0): Amber / Gold
  "#2563eb", // Level 1 (Direct retweets): Royal Blue
  "#7c3aed", // Level 2 (Secondary spread): Purple
  "#db2777", // Level 3: Rose
  "#059669", // Level 4+: Emerald
];

function getDepthColor(depth) {
  if (depth === 0) return DEPTH_COLORS[0];
  return DEPTH_COLORS[Math.min(depth, DEPTH_COLORS.length - 1)];
}

export default function SocialNetworkPage({ onBack }) {
  const [dataset, setDataset] = useState("twitter15");
  const [postId, setPostId] = useState("");
  const [maxNodes, setMaxNodes] = useState(120);

  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [selectedNode, setSelectedNode] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!postId) return;
    let isMounted = true;
    setLoading(true);
    setError("");

    fetch(`${API_URL}/api/sna/graph/${dataset}/${postId}?max_nodes=${maxNodes}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!isMounted) return;
        setGraphData(data);
        const root = data.nodes.find((n) => n.is_root) || data.nodes[0];
        setSelectedNode(root || null);
        setZoom(1);
        setPan({ x: 0, y: 0 });
      })
      .catch((err) => {
        if (isMounted) {
          console.error("Failed to fetch propagation graph:", err);
          setError("Failed to construct propagation graph for the selected post.");
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [dataset, postId, maxNodes]);

  // Compute radial layout positions in SVG space (900x650)
  const layout = useMemo(() => {
    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
      return { nodePositions: {}, width: 900, height: 650, centerX: 450, centerY: 325 };
    }

    const width = 900;
    const height = 650;
    const centerX = width / 2;
    const centerY = height / 2;

    const nodePositions = {};
    const depthGroups = {};

    graphData.nodes.forEach((node) => {
      const d = node.depth || 0;
      if (!depthGroups[d]) depthGroups[d] = [];
      depthGroups[d].push(node);
    });

    // Root at center
    if (depthGroups[0] && depthGroups[0].length > 0) {
      depthGroups[0].forEach((node) => {
        nodePositions[node.id] = { x: centerX, y: centerY };
      });
    }

    // Concentric radial rings for subsequent depths
    const sortedDepths = Object.keys(depthGroups)
      .map(Number)
      .filter((d) => d > 0)
      .sort((a, b) => a - b);

    const baseRadius = 110;
    const radiusStep = 90;

    sortedDepths.forEach((depth, dIndex) => {
      const group = depthGroups[depth];
      const radius = baseRadius + dIndex * radiusStep;
      const count = group.length;

      group.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / count + depth * 0.4;
        const r = radius + ((node.out_degree % 4) - 2) * 10;
        nodePositions[node.id] = {
          x: centerX + r * Math.cos(angle),
          y: centerY + r * Math.sin(angle),
        };
      });
    });

    return { nodePositions, width, height, centerX, centerY };
  }, [graphData]);

  // Pan handlers
  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div className="sna-page-container">
      {/* Page Header */}
      <div className="sna-page-header">
        <button type="button" className="sna-back-button" onClick={onBack}>
          ← Back to Dashboard
        </button>
        <div className="sna-header-titles">
          <h1>Social Network Propagation Graph</h1>
          <p>
            Visualizes the observed spread of a selected post across users in the rumor propagation tree.
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

      {/* Graph Controls Bar */}
      <div className="sna-graph-controls-card">
        <div className="sna-controls-left">
          <span className="sna-control-label">Node Limit:</span>
          <div className="sna-pill-group">
            {[60, 100, 150, 250].map((limit) => (
              <button
                key={limit}
                type="button"
                className={`sna-pill-btn ${maxNodes === limit ? "active" : ""}`}
                onClick={() => setMaxNodes(limit)}
              >
                {limit}
              </button>
            ))}
          </div>
          {graphData?.is_truncated && (
            <span className="sna-truncation-badge">
              Showing {graphData.displayed_nodes} of {graphData.total_nodes} nodes
            </span>
          )}
        </div>

        <div className="sna-controls-right">
          <button
            type="button"
            className="sna-zoom-btn"
            onClick={() => setZoom((z) => Math.min(2.5, z + 0.2))}
            title="Zoom In"
          >
            + Zoom In
          </button>
          <button
            type="button"
            className="sna-zoom-btn"
            onClick={() => setZoom((z) => Math.max(0.4, z - 0.2))}
            title="Zoom Out"
          >
            − Zoom Out
          </button>
          <button
            type="button"
            className="sna-zoom-btn"
            onClick={() => {
              setZoom(1);
              setPan({ x: 0, y: 0 });
            }}
            title="Reset View"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Main Graph & Inspector Grid */}
      <div className="sna-graph-layout-grid">
        {/* Interactive SVG Canvas */}
        <div
          className="sna-canvas-card"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {loading && (
            <div className="sna-canvas-overlay">
              <span className="spinner" />
              <p>Loading propagation tree...</p>
            </div>
          )}

          {error && (
            <div className="sna-canvas-overlay error">
              <p>{error}</p>
            </div>
          )}

          {!loading && !error && graphData && graphData.nodes.length === 0 && (
            <div className="sna-canvas-overlay">
              <p>No propagation nodes available for this post.</p>
            </div>
          )}

          {graphData && graphData.nodes.length > 0 && (
            <svg
              className="sna-graph-svg"
              viewBox={`0 0 ${layout.width} ${layout.height}`}
              style={{ cursor: isDragging ? "grabbing" : "grab" }}
            >
              <defs>
                <marker
                  id="edge-arrow"
                  viewBox="0 0 10 10"
                  refX="18"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1 L 9 5 L 0 9 z" fill="#94a3b8" />
                </marker>
                <marker
                  id="edge-arrow-active"
                  viewBox="0 0 10 10"
                  refX="18"
                  refY="5"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1 L 9 5 L 0 9 z" fill="#2563eb" />
                </marker>
              </defs>

              <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
                {/* Concentric Depth Guide Rings */}
                {[110, 200, 290, 380].map((r) => (
                  <circle
                    key={r}
                    cx={layout.centerX}
                    cy={layout.centerY}
                    r={r}
                    fill="none"
                    stroke="#e2e8f0"
                    strokeDasharray="3 3"
                    strokeWidth="1"
                  />
                ))}

                {/* Directed Edges */}
                {graphData.edges.map((edge, idx) => {
                  const p1 = layout.nodePositions[edge.source];
                  const p2 = layout.nodePositions[edge.target];
                  if (!p1 || !p2) return null;

                  const isConnectedToSelected =
                    selectedNode &&
                    (selectedNode.id === edge.source || selectedNode.id === edge.target);

                  return (
                    <line
                      key={`${edge.source}-${edge.target}-${idx}`}
                      x1={p1.x}
                      y1={p1.y}
                      x2={p2.x}
                      y2={p2.y}
                      stroke={isConnectedToSelected ? "#2563eb" : "#cbd5e1"}
                      strokeWidth={isConnectedToSelected ? 2 : 1}
                      strokeOpacity={isConnectedToSelected ? 0.95 : 0.6}
                      markerEnd={
                        isConnectedToSelected
                          ? "url(#edge-arrow-active)"
                          : "url(#edge-arrow)"
                      }
                    />
                  );
                })}

                {/* Nodes */}
                {graphData.nodes.map((node) => {
                  const pos = layout.nodePositions[node.id];
                  if (!pos) return null;

                  const isSelected = selectedNode?.id === node.id;
                  const isRoot = node.is_root;
                  const radius = isRoot
                    ? 13
                    : Math.min(11, 6 + Math.sqrt(node.out_degree + 1) * 2);
                  const color = getDepthColor(node.depth);

                  return (
                    <g
                      key={node.id}
                      transform={`translate(${pos.x}, ${pos.y})`}
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedNode(node);
                      }}
                      style={{ cursor: "pointer" }}
                    >
                      {/* Selection Ring */}
                      {isSelected && (
                        <circle
                          r={radius + 5}
                          fill="none"
                          stroke="#1e40af"
                          strokeWidth="2"
                          strokeDasharray="3 3"
                        />
                      )}

                      {/* Node Circle */}
                      <circle
                        r={radius}
                        fill={color}
                        stroke="#ffffff"
                        strokeWidth="2"
                        className="sna-node-circle"
                      />

                      {/* Root Label */}
                      {isRoot && (
                        <text
                          y={-radius - 4}
                          textAnchor="middle"
                          fill="#b45309"
                          fontSize="10"
                          fontWeight="700"
                        >
                          ROOT
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            </svg>
          )}

          {/* Simple Legend */}
          <div className="sna-depth-legend">
            <span className="sna-legend-title">Legend:</span>
            <span className="sna-legend-item">
              <span className="sna-legend-dot" style={{ background: DEPTH_COLORS[0] }} />
              Root Author (Hop 0)
            </span>
            <span className="sna-legend-item">
              <span className="sna-legend-dot" style={{ background: DEPTH_COLORS[1] }} />
              Level 1 (Direct Retweet)
            </span>
            <span className="sna-legend-item">
              <span className="sna-legend-dot" style={{ background: DEPTH_COLORS[2] }} />
              Level 2 (Secondary Spread)
            </span>
            <span className="sna-legend-item">
              <span className="sna-legend-dot" style={{ background: DEPTH_COLORS[3] }} />
              Level 3+
            </span>
          </div>
        </div>

        {/* Right Sidebar: Graph Summary + Selected Node Inspector */}
        <div className="sna-sidebar-column">
          {/* Graph Summary Card */}
          {graphData && (
            <div className="sna-info-card">
              <h3>Graph Summary</h3>
              <div className="sna-summary-grid">
                <div className="sna-summary-item">
                  <span className="sna-summary-label">Dataset</span>
                  <span className="sna-summary-val">{graphData.dataset?.toUpperCase()}</span>
                </div>
                <div className="sna-summary-item">
                  <span className="sna-summary-label">Post ID</span>
                  <span className="sna-summary-val code-font">{graphData.post_id}</span>
                </div>
                <div className="sna-summary-item">
                  <span className="sna-summary-label">Veracity</span>
                  <span className={`sna-veracity-badge badge-${graphData.label}`}>
                    {graphData.label?.toUpperCase()}
                  </span>
                </div>
                <div className="sna-summary-item">
                  <span className="sna-summary-label">Observed Nodes</span>
                  <span className="sna-summary-val font-bold">{graphData.total_nodes}</span>
                </div>
                <div className="sna-summary-item">
                  <span className="sna-summary-label">Propagation Edges</span>
                  <span className="sna-summary-val font-bold">{graphData.total_edges}</span>
                </div>
                <div className="sna-summary-item">
                  <span className="sna-summary-label">Displayed Nodes</span>
                  <span className="sna-summary-val">{graphData.displayed_nodes}</span>
                </div>
              </div>
            </div>
          )}

          {/* Selected Node Details Card */}
          <div className="sna-info-card">
            <h3>Selected Node Details</h3>
            {selectedNode ? (
              <div className="sna-inspector-body">
                <div className="sna-prop-row">
                  <span className="sna-prop-label">User ID (uid):</span>
                  <span className="sna-prop-val code-font font-bold">{selectedNode.uid}</span>
                </div>
                <div className="sna-prop-row">
                  <span className="sna-prop-label">Role:</span>
                  <span className="sna-prop-val">
                    {selectedNode.is_root
                      ? "Source Author (Root)"
                      : selectedNode.out_degree > 3
                      ? `Active Forwarder (${selectedNode.out_degree} retweets)`
                      : "Cascade Participant"}
                  </span>
                </div>
                <div className="sna-prop-row">
                  <span className="sna-prop-label">Cascade Depth:</span>
                  <span className="sna-prop-val">
                    {selectedNode.depth === 0
                      ? "Hop 0 (Root)"
                      : `Hop ${selectedNode.depth} (Level ${selectedNode.depth})`}
                  </span>
                </div>
                <div className="sna-prop-row">
                  <span className="sna-prop-label">Time Delay:</span>
                  <span className="sna-prop-val">
                    {selectedNode.delay} min ({Math.round((selectedNode.delay / 60) * 10) / 10} hrs)
                  </span>
                </div>
                <div className="sna-prop-row">
                  <span className="sna-prop-label">Out-Degree:</span>
                  <span className="sna-prop-val highlight-blue">
                    {selectedNode.out_degree} direct retweets
                  </span>
                </div>
                <div className="sna-prop-row">
                  <span className="sna-prop-label">In-Degree:</span>
                  <span className="sna-prop-val">{selectedNode.in_degree}</span>
                </div>
                <div className="sna-prop-row">
                  <span className="sna-prop-label">Total Degree:</span>
                  <span className="sna-prop-val">{selectedNode.total_degree}</span>
                </div>
              </div>
            ) : (
              <div className="sna-inspector-empty">
                <p>Click any node on the graph to inspect its details.</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Academic Note Footer */}
      <div className="sna-footnote">
        <p>
          Note: This graph reflects the observed parent-child propagation tree from the Twitter15/Twitter16 benchmark.
          It represents recorded forwarding relationships for this post, not the complete Twitter follower network.
        </p>
      </div>
    </div>
  );
}
