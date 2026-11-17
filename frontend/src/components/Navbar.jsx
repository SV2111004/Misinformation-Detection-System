import { useState } from "react";

export default function Navbar({ currentPath, onNavigate, isBackendOnline = true }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { path: "/", label: "Dashboard", badge: null },
    { path: "/claim-analysis", label: "Claim Classifier", badge: "DistilBERT" },
    { path: "/evidence", label: "Evidence & RAG", badge: "ChromaDB" },
    { path: "/agentic-analysis", label: "Multi-Agent Workflow", badge: "5 Agents" },
    { path: "/influence-analysis", label: "Social Influence", badge: "FakeNewsNet" },
    { path: "/social-network", label: "Propagation Graph", badge: "SNA" },
    { path: "/propagation-analysis", label: "Cascade Kinetics", badge: "Kinetics" },
  ];

  const handleNavClick = (path) => {
    onNavigate(path);
    setMobileMenuOpen(false);
  };

  return (
    <header className="app-navbar">
      <div className="navbar-top-row">
        <div
          className="navbar-brand"
          onClick={() => handleNavClick("/")}
          role="button"
          tabIndex={0}
        >
          <div className="brand-logo-mark">V</div>
          <div className="brand-titles">
            <div className="brand-main-title">
              <h2>VERITAS</h2>
              <span className="brand-tag">Research System</span>
            </div>
            <span className="brand-subtitle">
              Political Misinformation Detection, Evidence Retrieval &amp; Rumor Propagation
            </span>
          </div>
        </div>

        <div className="navbar-top-right">
          <div className={`system-status-indicator ${isBackendOnline ? "status-online" : "status-offline"}`}>
            <span className="status-pulse-dot" />
            <span className="status-label-text">
              {isBackendOnline ? "FastAPI Backend Online" : "Connecting Backend..."}
            </span>
          </div>

          <button
            type="button"
            className="mobile-menu-toggle"
            onClick={() => setMobileMenuOpen((prev) => !prev)}
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? "✕ Close" : "☰ Menu"}
          </button>
        </div>
      </div>

      <nav className={`navbar-tabs-bar ${mobileMenuOpen ? "mobile-open" : ""}`}>
        <ul className="nav-tabs-list">
          {navItems.map((item) => {
            const isActive = currentPath === item.path;
            return (
              <li key={item.path} className="nav-tab-item">
                <button
                  type="button"
                  className={`nav-tab-link ${isActive ? "active" : ""}`}
                  onClick={() => handleNavClick(item.path)}
                  aria-current={isActive ? "page" : undefined}
                >
                  <span className="tab-label">{item.label}</span>
                  {item.badge && <span className="tab-pill-badge">{item.badge}</span>}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
