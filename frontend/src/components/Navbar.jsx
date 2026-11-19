import { useState } from "react";

export default function Navbar({ currentPath, onNavigate, isBackendOnline = true }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { path: "/", label: "Dashboard" },
    { path: "/agentic-analysis", label: "Multi-Agent Workflow" },
    { path: "/claim-analysis", label: "Claim Classifier" },
    { path: "/evidence", label: "Evidence & RAG" },
    { path: "/influence-analysis", label: "Social Influence" },
    { path: "/social-network", label: "Propagation Graph" },
    { path: "/propagation-analysis", label: "Cascade Kinetics" },
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
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") handleNavClick("/");
          }}
        >
          <div className="brand-logo-mark" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              <path d="m9 12 2 2 4-4" />
            </svg>
          </div>
          <div className="brand-titles">
            <div className="brand-main-title">
              <h1>Misinformation Detection &amp; Social Network Propagation</h1>
            </div>
            <span className="brand-subtitle">Research &amp; Verification Platform</span>
          </div>
        </div>

        <div className="navbar-top-right">
          <div className={`system-status-indicator ${isBackendOnline ? "status-online" : "status-offline"}`}>
            <span className="status-pulse-dot" />
            <span className="status-label-text">
              {isBackendOnline ? "System Online" : "Connecting..."}
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

      <nav className={`navbar-tabs-bar ${mobileMenuOpen ? "mobile-open" : ""}`} aria-label="Main Navigation">
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
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
