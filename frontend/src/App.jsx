import { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import DashboardPage from "./pages/DashboardPage";
import ClaimAnalysisPage from "./pages/ClaimAnalysisPage";
import ModelComparisonPage from "./pages/ModelComparisonPage";
import EvidenceResearchPage from "./pages/EvidenceResearchPage";
import AgenticAnalysisPage from "./pages/AgenticAnalysisPage";
import HistoricalInfluencePage from "./pages/HistoricalInfluencePage";
import SocialNetworkPage from "./sna/SocialNetworkPage";
import PropagationAnalysisPage from "./sna/PropagationAnalysisPage";
import { API_URL } from "./utils/constants";
import "./App.css";
import "./sna/sna.css";

export default function App() {
  const [currentPath, setCurrentPath] = useState(() => {
    return window.location.pathname || "/";
  });

  const [claim, setClaim] = useState("");
  const [isBackendOnline, setIsBackendOnline] = useState(true);

  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem("misinformation-history");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Client-side routing with smooth transition and popstate listener
  const navigate = (path) => {
    window.history.pushState({}, "", path);
    setCurrentPath(path);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(window.location.pathname || "/");
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // Periodic health check for system status indicator
  useEffect(() => {
    let isMounted = true;
    const checkHealth = () => {
      fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(4000) })
        .then((res) => {
          if (isMounted) setIsBackendOnline(res.ok);
        })
        .catch(() => {
          if (isMounted) setIsBackendOnline(false);
        });
    };

    checkHealth();
    const interval = setInterval(checkHealth, 25000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Save history to localStorage
  const saveHistory = (data) => {
    if (!data?.claim) return;

    const newItem = {
      id: Date.now(),
      claim: data.claim,
      baseline: data.baseline?.label || "N/A",
      distilbert: data.distilbert?.label || "N/A",
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setHistory((prev) => {
      const updated = [newItem, ...prev.filter((h) => h.claim !== data.claim)].slice(0, 10);
      try {
        localStorage.setItem("misinformation-history", JSON.stringify(updated));
      } catch (err) {
        console.warn("Could not save evaluation history:", err);
      }
      return updated;
    });
  };

  return (
    <div className="app-root-shell">
      {/* Top Application Header & Navigation Bar */}
      <Navbar
        currentPath={currentPath}
        onNavigate={navigate}
        isBackendOnline={isBackendOnline}
      />

      {/* Main Page Content Area Routed by Path */}
      <main className="app-main-viewport">
        {currentPath === "/" && (
          <DashboardPage
            onNavigate={navigate}
            history={history}
          />
        )}

        {currentPath === "/claim-analysis" && (
          <ClaimAnalysisPage
            claim={claim}
            setClaim={setClaim}
            onSaveHistory={saveHistory}
            onNavigate={navigate}
          />
        )}

        {currentPath === "/model-comparison" && (
          <ModelComparisonPage
            claim={claim}
            setClaim={setClaim}
            onSaveHistory={saveHistory}
            onNavigate={navigate}
          />
        )}

        {currentPath === "/evidence" && (
          <EvidenceResearchPage
            claim={claim}
            setClaim={setClaim}
            onNavigate={navigate}
          />
        )}

        {currentPath === "/agentic-analysis" && (
          <AgenticAnalysisPage
            claim={claim}
            setClaim={setClaim}
            onSaveHistory={saveHistory}
            onNavigate={navigate}
          />
        )}

        {currentPath === "/influence-analysis" && (
          <HistoricalInfluencePage
            claim={claim}
            setClaim={setClaim}
            onNavigate={navigate}
          />
        )}

        {currentPath === "/social-network" && (
          <SocialNetworkPage onBack={() => navigate("/")} />
        )}

        {currentPath === "/propagation-analysis" && (
          <PropagationAnalysisPage onBack={() => navigate("/")} />
        )}
      </main>

      {/* Professional Academic Footer */}
      <footer className="app-footer">
        <div className="footer-content-row">
          <div className="footer-left">
            <strong>VERITAS · Misinformation Detection &amp; Diffusion Intelligence System</strong>
            <p>
              Undergraduate Engineering Major Project · Fine-Tuned DistilBERT, LIAR-PLUS Hybrid RAG, FakeNewsNet, Twitter15/16 Benchmarks.
            </p>
          </div>
          <div className="footer-right">
            <span>Powered by FastAPI + PyTorch + ChromaDB + NetworkX + React</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
