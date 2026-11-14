export default function SnaNavCards({ onNavigate }) {
  const cards = [
    {
      id: "social-network",
      path: "/social-network",
      step: "Feature 01",
      title: "Social Network / Propagation Graph",
      description:
        "Visualize how a selected rumor post propagates across users through observed parent-child forwarding relationships.",
      actionText: "Open Propagation Graph →",
    },
    {
      id: "propagation-analysis",
      path: "/propagation-analysis",
      step: "Feature 02",
      title: "Propagation Analysis",
      description:
        "Examine cascade size, propagation depth levels, time delays, and cumulative adoption trajectory over time.",
      actionText: "View Propagation Analysis →",
    },
  ];

  return (
    <section className="sna-dashboard-section" id="sna-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">SOCIAL NETWORK ANALYSIS &amp; PROPAGATION</span>
          <h2>Rumor Propagation Cascades</h2>
          <p className="section-subtitle">
            Reconstruct and analyze observed user-to-user diffusion trees from the Twitter15 and Twitter16 rumor detection benchmarks.
          </p>
        </div>
        <div className="heading-tags">
          <span className="corpus-tag">Twitter15: 1,490 Trees</span>
          <span className="corpus-tag">Twitter16: 818 Trees</span>
        </div>
      </div>

      <div className="sna-cards-grid sna-cards-two-col">
        {cards.map((card) => (
          <div
            key={card.id}
            className="sna-dashboard-card"
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
            <div className="sna-card-top">
              <span className="sna-card-step">{card.step}</span>
            </div>

            <div className="sna-card-body">
              <h3>{card.title}</h3>
              <p className="sna-card-description">{card.description}</p>
            </div>

            <div className="sna-card-action">
              <span>{card.actionText}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
