import { getLabelInfo } from "../utils/constants";

const SPECTRUM_ITEMS_3 = [
  { key: "misleading", label: "Misleading", level: 2 },
  { key: "half-true", label: "Half-True", level: 4 },
  { key: "factual", label: "Factual", level: 6 },
];

export default function TruthScale({
  activeLabel,
  title = null,
  showSecondary = false,
  secondaryLabel = null,
}) {
  const normActive = String(activeLabel || "").toLowerCase().trim();
  const normSecondary = String(secondaryLabel || "").toLowerCase().trim();

  const items = SPECTRUM_ITEMS_3;
  const activeInfo = getLabelInfo(activeLabel);

  const displayTitle = title || "Socio-Economic Policy Veracity Scale (3-Class)";

  return (
    <div className="truth-scale-card">
      <div className="truth-scale-header">
        <div>
          <span className="scale-eyebrow">TARGET POLICY DOMAIN SCALE</span>
          <h4>{displayTitle}</h4>
        </div>
        {activeLabel && (
          <span className="scale-active-badge">
            Predicted: <strong>{activeInfo.title}</strong>
          </span>
        )}
      </div>

      <div className="scale-track-container track-3class">
        {items.map((item) => {
          const isPrimary =
            normActive === item.key ||
            (item.key === "misleading" && ["pants-fire", "false"].includes(normActive)) ||
            (item.key === "half-true" && ["barely-true", "half-true"].includes(normActive)) ||
            (item.key === "factual" && ["mostly-true", "true"].includes(normActive));

          const isSecondary =
            showSecondary &&
            (normSecondary === item.key ||
              (item.key === "misleading" && ["pants-fire", "false"].includes(normSecondary)) ||
              (item.key === "half-true" && ["barely-true", "half-true"].includes(normSecondary)) ||
              (item.key === "factual" && ["mostly-true", "true"].includes(normSecondary)));

          let dotClass = "scale-dot";
          if (isPrimary && isSecondary) {
            dotClass += " dot-both-models";
          } else if (isPrimary) {
            dotClass += ` dot-level-${item.level} dot-active`;
          } else if (isSecondary) {
            dotClass += " dot-secondary-model";
          }

          return (
            <div
              key={item.key}
              className={`scale-step-item ${isPrimary ? "step-primary-active" : ""} ${
                isSecondary ? "step-secondary-active" : ""
              }`}
            >
              <div className={dotClass} />
              <span className="scale-step-label">{item.label}</span>
            </div>
          );
        })}
      </div>

      {showSecondary && (
        <div className="scale-dual-legend">
          <span className="legend-marker">
            <span className="legend-dot primary-dot" /> Primary Model
          </span>
          <span className="legend-marker">
            <span className="legend-dot secondary-dot" /> Comparative Baseline
          </span>
        </div>
      )}
    </div>
  );
}
