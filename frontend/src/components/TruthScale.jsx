import { getLabelInfo } from "../utils/constants";

const SPECTRUM_ITEMS = [
  { key: "pants-fire", label: "Pants on Fire", level: 1 },
  { key: "false", label: "False", level: 2 },
  { key: "barely-true", label: "Barely True", level: 3 },
  { key: "half-true", label: "Half True", level: 4 },
  { key: "mostly-true", label: "Mostly True", level: 5 },
  { key: "true", label: "True", level: 6 },
];

export default function TruthScale({
  activeLabel,
  secondaryLabel = null,
  title = "LIAR Benchmark Veracity Spectrum",
  showSecondary = false,
}) {
  const activeInfo = getLabelInfo(activeLabel);

  return (
    <div className="truth-scale-card">
      <div className="truth-scale-header">
        <div>
          <span className="scale-eyebrow">BENCHMARK SPECTRUM</span>
          <h4>{title}</h4>
        </div>
        {activeLabel && (
          <span className="scale-active-badge">
            Predicted: <strong>{activeInfo.title}</strong>
          </span>
        )}
      </div>

      <div className="scale-track-container">
        {SPECTRUM_ITEMS.map((item) => {
          const isPrimary = activeLabel === item.key;
          const isSecondary = showSecondary && secondaryLabel === item.key;

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
            <span className="legend-dot primary-dot" /> Primary Model (DistilBERT)
          </span>
          <span className="legend-marker">
            <span className="legend-dot secondary-dot" /> Baseline Model (TF-IDF)
          </span>
        </div>
      )}
    </div>
  );
}
