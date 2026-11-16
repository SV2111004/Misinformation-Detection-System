export default function ConfidenceBar({ value = 0, label = "Model Confidence", level = 0 }) {
  const percentage = Math.max(0, Math.min(100, (value || 0) * 100));

  return (
    <div className="confidence-meter-container">
      <div className="confidence-label-row">
        <span className="confidence-title">{label}</span>
        <strong className="confidence-value">{percentage.toFixed(1)}%</strong>
      </div>
      <div className="confidence-track">
        <div
          className={`confidence-fill level-fill-${level}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
