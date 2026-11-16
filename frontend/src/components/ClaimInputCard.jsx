import { EXAMPLES } from "../utils/constants";

export default function ClaimInputCard({
  claim,
  onClaimChange,
  onSubmit,
  loading = false,
  error = "",
  onClear,
  buttonLabel = "Analyze Claim",
  placeholder = "Enter a political statement, policy assertion, or news headline for verification...",
  showExamples = true,
}) {
  return (
    <div className="claim-input-card">
      <form onSubmit={onSubmit} className="claim-form">
        <div className="claim-input-header">
          <label htmlFor="claim-textarea" className="claim-input-label">
            Target Statement / Claim Text
          </label>
          <span className="character-counter">{claim.length} characters</span>
        </div>

        <textarea
          id="claim-textarea"
          value={claim}
          onChange={(e) => onClaimChange(e.target.value)}
          placeholder={placeholder}
          rows={4}
          className="claim-textarea"
        />

        <div className="claim-form-actions">
          {onClear && (
            <button
              type="button"
              className="btn-secondary"
              onClick={onClear}
              disabled={loading || !claim}
            >
              Clear
            </button>
          )}

          <button
            type="submit"
            className="btn-primary"
            disabled={loading || !claim.trim()}
          >
            {loading ? (
              <>
                <span className="btn-spinner" />
                Processing...
              </>
            ) : (
              <>
                {buttonLabel}
                <span className="btn-arrow">→</span>
              </>
            )}
          </button>
        </div>
      </form>

      {error && (
        <div className="claim-error-alert" role="alert">
          <span className="alert-icon">!</span>
          <div className="alert-message">
            <strong>Validation Error:</strong> {error}
          </div>
        </div>
      )}

      {showExamples && (
        <div className="claim-examples-container">
          <span className="examples-header">Illustrative Benchmark Claims:</span>
          <div className="examples-pill-list">
            {EXAMPLES.map((ex, idx) => (
              <button
                key={idx}
                type="button"
                className="example-pill-btn"
                onClick={() => onClaimChange(ex)}
                disabled={loading}
              >
                &ldquo;{ex.length > 70 ? `${ex.slice(0, 70)}...` : ex}&rdquo;
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
