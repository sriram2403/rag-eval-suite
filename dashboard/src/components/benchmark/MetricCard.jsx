import { cls, metricLabel } from '../../utils.js';

export default function MetricCard({ name, score, threshold, passRate }) {
  const c = cls(score, threshold);
  const pct = Math.round(score * 100);
  const r = 33;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - score);

  return (
    <div className={`ring-card ${c}`}>
      <div className="ring-wrap">
        <svg width="76" height="76" viewBox="0 0 76 76">
          <circle className="ring-bg"   cx="38" cy="38" r={r} />
          <circle
            className={`ring-fill ${c}`}
            cx="38" cy="38" r={r}
            strokeDasharray={circ}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="ring-center">
          <span className={`ring-num ${c}`}>{pct}</span>
          <span className="ring-pct">%</span>
        </div>
      </div>
      <div className="ring-name">{metricLabel(name)}</div>
      <div className="ring-meta">Pass {Math.round(passRate * 100)}%</div>
    </div>
  );
}
