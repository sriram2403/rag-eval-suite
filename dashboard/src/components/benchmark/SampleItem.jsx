import { useState } from 'react';
import { cls, metricLabel } from '../../utils.js';

export default function SampleItem({ result: res }) {
  const [open, setOpen] = useState(false);
  const ov = res.overall_score ?? 0;
  const c = cls(ov);

  return (
    <div className="sample-item">
      <div className="sample-hd" onClick={() => setOpen(o => !o)}>
        <span className="sample-id">{res.sample?.sample_id ?? '#?'}</span>
        <span className="sample-q">{res.sample?.question ?? ''}</span>
        <div className="sample-chips">
          {Object.entries(res.metric_results ?? {}).slice(0, 3).map(([mn, mr]) => {
            const mc = cls(mr.score, mr.threshold);
            return (
              <span key={mn} className={`sc ${mc}`}>
                {(mr.score * 100).toFixed(0)}%
              </span>
            );
          })}
        </div>
        <span className={`sc ${c}`} style={{ fontWeight: 800 }}>
          {(ov * 100).toFixed(0)}%
        </span>
        <span className={`sample-caret${open ? ' open' : ''}`}>▼</span>
      </div>

      {open && (
        <div className="sample-bd">
          {res.sample?.answer && (
            <div className="detail-block">
              <div className="detail-lbl">Generated Answer</div>
              <div className="detail-txt">{res.sample.answer}</div>
            </div>
          )}
          {res.sample?.ground_truth && (
            <div className="detail-block">
              <div className="detail-lbl">Ground Truth</div>
              <div className="detail-txt">{res.sample.ground_truth}</div>
            </div>
          )}
          {Object.values(res.metric_results ?? {}).some(mr => mr.explanation) && (
            <div className="expl-rows">
              {Object.entries(res.metric_results ?? {}).map(([mn, mr]) =>
                mr.explanation ? (
                  <div key={mn} className="expl-row">
                    <span className="expl-nm">{metricLabel(mn)}</span>
                    <span className="expl-tx">{mr.explanation}</span>
                  </div>
                ) : null
              )}
            </div>
          )}
          {res.errors?.length > 0 && (
            <div style={{ color: 'var(--fail)', fontSize: '11px' }}>
              Errors: {res.errors.join(', ')}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
