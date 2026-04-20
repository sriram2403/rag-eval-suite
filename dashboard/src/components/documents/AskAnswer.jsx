import { cls, metricLabel } from '../../utils.js';

function parseAnswer(raw) {
  if (raw.includes('FROM DOCUMENT') && raw.includes('EXTENDED REASONING')) {
    const docPart = raw.split('EXTENDED REASONING')[0]
      .replace('FROM DOCUMENT:', '').replace('📄', '').trim();
    const reasonPart = raw.split('EXTENDED REASONING:')[1]?.trim() ?? '';
    return { docPart, reasonPart };
  }
  return null;
}

export default function AskAnswer({ data }) {
  const raw    = data.answer ?? '';
  const parsed = parseAnswer(raw);

  return (
    <div className="answer-card">
      <div className="ans-section">
        <div className="ans-label">
          <span className="ans-dot doc" />
          From Document
        </div>
        <div className="ans-txt">
          {parsed ? parsed.docPart : raw}
        </div>
      </div>

      {parsed?.reasonPart && (
        <div className="ans-section">
          <div className="ans-label">
            <span className="ans-dot reason" />
            Extended Reasoning
          </div>
          <div className="ans-txt" style={{ color: 'var(--text2)' }}>
            {parsed.reasonPart}
          </div>
        </div>
      )}

      {data.contexts?.length > 0 && (
        <div className="ans-section">
          <div className="ans-label">
            <span className="ans-dot" style={{ background: 'var(--text3)' }} />
            Retrieved Context ({data.contexts.length} chunks)
          </div>
          <div className="chunk-list">
            {data.contexts.map((ctx, i) => (
              <div key={i} className="chunk-item">
                <div className="chunk-lbl">CHUNK {i + 1}</div>
                <div className="chunk-txt">{ctx.slice(0, 600)}{ctx.length > 600 ? '…' : ''}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.evaluation?.metric_results && (
        <div className="ans-section">
          <div className="ans-label">
            <span className="ans-dot" style={{ background: 'var(--accent)' }} />
            Evaluation Scores
          </div>
          <div className="eval-chips" style={{ marginBottom: 14 }}>
            {Object.entries(data.evaluation.metric_results).map(([mn, mr]) => {
              const mc = cls(mr.score, mr.threshold);
              return (
                <span key={mn} className={`eval-chip ${mc}`}>
                  {metricLabel(mn)}: {(mr.score * 100).toFixed(0)}%
                </span>
              );
            })}
          </div>
          {Object.entries(data.evaluation.metric_results).map(([mn, mr]) => {
            const mc = cls(mr.score, mr.threshold);
            return mr.explanation ? (
              <div key={mn} className="expl-entry">
                <div className={`expl-hd ${mc}`}>{metricLabel(mn)} — {(mr.score * 100).toFixed(0)}%</div>
                <div className="expl-bd">{mr.explanation}</div>
              </div>
            ) : null;
          })}
        </div>
      )}
    </div>
  );
}
