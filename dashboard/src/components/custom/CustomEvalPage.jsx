import { useState } from 'react';
import { api } from '../../api.js';
import { cls, metricLabel } from '../../utils.js';

export default function CustomEvalPage({ showToast }) {
  const [question,    setQuestion]    = useState('');
  const [groundTruth, setGroundTruth] = useState('');
  const [contexts,    setContexts]    = useState([]);
  const [ctxDraft,    setCtxDraft]    = useState('');
  const [answer,      setAnswer]      = useState('');
  const [pipeline,    setPipeline]    = useState('naive');
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);

  function addCtx() {
    if (!ctxDraft.trim()) return;
    setContexts(c => [...c, ctxDraft.trim()]);
    setCtxDraft('');
  }

  async function runEval() {
    if (!question || !groundTruth || contexts.length === 0) {
      showToast('Question, ground truth and at least one context required', 'error');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const data = await api.evaluateSample({ question, ground_truth: groundTruth, contexts, answer, pipeline });
      setResult(data);
      showToast('Evaluation complete!');
    } catch (e) {
      showToast(e.message, 'error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="eval-grid">

      {/* ── Input panel ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div className="gcard">
          <div className="eyebrow">Question & Answer</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="field">
              <label htmlFor="cQ">Question</label>
              <input id="cQ" type="text" value={question} placeholder="e.g. When did American Express start?" onChange={e => setQuestion(e.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="cGT">Ground Truth Answer</label>
              <input id="cGT" type="text" value={groundTruth} placeholder="The correct expected answer" onChange={e => setGroundTruth(e.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="cAns">Generated Answer (leave blank to auto-generate)</label>
              <input id="cAns" type="text" value={answer} placeholder="Leave blank — pipeline will generate" onChange={e => setAnswer(e.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="cPipe">Pipeline</label>
              <div className="sel-wrap">
                <select id="cPipe" value={pipeline} onChange={e => setPipeline(e.target.value)}>
                  <option value="naive">Naive RAG (TF-IDF)</option>
                  <option value="semantic">Semantic RAG</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <div className="gcard">
          <div className="eyebrow">Context Passages</div>
          {contexts.length > 0 && (
            <div className="ctx-list" style={{ marginBottom: 12 }}>
              {contexts.map((c, i) => (
                <div key={i} className="ctx-row">
                  <span className="ctx-num">#{i + 1}</span>
                  <span className="ctx-txt">{c}</span>
                  <button type="button" className="ctx-rm" onClick={() => setContexts(cs => cs.filter((_, j) => j !== i))}>✕</button>
                </div>
              ))}
            </div>
          )}
          <div className="ctx-add-row">
            <textarea
              value={ctxDraft}
              placeholder="Paste context passage..."
              onChange={e => setCtxDraft(e.target.value)}
            />
            <button type="button" className="btn-add" onClick={addCtx}>+ Add</button>
          </div>
        </div>

        <button type="button" className="btn-run" onClick={runEval} disabled={loading}>
          {loading ? <><div className="pulse-ring" style={{ width: 18, height: 18, borderWidth: 2 }} /> Evaluating...</> : '▶  Evaluate Sample'}
        </button>
      </div>

      {/* ── Results panel ── */}
      <div>
        {!result && !loading && (
          <div className="glass">
            <div className="empty">
              <div className="empty-ico">🔬</div>
              <div className="empty-ttl">No Results Yet</div>
              <div className="empty-sub">Fill in the form and click Evaluate Sample to score your RAG output.</div>
            </div>
          </div>
        )}
        {loading && (
          <div className="glass">
            <div className="running-card">
              <div className="pulse-ring" />
              <div className="running-label">SCORING WITH LLM JUDGE...</div>
            </div>
          </div>
        )}
        {result && (
          <div className="gcard">
            <div className="eyebrow">Evaluation Results</div>
            <div className="eval-result">
              {result.sample?.answer && (
                <div className="detail-block">
                  <div className="detail-lbl">Generated Answer</div>
                  <div className="detail-txt">{result.sample.answer}</div>
                </div>
              )}
              <div className="score-row">
                {Object.entries(result.metric_results ?? {}).map(([mn, mr]) => {
                  const c = cls(mr.score, mr.threshold);
                  return (
                    <div key={mn} className="score-block">
                      <div className="score-lbl">{metricLabel(mn)}</div>
                      <div className={`score-big ${c}`}>{(mr.score * 100).toFixed(0)}%</div>
                    </div>
                  );
                })}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {Object.entries(result.metric_results ?? {}).map(([mn, mr]) =>
                  mr.explanation ? (
                    <div key={mn} className="expl-pill">
                      <strong style={{ color: `var(--${cls(mr.score, mr.threshold)})` }}>
                        {metricLabel(mn)}
                      </strong>{' — '}{mr.explanation}
                    </div>
                  ) : null
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
