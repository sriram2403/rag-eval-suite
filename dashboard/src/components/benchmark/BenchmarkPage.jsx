import { useState } from 'react';
import { api } from '../../api.js';
import { cls } from '../../utils.js';
import MetricCard from './MetricCard.jsx';
import SampleItem from './SampleItem.jsx';
import RadarChart from './RadarChart.jsx';

const ALL_METRICS = [
  { id: 'faithfulness',        label: 'Fact Accuracy' },
  { id: 'answer_relevance',    label: 'Answers the Question' },
  { id: 'context_recall',      label: 'Info Coverage' },
  { id: 'groundedness',        label: 'Stays in Document' },
  { id: 'context_precision',   label: 'Source Quality' },
  { id: 'semantic_similarity', label: 'Meaning Match' },
  { id: 'rouge_l',             label: 'Word Overlap' },
];

const GUIDE = [
  ['Fact Accuracy',        'every claim is backed by the document'],
  ['Answers the Question', 'the answer addresses what was asked'],
  ['Info Coverage',        'retrieval found all relevant information'],
  ['Stays in Document',    'answer does not go beyond the source'],
  ['Source Quality',       'retrieved chunks are on-topic'],
];

export default function BenchmarkPage({ showToast }) {
  const [pipeline, setPipeline]           = useState('naive');
  const [dataset, setDataset]             = useState('science');
  const [maxSamples, setMaxSamples]       = useState(3);
  const [useLlm, setUseLlm]               = useState(true);
  const [selectedMetrics, setSelectedMetrics] = useState(
    new Set(['faithfulness', 'answer_relevance', 'context_recall', 'groundedness'])
  );
  const [running, setRunning] = useState(false);
  const [report, setReport]   = useState(null);

  function toggleMetric(id) {
    setSelectedMetrics(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  async function runBenchmark() {
    if (selectedMetrics.size === 0) { showToast('Select at least one metric', 'error'); return; }
    setRunning(true);
    setReport(null);
    try {
      const { job_id } = await api.startBenchmark({
        pipeline, dataset, use_llm: useLlm,
        max_samples: maxSamples,
        metrics: [...selectedMetrics],
      });
      while (true) {
        await new Promise(r => setTimeout(r, 1500));
        const st = await api.pollJob(job_id);
        if (st.status === 'completed') break;
        if (st.status === 'failed') throw new Error(st.error || 'Job failed');
      }
      const result = await api.getResults(job_id);
      setReport(result);
      showToast('Benchmark complete!');
    } catch (e) {
      showToast(e.message, 'error');
    } finally {
      setRunning(false);
    }
  }

  const scores  = report?.aggregate_scores ?? {};
  const overall = Object.keys(scores).length
    ? Object.values(scores).reduce((a, b) => a + b, 0) / Object.keys(scores).length
    : 0;
  const oc = cls(overall);

  return (
    <div className="bench-layout">

      {/* ── Config panel ── */}
      <div className="config-panel">
        <div className="gcard">
          <div className="eyebrow">Pipeline</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="field">
              <label htmlFor="pipelineSelect">RAG Pipeline</label>
              <div className="sel-wrap">
                <select id="pipelineSelect" value={pipeline} onChange={e => setPipeline(e.target.value)}>
                  <option value="naive">Naive RAG (TF-IDF)</option>
                  <option value="semantic">Semantic RAG (Embeddings)</option>
                  <option value="mock">Mock Pipeline</option>
                </select>
              </div>
            </div>

            <div className="field">
              <label htmlFor="datasetSelect">Dataset</label>
              <div className="sel-wrap">
                <select id="datasetSelect" value={dataset} onChange={e => setDataset(e.target.value)}>
                  <option value="science">Science Q&A</option>
                  <option value="tech">Technology Q&A</option>
                  <option value="hallucination">Hallucination Tests</option>
                  <option value="all">All Datasets</option>
                </select>
              </div>
            </div>

            <div className="field">
              <label htmlFor="maxSamples">Max Samples</label>
              <input
                id="maxSamples"
                type="number"
                value={maxSamples}
                min={1} max={50}
                onChange={e => setMaxSamples(parseInt(e.target.value) || 3)}
              />
            </div>

            <div className="toggle-row">
              <span className="toggle-label">LLM-as-Judge</span>
              <button
                type="button"
                className={`toggle${useLlm ? ' on' : ''}`}
                onClick={() => setUseLlm(v => !v)}
              />
            </div>
          </div>
        </div>

        <div className="gcard">
          <div className="eyebrow">Metrics</div>
          <div className="m-chips">
            {ALL_METRICS.map(m => (
              <button
                key={m.id}
                type="button"
                className={`m-chip${selectedMetrics.has(m.id) ? ' on' : ''}`}
                onClick={() => toggleMetric(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <button type="button" className="btn-run" disabled={running} onClick={runBenchmark}>
          {running ? <><div className="pulse-ring" style={{ width: 18, height: 18, borderWidth: 2 }} /> Running...</> : '▶  Run Benchmark'}
        </button>

        <div className="gcard">
          <div className="eyebrow">Metric Guide</div>
          <div className="guide-list">
            {GUIDE.map(([nm, desc]) => (
              <div key={nm} className="guide-row">
                <div className="guide-dot" />
                <div>
                  <div className="guide-nm">{nm}</div>
                  <div className="guide-desc">{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Results area ── */}
      <div className="results-area">
        {!report && !running && (
          <div className="glass">
            <div className="empty">
              <div className="empty-ico">◎</div>
              <div className="empty-ttl">Ready to Evaluate</div>
              <div className="empty-sub">Configure your pipeline and metrics, then click Run Benchmark.</div>
            </div>
          </div>
        )}

        {running && (
          <div className="glass">
            <div className="running-card">
              <div className="pulse-ring" />
              <div className="running-label">EVALUATING · {dataset.toUpperCase()} · {selectedMetrics.size} METRICS</div>
            </div>
          </div>
        )}

        {report && (
          <>
            <div className="gcard">
              <div className="overall-banner" style={{ padding: 0 }}>
                <div>
                  <div className="score-label">Overall Score</div>
                  <div className={`score-giant ${oc}`}>
                    {(overall * 100).toFixed(1)}<span style={{ fontSize: 22 }}>%</span>
                  </div>
                </div>
                <div className="divider-v" />
                <div className="banner-stats">
                  <div className="bstat">
                    <span className="bstat-v">{report.total_samples}</span>
                    <span className="bstat-l">Samples</span>
                  </div>
                  <div className="bstat">
                    <span className="bstat-v">{report.passed_samples}</span>
                    <span className="bstat-l">Passed</span>
                  </div>
                  <div className="bstat">
                    <span className="bstat-v">{report.run_duration_s?.toFixed(1)}s</span>
                    <span className="bstat-l">Duration</span>
                  </div>
                </div>
                <div className="pipeline-tag">{report.pipeline_name}</div>
              </div>
            </div>

            <div>
              <div className="eyebrow">Metric Scores</div>
              <div className="metric-bento">
                {Object.entries(report.aggregate_scores).map(([m, score]) => (
                  <MetricCard
                    key={m}
                    name={m}
                    score={score}
                    threshold={report.config?.thresholds?.[m] ?? 0.7}
                    passRate={report.pass_rates?.[m] ?? 0}
                  />
                ))}
              </div>
            </div>

            <div className="radar-card">
              <div className="eyebrow" style={{ marginBottom: 0 }}>Score Radar</div>
              <RadarChart scores={report.aggregate_scores} thresholds={report.config?.thresholds ?? {}} />
            </div>

            {report.sample_results?.length > 0 && (
              <div>
                <div className="eyebrow">{report.sample_results.length} Sample Results</div>
                <div className="sample-list">
                  {report.sample_results.map((res, i) => (
                    <SampleItem key={i} result={res} />
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
