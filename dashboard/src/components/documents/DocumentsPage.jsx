import { useState, useEffect } from 'react';
import { api } from '../../api.js';
import { getDocIcon } from '../../utils.js';
import UploadZone from './UploadZone.jsx';
import AskAnswer from './AskAnswer.jsx';

export default function DocumentsPage({ showToast }) {
  const [docs,         setDocs]         = useState([]);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [question,     setQuestion]     = useState('');
  const [filter,       setFilter]       = useState('');
  const [gt,           setGt]           = useState('');
  const [asking,       setAsking]       = useState(false);
  const [answerData,   setAnswerData]   = useState(null);

  useEffect(() => { loadDocs(); }, []);

  async function loadDocs() {
    try {
      const data = await api.getDocuments();
      setDocs(data.documents ?? []);
    } catch {
      showToast('Failed to load documents', 'error');
    }
  }

  async function handleUpload(file) {
    setUploadStatus({ text: `Uploading ${file.name}…`, ok: null });
    try {
      const data = await api.uploadFile(file);
      if (data.success) {
        setUploadStatus({ text: `✓ ${data.filename} — ${data.chunks} chunks, ${data.characters.toLocaleString()} chars`, ok: true });
        loadDocs();
        showToast(`${data.filename} uploaded!`);
      } else {
        throw new Error(data.detail || 'Upload failed');
      }
    } catch (e) {
      setUploadStatus({ text: `✗ ${e.message}`, ok: false });
      showToast('Upload failed: ' + e.message, 'error');
    }
  }

  async function handleDelete(filename) {
    if (!confirm(`Delete "${filename}"?`)) return;
    await api.deleteDoc(filename);
    loadDocs();
    showToast(`${filename} deleted`);
  }

  async function handleAsk() {
    if (!question) { showToast('Enter a question', 'error'); return; }
    setAsking(true);
    setAnswerData(null);
    try {
      const data = await api.ask({ question, filename: filter || null, ground_truth: gt });
      setAnswerData(data);
      showToast('Answer generated!');
    } catch (e) {
      showToast(e.message, 'error');
    } finally {
      setAsking(false);
    }
  }

  const statusColor = uploadStatus?.ok === true
    ? 'var(--pass)'
    : uploadStatus?.ok === false
    ? 'var(--fail)'
    : 'var(--text3)';

  return (
    <div className="doc-layout">

      {/* ── Left: upload + doc list ── */}
      <div className="doc-left">
        <UploadZone onUpload={handleUpload} />

        {uploadStatus && (
          <div className="upload-status" style={{ color: statusColor }}>
            {uploadStatus.text}
          </div>
        )}

        <div className="gcard">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <div className="eyebrow" style={{ marginBottom: 0 }}>Uploaded Documents</div>
            <button type="button" className="btn-sec" onClick={loadDocs}>↻ Refresh</button>
          </div>

          {docs.length === 0 ? (
            <div className="empty" style={{ padding: '24px 0' }}>
              <div className="empty-ico" style={{ fontSize: 24 }}>📭</div>
              <div className="empty-sub">No documents uploaded yet</div>
            </div>
          ) : (
            <div className="doc-list">
              {docs.map(d => (
                <div key={d.filename} className="doc-item">
                  <div className="doc-ico-wrap">{getDocIcon(d.filename)}</div>
                  <div className="doc-nm">{d.filename}</div>
                  <div className="doc-mt">{d.chunks} chunks</div>
                  <button type="button" className="doc-del" onClick={() => handleDelete(d.filename)}>✕</button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Right: ask + answer ── */}
      <div className="doc-right">
        <div className="gcard">
          <div className="eyebrow">Ask a Question</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="ask-row">
              <input
                type="text"
                value={question}
                placeholder="What does the document say about…?"
                onChange={e => setQuestion(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleAsk()}
              />
              <button type="button" className="btn-ask" onClick={handleAsk} disabled={asking}>
                {asking ? '…' : 'Ask'}
              </button>
            </div>

            <div className="field">
              <label htmlFor="docFilter">Filter by Document (optional)</label>
              <div className="sel-wrap">
                <select id="docFilter" value={filter} onChange={e => setFilter(e.target.value)}>
                  <option value="">Search all documents</option>
                  {docs.map(d => <option key={d.filename} value={d.filename}>{d.filename}</option>)}
                </select>
              </div>
            </div>

            <div className="field">
              <label htmlFor="docGT">Ground Truth (optional — enables evaluation scores)</label>
              <input
                id="docGT"
                type="text"
                value={gt}
                placeholder="Leave blank to skip evaluation"
                onChange={e => setGt(e.target.value)}
              />
            </div>
          </div>

          {asking && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 16, color: 'var(--text3)', fontSize: 12 }}>
              <div className="pulse-ring" style={{ width: 18, height: 18, borderWidth: 2 }} />
              Searching documents and generating answer…
            </div>
          )}
        </div>

        {answerData && <AskAnswer data={answerData} />}
      </div>
    </div>
  );
}
