async function json(resPromise) {
  const res = await resPromise;
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

const h = { 'Content-Type': 'application/json' };

export const api = {
  startBenchmark: (req) =>
    json(fetch('/api/benchmark', { method: 'POST', headers: h, body: JSON.stringify(req) })),

  pollJob: (id) => json(fetch(`/api/jobs/${id}`)),

  getResults: (id) => json(fetch(`/api/results/${id}`)),

  evaluateSample: (req) =>
    json(fetch('/api/evaluate-sample', { method: 'POST', headers: h, body: JSON.stringify(req) })),

  getDocuments: () => json(fetch('/api/documents')),

  uploadFile: (file) => {
    const fd = new FormData();
    fd.append('file', file);
    return json(fetch('/api/upload', { method: 'POST', body: fd }));
  },

  deleteDoc: (filename) =>
    fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' }),

  ask: (req) =>
    json(fetch('/api/ask', { method: 'POST', headers: h, body: JSON.stringify(req) })),
};
