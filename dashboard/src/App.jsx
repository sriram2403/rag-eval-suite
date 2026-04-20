import { useState } from 'react';
import Toast from './components/Toast.jsx';
import BenchmarkPage from './components/benchmark/BenchmarkPage.jsx';
import CustomEvalPage from './components/custom/CustomEvalPage.jsx';
import DocumentsPage from './components/documents/DocumentsPage.jsx';
import { useToast } from './hooks/useToast.js';

const TABS = [
  { id: 'benchmark', icon: '⚡', label: 'Benchmark' },
  { id: 'custom',    icon: '🔬', label: 'Evaluate' },
  { id: 'documents', icon: '📁', label: 'Documents' },
];

export default function App() {
  const [tab, setTab] = useState('benchmark');
  const { toast, showToast } = useToast();

  return (
    <div className="app">
      <nav className="nav">
        {TABS.map(t => (
          <button
            key={t.id}
            type="button"
            className={`nav-btn${tab === t.id ? ' active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            <span className="nav-icon">{t.icon}</span>
            {t.label}
          </button>
        ))}
      </nav>

      <div className="page">
        {tab === 'benchmark' && <BenchmarkPage showToast={showToast} />}
        {tab === 'custom'    && <CustomEvalPage showToast={showToast} />}
        {tab === 'documents' && <DocumentsPage showToast={showToast} />}
      </div>

      <Toast toast={toast} />
    </div>
  );
}
