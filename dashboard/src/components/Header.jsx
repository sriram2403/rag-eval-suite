const TABS = [
  { id: 'benchmark', label: 'Benchmark' },
  { id: 'custom', label: 'Custom Eval' },
  { id: 'documents', label: 'Documents' },
];

export default function Header({ tab, setTab }) {
  return (
    <header>
      <div className="logo">
        <div className="logo-mark">🔍</div>
        <div>
          <div className="logo-name">RAG Eval</div>
          <div className="logo-tag">Evaluation Framework</div>
        </div>
      </div>

      <nav className="header-nav">
        {TABS.map(t => (
          <button
            key={t.id}
            type="button"
            className={`nav-btn${tab === t.id ? ' active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="header-right">
        <div className="api-badge">
          <div className="api-dot" />
          API Connected
        </div>
      </div>
    </header>
  );
}
