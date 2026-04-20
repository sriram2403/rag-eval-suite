import { useRef, useState } from 'react';

const TYPES = ['PDF', 'DOCX', 'TXT', 'MD', 'YAML', 'XML', 'JSON', 'HTML', 'RTF'];

export default function UploadZone({ onUpload }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) onUpload(file);
  }

  return (
    <div
      className={`upload-zone${dragOver ? ' drag-over' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.doc,.txt,.md,.yaml,.yml,.xml,.json,.html,.htm,.rtf"
        className="file-input-hidden"
        onChange={e => { if (e.target.files[0]) onUpload(e.target.files[0]); }}
      />
      <div className="upload-ico">📂</div>
      <div className="upload-ttl">Drop file or click to upload</div>
      <div className="upload-sub">Chunks, embeds and stores in Supabase</div>
      <div className="type-badges">
        {TYPES.map(t => <span key={t} className="tbadge">{t}</span>)}
      </div>
    </div>
  );
}
