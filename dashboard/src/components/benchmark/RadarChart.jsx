import { useRef, useEffect } from 'react';

export default function RadarChart({ scores, thresholds }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2 + 10;
    const R = Math.min(W, H) / 2 - 55;
    ctx.clearRect(0, 0, W, H);

    const labels = Object.keys(scores);
    const vals = Object.values(scores);
    const thrVals = labels.map(l => thresholds[l] ?? 0.7);
    const n = labels.length;
    if (n < 3) return;

    const ang = i => (i / n) * Math.PI * 2 - Math.PI / 2;
    const pt = (i, r) => ({ x: cx + r * Math.cos(ang(i)), y: cy + r * Math.sin(ang(i)) });

    [0.25, 0.5, 0.75, 1.0].forEach(lvl => {
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const p = pt(i, R * lvl);
        i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 1;
      ctx.stroke();
      if (lvl < 1) {
        ctx.fillStyle = 'rgba(255,255,255,0.15)';
        ctx.font = '9px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.fillText(`${lvl * 100 | 0}%`, cx, cy - R * lvl + 3);
      }
    });

    for (let i = 0; i < n; i++) {
      const p = pt(i, R);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.stroke();
    }

    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const p = pt(i, R * thrVals[i]);
      i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.strokeStyle = 'rgba(240,165,0,0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const p = pt(i, R * vals[i]);
      i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(0,212,170,0.12)';
    ctx.fill();
    ctx.strokeStyle = '#00d4aa';
    ctx.lineWidth = 2;
    ctx.stroke();

    for (let i = 0; i < n; i++) {
      const p = pt(i, R * vals[i]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = vals[i] >= thrVals[i] ? '#00d4aa' : '#ff6b9d';
      ctx.fill();
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'rgba(200,210,220,0.7)';
    ctx.font = '9px JetBrains Mono';
    for (let i = 0; i < n; i++) {
      const p = pt(i, R + 28);
      const lines = labels[i].replace(/_/g, '\n').split('\n');
      lines.forEach((l, li) => {
        ctx.fillText(l, p.x, p.y + li * 12 - (lines.length - 1) * 6);
      });
    }
  }, [scores, thresholds]);

  return <canvas ref={canvasRef} width={360} height={320} />;
}
