import React from 'react';

export default function MetricCard({ title, value, subtext, status = 'neutral', icon: Icon }) {
  const getStatusClass = () => {
    if (status === 'bullish') return 'text-bullish';
    if (status === 'bearish') return 'text-bearish';
    return 'text-neutral';
  };

  return (
    <div className="metric-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <span className="metric-label">{title}</span>
        {Icon && <Icon size={18} color="var(--text-muted)" />}
      </div>
      <div className="metric-value">{value}</div>
      {subtext && <div className={`metric-sub ${getStatusClass()}`}>{subtext}</div>}
    </div>
  );
}

