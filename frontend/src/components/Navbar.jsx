import React from 'react';
import { Activity, Sparkles, Sun, Moon } from 'lucide-react';

export default function Navbar({
  selectedTicker,
  onSelectTicker,
  onOpenPredictor,
  healthStatus,
  availableTickers,
  theme,
  onToggleTheme,
}) {
  const tickers = availableTickers && availableTickers.length > 0
    ? availableTickers
    : ['AAPL', 'NVDA', 'MSFT', 'GOOG', 'AMZN', 'META'];

  return (
    <header className="navbar">
      <div className="nav-brand">
        <span className="brand-icon">📈</span>
        <div>
          <div className="brand-title">Financial Alpha & Sentiment Intelligence</div>
          <div className="brand-subtitle">NLP Price Direction Predictor • Institutional Terminal</div>
        </div>
      </div>

      <div className="nav-actions">
        <button
          className="theme-toggle-btn"
          onClick={onToggleTheme}
          title={theme === 'dark' ? 'Switch to Light (White Background)' : 'Switch to Dark (Black Background)'}
        >
          {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
          <span>{theme === 'dark' ? 'Light Theme' : 'Dark Theme'}</span>
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Ticker:</span>
          <select
            className="ticker-select"
            value={selectedTicker}
            onChange={(e) => onSelectTicker(e.target.value)}
          >
            {tickers.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>

        <button className="btn-primary" onClick={onOpenPredictor}>
          <Sparkles size={15} />
          Live Headline Predictor
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: healthStatus === 'online' ? 'var(--bullish)' : 'var(--bearish)',
              boxShadow: healthStatus === 'online' ? '0 0 8px var(--bullish)' : 'none',
            }}
          />
          API {healthStatus || 'connecting...'}
        </div>
      </div>
    </header>
  );
}
