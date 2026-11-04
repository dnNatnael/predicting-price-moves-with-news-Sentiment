import React, { useState } from 'react';
import { X, Sparkles, TrendingUp, TrendingDown, ArrowRight } from 'lucide-react';
import { predictHeadline } from '../services/api';

export default function LivePredictorModal({ isOpen, onClose, defaultTicker = 'AAPL' }) {
  const [headline, setHeadline] = useState(
    'Company beats Q4 revenue estimates by 9% and raises full-year profit guidance amid surging demand'
  );
  const [ticker, setTicker] = useState(defaultTicker);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const samples = [
    { text: 'Company beats Q4 earnings by 12% and raises annual dividend payout', ticker: 'AAPL' },
    { text: 'DOJ initiates antitrust investigation into artificial intelligence chip supply deals', ticker: 'NVDA' },
    { text: 'Revenues slump 15% missing consensus estimates amid persistent macroeconomic headwinds', ticker: 'AAPL' },
    { text: 'FDA grants expedited breakthrough designation for novel therapeutic candidate', ticker: 'NVDA' },
  ];

  const handlePredict = async () => {
    if (!headline.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictHeadline(headline, ticker);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sparkles size={20} color="var(--accent-blue)" />
            <h2 className="modal-title">Live Financial Headline Predictor</h2>
          </div>
          <button className="modal-close" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        {/* Input Form */}
        <div style={{ marginBottom: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase' }}>
              Target Headline:
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Ticker:</span>
              <select
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                className="ticker-select"
                style={{ padding: '4px 8px' }}
              >
                {['AAPL', 'NVDA', 'MSFT', 'GOOG', 'AMZN', 'META'].map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <textarea
            className="textarea-input"
            value={headline}
            onChange={(e) => setHeadline(e.target.value)}
            placeholder="Type any financial headline, earnings report, or merger rumor..."
          />

          {/* Sample Prompts */}
          <div style={{ marginBottom: '16px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              Or click a sample scenario:
            </span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {samples.map((s, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setHeadline(s.text);
                    setTicker(s.ticker);
                  }}
                  style={{
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-secondary)',
                    fontSize: '11px',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    textAlign: 'left',
                  }}
                >
                  {s.ticker}: "{s.text.slice(0, 42)}..."
                </button>
              ))}
            </div>
          </div>

          <button
            className="btn-primary"
            style={{ width: '100%', justifyContent: 'center', padding: '10px' }}
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? 'Evaluating Financial Tone & Model...' : 'Score Sentiment & Predict Price Move →'}
          </button>
        </div>

        {error && (
          <div style={{ padding: '12px', backgroundColor: 'var(--bearish-bg)', color: 'var(--bearish)', borderRadius: '8px', fontSize: '13px' }}>
            {error}
          </div>
        )}

        {/* Prediction Results */}
        {result && (
          <div style={{ marginTop: '20px', padding: '18px', backgroundColor: 'var(--bg-secondary)', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '700' }}>
                  Model Prediction ({result.model_used})
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                  {result.predicted_next_day_direction === 'UP' ? (
                    <TrendingUp size={24} color="var(--bullish)" />
                  ) : (
                    <TrendingDown size={24} color="var(--bearish)" />
                  )}
                  <span
                    style={{
                      fontSize: '22px',
                      fontWeight: '800',
                      color: result.predicted_next_day_direction === 'UP' ? 'var(--bullish)' : 'var(--bearish)',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    NEXT-DAY {result.predicted_next_day_direction}
                  </span>
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Direction Confidence</span>
                <div style={{ fontSize: '20px', fontWeight: '700', fontFamily: 'var(--font-mono)' }}>
                  {(result.up_probability * 100).toFixed(1)}% UP
                </div>
              </div>
            </div>

            {/* Confidence Bar */}
            <div style={{ height: '8px', width: '100%', backgroundColor: 'var(--bearish)', borderRadius: '4px', overflow: 'hidden', display: 'flex', marginBottom: '16px' }}>
              <div style={{ width: `${result.up_probability * 100}%`, height: '100%', backgroundColor: 'var(--bullish)' }} />
            </div>

            {/* Sub-scores Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '14px' }}>
              {[
                { name: 'Ensemble', val: result.sentiment_ensemble, highlight: true },
                { name: 'Fin VADER', val: result.sentiment_fin_vader },
                { name: 'Loughran-M.', val: result.sentiment_lm },
                { name: 'FinBERT', val: result.sentiment_finbert },
              ].map((s) => (
                <div
                  key={s.name}
                  style={{
                    padding: '8px 10px',
                    backgroundColor: 'var(--bg-card)',
                    borderRadius: '6px',
                    border: s.highlight ? '1px solid var(--accent-blue)' : '1px solid var(--border-color)',
                  }}
                >
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{s.name}</div>
                  <div style={{ fontSize: '14px', fontWeight: '700', fontFamily: 'var(--font-mono)', marginTop: '2px', color: s.val > 0 ? 'var(--bullish)' : s.val < 0 ? 'var(--bearish)' : 'inherit' }}>
                    {s.val > 0 ? `+${s.val.toFixed(3)}` : s.val.toFixed(3)}
                  </div>
                </div>
              ))}
            </div>

            {/* Financial Explanation */}
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              {result.explanation}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

