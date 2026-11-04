import React, { useState } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, ArrowRight, BarChart2, TrendingUp, Sparkles } from 'lucide-react';
import { uploadPriceFile, uploadNewsFile } from '../services/api';

export default function DataIngestionTab({ onTickerAdded, onSelectTab }) {
  // Price Upload States
  const [priceFile, setPriceFile] = useState(null);
  const [priceTicker, setPriceTicker] = useState('TSLA');
  const [priceLoading, setPriceLoading] = useState(false);
  const [priceResult, setPriceResult] = useState(null);
  const [priceError, setPriceError] = useState(null);

  // News Upload States
  const [newsFile, setNewsFile] = useState(null);
  const [newsTicker, setNewsTicker] = useState('AAPL');
  const [newsLoading, setNewsLoading] = useState(false);
  const [newsResult, setNewsResult] = useState(null);
  const [newsError, setNewsError] = useState(null);

  const handlePriceUpload = async (e) => {
    e.preventDefault();
    if (!priceFile) {
      setPriceError('Please select a CSV file to upload.');
      return;
    }
    if (!priceTicker.trim()) {
      setPriceError('Please specify a stock ticker symbol (e.g. TSLA).');
      return;
    }

    setPriceLoading(true);
    setPriceError(null);
    setPriceResult(null);

    try {
      const res = await uploadPriceFile(priceFile, priceTicker);
      setPriceResult(res);
      if (onTickerAdded) {
        onTickerAdded(res.ticker);
      }
    } catch (err) {
      setPriceError(err.message);
    } finally {
      setPriceLoading(false);
    }
  };

  const handleNewsUpload = async (e) => {
    e.preventDefault();
    if (!newsFile) {
      setNewsError('Please select a CSV file containing headlines.');
      return;
    }

    setNewsLoading(true);
    setNewsError(null);
    setNewsResult(null);

    try {
      const res = await uploadNewsFile(newsFile, newsTicker);
      setNewsResult(res);
    } catch (err) {
      setNewsError(err.message);
    } finally {
      setNewsLoading(false);
    }
  };

  return (
    <div>
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '6px' }}>
          Data Ingestion & Pipeline Integration
        </h2>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
          Upload new market price series or raw financial news datasets. The automated backend calculates technical indicators, 
          scores headlines across 4 financial sentiment engines (Financial VADER, Loughran-McDonald, FinBERT, Ensemble), 
          and aligns them with price moves for immediate predictive analysis.
        </p>
      </div>

      <div className="grid-cols-2">
        {/* Card 1: Stock Price History Upload */}
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingUp size={20} color="var(--accent-blue)" />
              <div className="card-title">1. Ingest Stock Price History (OHLCV)</div>
            </div>
            <span className="badge badge-bullish">Price Data</span>
          </div>

          <form onSubmit={handlePriceUpload}>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '600', marginBottom: '6px' }}>
                Stock Ticker Symbol:
              </label>
              <input
                type="text"
                value={priceTicker}
                onChange={(e) => setPriceTicker(e.target.value.toUpperCase())}
                placeholder="e.g. TSLA, MSFT, SPY, BTC"
                className="ticker-select"
                style={{ width: '100%', padding: '10px 14px' }}
                required
              />
            </div>

            {/* Dropzone */}
            <div
              style={{
                border: '2px dashed var(--border-color)',
                borderRadius: '8px',
                padding: '24px',
                textAlign: 'center',
                backgroundColor: 'var(--bg-secondary)',
                marginBottom: '16px',
                cursor: 'pointer',
              }}
              onClick={() => document.getElementById('price-file-input').click()}
            >
              <UploadCloud size={32} color="var(--accent-blue)" style={{ margin: '0 auto 8px auto' }} />
              <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)' }}>
                {priceFile ? priceFile.name : 'Click or drag price CSV here'}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {priceFile ? `${(priceFile.size / 1024).toFixed(1)} KB` : 'Expected columns: Date, Close (or Open, High, Low, Volume)'}
              </div>
              <input
                id="price-file-input"
                type="file"
                accept=".csv"
                style={{ display: 'none' }}
                onChange={(e) => setPriceFile(e.target.files[0])}
              />
            </div>

            <button
              type="submit"
              className="btn-primary"
              style={{ width: '100%', justifyContent: 'center', padding: '10px' }}
              disabled={priceLoading}
            >
              {priceLoading ? 'Processing Prices & Technicals...' : 'Ingest Price & Compute Indicators →'}
            </button>
          </form>

          {priceError && (
            <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'var(--bearish-bg)', color: 'var(--bearish)', borderRadius: '8px', fontSize: '12px' }}>
              <AlertCircle size={14} style={{ display: 'inline', marginRight: '6px' }} />
              {priceError}
            </div>
          )}

          {priceResult && (
            <div style={{ marginTop: '16px', padding: '16px', backgroundColor: 'var(--bg-secondary)', borderRadius: '8px', border: '1px solid var(--bullish)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--bullish)', fontWeight: '600', marginBottom: '8px' }}>
                <CheckCircle2 size={16} />
                {priceResult.message}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px', marginBottom: '12px' }}>
                <div><strong>Total Bars:</strong> {priceResult.total_bars?.toLocaleString()}</div>
                <div><strong>Latest Close:</strong> ${priceResult.latest_close}</div>
                <div><strong>Start Date:</strong> {priceResult.start_date}</div>
                <div><strong>End Date:</strong> {priceResult.end_date}</div>
              </div>
              <button
                className="btn-primary"
                style={{ width: '100%', padding: '8px', fontSize: '12px', backgroundColor: 'var(--accent-blue)' }}
                onClick={() => onSelectTab('technical')}
              >
                View in Technical Chart Tab →
              </button>
            </div>
          )}
        </div>

        {/* Card 2: Financial News Dataset Upload */}
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <FileText size={20} color="var(--accent-blue)" />
              <div className="card-title">2. Ingest Financial News Headlines (CSV)</div>
            </div>
            <span className="badge badge-bullish">NLP Sentiment</span>
          </div>

          <form onSubmit={handleNewsUpload}>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '600', marginBottom: '6px' }}>
                Associated Stock Ticker (Optional):
              </label>
              <input
                type="text"
                value={newsTicker}
                onChange={(e) => setNewsTicker(e.target.value.toUpperCase())}
                placeholder="e.g. AAPL, NVDA, TSLA (or leave blank)"
                className="ticker-select"
                style={{ width: '100%', padding: '10px 14px' }}
              />
            </div>

            {/* Dropzone */}
            <div
              style={{
                border: '2px dashed var(--border-color)',
                borderRadius: '8px',
                padding: '24px',
                textAlign: 'center',
                backgroundColor: 'var(--bg-secondary)',
                marginBottom: '16px',
                cursor: 'pointer',
              }}
              onClick={() => document.getElementById('news-file-input').click()}
            >
              <UploadCloud size={32} color="var(--accent-blue)" style={{ margin: '0 auto 8px auto' }} />
              <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)' }}>
                {newsFile ? newsFile.name : 'Click or drag news CSV here'}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {newsFile ? `${(newsFile.size / 1024).toFixed(1)} KB` : 'Expected columns: headline (or title), date, publisher'}
              </div>
              <input
                id="news-file-input"
                type="file"
                accept=".csv"
                style={{ display: 'none' }}
                onChange={(e) => setNewsFile(e.target.files[0])}
              />
            </div>

            <button
              type="submit"
              className="btn-primary"
              style={{ width: '100%', justifyContent: 'center', padding: '10px' }}
              disabled={newsLoading}
            >
              {newsLoading ? 'Scoring Financial Sentiment across 4 Engines...' : 'Score Sentiment & Ingest News →'}
            </button>
          </form>

          {newsError && (
            <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'var(--bearish-bg)', color: 'var(--bearish)', borderRadius: '8px', fontSize: '12px' }}>
              <AlertCircle size={14} style={{ display: 'inline', marginRight: '6px' }} />
              {newsError}
            </div>
          )}

          {newsResult && (
            <div style={{ marginTop: '16px', padding: '16px', backgroundColor: 'var(--bg-secondary)', borderRadius: '8px', border: '1px solid var(--bullish)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--bullish)', fontWeight: '600', marginBottom: '8px' }}>
                <CheckCircle2 size={16} />
                {newsResult.message}
              </div>

              {/* Sentiment Breakdown */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginBottom: '12px' }}>
                <div style={{ padding: '8px', backgroundColor: 'var(--bg-card)', borderRadius: '6px', textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: 'var(--bullish)', fontWeight: '600' }}>Positive</div>
                  <div style={{ fontSize: '16px', fontWeight: '700', fontFamily: 'var(--font-mono)' }}>
                    {newsResult.sentiment_breakdown?.positive?.percentage}%
                  </div>
                </div>
                <div style={{ padding: '8px', backgroundColor: 'var(--bg-card)', borderRadius: '6px', textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: 'var(--neutral)', fontWeight: '600' }}>Neutral</div>
                  <div style={{ fontSize: '16px', fontWeight: '700', fontFamily: 'var(--font-mono)' }}>
                    {newsResult.sentiment_breakdown?.neutral?.percentage}%
                  </div>
                </div>
                <div style={{ padding: '8px', backgroundColor: 'var(--bg-card)', borderRadius: '6px', textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: 'var(--bearish)', fontWeight: '600' }}>Negative</div>
                  <div style={{ fontSize: '16px', fontWeight: '700', fontFamily: 'var(--font-mono)' }}>
                    {newsResult.sentiment_breakdown?.negative?.percentage}%
                  </div>
                </div>
              </div>

              {newsResult.top_bullish_sample && (
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '8px', lineHeight: '1.4' }}>
                  <strong style={{ color: 'var(--bullish)' }}>Bullish Highlight:</strong> "{newsResult.top_bullish_sample.slice(0, 80)}..."
                </div>
              )}

              <button
                className="btn-primary"
                style={{ width: '100%', padding: '8px', fontSize: '12px', backgroundColor: 'var(--accent-blue)' }}
                onClick={() => onSelectTab('sentiment')}
              >
                View Scored Articles in Sentiment Tab →
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
