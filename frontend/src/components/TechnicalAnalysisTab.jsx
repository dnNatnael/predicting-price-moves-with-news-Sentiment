import React, { useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';

export default function TechnicalAnalysisTab({ ticker, prices }) {
  const [timeframe, setTimeframe] = useState(120); // 120 bars default

  const displayPrices = (prices || []).slice(-timeframe);

  return (
    <div>
      {/* Price & Moving Averages Chart */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-header">
          <div>
            <div className="card-title">Technical Analysis & Price History ({ticker})</div>
            <div className="card-subtitle">Daily Close with SMA-20, SMA-50, and News Volume Integration</div>
          </div>

          <div style={{ display: 'flex', gap: '6px' }}>
            {[60, 120, 252].map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className="ticker-select"
                style={{
                  padding: '4px 10px',
                  backgroundColor: timeframe === tf ? 'var(--accent-blue)' : 'var(--bg-secondary)',
                  color: timeframe === tf ? '#ffffff' : 'var(--text-muted)',
                  border: '1px solid var(--border-color)',
                }}
              >
                {tf === 60 ? '3M' : tf === 120 ? '6M' : '1Y'}
              </button>
            ))}
          </div>
        </div>

        <div style={{ height: 360 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={displayPrices} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis dataKey="date" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="price" domain={['auto', 'auto']} stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="news" orientation="right" domain={[0, 'auto']} stroke="var(--text-muted)" hide />
              <Tooltip
                contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '8px', color: 'var(--text-primary)' }}
                formatter={(val, name) => [
                  typeof val === 'number' ? (name.includes('Return') ? `${(val * 100).toFixed(2)}%` : `$${val.toFixed(2)}`) : val,
                  name,
                ]}
              />
              <Legend verticalAlign="top" height={36} />
              <Bar yAxisId="news" dataKey="news_volume" fill="rgba(59, 130, 246, 0.15)" name="News Articles" />
              <Area yAxisId="price" type="monotone" dataKey="close" stroke="#3b82f6" fill="rgba(59, 130, 246, 0.08)" name="Close Price" />
              <Line yAxisId="price" type="monotone" dataKey="sma_20" stroke="#f59e0b" dot={false} strokeWidth={1.5} name="SMA 20" />
              <Line yAxisId="price" type="monotone" dataKey="sma_50" stroke="#8b5cf6" dot={false} strokeWidth={1.5} name="SMA 50" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Secondary Indicators: RSI & Sentiment Velocity */}
      <div className="grid-cols-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">RSI (Relative Strength Index - 14)</div>
              <div className="card-subtitle">Momentum oscillator with 70/30 overbought/oversold bands</div>
            </div>
          </div>

          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={displayPrices}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis dataKey="date" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 100]} stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '8px', color: 'var(--text-primary)' }}
                  formatter={(v) => [v.toFixed(1), 'RSI']}
                />
                <Line type="monotone" dataKey="rsi_14" stroke="#ec4899" dot={false} strokeWidth={1.5} name="RSI 14" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Daily News Sentiment Score</div>
              <div className="card-subtitle">Mean sentiment polarity of headlines mapped to each trading date</div>
            </div>
          </div>

          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={displayPrices}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis dataKey="date" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
                <YAxis domain={[-1, 1]} stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '8px', color: 'var(--text-primary)' }}
                  formatter={(v) => [v?.toFixed(3), 'Daily Sentiment']}
                />
                <Bar
                  dataKey="sentiment_mean"
                  name="Sentiment Score"
                  fill="#10b981"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

