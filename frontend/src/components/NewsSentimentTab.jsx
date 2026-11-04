import React, { useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { Search, CheckCircle, XCircle } from 'lucide-react';

export default function NewsSentimentTab({ benchmark, news }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterClass, setFilterClass] = useState('ALL');

  const filteredNews = (news || []).filter((item) => {
    const matchesSearch = item.headline.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          item.publisher.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesClass = filterClass === 'ALL' || item.ensemble_class.toUpperCase() === filterClass;
    return matchesSearch && matchesClass;
  });

  return (
    <div>
      {/* Benchmark Comparison Table & Chart */}
      <div className="grid-cols-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Sentiment Approach Benchmark (Step 5)</div>
              <div className="card-subtitle">Correlation with forward stock returns across 55,987 news articles</div>
            </div>
            <span className="badge badge-bullish">5 Approaches</span>
          </div>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Approach</th>
                  <th>Pearson r</th>
                  <th>Spearman ρ</th>
                  <th>Pos %</th>
                  <th>Neg %</th>
                  <th>p &lt; 0.05</th>
                </tr>
              </thead>
              <tbody>
                {(benchmark || []).map((row) => (
                  <tr key={row.approach}>
                    <td style={{ fontWeight: row.approach.includes('Ensemble') ? '700' : '500' }}>
                      {row.approach}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: row.pearson_r > 0.05 ? 'var(--bullish)' : 'inherit' }}>
                      {row.pearson_r > 0 ? `+${row.pearson_r.toFixed(4)}` : row.pearson_r.toFixed(4)}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>
                      {row.spearman_rho > 0 ? `+${row.spearman_rho.toFixed(4)}` : row.spearman_rho.toFixed(4)}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--bullish)' }}>{row.pct_positive.toFixed(1)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--bearish)' }}>{row.pct_negative.toFixed(1)}%</td>
                    <td>
                      {row.is_significant ? (
                        <CheckCircle size={16} color="var(--bullish)" />
                      ) : (
                        <XCircle size={16} color="var(--text-muted)" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Predictive Return Correlation Chart */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Predictive Alpha by Scoring Engine</div>
              <div className="card-subtitle">Pearson correlation (r) with actual stock price moves</div>
            </div>
          </div>

          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={benchmark || []}
                layout="vertical"
                margin={{ top: 10, right: 30, left: 60, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis type="number" stroke="var(--text-muted)" tickFormatter={(v) => v.toFixed(3)} />
                <YAxis dataKey="approach" type="category" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '8px', color: 'var(--text-primary)' }}
                  formatter={(val) => [`+${val.toFixed(4)}`, 'Pearson r']}
                />
                <Bar dataKey="pearson_r" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* News Feed Explorer */}
      <div className="card">
        <div className="card-header" style={{ flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div className="card-title">Multi-Scorer Headline Sentiment Explorer</div>
            <div className="card-subtitle">Real-time breakdown of financial lexicons and transformer outputs</div>
          </div>

          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <div style={{ position: 'relative' }}>
              <Search size={14} style={{ position: 'absolute', left: '10px', top: '10px', color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="Search headlines or publisher..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{
                  backgroundColor: 'var(--bg-secondary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  padding: '6px 12px 6px 30px',
                  color: 'var(--text-primary)',
                  fontSize: '13px',
                  outline: 'none',
                  width: '240px',
                }}
              />
            </div>

            <select
              value={filterClass}
              onChange={(e) => setFilterClass(e.target.value)}
              className="ticker-select"
              style={{ padding: '6px 12px' }}
            >
              <option value="ALL">All Tones</option>
              <option value="POSITIVE">Positive</option>
              <option value="NEUTRAL">Neutral</option>
              <option value="NEGATIVE">Negative</option>
            </select>
          </div>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Headline</th>
                <th>Publisher</th>
                <th>VADER</th>
                <th>L&M</th>
                <th>FinBERT</th>
                <th>Ensemble</th>
                <th>Tone</th>
              </tr>
            </thead>
            <tbody>
              {filteredNews.map((item, idx) => (
                <tr key={idx}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', whiteSpace: 'nowrap' }}>{item.date}</td>
                  <td style={{ fontWeight: '500' }}>{item.headline}</td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '12px' }}>{item.publisher}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{item.fin_vader > 0 ? `+${item.fin_vader}` : item.fin_vader}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{item.lm_sentiment > 0 ? `+${item.lm_sentiment}` : item.lm_sentiment}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{item.finbert > 0 ? `+${item.finbert}` : item.finbert}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: '700' }}>
                    {item.ensemble > 0 ? `+${item.ensemble}` : item.ensemble}
                  </td>
                  <td>
                    <span
                      className={`badge ${
                        item.ensemble_class === 'Positive'
                          ? 'badge-bullish'
                          : item.ensemble_class === 'Negative'
                          ? 'badge-bearish'
                          : 'badge-neutral'
                      }`}
                    >
                      {item.ensemble_class}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

