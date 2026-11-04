import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { CheckCircle, AlertCircle } from 'lucide-react';

export default function CorrelationMatrixTab({ ticker, correlations }) {
  const displayItems = (correlations || []).slice(0, 15);

  return (
    <div>
      <div className="card" style={{ marginBottom: '24px' }}>
        <div className="card-header">
          <div>
            <div className="card-title">Top Predictive Alpha Features ({ticker})</div>
            <div className="card-subtitle">Ranked by Pearson r and Spearman ρ correlation with forward stock price moves</div>
          </div>
          <span className="badge badge-bullish">Step 3 Analytics</span>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Feature Name</th>
                <th>Pearson r</th>
                <th>Spearman ρ</th>
                <th>p-value</th>
                <th>Significance (p &lt; 0.05)</th>
              </tr>
            </thead>
            <tbody>
              {displayItems.map((item, idx) => (
                <tr key={item.feature}>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>#{idx + 1}</td>
                  <td style={{ fontWeight: '600', fontFamily: 'var(--font-mono)', fontSize: '13px' }}>{item.feature}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: item.pearson_r > 0 ? 'var(--bullish)' : 'var(--bearish)' }}>
                    {item.pearson_r > 0 ? `+${item.pearson_r.toFixed(4)}` : item.pearson_r.toFixed(4)}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {item.spearman_rho > 0 ? `+${item.spearman_rho.toFixed(4)}` : item.spearman_rho.toFixed(4)}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>{item.p_value.toFixed(5)}</td>
                  <td>
                    {item.is_significant ? (
                      <span className="badge badge-bullish">
                        <CheckCircle size={12} /> Significant
                      </span>
                    ) : (
                      <span className="badge" style={{ backgroundColor: 'rgba(100, 116, 139, 0.2)', color: 'var(--text-muted)' }}>
                        p &gt; 0.05
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Correlation Chart */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Correlation Strength by Feature</div>
            <div className="card-subtitle">Visualizing absolute predictive magnitude across technical and sentiment factors</div>
          </div>
        </div>

        <div style={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={displayItems}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 120, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis type="number" stroke="var(--text-muted)" tickFormatter={(v) => v.toFixed(3)} />
              <YAxis dataKey="feature" type="category" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '8px', color: 'var(--text-primary)' }}
                formatter={(val) => [`${val > 0 ? '+' : ''}${val.toFixed(4)}`, 'Pearson r']}
              />
              <Bar
                dataKey="pearson_r"
                fill="#3b82f6"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

