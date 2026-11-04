import React, { useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { Trophy, Award } from 'lucide-react';

export default function ModelPerformanceTab({ ticker, models }) {
  const [targetMode, setTargetMode] = useState('binary');

  const modelRows = targetMode === 'binary' ? models?.binary_models || [] : models?.multiclass_models || [];
  const topFeatures = models?.top_features || [];

  return (
    <div>
      {/* Mode Switcher */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: '700' }}>Predictive Machine Learning Models ({ticker})</h2>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Strict 70% Train / 15% Validation / 15% Out-of-Sample Test Chronological Split
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => setTargetMode('binary')}
            className={`tab-btn ${targetMode === 'binary' ? 'active' : ''}`}
            style={{ padding: '8px 16px', borderRadius: '6px' }}
          >
            Binary (Up vs Down)
          </button>
          <button
            onClick={() => setTargetMode('multiclass')}
            className={`tab-btn ${targetMode === 'multiclass' ? 'active' : ''}`}
            style={{ padding: '8px 16px', borderRadius: '6px' }}
          >
            3-Class (Up / Neutral / Down)
          </button>
        </div>
      </div>

      {/* Model Benchmark Table */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div className="card-header">
          <div className="card-title">Out-of-Sample Test Performance Comparison</div>
          <span className="badge badge-bullish">Zero Future Data Leakage</span>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Test Accuracy</th>
                <th>Macro F1</th>
                <th>ROC-AUC</th>
                <th>Strategy Cumulative Return</th>
                <th>Strategy Sharpe</th>
              </tr>
            </thead>
            <tbody>
              {modelRows.map((m, idx) => (
                <tr key={m.model_name}>
                  <td style={{ fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {idx === 0 && <Trophy size={16} color="#f59e0b" />}
                    {m.model_name}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--bullish)' }}>
                    {(m.test_accuracy * 100).toFixed(2)}%
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{m.test_f1_macro.toFixed(4)}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {m.test_roc_auc ? m.test_roc_auc.toFixed(4) : 'N/A'}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: m.strategy_cumulative_return > 0 ? 'var(--bullish)' : 'var(--bearish)' }}>
                    {m.strategy_cumulative_return > 0 ? `+${(m.strategy_cumulative_return * 100).toFixed(2)}%` : `${(m.strategy_cumulative_return * 100).toFixed(2)}%`}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{m.strategy_sharpe_ratio.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Importance & Model Insights */}
      <div className="grid-cols-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Model Feature Importances (Gini / Weights)</div>
              <div className="card-subtitle">Contribution of each engineered feature to the final prediction</div>
            </div>
          </div>

          <div style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={topFeatures}
                layout="vertical"
                margin={{ top: 10, right: 30, left: 100, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis type="number" stroke="var(--text-muted)" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <YAxis dataKey="feature" type="category" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '8px', color: 'var(--text-primary)' }}
                  formatter={(val) => [`${(val * 100).toFixed(1)}%`, 'Importance']}
                />
                <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div className="card-header">
              <div className="card-title">Quant Methodology & Insights</div>
            </div>
            <ul style={{ paddingLeft: '20px', fontSize: '13px', lineHeight: '1.8', color: 'var(--text-secondary)' }}>
              <li>
                <strong style={{ color: 'var(--text-primary)' }}>Lookahead Bias Elimination:</strong> Train set strictly ends before validation; validation ends before test set. Standard scalers are fit purely on train data.
              </li>
              <li>
                <strong style={{ color: 'var(--text-primary)' }}>Class Imbalance Balancing:</strong> Automated inverse frequency weighting prevents market upward drift skew.
              </li>
              <li>
                <strong style={{ color: 'var(--text-primary)' }}>Ensemble Advantage:</strong> Gradient Boosting and Logistic Regression capitalize on non-linear interaction between pre-market sentiment velocity and technical momentum (RSI/MACD).
              </li>
            </ul>
          </div>

          <div style={{ padding: '14px', backgroundColor: 'var(--bg-secondary)', borderRadius: '8px', border: '1px solid var(--border-color)', marginTop: '16px' }}>
            <span style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: '700' }}>Production Status</span>
            <div style={{ fontSize: '13px', color: 'var(--bullish)', fontWeight: '600', marginTop: '4px' }}>
              ✓ Model weights, scaler, and top features serialized in <code>output/models/</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

