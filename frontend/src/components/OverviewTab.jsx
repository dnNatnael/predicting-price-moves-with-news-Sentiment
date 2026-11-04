import React from 'react';
import { Newspaper, TrendingUp, ShieldCheck, Zap, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import MetricCard from './MetricCard';

export default function OverviewTab({ overview, news, models, onSelectTab }) {
  if (!overview) return <div>Loading overview metrics...</div>;

  const sentimentData = [
    { name: 'Positive', value: overview.sentiment_distribution?.positive || 30.6, color: '#10b981' },
    { name: 'Neutral', value: overview.sentiment_distribution?.neutral || 41.5, color: '#8b5cf6' },
    { name: 'Negative', value: overview.sentiment_distribution?.negative || 27.9, color: '#ef4444' },
  ];

  return (
    <div>
      {/* KPI Cards */}
      <div className="grid-cols-4">
        <MetricCard
          title="Total Financial News"
          value={overview.total_news_articles?.toLocaleString()}
          subtext="Coverage across 55k+ articles"
          status="bullish"
          icon={Newspaper}
        />
        <MetricCard
          title="Historical Trading Bars"
          value={overview.total_trading_days?.toLocaleString()}
          subtext={`${overview.date_range?.start} → ${overview.date_range?.end}`}
          status="neutral"
          icon={TrendingUp}
        />
        <MetricCard
          title="Best Model Test Accuracy"
          value={`${(overview.best_model_accuracy * 100).toFixed(2)}%`}
          subtext={`Engine: ${overview.best_model_name}`}
          status="bullish"
          icon={Zap}
        />
        <MetricCard
          title="Walk-Forward Sharpe Ratio"
          value={overview.walk_forward_sharpe?.toFixed(2)}
          subtext={`CAGR: +${overview.walk_forward_cagr?.toFixed(1)}% | MDD: -${overview.walk_forward_mdd?.toFixed(1)}%`}
          status={overview.walk_forward_sharpe >= 1.0 ? 'bullish' : 'neutral'}
          icon={ShieldCheck}
        />
      </div>

      {/* Main Grid: Sentiment Distribution & Top Alpha Features */}
      <div className="grid-cols-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Ensemble Sentiment Polarity Share</div>
              <div className="card-subtitle">Synthesizing Financial VADER, Loughran-McDonald, FinBERT, & TextBlob</div>
            </div>
            <span className="badge badge-bullish">55,987 Headlines</span>
          </div>

          <div style={{ height: 260, display: 'flex', alignItems: 'center' }}>
            <ResponsiveContainer width="55%" height="100%">
              <PieChart>
                <Pie
                  data={sentimentData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {sentimentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '8px', color: 'var(--text-primary)' }}
                  formatter={(val) => [`${val.toFixed(1)}%`, 'Share']}
                />
              </PieChart>
            </ResponsiveContainer>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '45%' }}>
              {sentimentData.map((item) => (
                <div key={item.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: item.color }} />
                    <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{item.name}</span>
                  </div>
                  <span style={{ fontSize: '14px', fontWeight: '700', fontFamily: 'var(--font-mono)' }}>
                    {item.value.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Top Alpha Drivers */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Top Predictive Alpha Features</div>
              <div className="card-subtitle">Non-collinear features selected by correlation screening</div>
            </div>
            <button
              onClick={() => onSelectTab('correlations')}
              style={{ background: 'none', border: 'none', color: 'var(--accent-blue)', cursor: 'pointer', fontSize: '12px', fontWeight: '600' }}
            >
              View All →
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {(models?.top_features || []).slice(0, 5).map((f, i) => (
              <div
                key={f.feature}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '10px 14px',
                  backgroundColor: 'var(--bg-secondary)',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>#{i + 1}</span>
                  <span style={{ fontSize: '13px', fontWeight: '500' }}>{f.feature}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '80px', height: '6px', backgroundColor: 'var(--border-color)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ width: `${Math.min(100, f.importance * 300)}%`, height: '100%', backgroundColor: 'var(--accent-blue)' }} />
                  </div>
                  <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', fontWeight: '600' }}>
                    {(f.importance * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent High Impact News */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Recent Breaking Financial News</div>
            <div className="card-subtitle">Real-time sentiment scored across 4 financial NLP algorithms</div>
          </div>
          <button
            onClick={() => onSelectTab('sentiment')}
            style={{ background: 'none', border: 'none', color: 'var(--accent-blue)', cursor: 'pointer', fontSize: '12px', fontWeight: '600' }}
          >
            Explore All Headlines →
          </button>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Headline</th>
                <th>Publisher</th>
                <th>VADER</th>
                <th>FinBERT</th>
                <th>Ensemble</th>
                <th>Tone</th>
              </tr>
            </thead>
            <tbody>
              {(news || []).slice(0, 6).map((item, idx) => (
                <tr key={idx}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>{item.date}</td>
                  <td style={{ maxWidth: '420px', fontWeight: '500' }}>{item.headline}</td>
                  <td style={{ color: 'var(--text-muted)' }}>{item.publisher}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{item.fin_vader > 0 ? `+${item.fin_vader}` : item.fin_vader}</td>
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

