import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  HelpCircle,
  Cpu,
  LineChart,
  Network,
  MessageSquareText,
  History,
  UploadCloud,
  CheckCircle2,
  TrendingUp,
  ShieldAlert,
  Layers,
  ArrowRight,
  Terminal,
  FileSpreadsheet,
  ChevronDown,
  ChevronUp,
  Mail,
  Phone,
  UserCheck,
} from 'lucide-react';

export default function AboutTab({ onSelectTab }) {
  const [activeSection, setActiveSection] = useState('overview');

  const scrollToSection = (sectionId) => {
    setActiveSection(sectionId);
    const element = document.getElementById(`section-${sectionId}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Automatically update active section pill as the user scrolls
  useEffect(() => {
    const sections = ['overview', 'guide', 'schemas', 'findings', 'contact'];
    const handleScroll = () => {
      const scrollPos = window.scrollY + 180;
      for (const id of sections) {
        const el = document.getElementById(`section-${id}`);
        if (el) {
          const top = el.offsetTop;
          const height = el.offsetHeight;
          if (scrollPos >= top && scrollPos < top + height) {
            setActiveSection(id);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const stats = [
    { label: 'Raw News Articles', val: '1,400,000+', desc: 'Parsed from analyst ratings dataset' },
    { label: 'Scored Headlines', val: '55,987', desc: 'Across mega-cap tech equities' },
    { label: 'Alpha Features per Bar', val: '113', desc: 'NLP sentiment, momentum & technicals' },
    { label: 'Walk-Forward Horizon', val: '13 Years', desc: 'Continuous out-of-sample testing (2011–2023)' },
    { label: 'Out-of-Sample Sharpe', val: '1.08', desc: 'vs 1.02 Buy & Hold benchmark on NVDA' },
    { label: 'Automated Tests', val: '73 / 73 Passed', desc: '100% test coverage with 0 failures' },
  ];

  const pipelineStages = [
    {
      num: '01',
      title: 'Sentiment Preparation & Time Alignment',
      icon: MessageSquareText,
      desc: 'Standardizes timestamps to Eastern Market Time (UTC-4) and maps news into Pre-Market, Regular Market, and After-Hours sessions. Applies a strict 16:00 market close cutoff to eliminate lookahead bias.',
    },
    {
      num: '02',
      title: '113-Feature Alpha Engineering',
      icon: Layers,
      desc: 'Calculates rolling sentiment momentum (3D/7D/14D), positive/negative ratio spreads, topic distribution shares, volume acceleration, and technical indicators (SMA-20, SMA-50, RSI-14, MACD, Volume Z-scores).',
    },
    {
      num: '03',
      title: 'Correlation Analysis & Alpha Selection',
      icon: Network,
      desc: 'Computes parametric Pearson r and non-parametric Spearman rho across T-1 through T-5 forward price returns. Statistically filters significant alpha drivers (p < 0.05) to eliminate market noise.',
    },
    {
      num: '04',
      title: 'Predictive Machine Learning',
      icon: Cpu,
      desc: 'Trains Gradient Boosting, Random Forest, and regularized Logistic Regression for next-day direction forecasting. Handles financial class imbalance via class-weight adjustments and temporal train/test splits.',
    },
    {
      num: '05',
      title: 'Multi-Scorer NLP Benchmarking',
      icon: LineChart,
      desc: 'Compares TextBlob, Financial VADER (Wall Street lexicon), Loughran-McDonald (SEC 10-K dictionary), FinBERT (transformer), and an optimized Ensemble Composite achieving a >5x correlation lift over standard NLP.',
    },
    {
      num: '06',
      title: 'Walk-Forward Backtesting & Risk Validation',
      icon: History,
      desc: 'Executes a realistic 13-year walk-forward test (26 rolling 6-month out-of-sample windows). Re-trains models sequentially to mimic real-world hedge fund execution with zero lookahead leakage.',
    },
  ];

  const tabsGuide = [
    {
      tabId: 'overview',
      title: '1. Overview Tab',
      icon: Layers,
      purpose: 'Executive summary of market regime, sentiment posture, and top predictive factors.',
      howTo: [
        'Select any stock ticker in the top navigation to update all macro KPIs.',
        'View the 24-Hour Sentiment Posture donut chart (Bullish vs Bearish vs Neutral).',
        'Review the Top Alpha Drivers table ranked by correlation with forward returns.',
        'Read recent breaking headlines with live sentiment category badges.',
      ],
    },
    {
      tabId: 'sentiment',
      title: '2. Sentiment Engines Tab',
      icon: MessageSquareText,
      purpose: 'Compares the 4 quantitative NLP models against standard general English NLP.',
      howTo: [
        'Examine the multi-scorer benchmark table comparing TextBlob, Fin-VADER, Loughran-McDonald, FinBERT, and the Ensemble.',
        'Observe the correlation bar chart showing how domain-specific lexicons provide higher predictive correlation (r > +0.13).',
        'Use the searchable Headline Sentiment Explorer to inspect individual headline scores across all 4 engines.',
      ],
    },
    {
      tabId: 'technical',
      title: '3. Technical & Price Tab',
      icon: LineChart,
      purpose: 'Visualizes historical OHLCV price action combined with news sentiment volume overlay.',
      howTo: [
        'Toggle between 3M, 6M, and 1Y timeframe views.',
        'Inspect moving average crossovers (SMA-20 and SMA-50) alongside daily closing prices.',
        'Monitor the RSI-14 oscillator to detect overbought (>70) and oversold (<30) momentum zones.',
        'Correlate price swings with daily sentiment volume bars colored by net polarity.',
      ],
    },
    {
      tabId: 'correlations',
      title: '4. Alpha Correlations Tab',
      icon: Network,
      purpose: 'Detailed statistical evaluation of lead/lag relationships between sentiment and stock returns.',
      howTo: [
        'Review the Ranked Alpha Factor Scorecard sorted by statistical significance.',
        'Check Pearson r, Spearman rho, and p-values to verify non-randomness (p < 0.05).',
        'Analyze which sentiment lag (T-1, T-2, T-3) has the strongest impact on next-day returns.',
      ],
    },
    {
      tabId: 'models',
      title: '5. ML Models Tab',
      icon: Cpu,
      purpose: 'In-depth performance metrics for trained classifiers and feature importance rankings.',
      howTo: [
        'Toggle between Binary classification (Up vs Down) and 3-Class classification (Bullish, Flat, Bearish).',
        'Compare Accuracy, Precision, Recall, F1 Score, and ROC-AUC across Gradient Boosting, Random Forest, and Logistic Regression.',
        'Inspect Gini Feature Importance bars to see which technical and sentiment indicators the model relies on most.',
      ],
    },
    {
      tabId: 'backtest',
      title: '6. Walk-Forward Backtest Tab',
      icon: History,
      purpose: 'Realistic 13-year continuous out-of-sample trading simulation and risk metrics.',
      howTo: [
        'Toggle between Logarithmic and Linear equity curve scaling.',
        'Compare the Quantitative Sentiment Strategy against the benchmark Buy & Hold strategy.',
        'Inspect the Underwater Drawdown chart to see how negative sentiment signals shielded capital during market downturns.',
        'Review the 5-Fold TimeSeriesSplit cross-validation stability table.',
      ],
    },
    {
      tabId: 'ingest',
      title: '7. Ingest Data Tab',
      icon: UploadCloud,
      purpose: 'Dynamic ingestion pipeline for processing external stock price and news CSV files.',
      howTo: [
        'Drag and drop any stock price CSV (e.g., downloaded from Yahoo Finance or from the project data folder). The system automatically computes technical indicators and adds the ticker to the dashboard dropdown.',
        'Drag and drop any news headline CSV. The backend automatically scores all 4 NLP engines and calculates return correlations.',
      ],
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', position: 'relative' }}>
      {/* Hero Header */}
      <div
        className="card"
        style={{
          padding: '36px',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-color)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div style={{ maxWidth: '960px' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 14px',
              borderRadius: '20px',
              backgroundColor: 'var(--bg-secondary)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              fontSize: '12px',
              fontWeight: '600',
              marginBottom: '16px',
              letterSpacing: '0.5px',
            }}
          >
            <BookOpen size={14} /> QUANTITATIVE ALPHA RESEARCH PLATFORM
          </div>
          <h1
            style={{
              fontSize: '28px',
              fontWeight: '800',
              color: 'var(--text-primary)',
              marginBottom: '12px',
              lineHeight: '1.3',
            }}
          >
            Predicting Stock Price Moves with Financial News Sentiment
          </h1>
          <p
            style={{
              fontSize: '15px',
              color: 'var(--text-secondary)',
              lineHeight: '1.6',
              marginBottom: '24px',
            }}
          >
            An end-to-end quantitative trading intelligence system that extracts predictive market signals from 
            unstructured financial news headlines. Generic NLP dictionaries fail on Wall Street vocabulary; this 
            platform utilizes domain-specialized lexicons (Financial VADER, Loughran-McDonald) and transformer models 
            (FinBERT) to generate 113 alpha features per trading day, validating directionality through a 13-year 
            walk-forward backtest.
          </p>

          {/* Clean Navigation Buttons: Smooth scrolls down to each section */}
          <div
            style={{
              display: 'flex',
              gap: '12px',
              flexWrap: 'wrap',
              alignItems: 'center',
              backgroundColor: 'var(--bg-secondary)',
              padding: '12px 16px',
              borderRadius: '10px',
              border: '1px solid var(--border-color)',
            }}
          >
            <button
              className={`filter-btn ${activeSection === 'overview' ? 'active' : ''}`}
              onClick={() => scrollToSection('overview')}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Layers size={13} /> Project Overview & Architecture
              <ChevronDown size={12} />
            </button>
            <button
              className={`filter-btn ${activeSection === 'guide' ? 'active' : ''}`}
              onClick={() => scrollToSection('guide')}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <HelpCircle size={13} /> How to Use the Dashboard
              <ChevronDown size={12} />
            </button>
            <button
              className={`filter-btn ${activeSection === 'schemas' ? 'active' : ''}`}
              onClick={() => scrollToSection('schemas')}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <FileSpreadsheet size={13} /> CSV Schemas & Upload Specs
              <ChevronDown size={12} />
            </button>
            <button
              className={`filter-btn ${activeSection === 'findings' ? 'active' : ''}`}
              onClick={() => scrollToSection('findings')}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <TrendingUp size={13} /> Key Quantitative Findings
              <ChevronDown size={12} />
            </button>
            <button
              className={`filter-btn ${activeSection === 'contact' ? 'active' : ''}`}
              onClick={() => scrollToSection('contact')}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <UserCheck size={13} /> Author & Contact
              <ChevronDown size={12} />
            </button>
          </div>
        </div>
      </div>

      {/* KPI Stats Strip */}
      <div className="metrics-grid">
        {stats.map((s, idx) => (
          <div key={idx} className="metric-card">
            <div className="metric-title">{s.label}</div>
            <div className="metric-value" style={{ fontSize: '24px', color: 'var(--accent-cyan)' }}>
              {s.val}
            </div>
            <div className="metric-subtitle">{s.desc}</div>
          </div>
        ))}
      </div>

      {/* SECTION 1: Project Overview & Architecture */}
      <div
        id="section-overview"
        style={{ scrollMarginTop: '100px', display: 'flex', flexDirection: 'column', gap: '24px' }}
      >
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={20} color="var(--accent-blue)" /> 1. Project Overview & Quantitative Architecture
            </h2>
            <button
              className="filter-btn"
              onClick={scrollToTop}
              style={{ padding: '4px 10px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              <ChevronUp size={12} /> Back to Top
            </button>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '24px' }}>
            The system processes raw headlines and daily price bars through six mathematically rigorous stages designed to eliminate lookahead bias and data leakage.
          </p>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
              gap: '16px',
            }}
          >
            {pipelineStages.map((st) => {
              const Icon = st.icon;
              return (
                <div
                  key={st.num}
                  style={{
                    padding: '20px',
                    backgroundColor: 'var(--bg-secondary)',
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '10px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div
                      style={{
                        width: '32px',
                        height: '32px',
                        borderRadius: '6px',
                        backgroundColor: 'var(--bg-card)',
                        border: '1px solid var(--border-color)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'var(--accent-blue)',
                      }}
                    >
                      <Icon size={18} />
                    </div>
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '12px',
                        fontWeight: '700',
                        color: 'var(--text-muted)',
                      }}
                    >
                      STAGE {st.num}
                    </span>
                  </div>
                  <div style={{ fontSize: '15px', fontWeight: '600', color: 'var(--text-primary)' }}>
                    {st.title}
                  </div>
                  <div style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                    {st.desc}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* SECTION 2: How to Use the Dashboard */}
      <div
        id="section-guide"
        style={{ scrollMarginTop: '100px', display: 'flex', flexDirection: 'column', gap: '24px' }}
      >
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <HelpCircle size={20} color="var(--accent-cyan)" /> 2. How to Use the Dashboard
            </h2>
            <button
              className="filter-btn"
              onClick={scrollToTop}
              style={{ padding: '4px 10px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              <ChevronUp size={12} /> Back to Top
            </button>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '20px' }}>
            Complete walkthrough of header controls, live AI inference, and all functional dashboard tabs.
          </p>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
              gap: '16px',
              marginBottom: '24px',
            }}
          >
            <div
              style={{
                padding: '20px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
              }}
            >
              <div style={{ fontSize: '15px', fontWeight: '600', color: 'var(--accent-blue)', marginBottom: '8px' }}>
                Ticker Selector Dropdown (Top-Right)
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                Located on the top right of the navigation bar. Click to switch between supported equities (
                <code>AAPL</code>, <code>NVDA</code>, <code>MSFT</code>, <code>GOOG</code>, <code>AMZN</code>, <code>META</code>, <code>TSLA</code>). 
                Selecting any ticker instantly recalculates and reloads all dashboard tabs for that company.
              </p>
            </div>

            <div
              style={{
                padding: '20px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
              }}
            >
              <div style={{ fontSize: '15px', fontWeight: '600', color: 'var(--neutral)', marginBottom: '8px' }}>
                Live Headline Predictor Modal
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                Click the purple <strong>"Live Headline Predictor"</strong> button in the header. Enter any breaking 
                headline (e.g. <em>"Company beats earnings expectations, raises full-year guidance"</em>) to score it 
                instantly across all 4 NLP models and generate a real-time next-day price direction prediction.
              </p>
            </div>
          </div>

          <h3 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '16px', color: 'var(--text-primary)' }}>
            Tab-by-Tab Navigation Guide
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {tabsGuide.map((tg) => {
              const Icon = tg.icon;
              return (
                <div
                  key={tg.tabId}
                  style={{
                    padding: '20px',
                    backgroundColor: 'var(--bg-secondary)',
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '12px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div
                        style={{
                          width: '28px',
                          height: '28px',
                          borderRadius: '6px',
                          backgroundColor: 'var(--bg-card)',
                          border: '1px solid var(--border-color)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: 'var(--accent-blue)',
                        }}
                      >
                        <Icon size={16} />
                      </div>
                      <span style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)' }}>
                        {tg.title}
                      </span>
                    </div>
                    <button
                      className="filter-btn"
                      style={{ padding: '4px 12px', fontSize: '12px' }}
                      onClick={() => onSelectTab(tg.tabId)}
                    >
                      Go to Tab <ArrowRight size={12} style={{ display: 'inline', marginLeft: '4px' }} />
                    </button>
                  </div>

                  <div style={{ fontSize: '13px', color: 'var(--accent-cyan)', fontWeight: '500' }}>
                    Purpose: {tg.purpose}
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {tg.howTo.map((step, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: '8px',
                          fontSize: '13px',
                          color: 'var(--text-secondary)',
                        }}
                      >
                        <CheckCircle2 size={14} color="var(--bullish)" style={{ marginTop: '2px', flexShrink: 0 }} />
                        <span>{step}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* SECTION 3: CSV Schemas & Upload Specs */}
      <div
        id="section-schemas"
        style={{ scrollMarginTop: '100px', display: 'flex', flexDirection: 'column', gap: '24px' }}
      >
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <FileSpreadsheet size={20} color="var(--bullish)" /> 3. CSV Schemas & Upload Specifications
            </h2>
            <button
              className="filter-btn"
              onClick={scrollToTop}
              style={{ padding: '4px 10px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              <ChevronUp size={12} /> Back to Top
            </button>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '24px' }}>
            You can drag-and-drop your own stock price or news headline CSV files into <strong>Tab 7 ("7. Ingest Data")</strong>. 
            The platform automatically cleans, validates, and incorporates them into the live intelligence pipeline.
          </p>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
              gap: '20px',
            }}
          >
            {/* Price CSV Format */}
            <div
              style={{
                padding: '24px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <FileSpreadsheet size={18} color="var(--accent-blue)" />
                <span style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)' }}>
                  1. Stock Price CSV Schema
                </span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                Compatible with standard Yahoo Finance, Bloomberg, or AlphaVantage daily CSV exports.
              </p>

              <table className="data-table" style={{ marginBottom: '16px' }}>
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>Date</code></td>
                    <td>YYYY-MM-DD</td>
                    <td><span className="badge badge-bullish">Required</span></td>
                  </tr>
                  <tr>
                    <td><code>Close</code></td>
                    <td>Float</td>
                    <td><span className="badge badge-bullish">Required</span></td>
                  </tr>
                  <tr>
                    <td><code>Open</code></td>
                    <td>Float</td>
                    <td><span className="badge badge-neutral">Optional</span></td>
                  </tr>
                  <tr>
                    <td><code>High</code></td>
                    <td>Float</td>
                    <td><span className="badge badge-neutral">Optional</span></td>
                  </tr>
                  <tr>
                    <td><code>Low</code></td>
                    <td>Float</td>
                    <td><span className="badge badge-neutral">Optional</span></td>
                  </tr>
                  <tr>
                    <td><code>Volume</code></td>
                    <td>Integer / Float</td>
                    <td><span className="badge badge-neutral">Optional</span></td>
                  </tr>
                </tbody>
              </table>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                * Filename determines ticker (e.g. <code>AMD.csv</code> registers ticker <code>AMD</code>).
              </div>
            </div>

            {/* News CSV Format */}
            <div
              style={{
                padding: '24px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <MessageSquareText size={18} color="var(--neutral)" />
                <span style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)' }}>
                  2. News Headlines CSV Schema
                </span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                Compatible with Kaggle news datasets, SEC EDGAR RSS feeds, or custom scrapers.
              </p>

              <table className="data-table" style={{ marginBottom: '16px' }}>
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>Date</code></td>
                    <td>ISO or YYYY-MM-DD</td>
                    <td><span className="badge badge-bullish">Required</span></td>
                  </tr>
                  <tr>
                    <td><code>Headline</code> (or <code>title</code>)</td>
                    <td>String</td>
                    <td><span className="badge badge-bullish">Required</span></td>
                  </tr>
                  <tr>
                    <td><code>Publisher</code></td>
                    <td>String</td>
                    <td><span className="badge badge-neutral">Optional</span></td>
                  </tr>
                  <tr>
                    <td><code>Stock</code> (or <code>ticker</code>)</td>
                    <td>String</td>
                    <td><span className="badge badge-neutral">Optional</span></td>
                  </tr>
                </tbody>
              </table>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                * The engine will run Financial VADER, Loughran-McDonald, FinBERT, and the Ensemble on all ingested headlines.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 4: Key Quantitative Findings */}
      <div
        id="section-findings"
        style={{ scrollMarginTop: '100px', display: 'flex', flexDirection: 'column', gap: '24px' }}
      >
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingUp size={20} color="var(--accent-blue)" /> 4. Key Quantitative Findings
            </h2>
            <button
              className="filter-btn"
              onClick={scrollToTop}
              style={{ padding: '4px 10px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              <ChevronUp size={12} /> Back to Top
            </button>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '20px' }}>
            Key takeaways established across the 1.4M article dataset and 13-year out-of-sample walk-forward backtests:
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div
              style={{
                padding: '20px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '8px',
                borderLeft: '4px solid var(--bullish)',
              }}
            >
              <div style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '6px' }}>
                1. Financial Domain NLP Yields a &gt;5x Correlation Lift over General NLP
              </div>
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                General English dictionaries (TextBlob) produced an insignificant Pearson correlation of <code>r = +0.0253</code> with forward returns. In contrast, <strong>Financial VADER</strong> and <strong>FinBERT</strong> achieved <code>r = +0.1300</code>, and the <strong>Ensemble Composite</strong> achieved a Spearman rank correlation of <code>rho = +0.1487</code>, demonstrating that domain adaptation is required to capture market alpha.
              </div>
            </div>

            <div
              style={{
                padding: '20px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '8px',
                borderLeft: '4px solid var(--accent-blue)',
              }}
            >
              <div style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '6px' }}>
                2. Risk-Mitigation via Asymmetric Regime Rotation
              </div>
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                During prolonged market crashes (e.g. 2018 tech selloff and 2022 rate hike cycles), sentiment indicators turned strongly negative prior to the steepest price declines. Rotating to cash based on sentiment signals cushioned maximum drawdown on NVDA from <strong>66.34% down to 54.47%</strong> (a 12-percentage-point risk cushion) and raised the annualized Sharpe ratio to <strong>1.08</strong> (vs 1.02 Buy & Hold).
              </div>
            </div>

            <div
              style={{
                padding: '20px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '8px',
                borderLeft: '4px solid var(--neutral)',
              }}
            >
              <div style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '6px' }}>
                3. Temporal Decay of Sentiment Alpha
              </div>
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                Lead/lag correlation analysis confirmed that sentiment signals decay rapidly. The correlation is strongest at <strong>T-1 (next trading day)</strong> and decays by ~65% at <strong>T-3</strong>, approaching zero by <strong>T-5</strong>. This underscores the necessity of high-frequency headline ingestion and immediate session alignment.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 5: Author & Contact Information */}
      <div
        id="section-contact"
        style={{ scrollMarginTop: '100px', display: 'flex', flexDirection: 'column', gap: '24px' }}
      >
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <UserCheck size={20} color="var(--accent-blue)" /> 5. Project Author & Contact Information
            </h2>
            <button
              className="filter-btn"
              onClick={scrollToTop}
              style={{ padding: '4px 10px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              <ChevronUp size={12} /> Back to Top
            </button>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '24px' }}>
            For quantitative research inquiries, machine learning collaborations, or deployment discussions, reach out directly:
          </p>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
              gap: '16px',
            }}
          >
            {/* Email Link Card - Opens Gmail Compose directly for questions */}
            <div
              className="contact-card"
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '14px',
                padding: '22px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '10px',
                border: '1px solid var(--border-color)',
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div
                  style={{
                    width: '44px',
                    height: '44px',
                    borderRadius: '10px',
                    backgroundColor: 'var(--bg-card)',
                    border: '1px solid var(--border-color)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--accent-blue)',
                    flexShrink: 0,
                  }}
                >
                  <Mail size={22} />
                </div>
                <div style={{ overflow: 'hidden' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '700', letterSpacing: '0.05em' }}>
                    Email Address
                  </div>
                  <div style={{ fontSize: '15px', fontWeight: '600', color: 'var(--text-primary)', marginTop: '2px', textOverflow: 'ellipsis', overflow: 'hidden' }}>
                    natnael.yilma27@gmail.com
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '4px' }}>
                <a
                  href="https://mail.google.com/mail/?view=cm&fs=1&to=natnael.yilma27@gmail.com&su=Question%20regarding%20Financial%20News%20Sentiment%20Project"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="filter-btn active"
                  style={{ padding: '6px 14px', fontSize: '12px', textDecoration: 'none' }}
                >
                  ✉️ Compose in Gmail
                </a>
                <a
                  href="mailto:natnael.yilma27@gmail.com?subject=Question%20regarding%20Financial%20News%20Sentiment%20Project"
                  className="filter-btn"
                  style={{ padding: '6px 12px', fontSize: '12px', textDecoration: 'none' }}
                >
                  Default Mail App
                </a>
              </div>
            </div>

            {/* Phone Link Card */}
            <div
              className="contact-card"
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '14px',
                padding: '22px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '10px',
                border: '1px solid var(--border-color)',
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div
                  style={{
                    width: '44px',
                    height: '44px',
                    borderRadius: '10px',
                    backgroundColor: 'var(--bg-card)',
                    border: '1px solid var(--border-color)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--bullish)',
                    flexShrink: 0,
                  }}
                >
                  <Phone size={22} />
                </div>
                <div style={{ overflow: 'hidden' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '700', letterSpacing: '0.05em' }}>
                    Phone (Click to Call)
                  </div>
                  <div style={{ fontSize: '15px', fontWeight: '600', color: 'var(--text-primary)', marginTop: '2px', fontFamily: 'var(--font-mono)' }}>
                    +251 929 709 562
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '4px' }}>
                <a
                  href="tel:+251929709562"
                  className="filter-btn active"
                  style={{ padding: '6px 14px', fontSize: '12px', textDecoration: 'none' }}
                >
                  📞 Call Now
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
