import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import OverviewTab from './components/OverviewTab';
import NewsSentimentTab from './components/NewsSentimentTab';
import TechnicalAnalysisTab from './components/TechnicalAnalysisTab';
import CorrelationMatrixTab from './components/CorrelationMatrixTab';
import ModelPerformanceTab from './components/ModelPerformanceTab';
import WalkForwardBacktestTab from './components/WalkForwardBacktestTab';
import DataIngestionTab from './components/DataIngestionTab';
import AboutTab from './components/AboutTab';
import LivePredictorModal from './components/LivePredictorModal';

import {
  fetchHealth,
  fetchOverview,
  fetchSentimentBenchmark,
  fetchNewsFeed,
  fetchPrices,
  fetchCorrelations,
  fetchModels,
  fetchValidation,
} from './services/api';

import { LayoutDashboard, MessageSquareText, LineChart, Network, Cpu, History, UploadCloud, BookOpen, Mail, Phone } from 'lucide-react';

export default function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [availableTickers, setAvailableTickers] = useState(['AAPL', 'NVDA', 'MSFT', 'GOOG', 'AMZN', 'META']);
  const [activeTab, setActiveTab] = useState('overview');
  const [isPredictorOpen, setIsPredictorOpen] = useState(false);
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('app_theme') || 'dark';
  });

  useEffect(() => {
    document.body.className = theme === 'light' ? 'light-theme' : 'dark-theme';
    localStorage.setItem('app_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Data States
  const [healthStatus, setHealthStatus] = useState('online');
  const [overview, setOverview] = useState(null);
  const [benchmark, setBenchmark] = useState([]);
  const [news, setNews] = useState([]);
  const [prices, setPrices] = useState([]);
  const [correlations, setCorrelations] = useState([]);
  const [models, setModels] = useState(null);
  const [validation, setValidation] = useState(null);
  const [loading, setLoading] = useState(true);

  // Initial Health check & Available tickers
  useEffect(() => {
    fetchHealth()
      .then((data) => {
        setHealthStatus(data.status);
        if (data.available_tickers && data.available_tickers.length > 0) {
          setAvailableTickers(data.available_tickers);
        }
      })
      .catch(() => setHealthStatus('offline'));
  }, []);

  const handleTickerAdded = (newTicker) => {
    if (!availableTickers.includes(newTicker)) {
      setAvailableTickers((prev) => [newTicker, ...prev]);
    }
    setTicker(newTicker);
  };

  // Fetch all ticker data on change
  useEffect(() => {
    let isMounted = true;
    setLoading(true);

    Promise.all([
      fetchOverview(ticker),
      fetchSentimentBenchmark(ticker),
      fetchNewsFeed(ticker, 40),
      fetchPrices(ticker, 252),
      fetchCorrelations(ticker),
      fetchModels(ticker),
      fetchValidation(ticker),
    ])
      .then(([ov, bm, nw, pr, cr, md, vl]) => {
        if (!isMounted) return;
        setOverview(ov);
        setBenchmark(bm);
        setNews(nw);
        setPrices(pr);
        setCorrelations(cr);
        setModels(md);
        setValidation(vl);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching dashboard data:', err);
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [ticker]);

  const tabs = [
    { id: 'overview', label: '1. Overview', icon: LayoutDashboard },
    { id: 'sentiment', label: '2. Sentiment Engines', icon: MessageSquareText },
    { id: 'technical', label: '3. Technical & Price', icon: LineChart },
    { id: 'correlations', label: '4. Alpha Correlations', icon: Network },
    { id: 'models', label: '5. ML Models', icon: Cpu },
    { id: 'backtest', label: '6. Walk-Forward Backtest', icon: History },
    { id: 'ingest', label: '7. Ingest Data', icon: UploadCloud },
    { id: 'about', label: '8. About & Guide', icon: BookOpen },
  ];

  return (
    <div className="dashboard-container">
      <Navbar
        selectedTicker={ticker}
        onSelectTicker={setTicker}
        onOpenPredictor={() => setIsPredictorOpen(true)}
        healthStatus={healthStatus}
        availableTickers={availableTickers}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main className="content-area">
        {/* Navigation Tabs */}
        <div className="tabs-nav">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon size={16} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab Views */}
        {loading && activeTab !== 'ingest' && activeTab !== 'about' ? (
          <div style={{ textAlign: 'center', padding: '60px', color: 'var(--text-muted)' }}>
            <div style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>
              Loading Quantitative Intelligence for {ticker}...
            </div>
            <div style={{ fontSize: '12px' }}>Aligning news sentiment, technical indicators, and backtest results</div>
          </div>
        ) : (
          <>
            {activeTab === 'overview' && (
              <OverviewTab
                overview={overview}
                news={news}
                models={models}
                onSelectTab={setActiveTab}
              />
            )}

            {activeTab === 'sentiment' && (
              <NewsSentimentTab
                benchmark={benchmark}
                news={news}
              />
            )}

            {activeTab === 'technical' && (
              <TechnicalAnalysisTab
                ticker={ticker}
                prices={prices}
              />
            )}

            {activeTab === 'correlations' && (
              <CorrelationMatrixTab
                ticker={ticker}
                correlations={correlations}
              />
            )}

            {activeTab === 'models' && (
              <ModelPerformanceTab
                ticker={ticker}
                models={models}
              />
            )}

            {activeTab === 'backtest' && (
              <WalkForwardBacktestTab
                ticker={ticker}
                validation={validation}
              />
            )}

            {activeTab === 'ingest' && (
              <DataIngestionTab
                onTickerAdded={handleTickerAdded}
                onSelectTab={setActiveTab}
              />
            )}

            {activeTab === 'about' && (
              <AboutTab
                onSelectTab={setActiveTab}
              />
            )}
          </>
        )}
      </main>

      {/* Institutional Portfolio Footer with Contact Links */}
      <footer className="dashboard-footer">
        <div className="footer-content">
          <div className="footer-left">
            <span className="footer-title">Quantitative Alpha & Sentiment Intelligence Terminal</span>
            <span className="footer-desc">End-to-End Financial NLP • 13-Year Walk-Forward Backtesting • ML Direction Prediction</span>
          </div>
          <div className="footer-contacts">
            <a
              href="https://mail.google.com/mail/?view=cm&fs=1&to=natnael.yilma27@gmail.com&su=Question%20regarding%20Financial%20News%20Sentiment%20Project"
              target="_blank"
              rel="noopener noreferrer"
              className="contact-link"
              title="Compose Email to Natnael Yilma"
            >
              <Mail size={14} color="var(--accent-blue)" />
              <span>natnael.yilma27@gmail.com</span>
            </a>
            <a href="tel:+251929709562" className="contact-link" title="Call Phone Number">
              <Phone size={14} color="var(--bullish)" />
              <span>+251 929 709 562</span>
            </a>
          </div>
        </div>
      </footer>

      {/* Live Inference Modal */}
      <LivePredictorModal
        isOpen={isPredictorOpen}
        onClose={() => setIsPredictorOpen(false)}
        defaultTicker={ticker}
      />
    </div>
  );
}

