"""
Unit tests for CorrelationAnalyzer module (Step 3: Correlation Analysis)
Verifies:
1. Statistical correlation calculation (Pearson, Spearman, Fisher z CI)
2. Multi-lag and multi-horizon correlation analysis
3. Topic-specific price movement drivers
4. Publisher market influence and alpha
5. Market session predictive power
6. Top predictive feature scorecard
7. End-to-end Step 3 execution
"""

import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
import pytest
from correlation_analyzer import CorrelationAnalyzer, CorrelationMetric


@pytest.fixture
def mock_features_df():
    """Create synthetic engineered features DataFrame with correlated targets."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    # True latent signal: sentiment positively affects next day return
    sentiment = np.random.normal(0.05, 0.3, n)
    noise = np.random.normal(0, 0.01, n)
    returns = 0.02 * sentiment + noise

    df = pd.DataFrame({
        "Date": dates,
        "effective_date": dates,
        "daily_return": returns,
        "target_next_day_return": np.roll(returns, -1),
        "target_3d_return": np.roll(returns, -3),
        "sentiment_mean": sentiment,
        "sentiment_lag_1": np.roll(sentiment, 1),
        "sentiment_lag_2": np.roll(sentiment, 2),
        "sentiment_lag_3": np.roll(sentiment, 3),
        "sentiment_publisher_weighted": sentiment * 1.05,
        "sentiment_publisher_adjusted": sentiment - 0.01,
        "sentiment_velocity_1d": sentiment - np.roll(sentiment, 1),
        "sentiment_pre_market": sentiment * 0.8,
        "sentiment_regular_hours": sentiment * 0.9,
        "sentiment_post_market": sentiment * 1.1,
        "sentiment_topic_0": sentiment * 0.7,
        "sentiment_topic_1": -sentiment * 0.5,
        "sentiment_earnings": sentiment * 1.2,
        "sentiment_mergers": sentiment * 0.6,
        "news_volume": np.random.randint(1, 20, n),
        "news_volume_surge_zscore": np.random.normal(0, 1, n),
        "rsi_14": np.random.uniform(30, 70, n),
        "macd": np.random.normal(0, 0.5, n),
    })
    return df


@pytest.fixture
def mock_news_df():
    """Create synthetic news DataFrame for publisher analysis."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "headline": [f"Headline {i}" for i in range(50)],
        "date": list(dates) * 5,
        "effective_date": list(dates) * 5,
        "publisher": ["Benzinga Newsdesk"] * 25 + ["Lisa Levin"] * 25,
        "sentiment": np.random.normal(0.1, 0.2, 50),
        "stock": ["AAPL"] * 50,
    })


class TestCorrelationAnalyzer:
    """Test suite for CorrelationAnalyzer."""

    def test_calculate_stat_correlation(self):
        np.random.seed(42)
        x = pd.Series(np.linspace(1, 10, 50))
        y = x * 2.0 + np.random.normal(0, 0.1, 50)

        metric = CorrelationAnalyzer.calculate_stat_correlation(x, y, "x", "y")

        assert isinstance(metric, CorrelationMetric)
        assert metric.pearson_r > 0.95
        assert metric.pearson_pvalue < 0.001
        assert metric.is_significant == True
        assert metric.ci_lower_95 > 0.90

    def test_analyze_lagged_correlations(self, mock_features_df, tmp_path):
        analyzer = CorrelationAnalyzer(mock_features_df, ticker="AAPL", output_dir=tmp_path)
        lags_df = analyzer.analyze_lagged_correlations(save=False)

        assert isinstance(lags_df, pd.DataFrame)
        assert len(lags_df) > 0
        assert "feature_name" in lags_df.columns
        assert "pearson_r" in lags_df.columns
        assert "spearman_rho" in lags_df.columns
        assert "ci_lower_95" in lags_df.columns

    def test_analyze_topic_price_drivers(self, mock_features_df, tmp_path):
        analyzer = CorrelationAnalyzer(mock_features_df, ticker="AAPL", output_dir=tmp_path)
        topic_df = analyzer.analyze_topic_price_drivers(target_col="target_next_day_return", save=False)

        assert isinstance(topic_df, pd.DataFrame)
        assert len(topic_df) > 0
        assert "sentiment_earnings" in topic_df["feature_name"].values

    def test_analyze_publisher_market_influence(self, mock_features_df, mock_news_df, tmp_path):
        analyzer = CorrelationAnalyzer(mock_features_df, news_df=mock_news_df, ticker="AAPL", output_dir=tmp_path)
        pub_df = analyzer.analyze_publisher_market_influence(min_articles=5, save=False)

        assert isinstance(pub_df, pd.DataFrame)
        assert len(pub_df) > 0
        assert "publisher" in pub_df.columns
        assert "article_count" in pub_df.columns
        assert "pearson_r" in pub_df.columns

    def test_analyze_session_predictive_power(self, mock_features_df, tmp_path):
        analyzer = CorrelationAnalyzer(mock_features_df, ticker="AAPL", output_dir=tmp_path)
        sess_df = analyzer.analyze_session_predictive_power(target_col="target_next_day_return", save=False)

        assert isinstance(sess_df, pd.DataFrame)
        assert len(sess_df) > 0
        assert "sentiment_pre_market" in sess_df["feature_name"].values

    def test_build_top_features_scorecard(self, mock_features_df, tmp_path):
        analyzer = CorrelationAnalyzer(mock_features_df, ticker="AAPL", output_dir=tmp_path)
        scorecard = analyzer.build_top_features_scorecard(target_col="target_next_day_return", top_n=10, save=False)

        assert isinstance(scorecard, pd.DataFrame)
        assert len(scorecard) > 0
        assert "abs_pearson_r" in scorecard.columns
        # Verify descending order
        assert scorecard["abs_pearson_r"].is_monotonic_decreasing

    def test_end_to_end_step3_execution(self, mock_features_df, mock_news_df, tmp_path):
        analyzer = CorrelationAnalyzer(mock_features_df, news_df=mock_news_df, ticker="AAPL", output_dir=tmp_path)
        results = analyzer.run_full_correlation_analysis(target_col="target_next_day_return", save=True)

        assert "lags" in results
        assert "topics" in results
        assert "publishers" in results
        assert "sessions" in results
        assert "scorecard" in results

        # Verify output files
        assert (tmp_path / "data" / "lag_correlations_AAPL.csv").exists()
        assert (tmp_path / "data" / "topic_price_drivers_AAPL.csv").exists()
        assert (tmp_path / "data" / "publisher_market_influence_AAPL.csv").exists()
        assert (tmp_path / "data" / "session_predictive_power_AAPL.csv").exists()
        assert (tmp_path / "data" / "top_predictive_features_AAPL.csv").exists()
        assert (tmp_path / "data" / "correlation_summary_AAPL.json").exists()

