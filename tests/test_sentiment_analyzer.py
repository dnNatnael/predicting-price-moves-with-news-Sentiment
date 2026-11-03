"""
Unit tests for SentimentAnalyzer module
Tests all 4 components of Sentiment Analysis Preparation:
1. Topic-based sentiment features
2. Publisher-specific sentiment patterns
3. Sentiment trends around news spikes
4. Time-based and market session sentiment features
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
import pytest
from sentiment_analyzer import SentimentAnalyzer


@pytest.fixture
def mock_news_df():
    """Create synthetic financial news DataFrame covering multiple dates, publishers, and sessions."""
    records = [
        # Pre-market earnings (Bullish)
        {
            "headline": "Apple reports record Q4 earnings beat with strong revenue guidance",
            "date": "2023-01-05 07:15:00",
            "publisher": "Benzinga Newsdesk",
            "stock": "AAPL",
            "lda_topic": 0,
        },
        # Regular hours price target
        {
            "headline": "Morgan Stanley upgrades Microsoft price target to 350",
            "date": "2023-01-05 10:30:00",
            "publisher": "Benzinga Newsdesk",
            "stock": "MSFT",
            "lda_topic": 1,
        },
        # Post-market merger (Should affect next day 2023-01-06)
        {
            "headline": "Google announces agreement to acquire cloud security firm",
            "date": "2023-01-05 17:00:00",
            "publisher": "Lisa Levin",
            "stock": "GOOGL",
            "lda_topic": 2,
        },
        # Overnight negative news
        {
            "headline": "FDA rejects drug trial application citing safety concerns",
            "date": "2023-01-06 02:00:00",
            "publisher": "Lisa Levin",
            "stock": "AAPL",
            "lda_topic": 3,
        },
        # Regular hours negative
        {
            "headline": "Tech stocks plunge amid market volatility and recession fears",
            "date": "2023-01-06 14:00:00",
            "publisher": "Vick Meyer",
            "stock": "AAPL",
            "lda_topic": 4,
        },
        # Friday post-market
        {
            "headline": "Tesla shares surge to 52-week highs on record deliveries",
            "date": "2023-01-06 16:30:00",
            "publisher": "Benzinga Newsdesk",
            "stock": "TSLA",
            "lda_topic": 0,
        },
    ]
    # Multiply records to give enough volume for statistical aggregations
    df = pd.DataFrame(records * 10)
    # Add random slight time jitter
    return df


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer."""

    def test_datetime_and_timezone_handling(self, mock_news_df, tmp_path):
        analyzer = SentimentAnalyzer(mock_news_df, output_dir=tmp_path)
        df = analyzer.df

        assert "effective_date" in df.columns
        assert "hour_market" in df.columns
        assert "market_session" in df.columns

        # Verify that news published at 17:00 ET was attributed to next day
        post_market_row = df[df["hour_market"] == 17].iloc[0]
        assert post_market_row["effective_date"] > post_market_row["date_only"]

    def test_market_session_classification(self, mock_news_df, tmp_path):
        analyzer = SentimentAnalyzer(mock_news_df, output_dir=tmp_path)
        sessions = set(analyzer.df["market_session"].unique())

        assert "Pre-Market" in sessions
        assert "Regular-Hours" in sessions
        assert "Post-Market" in sessions
        assert "Overnight" in sessions

    def test_headline_sentiment_scoring(self, mock_news_df, tmp_path):
        analyzer = SentimentAnalyzer(mock_news_df, output_dir=tmp_path)
        scored = analyzer.score_sentiment()

        assert "sentiment_vader_compound" in scored.columns
        assert "sentiment_vader_pos" in scored.columns
        assert "sentiment_vader_neg" in scored.columns
        assert "sentiment_textblob_polarity" in scored.columns
        assert "sentiment_class" in scored.columns

        # Check score bounds
        assert scored["sentiment"].between(-1.0, 1.0).all()
        assert set(scored["sentiment_class"].unique()).issubset({"Positive", "Negative", "Neutral"})

    def test_topic_based_sentiment_features(self, mock_news_df, tmp_path):
        analyzer = SentimentAnalyzer(mock_news_df, output_dir=tmp_path)
        topic_features = analyzer.create_topic_sentiment_features(topic_col="lda_topic", save=False)

        assert "effective_date" in topic_features.columns
        # Topic columns should be prefixed with sentiment_topic_
        topic_cols = [c for c in topic_features.columns if c.startswith("sentiment_topic_")]
        assert len(topic_cols) > 0
        assert len(topic_features) > 0

    def test_financial_category_mapping(self, mock_news_df, tmp_path):
        analyzer = SentimentAnalyzer(mock_news_df, output_dir=tmp_path)
        df = analyzer.map_financial_topic_categories()

        assert "financial_category" in df.columns
        cats = df["financial_category"].unique()
        # Earnings headline should be classified into earnings
        earnings_row = df[df["headline"].str.contains("earnings", case=False)].iloc[0]
        assert earnings_row["financial_category"] == "earnings"

    def test_publisher_sentiment_patterns(self, mock_news_df, tmp_path):
        analyzer = SentimentAnalyzer(mock_news_df, output_dir=tmp_path)
        pub_stats = analyzer.analyze_publisher_sentiment(min_articles=5, save=False)

        assert isinstance(pub_stats, pd.DataFrame)
        assert "publisher" in pub_stats.columns
        assert "mean_sentiment" in pub_stats.columns
        assert "sentiment_bias" in pub_stats.columns
        assert "positivity_ratio" in pub_stats.columns
        assert "bullish_lean" in pub_stats.columns

        # Verify publisher-adjusted sentiment column was added to main df
        assert "sentiment_publisher_adjusted" in analyzer.df.columns
        assert "publisher_sentiment_bias" in analyzer.df.columns

    def test_news_spike_sentiment_analysis(self, mock_news_df, tmp_path):
        analyzer = SentimentAnalyzer(mock_news_df, output_dir=tmp_path)
        spike_res = analyzer.analyze_spike_sentiment(threshold_std=1.0, save=False)

        assert "comparison" in spike_res
        assert "trajectory" in spike_res
        comp = spike_res["comparison"]
        assert "spike_mean_sentiment" in comp
        assert "normal_mean_sentiment" in comp
        assert "p_value" in comp

    def test_time_based_sentiment_features(self, mock_news_df, tmp_path):
        analyzer = SentimentAnalyzer(mock_news_df, output_dir=tmp_path)
        time_features = analyzer.create_time_based_sentiment_features(save=False)

        assert "effective_date" in time_features.columns
        session_cols = [c for c in time_features.columns if c.startswith("sentiment_session_")]
        assert len(session_cols) > 0
        assert "is_monday" in time_features.columns
        assert "is_friday" in time_features.columns

    def test_run_full_preparation_pipeline(self, mock_news_df, tmp_path):
        analyzer = SentimentAnalyzer(mock_news_df, output_dir=tmp_path)
        results = analyzer.run_full_preparation(save_enriched_articles=True)

        assert "topic_features" in results
        assert "publisher_patterns" in results
        assert "spike_stats" in results
        assert "time_features" in results

        # Verify output files were created
        assert (tmp_path / "data" / "articles_with_sentiment.csv").exists()
        assert (tmp_path / "data" / "topic_daily_sentiment.csv").exists()
        assert (tmp_path / "data" / "publisher_sentiment_patterns.csv").exists()
        assert (tmp_path / "data" / "time_based_sentiment_features.csv").exists()
        assert (tmp_path / "sentiment_preparation_report.txt").exists()

