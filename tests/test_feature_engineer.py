"""
Unit tests for FeatureEngineer module (Step 2: Feature Engineering)
Verifies:
1. Headline length & word count statistics
2. Topic exposure and financial category indicators
3. Publisher credibility weights & adjusted sentiment
4. Temporal and session features (day of week, minutes from open)
5. Stock-specific news volume dynamics
6. Sentiment momentum & lags
7. Stock price alignment, technical indicators, and forward targets
"""

import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
import pytest
from feature_engineer import FeatureEngineer


@pytest.fixture
def sample_news_df():
    """Create sample news DataFrame with headlines, dates, and publishers."""
    dates = [
        "2023-01-03 08:00:00",  # Pre-market
        "2023-01-03 10:15:00",  # Regular hours
        "2023-01-03 16:30:00",  # Post-market -> moves to Jan 4
        "2023-01-04 09:00:00",  # Pre-market
        "2023-01-04 11:30:00",  # Regular hours
        "2023-01-05 14:00:00",  # Regular hours
        "2023-01-06 08:30:00",  # Friday pre-market
        "2023-01-06 15:00:00",  # Friday regular hours
    ]
    headlines = [
        "Apple beats earnings expectations with surge in revenue",
        "Analyst raises price target for AAPL citing cloud growth",
        "Merger agreement signed with tech supplier after market close",
        "FDA approves clinical trial for innovative medical device",
        "Apple announces product launch event for new iPhone",
        "Market volatility impacts tech stocks as rates rise",
        "Strong holiday sales boost quarterly earnings report",
        "Wall Street analyst issues upgrade rating for Apple",
    ]
    publishers = [
        "Benzinga Newsdesk",
        "Lisa Levin",
        "Benzinga Newsdesk",
        "Lisa Levin",
        "Reuters",
        "Benzinga Newsdesk",
        "Reuters",
        "Lisa Levin",
    ]
    return pd.DataFrame({
        "headline": headlines,
        "date": dates,
        "publisher": publishers,
        "stock": ["AAPL"] * 8,
        "lda_topic": [0, 1, 2, 3, 4, 0, 1, 2],
        "sentiment": [0.65, 0.45, 0.30, 0.50, 0.20, -0.40, 0.70, 0.35],
    })


@pytest.fixture
def sample_price_df():
    """Create sample OHLCV price DataFrame."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    prices = [150.0, 151.5, 153.0, 152.0, 154.5, 156.0, 155.0, 157.0, 158.5, 160.0]
    return pd.DataFrame({
        "Date": dates,
        "Open": [p - 1.0 for p in prices],
        "High": [p + 2.0 for p in prices],
        "Low": [p - 2.0 for p in prices],
        "Close": prices,
        "Volume": [1000000 + i * 50000 for i in range(10)],
    })


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    def test_text_features(self, sample_news_df, tmp_path):
        engineer = FeatureEngineer(sample_news_df, ticker="AAPL", output_dir=tmp_path)
        text_df = engineer.build_text_features()

        assert "effective_date" in text_df.columns
        assert "headline_length_mean" in text_df.columns
        assert "headline_word_count_mean" in text_df.columns
        assert (text_df["headline_length_mean"] > 0).all()

    def test_topic_features(self, sample_news_df, tmp_path):
        engineer = FeatureEngineer(sample_news_df, ticker="AAPL", output_dir=tmp_path)
        topic_df = engineer.build_topic_features()

        assert "effective_date" in topic_df.columns
        assert "cat_count_earnings" in topic_df.columns
        assert "has_earnings" in topic_df.columns
        assert "sentiment_earnings" in topic_df.columns
        assert "dominant_topic" in topic_df.columns

    def test_publisher_weighting_features(self, sample_news_df, tmp_path):
        engineer = FeatureEngineer(sample_news_df, ticker="AAPL", output_dir=tmp_path)
        daily_pub, pub_stats = engineer.build_publisher_weighting_features()

        assert "effective_date" in daily_pub.columns
        assert "sentiment_publisher_weighted" in daily_pub.columns
        assert "sentiment_publisher_adjusted" in daily_pub.columns
        assert "publisher_concentration_hhi" in daily_pub.columns

        assert "publisher" in pub_stats.columns
        assert "credibility_weight" in pub_stats.columns

    def test_temporal_features(self, sample_news_df, tmp_path):
        engineer = FeatureEngineer(sample_news_df, ticker="AAPL", output_dir=tmp_path)
        temp_df = engineer.build_temporal_features()

        assert "effective_date" in temp_df.columns
        assert "is_monday" in temp_df.columns
        assert "is_friday" in temp_df.columns
        assert "month_sin" in temp_df.columns
        assert "mean_minutes_from_open" in temp_df.columns
        assert "sentiment_pre_market" in temp_df.columns

    def test_news_volume_features(self, sample_news_df, tmp_path):
        engineer = FeatureEngineer(sample_news_df, ticker="AAPL", output_dir=tmp_path)
        vol_df = engineer.build_news_volume_features()

        assert "news_volume" in vol_df.columns
        assert "news_volume_sma_7" in vol_df.columns
        assert "news_volume_surge_zscore" in vol_df.columns
        assert "is_news_volume_spike" in vol_df.columns
        assert "sentiment_pos_ratio" in vol_df.columns

    def test_sentiment_momentum_features(self, sample_news_df, tmp_path):
        engineer = FeatureEngineer(sample_news_df, ticker="AAPL", output_dir=tmp_path)
        vol_df = engineer.build_news_volume_features()
        momentum_df = engineer.build_sentiment_momentum_features(vol_df)

        assert "sentiment_lag_1" in momentum_df.columns
        assert "sentiment_lag_2" in momentum_df.columns
        assert "sentiment_sma_3" in momentum_df.columns
        assert "sentiment_velocity_1d" in momentum_df.columns

    def test_price_alignment_and_technical_indicators(self, sample_news_df, sample_price_df, tmp_path):
        engineer = FeatureEngineer(sample_news_df, price_df=sample_price_df, ticker="AAPL", output_dir=tmp_path)
        daily_news = engineer.build_news_volume_features()
        merged = engineer.align_with_price_and_indicators(daily_news)

        assert "daily_return" in merged.columns
        assert "sma_20" in merged.columns
        assert "rsi_14" in merged.columns
        assert "macd" in merged.columns
        assert "bollinger_width" in merged.columns
        assert "atr_14" in merged.columns

        # Targets
        assert "target_next_day_return" in merged.columns
        assert "target_next_day_direction" in merged.columns
        assert "target_next_day_class" in merged.columns
        assert set(merged["target_next_day_direction"].dropna().unique()).issubset({0, 1})

    def test_end_to_end_feature_pipeline(self, sample_news_df, sample_price_df, tmp_path):
        engineer = FeatureEngineer(sample_news_df, price_df=sample_price_df, ticker="AAPL", output_dir=tmp_path)
        master = engineer.run_feature_pipeline(save=True)

        assert len(master) > 0
        assert (tmp_path / "data" / "engineered_features_AAPL.csv").exists()
        assert (tmp_path / "data" / "publisher_credibility_weights.csv").exists()
        assert (tmp_path / "data" / "feature_summary_AAPL.json").exists()

