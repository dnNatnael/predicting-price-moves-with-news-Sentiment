"""
Unit tests for Step 6: Validation Strategy Module.
Tests TimeSeriesSplit cross-validation, rolling walk-forward analysis, and regime robustness diagnostics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation_strategy import ValidationStrategy, RobustnessMetrics, FoldMetrics


@pytest.fixture
def sample_features_df():
    """Generate realistic synthetic features DataFrame for 500 trading days."""
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Synthetic price series with random walk + drift
    returns = np.random.normal(loc=0.0005, scale=0.015, size=n_days)
    prices = 100.0 * np.exp(np.cumsum(returns))

    # Predictive sentiment signal correlated with returns
    sentiment = returns * 10.0 + np.random.normal(0, 0.5, size=n_days)

    df = pd.DataFrame({
        "Date": dates,
        "Close": prices,
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Volume": np.random.randint(1_000_000, 5_000_000, size=n_days),
        "target_next_day_return": np.roll(returns, -1),
        "target_next_day_direction": (np.roll(returns, -1) > 0).astype(int),
        "sentiment_mean": sentiment,
        "sentiment_fin_vader": sentiment * 1.1,
        "sentiment_lag_1": np.roll(sentiment, 1),
        "sentiment_lag_2": np.roll(sentiment, 2),
        "rsi_14": np.random.uniform(30, 70, size=n_days),
        "macd": np.random.normal(0, 1, size=n_days),
        "volatility_10d": np.random.uniform(0.01, 0.03, size=n_days),
        "news_volume": np.random.randint(5, 50, size=n_days),
    })

    # Drop last row where target is rolled over
    return df.iloc[:-1].reset_index(drop=True)


class TestValidationStrategy:
    """Test suite for ValidationStrategy."""

    def test_initialization_and_cleaning(self, sample_features_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ValidationStrategy(
                features_df=sample_features_df,
                ticker="TEST",
                target_mode="binary",
                output_dir=tmpdir,
            )
            assert validator.ticker == "TEST"
            assert validator.target_mode == "binary"
            assert len(validator.df) == len(sample_features_df)
            assert "target_next_day_direction" in validator.df.columns

    def test_feature_selection_no_leakage(self, sample_features_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ValidationStrategy(
                features_df=sample_features_df,
                output_dir=tmpdir,
            )
            train_slice = validator.df.iloc[:200]
            selected = validator._select_train_features(
                train_df=train_slice,
                target_col="target_next_day_direction",
                top_k=5,
            )
            assert isinstance(selected, list)
            assert len(selected) <= 5
            # Ensure target and Date are not in selected features
            assert "Date" not in selected
            assert "target_next_day_return" not in selected
            assert "target_next_day_direction" not in selected

    def test_time_series_cv(self, sample_features_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ValidationStrategy(
                features_df=sample_features_df,
                ticker="AAPL",
                output_dir=tmpdir,
            )
            cv_df = validator.run_time_series_cv(
                n_splits=3,
                model_type="logistic_regression",
                top_k=5,
                save=True,
            )
            assert isinstance(cv_df, pd.DataFrame)
            assert len(cv_df) == 3
            assert set(cv_df.columns).issuperset({
                "fold", "train_size", "test_size", "accuracy",
                "balanced_accuracy", "f1_macro", "roc_auc", "strategy_return"
            })
            # Verify expanding window property
            assert cv_df["train_size"].iloc[0] < cv_df["train_size"].iloc[1] < cv_df["train_size"].iloc[2]
            # Check CSV export
            assert (Path(tmpdir) / "data" / "time_series_cv_results_AAPL.csv").exists()

    def test_walk_forward_analysis(self, sample_features_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ValidationStrategy(
                features_df=sample_features_df,
                ticker="AAPL",
                output_dir=tmpdir,
            )
            trades_df, metrics = validator.run_walk_forward_analysis(
                train_window_bars=200,
                test_window_bars=50,
                step_bars=50,
                model_type="logistic_regression",
                top_k=5,
                save=True,
            )
            assert isinstance(trades_df, pd.DataFrame)
            assert len(trades_df) > 0
            assert "strategy_equity" in trades_df.columns
            assert "benchmark_equity" in trades_df.columns
            assert "is_correct" in trades_df.columns
            assert "signal" in trades_df.columns

            assert isinstance(metrics, RobustnessMetrics)
            assert metrics.total_bars == len(trades_df)
            assert 0.0 <= metrics.win_rate <= 1.0
            assert 0.0 <= metrics.bull_regime_accuracy <= 1.0
            assert 0.0 <= metrics.bear_regime_accuracy <= 1.0
            assert metrics.max_drawdown >= 0.0

            # Check file exports
            assert (Path(tmpdir) / "data" / "walk_forward_trades_AAPL.csv").exists()
            assert (Path(tmpdir) / "data" / "walk_forward_metrics_AAPL.json").exists()
            assert (Path(tmpdir) / "figures" / "walk_forward_validation_AAPL.png").exists()

