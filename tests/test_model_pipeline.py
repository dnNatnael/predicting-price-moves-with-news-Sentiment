"""
Unit tests for ModelPipeline module (Step 4: Model Preparation & Training)
Verifies:
1. Feature selection (correlation & non-collinear filtering)
2. Strict temporal chronological splitting (Train/Val/Test)
3. Class imbalance balancing
4. Binary and multi-class model training
5. Model evaluation metrics & strategy simulation
6. Model artifact serialization (.joblib) for dashboard inference
"""

import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
import pytest
from model_pipeline import ModelPipeline, ModelMetrics


@pytest.fixture
def mock_ml_df():
    """Create synthetic historical feature DataFrame covering 200 trading days."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    # Generate signals and returns
    sentiment = np.random.normal(0.05, 0.25, n)
    rsi = np.random.uniform(30, 70, n)
    daily_ret = 0.015 * sentiment + np.random.normal(0, 0.01, n)
    next_ret = np.roll(daily_ret, -1)

    df = pd.DataFrame({
        "Date": dates,
        "effective_date": dates,
        "daily_return": daily_ret,
        "target_next_day_return": next_ret,
        "target_next_day_direction": (next_ret > 0).astype(int),
        "target_next_day_class": ["Up" if r > 0.005 else ("Down" if r < -0.005 else "Neutral") for r in next_ret],
        "sentiment_mean": sentiment,
        "sentiment_lag_1": np.roll(sentiment, 1),
        "sentiment_lag_2": np.roll(sentiment, 2),
        "sentiment_publisher_weighted": sentiment * 1.02,
        "sentiment_velocity_1d": sentiment - np.roll(sentiment, 1),
        "news_volume": np.random.randint(1, 15, n),
        "news_volume_surge_zscore": np.random.normal(0, 1, n),
        "rsi_14": rsi,
        "macd": np.random.normal(0, 0.5, n),
        "bollinger_pband": np.random.uniform(0, 1, n),
        "is_monday": (dates.day_name() == "Monday").astype(int),
        "is_friday": (dates.day_name() == "Friday").astype(int),
    })
    return df


class TestModelPipeline:
    """Test suite for ModelPipeline class."""

    def test_feature_selection(self, mock_ml_df, tmp_path):
        pipeline = ModelPipeline(mock_ml_df, ticker="AAPL", target_mode="binary", output_dir=tmp_path)
        selected = pipeline.select_features(top_k=8)

        assert isinstance(selected, list)
        assert len(selected) > 0
        assert len(selected) <= 8
        assert "target_next_day_direction" not in selected
        assert "Date" not in selected

    def test_temporal_splits(self, mock_ml_df, tmp_path):
        pipeline = ModelPipeline(mock_ml_df, ticker="AAPL", target_mode="binary", output_dir=tmp_path)
        X_train, X_val, X_test = pipeline.create_temporal_splits(top_k_features=8)

        # Check chronological separation (no index overlap)
        assert len(set(X_train.index).intersection(set(X_val.index))) == 0
        assert len(set(X_val.index).intersection(set(X_test.index))) == 0

        # Check ordering
        assert X_train.index.max() < X_val.index.min()
        assert X_val.index.max() < X_test.index.min()

        # Check class weights computed
        assert pipeline.class_weights_dict is not None
        assert 1 in pipeline.class_weights_dict
        assert 0 in pipeline.class_weights_dict

    def test_train_models(self, mock_ml_df, tmp_path):
        pipeline = ModelPipeline(mock_ml_df, ticker="AAPL", target_mode="binary", output_dir=tmp_path)
        pipeline.create_temporal_splits(top_k_features=8)
        models = pipeline.train_models()

        assert "Logistic_Regression" in models
        assert "Random_Forest" in models
        assert "Gradient_Boosting" in models

    def test_evaluate_models_binary(self, mock_ml_df, tmp_path):
        pipeline = ModelPipeline(mock_ml_df, ticker="AAPL", target_mode="binary", output_dir=tmp_path)
        pipeline.create_temporal_splits(top_k_features=8)
        pipeline.train_models()
        eval_df = pipeline.evaluate_models(save=False)

        assert isinstance(eval_df, pd.DataFrame)
        assert len(eval_df) == 3
        assert "test_accuracy" in eval_df.columns
        assert "test_f1_macro" in eval_df.columns
        assert "strategy_cumulative_return" in eval_df.columns
        assert (eval_df["test_accuracy"] >= 0).all()

    def test_evaluate_models_multiclass(self, mock_ml_df, tmp_path):
        pipeline = ModelPipeline(mock_ml_df, ticker="AAPL", target_mode="multiclass", output_dir=tmp_path)
        pipeline.create_temporal_splits(top_k_features=8)
        pipeline.train_models()
        eval_df = pipeline.evaluate_models(save=False)

        assert isinstance(eval_df, pd.DataFrame)
        assert len(eval_df) == 3
        assert pipeline.target_mode == "multiclass"

    def test_end_to_end_step4_execution(self, mock_ml_df, tmp_path):
        pipeline = ModelPipeline(mock_ml_df, ticker="AAPL", target_mode="binary", output_dir=tmp_path)
        results = pipeline.run_full_model_pipeline(top_k_features=8, save=True)

        assert "eval_df" in results
        assert "best_model_name" in results

        # Verify saved artifacts
        assert (tmp_path / "models" / "best_model_AAPL_binary.joblib").exists()
        assert (tmp_path / "models" / "scaler_AAPL.joblib").exists()
        assert (tmp_path / "data" / "model_comparison_AAPL_binary.csv").exists()
        assert (tmp_path / "data" / "model_evaluation_AAPL_binary.json").exists()
        assert (tmp_path / "data" / "feature_importances_AAPL_binary.csv").exists()
        assert (tmp_path / "figures" / "model_evaluation_AAPL_binary.png").exists()

