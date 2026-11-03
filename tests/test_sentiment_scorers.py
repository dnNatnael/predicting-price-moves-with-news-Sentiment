"""
Unit tests for Step 5: Financial Sentiment Scoring Approaches.
Tests FinancialVADER, LoughranMcDonaldDictionary, FinBERTScorer fallback, and EnsembleSentimentScorer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sentiment_scorers import (
    FinancialVADER,
    LoughranMcDonaldDictionary,
    FinBERTScorer,
    EnsembleSentimentScorer,
    SentimentApproachMetrics,
)


class TestFinancialVADER:
    """Test suite for FinancialVADER."""

    def test_bullish_lexicon(self):
        vader = FinancialVADER()
        score = vader.score_text("Company beats earnings estimates and raises guidance")
        assert score > 0.4, f"Expected strong positive score, got {score}"

    def test_bearish_lexicon(self):
        vader = FinancialVADER()
        score = vader.score_text("Company misses revenue estimates and files for bankruptcy")
        assert score < -0.4, f"Expected strong negative score, got {score}"

    def test_empty_and_nan_handling(self):
        vader = FinancialVADER()
        assert vader.score_text("") == 0.0
        assert vader.score_text(None) == 0.0
        assert vader.score_text(np.nan) == 0.0

    def test_series_scoring(self):
        vader = FinancialVADER()
        series = pd.Series([
            "Strong record profit and dividend hike",
            "Disastrous loss and debt default",
            "Neutral quarterly report filed",
        ])
        scores = vader.score_series(series)
        assert len(scores) == 3
        assert scores.iloc[0] > 0
        assert scores.iloc[1] < 0


class TestLoughranMcDonaldDictionary:
    """Test suite for LoughranMcDonaldDictionary."""

    def test_positive_terms(self):
        res = LoughranMcDonaldDictionary.score_text("Achieved record profit and exceptional growth")
        assert res["lm_pos_count"] >= 3
        assert res["lm_neg_count"] == 0
        assert res["lm_sentiment"] == 1.0

    def test_negative_terms(self):
        res = LoughranMcDonaldDictionary.score_text("Suffered catastrophic deficit and severe default loss")
        assert res["lm_neg_count"] >= 3
        assert res["lm_pos_count"] == 0
        assert res["lm_sentiment"] == -1.0

    def test_uncertainty_terms(self):
        res = LoughranMcDonaldDictionary.score_text("Estimated forecast pending preliminary approximate audit")
        assert res["lm_uncertainty_count"] >= 3

    def test_neutral_and_empty(self):
        res = LoughranMcDonaldDictionary.score_text("The meeting is scheduled on Tuesday at noon.")
        assert res["lm_sentiment"] == 0.0
        assert res["lm_pos_count"] == 0
        assert res["lm_neg_count"] == 0

        res_none = LoughranMcDonaldDictionary.score_text(None)
        assert res_none["lm_sentiment"] == 0.0

    def test_series_scoring(self):
        series = pd.Series([
            "Outstanding victory and profitable turnaround",
            "Severe turmoil and critical decline",
        ])
        df = LoughranMcDonaldDictionary.score_series(series)
        assert isinstance(df, pd.DataFrame)
        assert "lm_sentiment" in df.columns
        assert df["lm_sentiment"].iloc[0] > 0
        assert df["lm_sentiment"].iloc[1] < 0


class TestFinBERTScorer:
    """Test suite for FinBERTScorer (heuristic fallback and interface)."""

    def test_fallback_scoring(self):
        scorer = FinBERTScorer()
        texts = [
            "Company beats quarterly forecasts with booming sales",
            "Shares plunge amid major fraud investigation",
        ]
        df = scorer.score_batch(texts)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"finbert_pos", "finbert_neg", "finbert_neu", "finbert_sentiment"}
        assert len(df) == 2
        assert df["finbert_sentiment"].iloc[0] > 0
        assert df["finbert_sentiment"].iloc[1] < 0


class TestEnsembleSentimentScorer:
    """Test suite for EnsembleSentimentScorer."""

    @pytest.fixture
    def sample_data(self):
        news_df = pd.DataFrame({
            "headline": [
                "Tech giant beats earnings views and surges on strong forecast",
                "Firm misses revenue and lowers annual guidance",
                "Company holds annual shareholder meeting",
                "Strong dividend hike boosts investor confidence",
                "SEC launches probe following whistleblower complaint",
            ],
            "date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "effective_date": pd.date_range("2023-01-01", periods=5, freq="D"),
        })
        prices_df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "daily_return": [0.035, -0.040, 0.001, 0.025, -0.030],
        })
        return news_df, prices_df

    def test_ensemble_weights_normalization(self, sample_data):
        news_df, _ = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = EnsembleSentimentScorer(
                news_df=news_df,
                weights={"fin_vader": 3.0, "loughran_mcdonald": 3.0, "finbert": 2.0, "textblob": 2.0},
                output_dir=tmpdir,
            )
            assert pytest.approx(sum(scorer.weights.values()), rel=1e-5) == 1.0

    def test_run_all_scorers_and_benchmark(self, sample_data):
        news_df, prices_df = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = EnsembleSentimentScorer(news_df=news_df, output_dir=tmpdir)
            enriched_df = scorer.run_all_scorers(use_finbert_model=False)

            # Check that columns were added
            assert "sentiment_fin_vader" in enriched_df.columns
            assert "lm_sentiment" in enriched_df.columns
            assert "sentiment_textblob" in enriched_df.columns
            assert "finbert_sentiment" in enriched_df.columns
            assert "sentiment_ensemble" in enriched_df.columns
            assert "sentiment_ensemble_class" in enriched_df.columns

            # Check directional logic
            assert enriched_df["sentiment_ensemble"].iloc[0] > 0  # "beats"
            assert enriched_df["sentiment_ensemble"].iloc[1] < 0  # "misses"

            # Run benchmark against returns
            comp_df = scorer.benchmark_sentiment_approaches(price_returns_df=prices_df, save=True)
            assert isinstance(comp_df, pd.DataFrame)
            assert len(comp_df) == 5
            assert "pearson_r_with_return" in comp_df.columns
            assert "spearman_rho_with_return" in comp_df.columns

            # Check saved files
            assert (Path(tmpdir) / "data" / "sentiment_approach_comparison.csv").exists()
            assert (Path(tmpdir) / "data" / "sentiment_benchmark_summary.json").exists()
            assert (Path(tmpdir) / "figures" / "sentiment_approaches_comparison.png").exists()

