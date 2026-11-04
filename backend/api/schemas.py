"""
Pydantic schemas for the FastAPI backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    available_tickers: List[str]
    api_version: str


class OverviewResponse(BaseModel):
    ticker: str
    total_news_articles: int
    total_trading_days: int
    date_range: Dict[str, str]
    best_model_name: str
    best_model_accuracy: float
    walk_forward_sharpe: float
    walk_forward_cagr: float
    walk_forward_mdd: float
    sentiment_distribution: Dict[str, float]


class SentimentBenchmarkRow(BaseModel):
    approach: str
    mean_score: float
    pct_positive: float
    pct_neutral: float
    pct_negative: float
    pearson_r: float
    spearman_rho: float
    is_significant: bool


class NewsArticle(BaseModel):
    headline: str
    date: str
    publisher: Optional[str] = "Unknown"
    stock: Optional[str] = None
    fin_vader: float
    lm_sentiment: float
    finbert: float
    ensemble: float
    ensemble_class: str


class PriceBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    daily_return: Optional[float] = 0.0
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    sentiment_mean: Optional[float] = None
    news_volume: Optional[int] = 0


class CorrelationItem(BaseModel):
    feature: str
    pearson_r: float
    spearman_rho: float
    p_value: float
    is_significant: bool


class ModelEvaluationRow(BaseModel):
    model_name: str
    test_accuracy: float
    test_f1_macro: float
    test_roc_auc: Optional[float]
    strategy_cumulative_return: float
    strategy_sharpe_ratio: float


class FeatureImportanceRow(BaseModel):
    feature: str
    importance: float


class WalkForwardMetricsResponse(BaseModel):
    ticker: str
    total_bars: int
    cumulative_return: float
    benchmark_cumulative_return: float
    annualized_return: float
    benchmark_annualized_return: float
    sharpe_ratio: float
    benchmark_sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    benchmark_max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    bull_regime_accuracy: float
    bear_regime_accuracy: float
    high_vol_accuracy: float
    low_vol_accuracy: float


class WalkForwardEquityPoint(BaseModel):
    date: str
    strategy_equity: float
    benchmark_equity: float
    strategy_drawdown: float
    benchmark_drawdown: float
    signal: int
    is_correct: int


class PredictionRequest(BaseModel):
    headline: str = Field(..., json_schema_extra={"example": "Apple beats Q4 revenue expectations by 8% and raises annual dividend"})
    ticker: str = Field(default="AAPL", json_schema_extra={"example": "AAPL"})


class PredictionResponse(BaseModel):
    headline: str
    ticker: str
    sentiment_fin_vader: float
    sentiment_lm: float
    sentiment_finbert: float
    sentiment_ensemble: float
    sentiment_label: str
    predicted_next_day_direction: str  # "UP" or "DOWN"
    up_probability: float
    down_probability: float
    model_used: str
    explanation: str


class UploadPriceResponse(BaseModel):
    status: str
    ticker: str
    total_bars: int
    start_date: str
    end_date: str
    latest_close: float
    indicators_computed: List[str]
    message: str


class UploadNewsResponse(BaseModel):
    status: str
    ticker: Optional[str]
    total_articles: int
    sentiment_breakdown: Dict[str, Any]
    mean_ensemble_score: float
    top_bullish_sample: Optional[str] = None
    top_bearish_sample: Optional[str] = None
    return_correlation: Optional[float] = None
    message: str
