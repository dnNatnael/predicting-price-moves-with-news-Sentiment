"""
Compute correlation between aggregated news sentiment and daily stock returns.

Steps:
- Clean and align news + price dates
- Score sentiment per headline with TextBlob
- Average sentiment per date with configurable windows
- Compute daily returns from price data
- Merge and compute Pearson/Spearman correlation with p-values
- Handle timezone issues for news publication times
- Test different sentiment aggregation windows (T-1 vs same-day)

Robustness Features:
- Timezone handling for news timestamps
- Multiple sentiment aggregation strategies
- Statistical significance testing (p-values)
- Confidence intervals for correlations
- Configurable date ranges via CLI
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class CorrelationResult:
    """Container for correlation analysis results"""
    pearson_corr: float
    spearman_corr: float
    pearson_pvalue: float
    spearman_pvalue: float
    confidence_interval_95: Tuple[float, float]
    n_observations: int
    aggregation_method: str
    bootstrap_ci_95: Tuple[float, float] = field(default_factory=lambda: (np.nan, np.nan))
    sentiment_method: str = "textblob"


def load_news(news_path: Path, ticker: str, 
              start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load and filter news data for a specific ticker
    
    Args:
        news_path: Path to news CSV file
        ticker: Stock ticker symbol
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        Filtered DataFrame with news data
    """
    news = pd.read_csv(
        news_path,
        usecols=["headline", "date", "date_only", "stock", "hour"],
        parse_dates=["date", "date_only"],
    )
    news = news[news["stock"] == ticker].copy()
    news["headline"] = news["headline"].fillna("")
    
    # Apply date filters
    if start_date:
        start = pd.to_datetime(start_date)
        news = news[news["date_only"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        news = news[news["date_only"] <= end]
    
    return news


def handle_timezone(news: pd.DataFrame, 
                   news_timezone: str = "America/New_York",
                   market_timezone: str = "America/New_York") -> pd.DataFrame:
    """
    Handle timezone conversion for news timestamps
    
    News is assumed to be published in the specified timezone (default: ET)
    Market closes at 4 PM ET, so news after 4 PM may affect next day prices
    
    Args:
        news: DataFrame with 'date' column
        news_timezone: Timezone of news timestamps
        market_timezone: Timezone of market (for market hours)
        
    Returns:
        DataFrame with timezone-adjusted dates
    """
    news = news.copy()
    
    # Check if date column is timezone-naive
    if news["date"].dt.tz is None:
        # Localize to news timezone
        news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(news_timezone)
    
    # Convert to market timezone
    news["date_market_tz"] = news["date"].dt.tz_convert(market_timezone)
    
    # Extract hour in market timezone
    news["hour_market"] = news["date_market_tz"].dt.hour
    
    # News published after 4 PM ET (market close) affects next trading day
    # This is a key insight: post-market news should be attributed to next day
    news["effective_date"] = np.where(
        news["hour_market"] >= 16,  # After 4 PM
        news["date_only"] + pd.Timedelta(days=1),
        news["date_only"]
    )
    
    return news


def score_sentiment(news: pd.DataFrame, method: str = "textblob") -> pd.DataFrame:
    """
    Score sentiment of each headline using TextBlob or VADER
    
    Args:
        news: DataFrame with 'headline' column
        method: Sentiment method ('textblob' or 'vader')
        
    Returns:
        DataFrame with added 'sentiment' column
    """
    news = news.copy()
    
    if method == "textblob":
        news["sentiment"] = news["headline"].apply(
            lambda text: TextBlob(str(text)).sentiment.polarity
        )
    elif method == "vader":
        # Initialize VADER analyzer
        analyzer = SentimentIntensityAnalyzer()
        news["sentiment"] = news["headline"].apply(
            lambda text: analyzer.polarity_scores(str(text))["compound"]
        )
    else:
        raise ValueError(f"Unknown sentiment method: {method}. Use 'textblob' or 'vader'")
    
    return news


def compute_bootstrap_ci(x: np.ndarray, y: np.ndarray, 
                         n_bootstrap: int = 1000,
                         confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for Pearson correlation
    
    Args:
        x: First variable (sentiment)
        y: Second variable (returns)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    n = len(x)
    bootstrap_corrs = []
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        
        # Skip if insufficient variance
        if np.std(x_boot) == 0 or np.std(y_boot) == 0:
            continue
            
        corr, _ = stats.pearsonr(x_boot, y_boot)
        bootstrap_corrs.append(corr)
    
    if len(bootstrap_corrs) < 10:
        return (np.nan, np.nan)
    
    # Calculate percentile-based confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_corrs, lower_percentile)
    ci_upper = np.percentile(bootstrap_corrs, upper_percentile)
    
    return (ci_lower, ci_upper)


def aggregate_daily_sentiment(news: pd.DataFrame, 
                              method: str = "mean",
                              date_column: str = "effective_date") -> pd.DataFrame:
    """
    Aggregate sentiment scores by date using various methods
    
    Args:
        news: DataFrame with sentiment scores
        method: Aggregation method ('mean', 'median', 'weighted')
        date_column: Column to use for grouping ('effective_date', 'date_only')
        
    Returns:
        DataFrame with aggregated daily sentiment
    """
    if date_column not in news.columns:
        if "effective_date" in news.columns:
            date_column = "effective_date"
        elif "date_only" in news.columns:
            date_column = "date_only"
        elif "date" in news.columns:
            date_column = "date"
        else:
            raise KeyError(f"Date column '{date_column}' not found in DataFrame")

    if method == "mean":
        sentiment = (
            news.groupby(date_column)["sentiment"]
            .mean()
            .rename("avg_daily_sentiment")
            .reset_index()
        )
    elif method == "median":
        sentiment = (
            news.groupby(date_column)["sentiment"]
            .median()
            .rename("avg_daily_sentiment")
            .reset_index()
        )
    elif method == "weighted":
        # Weight by headline length (longer headlines = more weight)
        news_copy = news.copy()
        news_copy["weight"] = news_copy["headline"].str.len()
        weighted_sum = news_copy.groupby(date_column).apply(
            lambda x: (x["sentiment"] * x["weight"]).sum() / x["weight"].sum()
        )
        sentiment = weighted_sum.rename("avg_daily_sentiment").reset_index()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    sentiment.columns = ["date_only", "avg_daily_sentiment"]
    return sentiment


def create_lagged_sentiment(sentiment: pd.DataFrame, 
                            lag: int = 1) -> pd.DataFrame:
    """
    Create lagged sentiment features
    
    Args:
        sentiment: DataFrame with daily sentiment
        lag: Number of days to lag (1 = previous day)
        
    Returns:
        DataFrame with lagged sentiment
    """
    sentiment = sentiment.copy()
    sentiment = sentiment.sort_values("date_only")
    sentiment[f"sentiment_lag_{lag}"] = sentiment["avg_daily_sentiment"].shift(lag)
    return sentiment


def load_prices(price_path: Path, 
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess price data
    
    Args:
        price_path: Path to price CSV file
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        
    Returns:
        DataFrame with daily returns
    """
    prices = pd.read_csv(price_path, parse_dates=["Date"])
    prices = prices.rename(columns={"Date": "date_only"})
    prices["daily_return"] = prices["Close"].pct_change()
    
    # Apply date filters
    if start_date:
        start = pd.to_datetime(start_date)
        prices = prices[prices["date_only"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        prices = prices[prices["date_only"] <= end]
    
    return prices[["date_only", "daily_return"]].dropna()


def compute_correlation_with_stats(sentiment: pd.DataFrame, 
                                   returns: pd.DataFrame,
                                   use_lagged: bool = False,
                                   lag: int = 1,
                                   sentiment_method: str = "textblob") -> CorrelationResult:
    """
    Compute correlation with statistical significance
    
    Args:
        sentiment: DataFrame with sentiment scores
        returns: DataFrame with daily returns
        use_lagged: Whether to use lagged sentiment
        lag: Lag in days for sentiment
        sentiment_method: Method used for sentiment analysis
        
    Returns:
        CorrelationResult with statistics
    """
    merged = pd.merge(sentiment, returns, on="date_only", how="inner").dropna()
    
    if merged.empty:
        raise ValueError("No overlapping dates between sentiment and returns.")
    
    # Determine which sentiment column to use
    if use_lagged:
        sentiment_col = f"sentiment_lag_{lag}"
        if sentiment_col not in merged.columns:
            # Create lag if not exists
            merged = create_lagged_sentiment(merged, lag)
        sentiment_values = merged[sentiment_col].dropna()
        returns_values = merged.loc[sentiment_values.index, "daily_return"]
    else:
        sentiment_values = merged["avg_daily_sentiment"]
        returns_values = merged["daily_return"]
    
    # Remove any remaining NaN
    valid_idx = ~(sentiment_values.isna() | returns_values.isna())
    sentiment_values = sentiment_values[valid_idx]
    returns_values = returns_values[valid_idx]
    
    if len(sentiment_values) < 10:
        raise ValueError("Insufficient observations for correlation analysis (minimum 10 required)")
    
    # Convert to numpy arrays for bootstrap
    x = sentiment_values.values
    y = returns_values.values
    
    # Pearson correlation with p-value
    pearson_corr, pearson_pvalue = stats.pearsonr(x, y)
    
    # Spearman correlation (rank-based, more robust to outliers)
    spearman_corr, spearman_pvalue = stats.spearmanr(x, y)
    
    # Confidence interval for Pearson correlation (Fisher z-transformation)
    n = len(x)
    z = np.arctanh(pearson_corr)  # Fisher z-transformation
    se = 1 / np.sqrt(n - 3)
    z_lower = z - 1.96 * se
    z_upper = z + 1.96 * se
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)
    
    # Bootstrap confidence interval
    bootstrap_ci = compute_bootstrap_ci(x, y, n_bootstrap=1000, confidence=0.95)
    
    method = f"lag_{lag}" if use_lagged else "same_day"
    
    return CorrelationResult(
        pearson_corr=pearson_corr,
        spearman_corr=spearman_corr,
        pearson_pvalue=pearson_pvalue,
        spearman_pvalue=spearman_pvalue,
        confidence_interval_95=(ci_lower, ci_upper),
        bootstrap_ci_95=bootstrap_ci,
        n_observations=n,
        aggregation_method=method,
        sentiment_method=sentiment_method
    )


def run(news_path: Path, price_path: Path, ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_timezone: bool = True,
        aggregation_method: str = "mean",
        use_lagged: bool = False,
        lag: int = 1,
        sentiment_method: str = "textblob",
        verbose: bool = True) -> CorrelationResult:
    """
    Run the complete correlation analysis
    
    Args:
        news_path: Path to news CSV
        price_path: Path to price CSV
        ticker: Stock ticker
        start_date: Start date filter
        end_date: End date filter
        use_timezone: Whether to handle timezone for news
        aggregation_method: How to aggregate sentiment ('mean', 'median', 'weighted')
        use_lagged: Whether to use lagged sentiment
        lag: Days to lag sentiment
        sentiment_method: Sentiment method ('textblob' or 'vader')
        verbose: Print results
        
    Returns:
        CorrelationResult with statistics
    """
    # Load data
    news = load_news(news_path, ticker, start_date, end_date)
    returns = load_prices(price_path, start_date, end_date)
    
    if verbose:
        print(f"Ticker: {ticker}")
        print(f"News articles: {len(news)}")
        print(f"Trading days: {len(returns)}")
    
    # Handle timezone
    if use_timezone:
        news = handle_timezone(news)
        date_col = "effective_date"
    else:
        date_col = "date_only"
    
    # Score sentiment (supports both TextBlob and VADER)
    news = score_sentiment(news, method=sentiment_method)
    
    # Aggregate sentiment
    sentiment = aggregate_daily_sentiment(news, method=aggregation_method, 
                                         date_column=date_col)
    
    if verbose:
        print(f"Days with sentiment: {len(sentiment)}")
    
    # Compute correlation with statistics
    result = compute_correlation_with_stats(sentiment, returns, 
                                           use_lagged=use_lagged, 
                                           lag=lag,
                                           sentiment_method=sentiment_method)
    
    if verbose:
        print(f"\nCorrelation Analysis Results:")
        print(f"  Sentiment Method: {result.sentiment_method}")
        print(f"  Aggregation: {result.aggregation_method}")
        print(f"  Observations: {result.n_observations}")
        print(f"  Pearson correlation: {result.pearson_corr:.4f}")
        print(f"  Pearson p-value: {result.pearson_pvalue:.4f}")
        print(f"  Spearman correlation: {result.spearman_corr:.4f}")
        print(f"  Spearman p-value: {result.spearman_pvalue:.4f}")
        print(f"  95% CI (Fisher z): [{result.confidence_interval_95[0]:.4f}, {result.confidence_interval_95[1]:.4f}]")
        print(f"  95% CI (Bootstrap): [{result.bootstrap_ci_95[0]:.4f}, {result.bootstrap_ci_95[1]:.4f}]")
        
        # Interpret significance
        if result.pearson_pvalue < 0.05:
            print("  Significance: STATISTICALLY SIGNIFICANT at α=0.05")
        else:
            print("  Significance: NOT significant at α=0.05")
    
    return result


def compute_rolling_correlation(sentiment: pd.DataFrame, 
                                 returns: pd.DataFrame,
                                 window: int = 30,
                                 use_lagged: bool = True,
                                 lag: int = 1) -> pd.DataFrame:
    """
    Compute rolling correlation between sentiment and returns
    
    Args:
        sentiment: DataFrame with daily sentiment
        returns: DataFrame with daily returns
        window: Rolling window size (default 30 days)
        use_lagged: Whether to use lagged sentiment
        lag: Lag in days for sentiment
        
    Returns:
        DataFrame with rolling correlations and p-values
    """
    # Merge sentiment and returns
    merged = pd.merge(sentiment, returns, on="date_only", how="inner")
    merged = merged.sort_values("date_only")
    
    # Create lagged sentiment if needed
    if use_lagged:
        merged = create_lagged_sentiment(merged, lag)
        sentiment_col = f"sentiment_lag_{lag}"
    else:
        sentiment_col = "avg_daily_sentiment"
    
    # Compute rolling correlation
    rolling_corrs = []
    rolling_pvalues = []
    dates = []
    
    for i in range(window, len(merged) + 1):
        window_data = merged.iloc[i - window:i]
        x = window_data[sentiment_col].dropna()
        y = window_data.loc[x.index, "daily_return"]
        
        if len(x) >= 10 and len(y) >= 10:
            corr, pval = stats.pearsonr(x, y)
        else:
            corr, pval = np.nan, np.nan
            
        rolling_corrs.append(corr)
        rolling_pvalues.append(pval)
        dates.append(merged.iloc[i - 1]["date_only"])
    
    return pd.DataFrame({
        "date": dates,
        "rolling_correlation": rolling_corrs,
        "rolling_pvalue": rolling_pvalues
    })


def plot_rolling_correlation(rolling_df: pd.DataFrame, 
                             ticker: str,
                             output_path: Optional[Path] = None) -> None:
    """
    Plot rolling correlation over time
    
    Args:
        rolling_df: DataFrame with rolling correlation results
        ticker: Stock ticker
        output_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot correlation
    axes[0].plot(rolling_df["date"], rolling_df["rolling_correlation"], 
                 label="Rolling Correlation (30-day)", color="blue", linewidth=1.5)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].fill_between(rolling_df["date"], 
                         rolling_df["rolling_correlation"], 0,
                         alpha=0.3)
    axes[0].set_ylabel("Pearson Correlation")
    axes[0].set_title(f"{ticker}: Rolling Correlation - Sentiment vs Returns (30-day window)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add significance threshold lines
    axes[0].axhline(y=0.3, color="green", linestyle=":", alpha=0.5, label="Weak ±0.3")
    axes[0].axhline(y=-0.3, color="green", linestyle=":", alpha=0.5)
    
    # Plot p-values
    axes[1].plot(rolling_df["date"], rolling_df["rolling_pvalue"], 
                 label="P-value", color="red", linewidth=1.5)
    axes[1].axhline(y=0.05, color="orange", linestyle="--", 
                     label="α = 0.05", alpha=0.7)
    axes[1].axhline(y=0.10, color="yellow", linestyle="--", 
                     label="α = 0.10", alpha=0.7)
    axes[1].set_ylabel("P-value")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Statistical Significance of Rolling Correlations")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved rolling correlation plot to {output_path}")
    
    plt.close()


def compute_sector_correlations(news_data: pd.DataFrame,
                                prices_data: dict,
                                sector_mapping: dict,
                                sentiment_method: str = "textblob",
                                use_lagged: bool = True,
                                lag: int = 1) -> pd.DataFrame:
    """
    Compute correlation by sector for multiple tickers
    
    Args:
        news_data: DataFrame with news articles containing stock column
        prices_data: Dictionary mapping ticker to price DataFrame
        sector_mapping: Dictionary mapping ticker to sector
        sentiment_method: Sentiment method to use
        use_lagged: Whether to use lagged sentiment
        lag: Lag in days
        
    Returns:
        DataFrame with correlations by sector
    """
    results = []
    
    for ticker, prices in prices_data.items():
        if ticker not in sector_mapping:
            continue
            
        sector = sector_mapping[ticker]
        
        # Get news for this ticker
        ticker_news = news_data[news_data["stock"] == ticker].copy()
        
        if len(ticker_news) < 10:
            continue
        
        # Score sentiment
        ticker_news = score_sentiment(ticker_news, method=sentiment_method)
        
        # Handle timezone
        ticker_news = handle_timezone(ticker_news)
        
        # Aggregate sentiment
        sentiment = aggregate_daily_sentiment(ticker_news, 
                                               date_column="effective_date")
        
        # Compute returns
        prices = prices.copy()
        prices["daily_return"] = prices["Close"].pct_change()
        prices = prices[["date_only", "daily_return"]].dropna()
        
        if len(sentiment) < 10 or len(prices) < 10:
            continue
        
        # Compute correlation
        try:
            result = compute_correlation_with_stats(
                sentiment, prices, 
                use_lagged=use_lagged, 
                lag=lag,
                sentiment_method=sentiment_method
            )
            
            results.append({
                "ticker": ticker,
                "sector": sector,
                "pearson_corr": result.pearson_corr,
                "pearson_pvalue": result.pearson_pvalue,
                "spearman_corr": result.spearman_corr,
                "n_observations": result.n_observations
            })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return pd.DataFrame(results)


def plot_sector_heatmap(sector_results: pd.DataFrame,
                        output_path: Optional[Path] = None) -> None:
    """
    Plot correlation heatmap by sector
    
    Args:
        sector_results: DataFrame with correlation results by sector
        output_path: Optional path to save the plot
    """
    if sector_results.empty:
        print("No sector results to plot")
        return
    
    # Aggregate by sector (mean correlation)
    sector_corr = sector_results.groupby("sector").agg({
        "pearson_corr": "mean",
        "spearman_corr": "mean",
        "ticker": "count"
    }).rename(columns={"ticker": "n_tickers"})
    
    # Create heatmap data
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pearson correlation heatmap
    corr_matrix = sector_corr[["pearson_corr"]].T
    sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn", center=0,
                vmin=-0.5, vmax=0.5, ax=axes[0], fmt=".3f")
    axes[0].set_title("Average Pearson Correlation by Sector")
    axes[0].set_ylabel("")
    
    # Spearman correlation heatmap
    corr_matrix = sector_corr[["spearman_corr"]].T
    sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn", center=0,
                vmin=-0.5, vmax=0.5, ax=axes[1], fmt=".3f")
    axes[1].set_title("Average Spearman Correlation by Sector")
    axes[1].set_ylabel("")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved sector heatmap to {output_path}")
    
    plt.close()


def run_comprehensive_analysis(news_path: Path, price_path: Path, ticker: str,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               sentiment_method: str = "textblob") -> dict:
    """
    Run multiple correlation analyses with different configurations
    
    Tests:
    - Same-day vs lagged sentiment
    - Different aggregation methods
    - Different sentiment methods (TextBlob vs VADER)
    
    Args:
        news_path: Path to news CSV
        price_path: Path to price CSV
        ticker: Stock ticker
        start_date: Start date filter
        end_date: End date filter
        sentiment_method: Sentiment method ('textblob' or 'vader')
        
    Returns:
        Dictionary with results from all configurations
    """
    results = {}
    
    # Test same-day vs lagged
    for use_lagged in [False, True]:
        for lag in [1, 2, 3]:
            key = f"{sentiment_method}_{'lag_' + str(lag) if use_lagged else 'same_day'}"
            try:
                result = run(news_path, price_path, ticker, 
                           start_date, end_date,
                           use_timezone=True,
                           aggregation_method="mean",
                           use_lagged=use_lagged,
                           lag=lag,
                           sentiment_method=sentiment_method,
                           verbose=False)
                results[key] = result
            except Exception as e:
                results[key] = {"error": str(e)}
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate news sentiment with stock price movements."
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker symbol to filter news and pick price file (default: AAPL).",
    )
    parser.add_argument(
        "--news-path",
        type=Path,
        default=Path("output/data/articles_with_topics.csv"),
        help="Path to the news CSV file.",
    )
    parser.add_argument(
        "--price-path",
        type=Path,
        default=Path("data") / "AAPL.csv",
        help="Path to the price CSV file (default assumes ticker.csv inside data/).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for analysis (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for analysis (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--no-timezone",
        action="store_true",
        help="Disable timezone handling for news timestamps.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "median", "weighted"],
        default="mean",
        help="Sentiment aggregation method (default: mean).",
    )
    parser.add_argument(
        "--sentiment-method",
        type=str,
        choices=["textblob", "vader"],
        default="textblob",
        help="Sentiment analysis method (default: textblob).",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        help="Number of days to lag sentiment (default: 1).",
    )
    parser.add_argument(
        "--use-lagged",
        action="store_true",
        help="Use lagged sentiment instead of same-day.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=30,
        help="Rolling window size for correlation (default: 30 days).",
    )
    parser.add_argument(
        "--rolling-plot",
        action="store_true",
        help="Generate rolling correlation plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/correlation"),
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--sector-heatmap",
        action="store_true",
        help="Generate sector correlation heatmap.",
    )
    parser.add_argument(
        "--sector-mapping",
        type=str,
        default=None,
        help="JSON file with ticker-to-sector mapping for sector analysis.",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive analysis with multiple configurations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Resolve price path
    resolved_price_path = args.price_path
    if resolved_price_path.is_dir():
        resolved_price_path = resolved_price_path / f"{args.ticker.upper()}.csv"
    
    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.comprehensive:
        print("Running comprehensive correlation analysis...")
        results = run_comprehensive_analysis(
            args.news_path, resolved_price_path, args.ticker,
            args.start_date, args.end_date,
            sentiment_method=args.sentiment_method
        )
        print("\nSummary of all configurations:")
        print("-" * 60)
        for key, result in results.items():
            if isinstance(result, dict) and "error" in result:
                print(f"{key}: Error - {result['error']}")
            else:
                sig = "SIGNIFICANT" if result.pearson_pvalue < 0.05 else "not sig"
                print(f"{key}: r={result.pearson_corr:.4f}, p={result.pearson_pvalue:.4f} ({sig})")
    else:
        result = run(
            args.news_path, resolved_price_path, args.ticker,
            args.start_date, args.end_date,
            use_timezone=not args.no_timezone,
            aggregation_method=args.aggregation,
            use_lagged=args.use_lagged,
            lag=args.lag,
            sentiment_method=args.sentiment_method,
            verbose=True
        )
        
        # Generate rolling correlation plot if requested
        if args.rolling_plot:
            print("\nGenerating rolling correlation plot...")
            
            # Load and process data for rolling correlation
            news = load_news(args.news_path, args.ticker, args.start_date, args.end_date)
            returns = load_prices(resolved_price_path, args.start_date, args.end_date)
            
            # Handle timezone and score sentiment
            news = handle_timezone(news)
            news = score_sentiment(news, method=args.sentiment_method)
            
            # Aggregate sentiment
            sentiment = aggregate_daily_sentiment(news, method=args.aggregation, 
                                                 date_column="effective_date")
            
            # Compute rolling correlation
            rolling_df = compute_rolling_correlation(
                sentiment, returns,
                window=args.rolling_window,
                use_lagged=args.use_lagged,
                lag=args.lag
            )
            
            # Plot rolling correlation
            output_plot_path = args.output_dir / f"rolling_correlation_{args.ticker}.png"
            plot_rolling_correlation(rolling_df, args.ticker, output_plot_path)
            
            print(f"Rolling correlation plot saved to {output_plot_path}")
    
    # Generate sector heatmap if requested
    if args.sector_heatmap:
        if not args.sector_mapping:
            print("Error: --sector-mapping required for sector heatmap")
        else:
            print("\nGenerating sector correlation heatmap...")
            
            # Load sector mapping
            with open(args.sector_mapping, 'r') as f:
                sector_mapping = json.load(f)
            
            # Load all news data
            news_data = pd.read_csv(
                args.news_path,
                usecols=["headline", "date", "date_only", "stock", "hour"],
                parse_dates=["date", "date_only"],
            )
            news_data["headline"] = news_data["headline"].fillna("")
            
            # Load price data for all tickers
            prices_data = {}
            data_dir = args.price_path if args.price_path.is_dir() else args.price_path.parent
            
            for ticker in sector_mapping.keys():
                price_file = data_dir / f"{ticker}.csv"
                if price_file.exists():
                    prices_data[ticker] = pd.read_csv(price_file, parse_dates=["Date"])
                    prices_data[ticker] = prices_data[ticker].rename(columns={"Date": "date_only"})
            
            # Compute sector correlations
            sector_results = compute_sector_correlations(
                news_data, prices_data, sector_mapping,
                sentiment_method=args.sentiment_method,
                use_lagged=args.use_lagged,
                lag=args.lag
            )
            
            # Plot sector heatmap
            output_heatmap_path = args.output_dir / "sector_correlation_heatmap.png"
            plot_sector_heatmap(sector_results, output_heatmap_path)
            
            print(f"Sector heatmap saved to {output_heatmap_path}")
            print("\nSector Results:")
            print(sector_results)
