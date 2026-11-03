"""
Feature Engineering Execution Script (Step 2)
Generates comprehensive ML-ready feature matrices combining:
- Headline text metrics (length, word count)
- Topic exposure and financial category indicators
- Publisher credibility weights and adjusted sentiment
- Temporal & market session metrics (minutes from open, session breakdown)
- Stock-specific news volume dynamics (surge z-score, rolling SMAs)
- Sentiment momentum & multi-day lags
- Technical market indicators & forward prediction targets

Usage:
    python scripts/run_feature_engineering.py --ticker AAPL
    python scripts/run_feature_engineering.py --ticker NVDA
    python scripts/run_feature_engineering.py --ticker AAPL --price-path data/AAPL.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from feature_engineer import FeatureEngineer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Feature Engineering for financial news and stock prices (Step 2)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol to process (default: AAPL).",
    )
    parser.add_argument(
        "--news-path",
        type=Path,
        default=None,
        help="Path to news CSV with sentiment scores (defaults to output/data/articles_with_sentiment.csv).",
    )
    parser.add_argument(
        "--price-path",
        type=Path,
        default=None,
        help="Path to stock OHLCV CSV file (defaults to data/{TICKER}.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save engineered features and charts (default: output).",
    )
    parser.add_argument(
        "--topic-col",
        type=str,
        default="lda_topic",
        help="Topic column name (default: lda_topic).",
    )
    return parser.parse_args()


def resolve_news_file(custom_path: Optional[Path]) -> Path:
    """Locate available news dataset in priority order."""
    if custom_path and custom_path.exists():
        return custom_path

    candidates = [
        project_root / "output" / "data" / "articles_with_sentiment.csv",
        project_root / "output" / "data" / "articles_with_topics.csv",
        project_root / "data" / "raw_analyst_ratings.csv",
        project_root / "data" / "financial_news.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError("Could not find a valid news file. Pass --news-path explicitly.")


def resolve_price_file(ticker: str, custom_path: Optional[Path]) -> Optional[Path]:
    """Locate stock price OHLCV file for ticker."""
    if custom_path and custom_path.exists():
        return custom_path

    ticker_file = project_root / "data" / f"{ticker.upper()}.csv"
    if ticker_file.exists():
        return ticker_file

    print(f"Notice: Price file not found at {ticker_file}. Processing news features without price alignment.")
    return None


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()

    news_file = resolve_news_file(args.news_path)
    price_file = resolve_price_file(ticker, args.price_path)

    print("=" * 80)
    print("STEP 2: FEATURE ENGINEERING EXECUTION")
    print(f"Ticker:        {ticker}")
    print(f"News Source:   {news_file}")
    print(f"Price Source:  {price_file if price_file else 'None'}")
    print(f"Output Dir:    {args.output_dir.resolve()}")
    print("=" * 80)

    # Load news
    print(f"Loading news from {news_file}...")
    news_df = pd.read_csv(news_file)
    print(f"Loaded {len(news_df):,} news articles.")

    # Load price if available
    price_df = None
    if price_file:
        print(f"Loading price history from {price_file}...")
        price_df = pd.read_csv(price_file)
        print(f"Loaded {len(price_df):,} price bars.")

    # Initialize FeatureEngineer
    engineer = FeatureEngineer(
        news_df=news_df,
        price_df=price_df,
        ticker=ticker,
        output_dir=args.output_dir,
    )

    # Run feature pipeline
    master_df = engineer.run_feature_pipeline(
        topic_col=args.topic_col,
        save=True
    )

    print(f"\nEngineered dataset preview (Last 5 rows):")
    cols_preview = [c for c in ["Date", "Close", "daily_return", "sentiment_mean", "sentiment_lag_1", "news_volume", "target_next_day_direction", "target_next_day_class"] if c in master_df.columns]
    print(master_df[cols_preview].tail(5))


if __name__ == "__main__":
    main()

