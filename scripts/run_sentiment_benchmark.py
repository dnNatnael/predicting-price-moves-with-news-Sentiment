"""
Step 5: Financial Sentiment Scoring Approaches Benchmark Runner
Executes Step 5 of the predictive modeling pipeline:
- Applies four distinct sentiment scoring approaches:
    1. Financial VADER (Wall Street lexicon enhanced)
    2. Loughran-McDonald (Academic financial disclosures dictionary)
    3. TextBlob (General English baseline)
    4. Financial Transformer (FinBERT / calibrated heuristic)
    5. Weighted Consensus Ensemble
- Benchmarks each approach against real stock price returns (Pearson r, Spearman rho, statistical significance)
- Exports comparison CSV, JSON summaries, and comparative visualizations.

Usage:
    python scripts/run_sentiment_benchmark.py --ticker AAPL
    python scripts/run_sentiment_benchmark.py --ticker NVDA --sample-size 5000
    python scripts/run_sentiment_benchmark.py --ticker AAPL --use-finbert-model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from sentiment_scorers import EnsembleSentimentScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-approach financial sentiment scoring and return benchmarking (Step 5)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol to benchmark against (default: AAPL).",
    )
    parser.add_argument(
        "--news-path",
        type=Path,
        default=None,
        help="Path to news dataset (defaults to output/data/articles_with_topics.csv or articles_with_sentiment.csv).",
    )
    parser.add_argument(
        "--price-path",
        type=Path,
        default=None,
        help="Path to stock price CSV (defaults to data/{TICKER}.csv).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional max number of articles to sample for faster scoring (default: all).",
    )
    parser.add_argument(
        "--use-finbert-model",
        action="store_true",
        default=False,
        help="Download and load full HuggingFace FinBERT transformer weights (defaults to False, using calibrated financial heuristic).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save comparison outputs and figures (default: output).",
    )
    return parser.parse_args()


def resolve_news_file(custom_path: Path | None) -> Path:
    """Locate available news dataset in priority order."""
    if custom_path and custom_path.exists():
        return custom_path

    candidates = [
        project_root / "output" / "data" / "articles_with_topics.csv",
        project_root / "output" / "data" / "articles_with_sentiment.csv",
        project_root / "data" / "raw_analyst_ratings.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError("Could not find a valid news file. Pass --news-path explicitly.")


def resolve_price_file(ticker: str, custom_path: Path | None) -> Path | None:
    """Locate stock price OHLCV file for ticker."""
    if custom_path and custom_path.exists():
        return custom_path

    ticker_file = project_root / "data" / f"{ticker.upper()}.csv"
    if ticker_file.exists():
        return ticker_file

    print(f"Notice: Price file not found at {ticker_file}. Running sentiment scoring without price alignment.")
    return None


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()

    news_file = resolve_news_file(args.news_path)
    price_file = resolve_price_file(ticker, args.price_path)

    print("=" * 80)
    print("STEP 5: FINANCIAL SENTIMENT SCORING APPROACHES BENCHMARK")
    print(f"Target Ticker:     {ticker}")
    print(f"News Source:       {news_file}")
    print(f"Price Source:      {price_file if price_file else 'None'}")
    print(f"FinBERT Model:     {'HuggingFace Transformer' if args.use_finbert_model else 'Calibrated Domain Engine'}")
    print(f"Output Directory:  {args.output_dir.resolve()}")
    print("=" * 80)

    # 1. Load news data
    print(f"\nLoading news data from {news_file}...")
    news_df = pd.read_csv(news_file)
    print(f"Loaded {len(news_df):,} raw articles.")

    # Filter by ticker if available
    if "stock" in news_df.columns:
        stock_matches = news_df[news_df["stock"].astype(str).str.upper() == ticker]
        if len(stock_matches) >= 50:
            print(f"Filtering news to ticker '{ticker}': {len(stock_matches):,} matching articles found.")
            news_df = stock_matches.copy()
        else:
            print(f"Ticker '{ticker}' has few specific matches ({len(stock_matches)}). Using full news corpus ({len(news_df):,} articles).")

    # Subsample if requested
    if args.sample_size and len(news_df) > args.sample_size:
        print(f"Subsampling to {args.sample_size:,} articles for accelerated evaluation...")
        news_df = news_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    # 2. Load stock prices if available
    price_df = None
    if price_file and price_file.exists():
        print(f"Loading price series from {price_file}...")
        price_df = pd.read_csv(price_file)
        if "daily_return" not in price_df.columns and "Close" in price_df.columns:
            price_df["daily_return"] = price_df["Close"].pct_change()
        print(f"Loaded {len(price_df):,} daily bars.")

    # 3. Initialize and execute EnsembleSentimentScorer
    scorer = EnsembleSentimentScorer(
        news_df=news_df,
        output_dir=args.output_dir,
    )

    enriched_df = scorer.run_all_scorers(use_finbert_model=args.use_finbert_model)
    comp_df = scorer.benchmark_sentiment_approaches(price_returns_df=price_df, save=True)

    # 4. Display formatted summary table
    print("\n" + "=" * 80)
    print("STEP 5 BENCHMARK RESULTS: SENTIMENT APPROACH COMPARISON")
    print("=" * 80)

    cols_to_show = [
        "approach_name",
        "mean_score",
        "pct_positive",
        "pct_neutral",
        "pct_negative",
        "pearson_r_with_return",
        "spearman_rho_with_return",
        "is_significant",
    ]
    formatted = comp_df[cols_to_show].copy()
    formatted.columns = [
        "Approach",
        "Mean",
        "Pos %",
        "Neu %",
        "Neg %",
        "Pearson r",
        "Spearman ρ",
        "Sig (p<0.05)",
    ]
    print(formatted.to_string(index=False))

    print("\n" + "=" * 80)
    print("KEY ARTIFACTS GENERATED:")
    print(f"1. Comparison CSV:   {args.output_dir / 'data' / 'sentiment_approach_comparison.csv'}")
    print(f"2. Enriched News:    {args.output_dir / 'data' / 'articles_with_multi_sentiment.csv'}")
    print(f"3. Summary JSON:     {args.output_dir / 'data' / 'sentiment_benchmark_summary.json'}")
    print(f"4. Comparison Chart: {args.output_dir / 'figures' / 'sentiment_approaches_comparison.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

