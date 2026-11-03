"""
Step 3: Advanced Correlation Analysis Execution Script
Executes systematic quantitative correlation analysis:
- Multi-lag & horizon correlation analysis (T to T-5 vs 1d, 3d, 5d returns)
- Topic-specific price movement drivers
- Publisher market influence & predictive alpha
- Market session predictive power (Pre-market vs Regular vs Post)
- Feature selection scorecard for ML modeling

Usage:
    python scripts/run_correlation_analysis.py --ticker AAPL
    python scripts/run_correlation_analysis.py --ticker NVDA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from correlation_analyzer import CorrelationAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Advanced Correlation Analysis (Step 3)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock symbol to analyze (default: AAPL).",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Path to engineered features CSV (default: output/data/engineered_features_{TICKER}.csv).",
    )
    parser.add_argument(
        "--news-path",
        type=Path,
        default=Path("output/data/articles_with_sentiment.csv"),
        help="Path to articles with sentiment CSV for publisher-level analysis.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target_next_day_return",
        help="Target column for correlation screening (default: target_next_day_return).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()

    # Resolve features path
    features_path = args.features_path
    if not features_path:
        features_path = args.output_dir / "data" / f"engineered_features_{ticker}.csv"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Engineered features not found at {features_path}. "
            f"Please run 'python scripts/run_feature_engineering.py --ticker {ticker}' first."
        )

    print("=" * 80)
    print("STEP 3: ADVANCED QUANTITATIVE CORRELATION ANALYSIS")
    print(f"Ticker:        {ticker}")
    print(f"Features:      {features_path}")
    print(f"News Source:   {args.news_path}")
    print(f"Target:        {args.target_col}")
    print(f"Output Dir:    {args.output_dir.resolve()}")
    print("=" * 80)

    # Load data
    features_df = pd.read_csv(features_path)
    print(f"Loaded {len(features_df):,} samples with {len(features_df.columns)} features.")

    news_df = None
    if args.news_path and args.news_path.exists():
        try:
            news_df = pd.read_csv(args.news_path)
            print(f"Loaded {len(news_df):,} articles for publisher-level analysis.")
        except Exception as e:
            print(f"Notice: Could not load news CSV ({e}). Proceeding without publisher breakdown.")

    # Initialize and run
    analyzer = CorrelationAnalyzer(
        features_df=features_df,
        news_df=news_df,
        ticker=ticker,
        output_dir=args.output_dir,
    )

    results = analyzer.run_full_correlation_analysis(
        target_col=args.target_col,
        save=True
    )

    print("\nSummary of Top 5 Predictive Features for Target Return:")
    top_5 = results["scorecard"].head(5)[["feature_name", "pearson_r", "pearson_pvalue", "spearman_rho", "is_significant"]]
    print(top_5.to_string(index=False))

    print("\nSummary of Generated Step 3 Artifacts:")
    print(f"  * Multi-Lag Correlations:   {args.output_dir / 'data' / f'lag_correlations_{ticker}.csv'}")
    print(f"  * Topic Drivers:            {args.output_dir / 'data' / f'topic_price_drivers_{ticker}.csv'}")
    print(f"  * Publisher Influence:      {args.output_dir / 'data' / f'publisher_market_influence_{ticker}.csv'}")
    print(f"  * Session Predictive Power: {args.output_dir / 'data' / f'session_predictive_power_{ticker}.csv'}")
    print(f"  * Top Features Scorecard:   {args.output_dir / 'data' / f'top_predictive_features_{ticker}.csv'}")
    print(f"  * JSON Summary Report:      {args.output_dir / 'data' / f'correlation_summary_{ticker}.json'}")
    print(f"  * Figures Directory:        {args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()

