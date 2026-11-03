"""
Sentiment Analysis Preparation Runner
Executes Step 1 of the predictive modeling pipeline:
- Headline sentiment scoring (VADER + TextBlob)
- Topic-based sentiment feature generation
- Publisher-specific sentiment patterns and bias extraction
- News spike sentiment dynamics
- Market session and time-based temporal sentiment features

Usage:
    python scripts/prepare_sentiment.py
    python scripts/prepare_sentiment.py --news-path output/data/articles_with_topics.csv --output-dir output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from sentiment_analyzer import SentimentAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sentiment Analysis Preparation for financial news dataset."
    )
    parser.add_argument(
        "--news-path",
        type=Path,
        default=None,
        help="Path to news CSV file with headlines and dates. (Defaults to output/data/articles_with_topics.csv or data/financial_news.csv)",
    )
    parser.add_argument(
        "--spikes-path",
        type=Path,
        default=Path("output/data/news_spikes.csv"),
        help="Optional path to news_spikes.csv from EDA.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save generated sentiment features and figures.",
    )
    parser.add_argument(
        "--topic-col",
        type=str,
        default=None,
        help="Column to use for topic grouping ('lda_topic', 'bertopic_label', or 'financial_category').",
    )
    parser.add_argument(
        "--min-publisher-articles",
        type=int,
        default=10,
        help="Minimum article threshold to profile a publisher's sentiment bias (default: 10).",
    )
    parser.add_argument(
        "--spike-threshold-std",
        type=float,
        default=2.0,
        help="Standard deviation threshold for volume spike detection (default: 2.0).",
    )
    return parser.parse_args()


def resolve_news_path(custom_path: Optional[Path]) -> Path:
    """Find available news file in priority order."""
    if custom_path and custom_path.exists():
        return custom_path

    candidates = [
        project_root / "output" / "data" / "articles_with_topics.csv",
        project_root / "data" / "financial_news.csv",
        project_root / "data" / "raw_analyst_ratings.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a valid news dataset. Please pass --news-path with a valid CSV file."
    )


def main() -> None:
    args = parse_args()

    news_file = resolve_news_path(args.news_path)
    print("=" * 80)
    print("FINANCIAL NEWS SENTIMENT ANALYSIS PREPARATION")
    print(f"Loading data from: {news_file}")
    print(f"Output directory:  {args.output_dir.resolve()}")
    print("=" * 80)

    # Load news
    df = pd.read_csv(news_file)
    print(f"Loaded {len(df):,} records.")

    # Initialize SentimentAnalyzer
    analyzer = SentimentAnalyzer(
        df=df,
        output_dir=args.output_dir,
    )

    # Check for optional spikes dataframe
    spikes_df = None
    if args.spikes_path and args.spikes_path.exists():
        try:
            spikes_df = pd.read_csv(args.spikes_path)
            print(f"✓ Loaded {len(spikes_df)} detected news spikes from {args.spikes_path}")
        except Exception as e:
            print(f"Notice: Could not load spikes CSV ({e}), will compute spikes automatically.")

    # Run full preparation
    results = analyzer.run_full_preparation(save_enriched_articles=True)

    print("\nSummary of Generated Artifacts:")
    print(f"  * Enriched Articles:       {args.output_dir / 'data' / 'articles_with_sentiment.csv'}")
    print(f"  * Topic Sentiment:         {args.output_dir / 'data' / 'topic_daily_sentiment.csv'}")
    print(f"  * Publisher Patterns:      {args.output_dir / 'data' / 'publisher_sentiment_patterns.csv'}")
    print(f"  * Spike Dynamics:          {args.output_dir / 'data' / 'spike_sentiment_analysis.csv'}")
    print(f"  * Time/Session Features:   {args.output_dir / 'data' / 'time_based_sentiment_features.csv'}")
    print(f"  * Visualizations:          {args.output_dir / 'figures'}")
    print(f"  * Report:                  {args.output_dir / 'sentiment_preparation_report.txt'}")


if __name__ == "__main__":
    main()

