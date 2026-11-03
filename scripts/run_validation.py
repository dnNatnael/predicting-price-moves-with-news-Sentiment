"""
Step 6: Quantitative Validation Strategy Runner
Executes Step 6 of the predictive modeling pipeline:
- K-fold expanding window TimeSeriesSplit cross-validation
- Rolling window Walk-Forward Analysis (WFA) across multi-year market cycles
- Quantitative risk/return metrics: Sharpe ratio, Sortino, Max Drawdown, Calmar, Win Rate
- Market regime robustness diagnostics (Bull vs Bear vs High/Low Volatility)
- Diagnostic chart export

Usage:
    python scripts/run_validation.py --ticker AAPL
    python scripts/run_validation.py --ticker NVDA --model-type gradient_boosting
    python scripts/run_validation.py --ticker AAPL --n-splits 5 --train-window 504 --test-window 126
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from validation_strategy import ValidationStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rigorous time-series cross-validation and walk-forward analysis (Step 6)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol to validate (default: AAPL).",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Path to engineered features CSV (defaults to output/data/engineered_features_{TICKER}.csv).",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["binary", "multiclass"],
        default="binary",
        help="Target classification mode: 'binary' (Up vs Down) or 'multiclass'.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gradient_boosting", "random_forest", "logistic_regression"],
        default="gradient_boosting",
        help="Classifier architecture to evaluate (default: gradient_boosting).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of expanding TimeSeriesSplit folds (default: 5).",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=504,
        help="Rolling train window size in bars for Walk-Forward Analysis (~252 bars/yr, default: 504).",
    )
    parser.add_argument(
        "--test-window",
        type=int,
        default=126,
        help="Out-of-sample forward test window size in bars (~126 bars/6 mo, default: 126).",
    )
    parser.add_argument(
        "--step-bars",
        type=int,
        default=126,
        help="Advance step size in bars for rolling walk-forward window (default: 126).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of non-collinear features selected on each training window (default: 25).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save validation outputs and figures (default: output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()

    features_path = args.features_path
    if not features_path:
        features_path = args.output_dir / "data" / f"engineered_features_{ticker}.csv"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Engineered features dataset not found at {features_path}. "
            f"Please run 'python scripts/run_feature_engineering.py --ticker {ticker}' first."
        )

    print("=" * 80)
    print("STEP 6: QUANTITATIVE TIME-SERIES VALIDATION PIPELINE")
    print(f"Ticker:            {ticker}")
    print(f"Features Dataset:  {features_path}")
    print(f"Target Mode:       {args.target_mode}")
    print(f"Model Type:        {args.model_type}")
    print(f"TimeSeriesSplit:   {args.n_splits} folds")
    print(f"Walk-Forward WFA:  Train={args.train_window} bars, Test={args.test_window} bars, Step={args.step_bars} bars")
    print(f"Output Directory:  {args.output_dir.resolve()}")
    print("=" * 80)

    # 1. Load engineered dataset
    print(f"\nLoading engineered dataset from {features_path}...")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df):,} trading bars across {df.shape[1]} features.")

    # 2. Initialize ValidationStrategy
    validator = ValidationStrategy(
        features_df=df,
        ticker=ticker,
        target_mode=args.target_mode,
        output_dir=args.output_dir,
    )

    # 3. Run TimeSeriesSplit cross-validation
    cv_df = validator.run_time_series_cv(
        n_splits=args.n_splits,
        model_type=args.model_type,
        top_k=args.top_k,
        save=True,
    )

    # 4. Run Rolling Walk-Forward Analysis
    trades_df, metrics = validator.run_walk_forward_analysis(
        train_window_bars=args.train_window,
        test_window_bars=args.test_window,
        step_bars=args.step_bars,
        model_type=args.model_type,
        top_k=args.top_k,
        save=True,
    )

    # 5. Summary Presentation
    print("\n" + "=" * 80)
    print(f"VALIDATION SUMMARY: {ticker} OUT-OF-SAMPLE RESULTS")
    print("=" * 80)
    print(f"Total Out-of-Sample Days:  {metrics.total_bars:,} bars (~{metrics.total_bars/252:.1f} years)")
    print(f"Out-of-Sample Accuracy:    {trades_df['is_correct'].mean()*100:.2f}%")
    print(f"Strategy Cumulative Return:{metrics.cumulative_return*100:+.2f}%  (vs Buy & Hold: {metrics.benchmark_cumulative_return*100:+.2f}%)")
    print(f"Strategy CAGR:             {metrics.annualized_return*100:+.2f}%  (vs Buy & Hold: {metrics.benchmark_annualized_return*100:+.2f}%)")
    print(f"Strategy Sharpe Ratio:     {metrics.sharpe_ratio:.2f}  (vs Buy & Hold: {metrics.benchmark_sharpe_ratio:.2f})")
    print(f"Strategy Sortino Ratio:    {metrics.sortino_ratio:.2f}")
    print(f"Strategy Max Drawdown:     {metrics.max_drawdown*100:.2f}%  (vs Buy & Hold: {metrics.benchmark_max_drawdown*100:.2f}%)")
    print(f"Strategy Calmar Ratio:     {metrics.calmar_ratio:.2f}")
    print(f"Win Rate:                  {metrics.win_rate*100:.2f}%")
    print(f"Profit Factor:             {metrics.profit_factor:.2f}")
    print("-" * 80)
    print("REGIME ACCURACY BREAKDOWN:")
    print(f"  • Bull Markets (Up Days):     {metrics.bull_regime_accuracy*100:.2f}%")
    print(f"  • Bear Markets (Down Days):   {metrics.bear_regime_accuracy*100:.2f}%")
    print(f"  • High Volatility Periods:    {metrics.high_vol_accuracy*100:.2f}%")
    print(f"  • Low Volatility Periods:     {metrics.low_vol_accuracy*100:.2f}%")
    print("=" * 80)

    print("\nSAVED STEP 6 ARTIFACTS:")
    print(f"1. Cross-Validation Results: {args.output_dir / 'data' / f'time_series_cv_results_{ticker}.csv'}")
    print(f"2. Walk-Forward Trades:      {args.output_dir / 'data' / f'walk_forward_trades_{ticker}.csv'}")
    print(f"3. Robustness Metrics:       {args.output_dir / 'data' / f'walk_forward_metrics_{ticker}.json'}")
    print(f"4. Diagnostic Dashboard:     {args.output_dir / 'figures' / f'walk_forward_validation_{ticker}.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

