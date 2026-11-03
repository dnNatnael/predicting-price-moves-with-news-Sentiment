"""
Step 4: Machine Learning Model Training & Evaluation Runner
Executes Step 4 of the predictive modeling pipeline:
- Temporal chronological train/val/test splitting (70% / 15% / 15%)
- Class imbalance balancing
- Multi-model training (Logistic Regression, Random Forest, Gradient Boosting)
- Out-of-sample performance evaluation (Accuracy, F1-macro, ROC-AUC)
- Trading strategy backtest simulation vs Buy-and-Hold
- Serialization of best model, scaler, and features for dashboard inference

Usage:
    python scripts/train_models.py --ticker AAPL --target-mode binary
    python scripts/train_models.py --ticker AAPL --target-mode multiclass
    python scripts/train_models.py --ticker NVDA --target-mode binary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from model_pipeline import ModelPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate machine learning models for stock price move prediction (Step 4)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL).",
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
        help="Target classification mode: 'binary' (Up vs Down) or 'multiclass' (Up vs Neutral vs Down).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of non-collinear predictive features to select (default: 25).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save models and evaluation outputs (default: output).",
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
    print("STEP 4: PREDICTIVE MACHINE LEARNING MODEL TRAINING")
    print(f"Ticker:        {ticker}")
    print(f"Target Mode:   {args.target_mode.upper()}")
    print(f"Features:      {features_path}")
    print(f"Top K Feats:   {args.top_k}")
    print(f"Output Dir:    {args.output_dir.resolve()}")
    print("=" * 80)

    # Load dataset
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df):,} trading days.")

    # Initialize ModelPipeline
    pipeline = ModelPipeline(
        features_df=df,
        ticker=ticker,
        target_mode=args.target_mode,
        output_dir=args.output_dir,
    )

    # Run complete modeling pipeline
    results = pipeline.run_full_model_pipeline(
        top_k_features=args.top_k,
        save=True
    )

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE TEST PERFORMANCE SUMMARY:")
    print("=" * 80)
    cols = ["model_name", "test_accuracy", "test_f1_macro", "test_roc_auc", "strategy_cumulative_return", "strategy_sharpe_ratio"]
    print(results["eval_df"][cols].to_string(index=False))

    print("\nGenerated Step 4 Artifacts:")
    print(f"  * Best Model:          {args.output_dir / 'models' / f'best_model_{ticker}_{args.target_mode}.joblib'}")
    print(f"  * Feature Scaler:      {args.output_dir / 'models' / f'scaler_{ticker}.joblib'}")
    print(f"  * Model Comparison:    {args.output_dir / 'data' / f'model_comparison_{ticker}_{args.target_mode}.csv'}")
    print(f"  * Evaluation JSON:     {args.output_dir / 'data' / f'model_evaluation_{ticker}_{args.target_mode}.json'}")
    print(f"  * Feature Importances: {args.output_dir / 'data' / f'feature_importances_{ticker}_{args.target_mode}.csv'}")
    print(f"  * Evaluation Figures:  {args.output_dir / 'figures' / f'model_evaluation_{ticker}_{args.target_mode}.png'}")


if __name__ == "__main__":
    main()

