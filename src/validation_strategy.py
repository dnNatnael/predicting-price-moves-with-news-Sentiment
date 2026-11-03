"""
Validation Strategy Module
Step 6 of the predictive modeling pipeline for financial news and stock prices.

This module implements rigorous quantitative time-series validation protocols:
1. Expanding Window Time-Series Cross-Validation (TimeSeriesSplit):
   - K-fold expanding window cross-validation preventing lookahead data leakage.
   - Scaler and feature selection are fit strictly within each training fold.
   - Evaluates fold-by-fold accuracy, balanced accuracy, F1-macro, and ROC-AUC.
2. Rolling Window Walk-Forward Analysis (WFA):
   - Fixed rolling train window (e.g., 504 bars / 2 years) and out-of-sample forward test window
     (e.g., 126 bars / 6 months), stepping forward across the entire historical timeline.
   - Generates a contiguous out-of-sample prediction series simulating real-world algorithmic trading.
3. Out-of-Sample Regime & Robustness Diagnostics:
   - Measures model performance across Bull markets, Bear markets, and High/Low volatility regimes.
   - Computes institutional risk/return metrics: Sharpe ratio, Sortino ratio, Maximum Drawdown (MDD),
     Calmar ratio, Win Rate, and Profit Factor.
4. Visualization & Artifact Serialization:
   - Exports fold metrics CSV, out-of-sample trade history CSV, summary JSON, and 4-panel diagnostic plot.

Author: Nova Financial Solutions / Quantitative Research
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FoldMetrics:
    """Metrics for an individual time-series cross-validation fold."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc: Optional[float]
    strategy_return: float
    benchmark_return: float


@dataclass
class RobustnessMetrics:
    """Institutional risk-adjusted performance metrics for out-of-sample strategy."""
    total_bars: int
    trading_days: int
    cumulative_return: float
    benchmark_cumulative_return: float
    annualized_return: float
    benchmark_annualized_return: float
    annualized_volatility: float
    benchmark_annualized_volatility: float
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


class ValidationStrategy:
    """
    Implements TimeSeriesSplit cross-validation, rolling walk-forward analysis,
    and out-of-sample regime robustness testing.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        ticker: str = "AAPL",
        target_mode: str = "binary",
        output_dir: Union[str, Path] = "output",
        random_state: int = 42,
    ) -> None:
        """
        Initialize the ValidationStrategy.

        Args:
            features_df: Engineered features DataFrame (from Step 2).
            ticker: Stock ticker symbol (default: 'AAPL').
            target_mode: 'binary' (1 = Up, 0 = Down) or 'multiclass' (0, 1, 2).
            output_dir: Output directory for reports and charts.
            random_state: Random seed for reproducibility.
        """
        self.ticker = ticker.upper()
        self.target_mode = target_mode.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self.df = features_df.copy()
        self._prepare_dataset()

        # Results containers
        self.cv_results: Optional[pd.DataFrame] = None
        self.walk_forward_trades: Optional[pd.DataFrame] = None
        self.robustness_metrics: Optional[RobustnessMetrics] = None

    def _prepare_dataset(self) -> None:
        """Clean dataset, ensure chronological ordering and binary/multiclass target alignment."""
        date_col = "Date" if "Date" in self.df.columns else "effective_date"
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")
            self.df = self.df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        if "target_next_day_class" in self.df.columns:
            self.df = self.df[self.df["target_next_day_class"] != "Unknown"]

        if "target_next_day_return" in self.df.columns:
            self.df = self.df.dropna(subset=["target_next_day_return"])

        # Target direction (binary: 1 = Up, 0 = Down)
        if "target_next_day_direction" not in self.df.columns and "target_next_day_return" in self.df.columns:
            self.df["target_next_day_direction"] = (self.df["target_next_day_return"] > 0).astype(int)

        # Target multiclass (0 = Down, 1 = Neutral, 2 = Up)
        if "target_multiclass" not in self.df.columns and "target_next_day_return" in self.df.columns:
            def map_class(ret: float) -> int:
                if ret > 0.005:
                    return 2
                elif ret < -0.005:
                    return 0
                return 1
            self.df["target_multiclass"] = self.df["target_next_day_return"].apply(map_class)

    def _get_candidate_features(self) -> List[str]:
        """Return numeric feature columns excluding metadata and target columns."""
        exclude_cols = {
            "Date", "effective_date", "date", "stock", "target_next_day_return",
            "target_next_day_direction", "target_next_day_class", "target_multiclass",
            "target_3d_return", "target_5d_return"
        }
        return [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    def _select_train_features(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        top_k: int = 25,
        max_correlation: float = 0.85
    ) -> List[str]:
        """
        Screen features strictly within the training slice to prevent lookahead leakage.
        """
        candidates = self._get_candidate_features()
        target_series = train_df[target_col].astype(float)

        corrs = []
        for col in candidates:
            series = train_df[col].astype(float)
            if series.std() == 0 or len(series.dropna()) < 10:
                continue
            r, _ = stats.pearsonr(series.fillna(0.0), target_series)
            if not np.isnan(r):
                corrs.append({"feature": col, "abs_r": abs(r)})

        if not corrs:
            return candidates[:top_k]

        corr_df = pd.DataFrame(corrs).sort_values("abs_r", ascending=False)
        selected: List[str] = []

        for _, row in corr_df.iterrows():
            feat = row["feature"]
            if len(selected) >= top_k:
                break

            # Multicollinearity check
            is_collinear = False
            for s in selected:
                pairwise = abs(train_df[feat].corr(train_df[s]))
                if pairwise > max_correlation:
                    is_collinear = True
                    break

            if not is_collinear:
                selected.append(feat)

        return selected if selected else candidates[:top_k]

    def _instantiate_model(self, model_type: str) -> Any:
        """Create fresh estimator instance for cross-validation / walk-forward step."""
        m_type = model_type.lower()
        if m_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
            )
        elif m_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif m_type == "logistic_regression":
            return LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'gradient_boosting', 'random_forest', or 'logistic_regression'.")

    # =========================================================================
    # 1. Time-Series Cross-Validation (Expanding Window)
    # =========================================================================
    def run_time_series_cv(
        self,
        n_splits: int = 5,
        model_type: str = "gradient_boosting",
        top_k: int = 25,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Run K-fold expanding window TimeSeriesSplit cross-validation.
        Guarantees strictly chronological training and zero future data leakage.

        Args:
            n_splits: Number of cross-validation folds (default: 5).
            model_type: Classifier architecture to evaluate.
            top_k: Top non-collinear features selected on each training fold.
            save: Whether to persist results to disk.

        Returns:
            DataFrame containing performance metrics across all folds.
        """
        print("=" * 80)
        print(f"STEP 6: TIME-SERIES CROSS-VALIDATION ({n_splits} EXPANDING FOLDS)")
        print(f"Ticker: {self.ticker} | Model: {model_type} | Target Mode: {self.target_mode}")
        print("=" * 80)

        target_col = "target_next_day_direction" if self.target_mode == "binary" else "target_multiclass"
        date_col = "Date" if "Date" in self.df.columns else "effective_date"

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_records: List[Dict[str, Any]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(self.df), start=1):
            train_sub = self.df.iloc[train_idx].copy()
            test_sub = self.df.iloc[test_idx].copy()

            # Date boundaries
            train_start = str(train_sub[date_col].min())[:10]
            train_end = str(train_sub[date_col].max())[:10]
            test_start = str(test_sub[date_col].min())[:10]
            test_end = str(test_sub[date_col].max())[:10]

            # Feature selection strictly on train slice
            selected_features = self._select_train_features(train_sub, target_col=target_col, top_k=top_k)

            X_train_raw = train_sub[selected_features].fillna(0.0)
            X_test_raw = test_sub[selected_features].fillna(0.0)
            y_train = train_sub[target_col].astype(int)
            y_test = test_sub[target_col].astype(int)

            # Fit scaler on train slice ONLY
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            # Fit model
            model = self._instantiate_model(model_type)
            model.fit(X_train, y_train)

            # Evaluate predictions
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # ROC-AUC
            roc = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)
                    if self.target_mode == "binary" and proba.shape[1] == 2:
                        roc = roc_auc_score(y_test, proba[:, 1])
                    else:
                        roc = roc_auc_score(y_test, proba, multi_class="ovr")
                except Exception:
                    roc = None

            # Out-of-sample trading return in this fold
            actual_returns = test_sub["target_next_day_return"].values
            if self.target_mode == "binary":
                signals = y_pred  # 1 = Long, 0 = Cash
            else:
                signals = (y_pred == 2).astype(int)

            strat_ret = float(np.prod(1.0 + signals * actual_returns) - 1.0)
            bench_ret = float(np.prod(1.0 + actual_returns) - 1.0)

            fold_metric = FoldMetrics(
                fold=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=len(train_idx),
                test_size=len(test_idx),
                accuracy=round(float(acc), 4),
                balanced_accuracy=round(float(bal_acc), 4),
                precision_macro=round(float(prec), 4),
                recall_macro=round(float(rec), 4),
                f1_macro=round(float(f1), 4),
                roc_auc=round(float(roc), 4) if roc is not None else None,
                strategy_return=round(float(strat_ret), 4),
                benchmark_return=round(float(bench_ret), 4),
            )
            fold_records.append(asdict(fold_metric))

            print(f"Fold {fold_idx}: Train {train_start} -> {train_end} ({len(train_idx)} bars) | "
                  f"Test {test_start} -> {test_end} ({len(test_idx)} bars) | "
                  f"Acc: {acc*100:.2f}% | F1: {f1:.4f} | ROC-AUC: {roc if roc else 0.0:.4f}")

        cv_df = pd.DataFrame(fold_records)
        self.cv_results = cv_df

        # Print aggregate summary
        mean_acc = cv_df["accuracy"].mean()
        std_acc = cv_df["accuracy"].std()
        mean_f1 = cv_df["f1_macro"].mean()
        mean_roc = cv_df["roc_auc"].dropna().mean() if not cv_df["roc_auc"].dropna().empty else 0.0

        print("-" * 80)
        print(f"TimeSeriesSplit Aggregate: Accuracy={mean_acc*100:.2f}% (±{std_acc*100:.2f}%), "
              f"Macro F1={mean_f1:.4f}, Mean ROC-AUC={mean_roc:.4f}")
        print("-" * 80)

        if save:
            out_file = self.output_dir / "data" / f"time_series_cv_results_{self.ticker}.csv"
            cv_df.to_csv(out_file, index=False)
            print(f"✓ Cross-validation results saved to: {out_file}")

        return cv_df

    # =========================================================================
    # 2. Rolling Window Walk-Forward Analysis (WFA)
    # =========================================================================
    def run_walk_forward_analysis(
        self,
        train_window_bars: int = 504,   # ~2 trading years
        test_window_bars: int = 126,    # ~6 months
        step_bars: int = 126,
        model_type: str = "gradient_boosting",
        top_k: int = 25,
        save: bool = True,
    ) -> Tuple[pd.DataFrame, RobustnessMetrics]:
        """
        Execute continuous rolling Walk-Forward Analysis (WFA).
        Simulates live algorithmic deployment across multi-year market cycles.

        Args:
            train_window_bars: Number of trailing historical bars used to train in each step.
            test_window_bars: Number of forward out-of-sample bars predicted before re-training.
            step_bars: Number of bars to advance the rolling window.
            model_type: Classifier architecture.
            top_k: Number of non-collinear features selected on each rolling window.
            save: Whether to persist out-of-sample trades and summary metrics.

        Returns:
            Tuple of (walk_forward_trades DataFrame, RobustnessMetrics).
        """
        print("\n" + "=" * 80)
        print(f"STEP 6: ROLLING WALK-FORWARD ANALYSIS (WFA)")
        print(f"Train Window: {train_window_bars} bars (~{train_window_bars/252:.1f} yrs) | "
              f"Test Window: {test_window_bars} bars (~{test_window_bars/252:.1f} yrs) | Step: {step_bars} bars")
        print("=" * 80)

        target_col = "target_next_day_direction" if self.target_mode == "binary" else "target_multiclass"
        date_col = "Date" if "Date" in self.df.columns else "effective_date"

        n = len(self.df)
        oos_trade_records: List[Dict[str, Any]] = []
        step_idx = 1

        for start_idx in range(0, n - train_window_bars, step_bars):
            train_end_idx = start_idx + train_window_bars
            test_end_idx = min(train_end_idx + test_window_bars, n)

            if test_end_idx - train_end_idx < 10:
                break

            train_slice = self.df.iloc[start_idx:train_end_idx].copy()
            test_slice = self.df.iloc[train_end_idx:test_end_idx].copy()

            # Select features on train slice ONLY
            features = self._select_train_features(train_slice, target_col=target_col, top_k=top_k)

            X_tr_raw = train_slice[features].fillna(0.0)
            X_te_raw = test_slice[features].fillna(0.0)
            y_tr = train_slice[target_col].astype(int)
            y_te = test_slice[target_col].astype(int)

            # Fit scaler on train slice ONLY
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            # Train fresh model
            model = self._instantiate_model(model_type)
            model.fit(X_tr, y_tr)

            # Predict on out-of-sample forward slice
            y_pred = model.predict(X_te)
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_te)
                    y_prob = proba[:, 1] if self.target_mode == "binary" else proba[:, 2]
                except Exception:
                    y_prob = None

            for i in range(len(test_slice)):
                bar_row = test_slice.iloc[i]
                pred_label = int(y_pred[i])
                actual_label = int(y_te.iloc[i])
                actual_ret = float(bar_row["target_next_day_return"])

                # Signal: 1 = Long, 0 = Cash
                if self.target_mode == "binary":
                    signal = 1 if pred_label == 1 else 0
                else:
                    signal = 1 if pred_label == 2 else 0

                strat_daily_ret = signal * actual_ret

                oos_trade_records.append({
                    "date": bar_row[date_col],
                    "close": float(bar_row["Close"]) if "Close" in bar_row else 0.0,
                    "actual_return": actual_ret,
                    "target_direction": actual_label,
                    "predicted_direction": pred_label,
                    "probability": round(float(y_prob[i]), 4) if y_prob is not None else 0.5,
                    "is_correct": int(pred_label == actual_label),
                    "signal": signal,
                    "strategy_daily_return": strat_daily_ret,
                    "benchmark_daily_return": actual_ret,
                    "step": step_idx,
                })

            step_idx += 1

        trades_df = pd.DataFrame(oos_trade_records).sort_values("date").reset_index(drop=True)
        self.walk_forward_trades = trades_df

        # Calculate Cumulative Returns & Drawdowns
        trades_df["strategy_equity"] = (1.0 + trades_df["strategy_daily_return"]).cumprod()
        trades_df["benchmark_equity"] = (1.0 + trades_df["benchmark_daily_return"]).cumprod()
        trades_df["strategy_peak"] = trades_df["strategy_equity"].cummax()
        trades_df["strategy_drawdown"] = (trades_df["strategy_equity"] - trades_df["strategy_peak"]) / trades_df["strategy_peak"]
        trades_df["benchmark_peak"] = trades_df["benchmark_equity"].cummax()
        trades_df["benchmark_drawdown"] = (trades_df["benchmark_equity"] - trades_df["benchmark_peak"]) / trades_df["benchmark_peak"]

        # Compute Institutional Risk/Return Metrics
        metrics = self._compute_robustness_metrics(trades_df)
        self.robustness_metrics = metrics

        print(f"✓ Walk-Forward Analysis complete across {len(trades_df):,} out-of-sample trading days ({step_idx-1} rolling steps).")
        print(f"  Out-of-Sample Accuracy:   {trades_df['is_correct'].mean()*100:.2f}%")
        print(f"  Strategy CAGR:            {metrics.annualized_return*100:.2f}% (vs Benchmark: {metrics.benchmark_annualized_return*100:.2f}%)")
        print(f"  Strategy Sharpe Ratio:    {metrics.sharpe_ratio:.2f} (vs Benchmark: {metrics.benchmark_sharpe_ratio:.2f})")
        print(f"  Strategy Max Drawdown:    {metrics.max_drawdown*100:.2f}% (vs Benchmark: {metrics.benchmark_max_drawdown*100:.2f}%)")
        print(f"  Strategy Calmar Ratio:    {metrics.calmar_ratio:.2f}")

        if save:
            trades_file = self.output_dir / "data" / f"walk_forward_trades_{self.ticker}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"✓ Walk-forward trade records saved to: {trades_file}")

            metrics_file = self.output_dir / "data" / f"walk_forward_metrics_{self.ticker}.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(asdict(metrics), f, indent=2)
            print(f"✓ Robustness metrics summary saved to: {metrics_file}")

            self.plot_validation_results(save=True)

        return trades_df, metrics

    # =========================================================================
    # 3. Robustness & Regime Diagnostics
    # =========================================================================
    def _compute_robustness_metrics(self, trades_df: pd.DataFrame) -> RobustnessMetrics:
        """Calculate quantitative risk, return, and regime robustness statistics."""
        n_days = len(trades_df)
        years = max(n_days / 252.0, 0.1)

        strat_cum = float(trades_df["strategy_equity"].iloc[-1] - 1.0)
        bench_cum = float(trades_df["benchmark_equity"].iloc[-1] - 1.0)

        strat_cagr = float((1.0 + strat_cum) ** (1.0 / years) - 1.0) if strat_cum > -1.0 else -1.0
        bench_cagr = float((1.0 + bench_cum) ** (1.0 / years) - 1.0) if bench_cum > -1.0 else -1.0

        strat_vol = float(trades_df["strategy_daily_return"].std() * np.sqrt(252))
        bench_vol = float(trades_df["benchmark_daily_return"].std() * np.sqrt(252))

        strat_sharpe = float(strat_cagr / strat_vol) if strat_vol > 0 else 0.0
        bench_sharpe = float(bench_cagr / bench_vol) if bench_vol > 0 else 0.0

        # Sortino Ratio (Downside deviation)
        neg_rets = trades_df.loc[trades_df["strategy_daily_return"] < 0, "strategy_daily_return"]
        downside_std = float(neg_rets.std() * np.sqrt(252)) if len(neg_rets) > 1 else 1e-4
        sortino = float(strat_cagr / downside_std) if downside_std > 0 else 0.0

        # Max Drawdown
        strat_mdd = float(abs(trades_df["strategy_drawdown"].min()))
        bench_mdd = float(abs(trades_df["benchmark_drawdown"].min()))

        calmar = float(strat_cagr / strat_mdd) if strat_mdd > 0 else 0.0

        # Win Rate & Profit Factor
        active_trades = trades_df[trades_df["signal"] > 0]
        win_rate = float((active_trades["strategy_daily_return"] > 0).mean()) if len(active_trades) > 0 else 0.0

        gains = active_trades.loc[active_trades["strategy_daily_return"] > 0, "strategy_daily_return"].sum()
        losses = abs(active_trades.loc[active_trades["strategy_daily_return"] < 0, "strategy_daily_return"].sum())
        profit_factor = float(gains / losses) if losses > 0 else float(gains)

        # Regime-Specific Accuracies
        bull_days = trades_df[trades_df["benchmark_daily_return"] > 0]
        bear_days = trades_df[trades_df["benchmark_daily_return"] <= 0]
        bull_acc = float(bull_days["is_correct"].mean()) if len(bull_days) > 0 else 0.0
        bear_acc = float(bear_days["is_correct"].mean()) if len(bear_days) > 0 else 0.0

        # Volatility Regimes (median return magnitude as high/low proxy)
        ret_mag = trades_df["benchmark_daily_return"].abs()
        med_mag = ret_mag.median()
        high_vol = trades_df[ret_mag > med_mag]
        low_vol = trades_df[ret_mag <= med_mag]
        high_vol_acc = float(high_vol["is_correct"].mean()) if len(high_vol) > 0 else 0.0
        low_vol_acc = float(low_vol["is_correct"].mean()) if len(low_vol) > 0 else 0.0

        return RobustnessMetrics(
            total_bars=n_days,
            trading_days=n_days,
            cumulative_return=round(strat_cum, 4),
            benchmark_cumulative_return=round(bench_cum, 4),
            annualized_return=round(strat_cagr, 4),
            benchmark_annualized_return=round(bench_cagr, 4),
            annualized_volatility=round(strat_vol, 4),
            benchmark_annualized_volatility=round(bench_vol, 4),
            sharpe_ratio=round(strat_sharpe, 4),
            benchmark_sharpe_ratio=round(bench_sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown=round(strat_mdd, 4),
            benchmark_max_drawdown=round(bench_mdd, 4),
            calmar_ratio=round(calmar, 4),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            bull_regime_accuracy=round(bull_acc, 4),
            bear_regime_accuracy=round(bear_acc, 4),
            high_vol_accuracy=round(high_vol_acc, 4),
            low_vol_accuracy=round(low_vol_acc, 4),
        )

    # =========================================================================
    # 4. Diagnostic Visualizations
    # =========================================================================
    def plot_validation_results(self, save: bool = True) -> None:
        """
        Generate 4-panel institutional validation diagnostic dashboard:
        1. TimeSeriesSplit CV Performance across expanding folds
        2. Walk-Forward Cumulative Equity Curves (Strategy vs Buy & Hold)
        3. Walk-Forward Underwater Drawdown comparison
        4. Regime Robustness (Bull vs Bear vs Volatility Breakdown)
        """
        if self.walk_forward_trades is None or self.walk_forward_trades.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        trades = self.walk_forward_trades
        trades_dates = pd.to_datetime(trades["date"])

        # Panel 1: TimeSeriesSplit CV Folds
        ax1 = axes[0, 0]
        if self.cv_results is not None and not self.cv_results.empty:
            cv_plot = self.cv_results.set_index("fold")[["accuracy", "f1_macro"]]
            cv_plot.plot(kind="bar", ax=ax1, color=["#3498db", "#2ecc71"], alpha=0.85)
            ax1.set_title(f"TimeSeriesSplit Cross-Validation Folds ({self.ticker})", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Metric Value", fontsize=11)
            ax1.set_xlabel("Fold Index", fontsize=11)
            ax1.set_ylim([0.0, 1.0])
            ax1.axhline(0.50, color="gray", linestyle="--", linewidth=0.8, label="Random Guess (50%)")
            ax1.legend(["Accuracy", "Macro F1", "Random Guess"], loc="lower right")
            ax1.grid(True, alpha=0.3, axis="y")
        else:
            ax1.text(0.5, 0.5, "TimeSeriesSplit Not Executed", ha="center", va="center")

        # Panel 2: Walk-Forward Out-of-Sample Cumulative Equity
        ax2 = axes[0, 1]
        ax2.plot(trades_dates, trades["strategy_equity"], label="News Sentiment Strategy", color="#27ae60", linewidth=1.8)
        ax2.plot(trades_dates, trades["benchmark_equity"], label=f"Buy & Hold {self.ticker}", color="#2980b9", linestyle="--", linewidth=1.5)
        ax2.set_title(f"Walk-Forward Cumulative Return (Out-of-Sample: {len(trades):,} Days)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Portfolio Growth ($1 Initial)", fontsize=11)
        ax2.set_yscale("log")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Underwater Drawdown
        ax3 = axes[1, 0]
        ax3.fill_between(trades_dates, trades["strategy_drawdown"] * 100, 0, color="#e74c3c", alpha=0.4, label="Strategy Drawdown")
        ax3.plot(trades_dates, trades["benchmark_drawdown"] * 100, color="#7f8c8d", linestyle=":", linewidth=1.2, label="Benchmark Drawdown")
        ax3.set_title("Walk-Forward Underwater Drawdown (%)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Drawdown (%)", fontsize=11)
        ax3.set_xlabel("Date", fontsize=11)
        ax3.legend(loc="lower left")
        ax3.grid(True, alpha=0.3)

        # Panel 4: Market Regime Breakdown
        ax4 = axes[1, 1]
        if self.robustness_metrics is not None:
            m = self.robustness_metrics
            regimes = ["Bull Days", "Bear Days", "High Volatility", "Low Volatility"]
            accs = [
                m.bull_regime_accuracy * 100,
                m.bear_regime_accuracy * 100,
                m.high_vol_accuracy * 100,
                m.low_vol_accuracy * 100,
            ]
            colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]
            bars = ax4.bar(regimes, accs, color=colors, alpha=0.85)
            ax4.set_title("Out-of-Sample Accuracy across Market Regimes (%)", fontsize=12, fontweight="bold")
            ax4.set_ylabel("Accuracy (%)", fontsize=11)
            ax4.set_ylim([0, 100])
            ax4.axhline(50.0, color="black", linestyle="--", linewidth=0.8, label="50% Baseline")
            ax4.grid(True, alpha=0.3, axis="y")

            # Value labels on bars
            for b in bars:
                h = b.get_height()
                ax4.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        if save:
            out_fig = self.output_dir / "figures" / f"walk_forward_validation_{self.ticker}.png"
            plt.savefig(out_fig, dpi=300, bbox_inches="tight")
            print(f"✓ Validation diagnostics chart saved to: {out_fig}")
        plt.close()

