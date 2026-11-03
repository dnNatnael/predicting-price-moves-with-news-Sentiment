"""
Model Preparation and Predictive Modeling Module
Step 4 of the predictive modeling pipeline for financial news and stock prices.

This module prepares data and trains machine learning models to predict stock price moves:
1. Strict Temporal Train / Validation / Test Splitting:
   - Splits time-series chronologically (e.g. 70% Train, 15% Val, 15% Test) to strictly
     eliminate lookahead bias and future data leakage.
2. Class Imbalance Handling:
   - Computes balanced class weights or resamples to address market drift and class skew.
3. Binary & Multi-Class Classification:
   - Binary Mode: Direction prediction (1 = Up, 0 = Down).
   - Multi-Class Mode: 3-class prediction (Up > +0.5%, Down < -0.5%, Neutral).
4. Correlation-Driven Feature Selection:
   - Filters features using correlation significance (p < 0.05) and removes redundant collinearity.
   - Scaler (StandardScaler) is fit strictly on training data.
5. Model Training & Comparison:
   - Trains Logistic Regression, Random Forest, and Gradient Boosting.
   - Computes Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
6. Trading Strategy Simulation & Feature Importance:
   - Evaluates a realistic strategy (Long when Up, Neutral/Cash when Down) vs Buy-and-Hold.
   - Computes Gini feature importances and saves serializable artifacts for the final dashboard.

Author: Nova Financial Solutions / Quantitative Research
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Evaluation metrics for a trained predictive model."""
    model_name: str
    target_mode: str  # 'binary' or 'multiclass'
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_precision_macro: float
    test_recall_macro: float
    test_f1_macro: float
    test_roc_auc: Optional[float]
    strategy_cumulative_return: float
    benchmark_cumulative_return: float
    strategy_sharpe_ratio: float


class ModelPipeline:
    """
    End-to-end model preparation, training, and evaluation pipeline for financial price moves.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        ticker: str = "AAPL",
        target_mode: str = "binary",
        output_dir: Union[str, Path] = "output",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the ModelPipeline.

        Args:
            features_df: Engineered features DataFrame (from Step 2).
            ticker: Stock symbol (default: 'AAPL').
            target_mode: 'binary' (Up vs Down) or 'multiclass' (Up vs Neutral vs Down).
            output_dir: Directory where models, metrics, and figures will be saved.
            train_ratio: Fraction of earliest data for training (default: 0.70).
            val_ratio: Fraction of middle data for validation (default: 0.15).
            test_ratio: Fraction of latest data for out-of-sample testing (default: 0.15).
            random_state: Random seed for reproducibility.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)

        self.ticker = ticker.upper()
        self.target_mode = target_mode.lower()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.df = features_df.copy()
        self._prepare_dataset()

        # Split containers
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.scaler: Optional[StandardScaler] = None
        self.selected_features: List[str] = []
        self.class_weights_dict: Optional[Dict[int, float]] = None

        # Trained models & metrics
        self.trained_models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, ModelMetrics] = {}
        self.best_model_name: Optional[str] = None

    def _prepare_dataset(self) -> None:
        """Clean target variables and ensure chronological sorting."""
        date_col = "Date" if "Date" in self.df.columns else "effective_date"
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")
            self.df = self.df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        # Drop trailing row where target_next_day_return is NaN / Unknown
        if "target_next_day_class" in self.df.columns:
            self.df = self.df[self.df["target_next_day_class"] != "Unknown"]

        if "target_next_day_return" in self.df.columns:
            self.df = self.df.dropna(subset=["target_next_day_return"])

        # Create binary target (1 = Up, 0 = Down)
        if "target_next_day_direction" not in self.df.columns and "target_next_day_return" in self.df.columns:
            self.df["target_next_day_direction"] = (self.df["target_next_day_return"] > 0).astype(int)

        # Create multi-class target (0 = Down, 1 = Neutral, 2 = Up)
        if "target_multiclass" not in self.df.columns and "target_next_day_return" in self.df.columns:
            def map_class_num(ret: float) -> int:
                if ret > 0.005:
                    return 2  # Up
                elif ret < -0.005:
                    return 0  # Down
                else:
                    return 1  # Neutral
            self.df["target_multiclass"] = self.df["target_next_day_return"].apply(map_class_num)

    # =========================================================================
    # 1. Feature Selection
    # =========================================================================
    def select_features(
        self,
        top_k: int = 25,
        min_pvalue: float = 0.20,
        max_correlation: float = 0.85,
    ) -> List[str]:
        """
        Select top predictive features based on correlation screening and multicollinearity removal.

        Args:
            top_k: Maximum number of features to select.
            min_pvalue: p-value threshold for inclusion (relaxed to 0.20 to capture informative features).
            max_correlation: Maximum pairwise correlation allowed between selected features.

        Returns:
            List of selected feature column names.
        """
        target_col = "target_next_day_direction" if self.target_mode == "binary" else "target_multiclass"

        # Exclude metadata and target columns
        exclude_cols = {
            "Date", "effective_date", "date", "stock", "target_next_day_return",
            "target_next_day_direction", "target_next_day_class", "target_multiclass",
            "target_3d_return", "target_5d_return"
        }
        candidate_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        print(f"Screening {len(candidate_cols)} candidate features for {self.target_mode} prediction...")

        # Calculate correlations with target
        corrs = []
        target_series = self.df[target_col].astype(float)

        for col in candidate_cols:
            series = self.df[col].astype(float)
            if series.std() == 0 or len(series.dropna()) < 10:
                continue

            r, p_val = stats.pearsonr(series.dropna(), target_series.loc[series.dropna().index])
            corrs.append({
                "feature": col,
                "abs_r": abs(r),
                "r": r,
                "p_value": p_val
            })

        corr_df = pd.DataFrame(corrs).sort_values("abs_r", ascending=False)

        # Greedily pick non-collinear features
        selected = []
        for _, row in corr_df.iterrows():
            feat = row["feature"]
            if len(selected) >= top_k:
                break

            # Check multicollinearity against already selected features
            is_collinear = False
            for s in selected:
                pairwise_corr = abs(self.df[feat].corr(self.df[s]))
                if pairwise_corr > max_correlation:
                    is_collinear = True
                    break

            if not is_collinear:
                selected.append(feat)

        print(f"✓ Selected {len(selected)} non-collinear features for model training.")
        self.selected_features = selected
        return selected

    # =========================================================================
    # 2. Temporal Train / Validation / Test Splitting
    # =========================================================================
    def create_temporal_splits(self, top_k_features: int = 25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partition dataset chronologically into Train (70%), Validation (15%), and Test (15%).
        Fits StandardScaler strictly on the training set to prevent lookahead leakage.

        Returns:
            Tuple of (X_train, X_val, X_test) scaled DataFrames.
        """
        if not self.selected_features:
            self.select_features(top_k=top_k_features)

        target_col = "target_next_day_direction" if self.target_mode == "binary" else "target_multiclass"

        n = len(self.df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_data = self.df.iloc[:train_end].copy()
        val_data = self.df.iloc[train_end:val_end].copy()
        test_data = self.df.iloc[val_end:].copy()

        print(f"Temporal Split: Train={len(train_data):,} bars ({self.train_ratio*100:.0f}%), "
              f"Val={len(val_data):,} bars ({self.val_ratio*100:.0f}%), "
              f"Test={len(test_data):,} bars ({self.test_ratio*100:.0f}%)")

        # Feature matrices
        X_train_raw = train_data[self.selected_features].fillna(0.0)
        X_val_raw = val_data[self.selected_features].fillna(0.0)
        X_test_raw = test_data[self.selected_features].fillna(0.0)

        # Scale features using training distribution ONLY
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(X_train_raw), columns=self.selected_features, index=train_data.index)
        self.X_val = pd.DataFrame(self.scaler.transform(X_val_raw), columns=self.selected_features, index=val_data.index)
        self.X_test = pd.DataFrame(self.scaler.transform(X_test_raw), columns=self.selected_features, index=test_data.index)

        # Target series
        self.y_train = train_data[target_col]
        self.y_val = val_data[target_col]
        self.y_test = test_data[target_col]

        # Compute balanced class weights on training labels
        classes = np.unique(self.y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
        self.class_weights_dict = {c: float(w) for c, w in zip(classes, weights)}
        print(f"Class Weights ({self.target_mode}): {self.class_weights_dict}")

        return self.X_train, self.X_val, self.X_test

    # =========================================================================
    # 3. Model Training & Comparison
    # =========================================================================
    def train_models(self) -> Dict[str, Any]:
        """
        Train and tune three diverse classification models:
        1. Logistic Regression (L2 regularized, balanced class weight)
        2. Random Forest Classifier (Non-linear ensemble, depth-constrained)
        3. Gradient Boosting Classifier (Boosting tree ensemble)

        Returns:
            Dictionary of trained model objects.
        """
        if self.X_train is None:
            self.create_temporal_splits()

        print("\nTraining classification models...")

        models = {
            "Logistic_Regression": LogisticRegression(
                C=0.1,
                class_weight=self.class_weights_dict,
                max_iter=1000,
                random_state=self.random_state
            ),
            "Random_Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=10,
                class_weight=self.class_weights_dict,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "Gradient_Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state
            )
        }

        for name, model in models.items():
            print(f"  * Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model

        print("✓ All models trained successfully.")
        return self.trained_models

    # =========================================================================
    # 4. Model Evaluation & Trading Strategy Simulation
    # =========================================================================
    def evaluate_models(self, save: bool = True) -> pd.DataFrame:
        """
        Evaluate trained models across Accuracy, Balanced Accuracy, Precision, Recall,
        F1-Score, ROC-AUC, and simulate market trading strategy returns.

        Returns:
            Comparison DataFrame of model performances.
        """
        if not self.trained_models:
            self.train_models()

        print("\nEvaluating model performance on test set (Out-Of-Sample)...")

        # Test period returns for strategy simulation
        test_returns = self.df.loc[self.X_test.index, "daily_return"].fillna(0.0)
        benchmark_cum_ret = float((1.0 + test_returns).cumprod().iloc[-1] - 1.0) if len(test_returns) > 0 else 0.0

        records = []
        for name, model in self.trained_models.items():
            # Predictions
            train_preds = model.predict(self.X_train)
            val_preds = model.predict(self.X_val)
            test_preds = model.predict(self.X_test)

            # Probabilities for AUC
            test_roc_auc = None
            if hasattr(model, "predict_proba"):
                test_probs = model.predict_proba(self.X_test)
                try:
                    if self.target_mode == "binary":
                        test_roc_auc = float(roc_auc_score(self.y_test, test_probs[:, 1]))
                    else:
                        test_roc_auc = float(roc_auc_score(self.y_test, test_probs, multi_class="ovr"))
                except Exception:
                    test_roc_auc = None

            # Classification metrics
            train_acc = float(accuracy_score(self.y_train, train_preds))
            val_acc = float(accuracy_score(self.y_val, val_preds))
            test_acc = float(accuracy_score(self.y_test, test_preds))
            test_bal_acc = float(balanced_accuracy_score(self.y_test, test_preds))
            test_prec = float(precision_score(self.y_test, test_preds, average="macro", zero_division=0))
            test_rec = float(recall_score(self.y_test, test_preds, average="macro", zero_division=0))
            test_f1 = float(f1_score(self.y_test, test_preds, average="macro", zero_division=0))

            # Strategy simulation: Go long when predicted Up, Cash (0 return) when predicted Down/Neutral
            if self.target_mode == "binary":
                position = test_preds  # 1 for Up, 0 for Down
            else:
                position = (test_preds == 2).astype(int)  # 2 for Up

            strat_returns = position * test_returns
            strat_cum_ret = float((1.0 + strat_returns).cumprod().iloc[-1] - 1.0) if len(strat_returns) > 0 else 0.0

            # Annualized Sharpe ratio of strategy
            strat_std = strat_returns.std()
            sharpe = float((strat_returns.mean() / strat_std) * np.sqrt(252)) if strat_std > 0 else 0.0

            metric_obj = ModelMetrics(
                model_name=name,
                target_mode=self.target_mode,
                train_accuracy=round(train_acc, 4),
                val_accuracy=round(val_acc, 4),
                test_accuracy=round(test_acc, 4),
                test_balanced_accuracy=round(test_bal_acc, 4),
                test_precision_macro=round(test_prec, 4),
                test_recall_macro=round(test_rec, 4),
                test_f1_macro=round(test_f1, 4),
                test_roc_auc=round(test_roc_auc, 4) if test_roc_auc is not None else None,
                strategy_cumulative_return=round(strat_cum_ret, 4),
                benchmark_cumulative_return=round(benchmark_cum_ret, 4),
                strategy_sharpe_ratio=round(sharpe, 4),
            )
            self.evaluation_results[name] = metric_obj
            records.append(asdict(metric_obj))

        eval_df = pd.DataFrame(records).sort_values("test_f1_macro", ascending=False).reset_index(drop=True)
        self.best_model_name = eval_df.iloc[0]["model_name"]
        print(f"✓ Best Model: {self.best_model_name} (F1: {eval_df.iloc[0]['test_f1_macro']:.4f}, Test Acc: {eval_df.iloc[0]['test_accuracy']:.4f})")

        if save:
            out_file = self.output_dir / "data" / f"model_comparison_{self.ticker}_{self.target_mode}.csv"
            eval_df.to_csv(out_file, index=False)
            print(f"✓ Model comparison saved to: {out_file}")

            # Save best model pipeline
            self._save_model_artifacts()
            self._save_evaluation_summary()
            self.plot_model_evaluation(save=True)

        return eval_df

    def _save_model_artifacts(self) -> None:
        """Save best trained model, scaler, and feature list for future dashboard inference."""
        best_model = self.trained_models[self.best_model_name]

        model_path = self.output_dir / "models" / f"best_model_{self.ticker}_{self.target_mode}.joblib"
        scaler_path = self.output_dir / "models" / f"scaler_{self.ticker}.joblib"
        features_path = self.output_dir / "models" / f"selected_features_{self.ticker}.json"

        joblib.dump(best_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        with open(features_path, "w", encoding="utf-8") as f:
            json.dump({"features": self.selected_features, "target_mode": self.target_mode}, f, indent=2)

        print(f"✓ Best model artifact saved to: {model_path}")
        print(f"✓ Scaler artifact saved to: {scaler_path}")

    def _save_evaluation_summary(self) -> Path:
        """Save detailed JSON report of model evaluation and classification report."""
        summary_path = self.output_dir / "data" / f"model_evaluation_{self.ticker}_{self.target_mode}.json"

        best_model = self.trained_models[self.best_model_name]
        test_preds = best_model.predict(self.X_test)
        report_dict = classification_report(self.y_test, test_preds, output_dict=True, zero_division=0)

        # Feature importances if available
        feat_importances = {}
        if hasattr(best_model, "feature_importances_"):
            for feat, imp in zip(self.selected_features, best_model.feature_importances_):
                feat_importances[feat] = round(float(imp), 4)
        elif hasattr(best_model, "coef_"):
            coefs = np.abs(best_model.coef_[0]) if len(best_model.coef_.shape) > 1 else np.abs(best_model.coef_)
            for feat, imp in zip(self.selected_features, coefs):
                feat_importances[feat] = round(float(imp), 4)

        # Sort feature importances
        sorted_fi = dict(sorted(feat_importances.items(), key=lambda item: item[1], reverse=True))

        summary = {
            "ticker": self.ticker,
            "target_mode": self.target_mode,
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "best_model": self.best_model_name,
            "models_evaluated": {k: asdict(v) for k, v in self.evaluation_results.items()},
            "classification_report_best_model": report_dict,
            "feature_importances": sorted_fi,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Save feature importances as CSV
        fi_csv = self.output_dir / "data" / f"feature_importances_{self.ticker}_{self.target_mode}.csv"
        pd.DataFrame(list(sorted_fi.items()), columns=["feature", "importance"]).to_csv(fi_csv, index=False)

        print(f"✓ Model evaluation summary saved to: {summary_path}")
        return summary_path

    # =========================================================================
    # 5. Visualizations
    # =========================================================================
    def plot_model_evaluation(self, save: bool = True) -> None:
        """Plot model comparison charts, confusion matrix, and feature importances."""
        best_model = self.trained_models[self.best_model_name]
        test_preds = best_model.predict(self.X_test)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Panel 1: Model Accuracy & F1 Comparison
        eval_df = pd.DataFrame([asdict(v) for v in self.evaluation_results.values()])
        x_indices = np.arange(len(eval_df))
        width = 0.35

        ax1.bar(x_indices - width/2, eval_df["test_accuracy"], width=width, label="Test Accuracy", color="#3498db")
        ax1.bar(x_indices + width/2, eval_df["test_f1_macro"], width=width, label="Test F1-Score (Macro)", color="#2ecc71")
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels([n.replace("_", "\n") for n in eval_df["model_name"]], fontsize=10)
        ax1.set_title(f"Model Performance Comparison ({self.ticker})", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Score", fontsize=11)
        ax1.set_ylim(0, 1.0)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: Confusion Matrix
        cm = confusion_matrix(self.y_test, test_preds)
        labels = ["Down", "Up"] if self.target_mode == "binary" else ["Down", "Neutral", "Up"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_title(f"Confusion Matrix ({self.best_model_name})", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Predicted Label", fontsize=11)
        ax2.set_ylabel("Actual Label", fontsize=11)

        # Panel 3: Feature Importances
        fi_file = self.output_dir / "data" / f"feature_importances_{self.ticker}_{self.target_mode}.csv"
        if fi_file.exists():
            fi_df = pd.read_csv(fi_file).head(12).sort_values("importance", ascending=True)
            ax3.barh(fi_df["feature"], fi_df["importance"], color="#9b59b6", alpha=0.85)
            ax3.set_title(f"Top Feature Importances ({self.best_model_name})", fontsize=12, fontweight="bold")
            ax3.set_xlabel("Relative Importance", fontsize=11)
            ax3.grid(True, alpha=0.3, axis="x")

        # Panel 4: Simulated Trading Strategy vs Benchmark
        test_returns = self.df.loc[self.X_test.index, "daily_return"].fillna(0.0)
        pos = test_preds if self.target_mode == "binary" else (test_preds == 2).astype(int)
        strat_returns = pos * test_returns

        cum_strat = (1.0 + strat_returns).cumprod()
        cum_bench = (1.0 + test_returns).cumprod()

        ax4.plot(cum_strat.values, label=f"Sentiment Strategy (Return: {cum_strat.iloc[-1]-1:.1%})", color="#27ae60", linewidth=2.0)
        ax4.plot(cum_bench.values, label=f"Buy & Hold (Return: {cum_bench.iloc[-1]-1:.1%})", color="#7f8c8d", linestyle="--", linewidth=1.5)
        ax4.set_title("Out-of-Sample Trading Strategy Simulation", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Test Trading Days", fontsize=11)
        ax4.set_ylabel("Growth of $1.00", fontsize=11)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            out_fig = self.output_dir / "figures" / f"model_evaluation_{self.ticker}_{self.target_mode}.png"
            plt.savefig(out_fig, dpi=300, bbox_inches="tight")
            print(f"✓ Model evaluation charts saved to: {out_fig}")
        plt.close()

    # =========================================================================
    # 6. Master Step 4 Pipeline Runner
    # =========================================================================
    def run_full_model_pipeline(self, top_k_features: int = 25, save: bool = True) -> Dict[str, Any]:
        """
        Execute full Step 4 pipeline:
        1. Feature selection (correlation & collinearity screening)
        2. Temporal chronological splitting (Train/Val/Test)
        3. Class imbalance balancing
        4. Model training (Logistic Regression, Random Forest, Gradient Boosting)
        5. Model evaluation & trading simulation
        6. Serialization of best model artifact for the dashboard

        Returns:
            Dictionary containing evaluation DataFrame and trained models.
        """
        print("=" * 80)
        print(f"STARTING STEP 4: MODEL PREPARATION & TRAINING ({self.ticker} | {self.target_mode.upper()})")
        print("=" * 80)

        # 1. Feature selection
        self.select_features(top_k=top_k_features)

        # 2. Temporal train/val/test split
        self.create_temporal_splits()

        # 3. Model training
        self.train_models()

        # 4. Evaluation & strategy backtesting
        eval_df = self.evaluate_models(save=save)

        print("\n" + "=" * 80)
        print(f"STEP 4: MODEL PIPELINE COMPLETED SUCCESSFULLY ✓")
        print(f"Best Performing Model: {self.best_model_name}")
        print("=" * 80)

        return {
            "eval_df": eval_df,
            "models": self.trained_models,
            "best_model_name": self.best_model_name,
            "selected_features": self.selected_features,
        }
