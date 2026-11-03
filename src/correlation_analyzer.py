"""
Advanced Correlation Analysis Module
Step 3 of the predictive modeling pipeline for financial news and stock prices.

This module conducts systematic quantitative correlation analysis between:
1. Multi-Lag & Multi-Horizon Sentiment vs Stock Returns:
   - Measures Pearson r, Spearman rho, p-values, and 95% bootstrap confidence intervals
     across lags (T, T-1, T-2, T-3, T-5) and forward return horizons (1d, 3d, 5d).
2. Topic-Specific Price Movement Drivers:
   - Dissects which news topics and financial categories (earnings, mergers, FDA, price targets)
     statistically drive price movements.
3. Publisher Market Influence & Alpha:
   - Evaluates individual publisher sentiment correlation with price reactions to identify
     which news outlets carry genuine market-moving impact.
4. Market Session Predictive Power:
   - Compares pre-market, regular hours, and post-market sentiment predictive power.
5. Feature Selection Screening:
   - Generates a ranked scorecard of top predictive features for Step 4 ML modeling.

Author: Nova Financial Solutions / Quantitative Research
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMetric:
    """Container for detailed statistical correlation metrics."""
    feature_name: str
    target_name: str
    pearson_r: float
    pearson_pvalue: float
    spearman_rho: float
    spearman_pvalue: float
    ci_lower_95: float
    ci_upper_95: float
    sample_size: int
    is_significant: bool  # p < 0.05


class CorrelationAnalyzer:
    """
    Advanced statistical correlation engine connecting financial news features
    with equity price dynamics and forward returns.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame] = None,
        ticker: str = "AAPL",
        output_dir: Union[str, Path] = "output",
    ) -> None:
        """
        Initialize CorrelationAnalyzer.

        Args:
            features_df: Engineered features DataFrame (from Step 2, e.g. engineered_features_AAPL.csv).
            news_df: Optional raw/enriched news DataFrame (for publisher-level granular correlation).
            ticker: Stock symbol (default: 'AAPL').
            output_dir: Directory where correlation tables and figures will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)

        self.ticker = ticker.upper()
        self.features_df = features_df.copy()
        self.news_df = news_df.copy() if news_df is not None else None

        self._ensure_clean_dates()

        # Cache for analysis results
        self.lag_correlations: Optional[pd.DataFrame] = None
        self.topic_correlations: Optional[pd.DataFrame] = None
        self.publisher_correlations: Optional[pd.DataFrame] = None
        self.session_correlations: Optional[pd.DataFrame] = None
        self.top_features_scorecard: Optional[pd.DataFrame] = None

    def _ensure_clean_dates(self) -> None:
        """Standardize date indexing and clean non-numeric artifacts."""
        date_col = "Date" if "Date" in self.features_df.columns else ("effective_date" if "effective_date" in self.features_df.columns else None)
        if date_col:
            self.features_df[date_col] = pd.to_datetime(self.features_df[date_col], errors="coerce")
            self.features_df = self.features_df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    @staticmethod
    def calculate_stat_correlation(
        x: pd.Series,
        y: pd.Series,
        feature_name: str = "feature",
        target_name: str = "target",
        confidence: float = 0.95
    ) -> CorrelationMetric:
        """
        Calculate Pearson and Spearman correlations with p-values and Fisher z-transform 95% CIs.
        """
        # Drop mutual NaNs and infinities
        valid_mask = ~(x.isna() | y.isna() | np.isinf(x) | np.isinf(y))
        x_clean = x[valid_mask].astype(float)
        y_clean = y[valid_mask].astype(float)
        n = len(x_clean)

        if n < 5 or x_clean.std() == 0 or y_clean.std() == 0:
            return CorrelationMetric(
                feature_name=feature_name,
                target_name=target_name,
                pearson_r=0.0,
                pearson_pvalue=1.0,
                spearman_rho=0.0,
                spearman_pvalue=1.0,
                ci_lower_95=0.0,
                ci_upper_95=0.0,
                sample_size=n,
                is_significant=False,
            )

        # Pearson
        r, p_pearson = stats.pearsonr(x_clean, y_clean)

        # Spearman
        rho, p_spearman = stats.spearmanr(x_clean, y_clean)

        # Fisher z-transformation for 95% Confidence Interval
        try:
            r_clamped = np.clip(r, -0.9999, 0.9999)
            z = np.arctanh(r_clamped)
            se = 1.0 / np.sqrt(n - 3)
            z_crit = stats.norm.ppf((1.0 + confidence) / 2.0)
            ci_lower = np.tanh(z - z_crit * se)
            ci_upper = np.tanh(z + z_crit * se)
        except Exception:
            ci_lower, ci_upper = r, r

        return CorrelationMetric(
            feature_name=feature_name,
            target_name=target_name,
            pearson_r=round(float(r), 4),
            pearson_pvalue=round(float(p_pearson), 6),
            spearman_rho=round(float(rho), 4),
            spearman_pvalue=round(float(p_spearman), 6),
            ci_lower_95=round(float(ci_lower), 4),
            ci_upper_95=round(float(ci_upper), 4),
            sample_size=int(n),
            is_significant=bool(p_pearson < 0.05),
        )

    # =========================================================================
    # 1. Multi-Lag & Multi-Horizon Correlation Analysis
    # =========================================================================
    def analyze_lagged_correlations(
        self,
        sentiment_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Analyze correlation across different sentiment lag structures (T, T-1, T-2, T-3, T-5)
        and forward return horizons (1d, 3d, 5d).

        Returns:
            DataFrame containing correlation stats for each lag and target combination.
        """
        if sentiment_cols is None:
            sentiment_cols = [
                "sentiment_mean",
                "sentiment_lag_1",
                "sentiment_lag_2",
                "sentiment_lag_3",
                "sentiment_lag_5",
                "sentiment_sma_3",
                "sentiment_sma_7",
                "sentiment_publisher_weighted",
                "sentiment_publisher_adjusted",
                "sentiment_velocity_1d",
            ]
        if target_cols is None:
            target_cols = [
                "daily_return",
                "target_next_day_return",
                "target_3d_return",
                "target_5d_return",
            ]

        # Filter available columns
        avail_sent = [c for c in sentiment_cols if c in self.features_df.columns]
        avail_target = [c for c in target_cols if c in self.features_df.columns]

        print(f"Analyzing multi-lag correlations ({len(avail_sent)} sentiment metrics vs {len(avail_target)} return horizons)...")

        records = []
        for sent_col in avail_sent:
            for targ_col in avail_target:
                metric = self.calculate_stat_correlation(
                    self.features_df[sent_col],
                    self.features_df[targ_col],
                    feature_name=sent_col,
                    target_name=targ_col,
                )
                records.append(asdict(metric))

        df_lags = pd.DataFrame(records).sort_values(by="pearson_r", ascending=False)
        self.lag_correlations = df_lags

        if save:
            out_file = self.output_dir / "data" / f"lag_correlations_{self.ticker}.csv"
            df_lags.to_csv(out_file, index=False)
            print(f"✓ Lag correlations saved to: {out_file}")
            self.plot_lagged_correlations(save=True)

        return df_lags

    def plot_lagged_correlations(self, save: bool = True) -> None:
        """Plot correlation comparison across sentiment lags (T through T-5)."""
        if self.lag_correlations is None or self.lag_correlations.empty:
            return

        # Filter for target_next_day_return
        target_name = "target_next_day_return"
        subset = self.lag_correlations[self.lag_correlations["target_name"] == target_name]

        if subset.empty:
            target_name = self.lag_correlations["target_name"].iloc[0]
            subset = self.lag_correlations[self.lag_correlations["target_name"] == target_name]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of Pearson correlations
        sorted_subset = subset.sort_values("pearson_r", ascending=True)
        colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in sorted_subset["pearson_r"]]

        ax1.barh(sorted_subset["feature_name"], sorted_subset["pearson_r"], color=colors, alpha=0.85)
        ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax1.set_title(f"Sentiment Lag vs {target_name} (Pearson r)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Pearson Correlation (r)", fontsize=11)
        ax1.grid(True, alpha=0.3, axis="x")

        # Error bar plot with 95% Confidence Intervals
        y_pos = np.arange(len(sorted_subset))
        ax2.errorbar(
            sorted_subset["pearson_r"],
            y_pos,
            xerr=[sorted_subset["pearson_r"] - sorted_subset["ci_lower_95"],
                  sorted_subset["ci_upper_95"] - sorted_subset["pearson_r"]],
            fmt="o",
            color="#2980b9",
            ecolor="#e67e22",
            elinewidth=2,
            capsize=4
        )
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_subset["feature_name"])
        ax2.axvline(0, color="gray", linestyle="--")
        ax2.set_title(f"95% Confidence Intervals ({target_name})", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Pearson r with 95% CI", fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            fig_path = self.output_dir / "figures" / f"lag_correlation_comparison_{self.ticker}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"✓ Lag correlation figure saved to: {fig_path}")
        plt.close()

    # =========================================================================
    # 2. Topic-Specific Price Movement Drivers
    # =========================================================================
    def analyze_topic_price_drivers(
        self,
        target_col: str = "target_next_day_return",
        save: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate which topics and financial categories (earnings, mergers, FDA, etc.)
        statistically drive price movements.

        Returns:
            Ranked DataFrame of topic features by correlation strength.
        """
        if target_col not in self.features_df.columns:
            target_col = "daily_return" if "daily_return" in self.features_df.columns else self.features_df.columns[1]

        topic_cols = [c for c in self.features_df.columns if c.startswith("sentiment_topic_") or c.startswith("sentiment_") and c in ["sentiment_earnings", "sentiment_mergers", "sentiment_fda", "sentiment_price_target", "sentiment_product_launch"]]

        print(f"Analyzing topic-specific price drivers against {target_col}...")

        records = []
        for col in topic_cols:
            metric = self.calculate_stat_correlation(
                self.features_df[col],
                self.features_df[target_col],
                feature_name=col,
                target_name=target_col
            )
            records.append(asdict(metric))

        df_topics = pd.DataFrame(records).sort_values(by="pearson_r", ascending=False)
        self.topic_correlations = df_topics

        if save:
            out_file = self.output_dir / "data" / f"topic_price_drivers_{self.ticker}.csv"
            df_topics.to_csv(out_file, index=False)
            print(f"✓ Topic price drivers saved to: {out_file}")
            self.plot_topic_correlations(target_col, save=True)

        return df_topics

    def plot_topic_correlations(self, target_col: str, save: bool = True) -> None:
        """Plot horizontal bar chart of topic sentiment correlation with price returns."""
        if self.topic_correlations is None or self.topic_correlations.empty:
            return

        sorted_df = self.topic_correlations.sort_values("pearson_r", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#27ae60" if r > 0 else "#c0392b" for r in sorted_df["pearson_r"]]

        bars = ax.barh(sorted_df["feature_name"], sorted_df["pearson_r"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Topic Sentiment Impact on {target_col} ({self.ticker})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Pearson Correlation (r)", fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")

        # Annotate significant bars
        for bar, p_val in zip(bars, sorted_df["pearson_pvalue"]):
            if p_val < 0.05:
                width = bar.get_width()
                offset = 0.005 if width >= 0 else -0.015
                ax.text(width + offset, bar.get_y() + bar.get_height() / 2, "★ p<0.05", va="center", color="#d35400", fontweight="bold", fontsize=9)

        plt.tight_layout()
        if save:
            fig_path = self.output_dir / "figures" / f"topic_price_correlations_{self.ticker}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"✓ Topic correlation figure saved to: {fig_path}")
        plt.close()

    # =========================================================================
    # 3. Publisher Market Influence & Alpha
    # =========================================================================
    def analyze_publisher_market_influence(
        self,
        min_articles: int = 5,
        target_col: str = "target_next_day_return",
        save: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate individual publisher market influence:
        Measures the correlation between each publisher's sentiment and the stock's forward return.

        Returns:
            DataFrame of publishers ranked by predictive correlation and statistical significance.
        """
        if self.news_df is None:
            print("Notice: No raw news DataFrame provided for granular publisher analysis. Skipping.")
            return pd.DataFrame()

        print(f"Analyzing individual publisher market influence against {target_col}...")

        # Prepare news dataset with effective date
        news = self.news_df.copy()
        if "effective_date" not in news.columns:
            news["date"] = pd.to_datetime(news["date"], errors="coerce", utc=True)
            news["effective_date"] = pd.to_datetime(news["date"].dt.date)

        news["effective_date"] = pd.to_datetime(news["effective_date"]).dt.tz_localize(None)

        # Merge with price returns
        date_col = "Date" if "Date" in self.features_df.columns else "effective_date"
        price_lookup = self.features_df.set_index(date_col)[target_col].to_dict()

        news["target_return"] = news["effective_date"].map(price_lookup)
        news_valid = news.dropna(subset=["sentiment", "target_return"])

        # Group by publisher
        records = []
        for pub, group in news_valid.groupby("publisher"):
            count = len(group)
            if count < min_articles:
                continue

            metric = self.calculate_stat_correlation(
                group["sentiment"],
                group["target_return"],
                feature_name=str(pub),
                target_name=target_col
            )
            rec = asdict(metric)
            rec["publisher"] = pub
            rec["article_count"] = count
            records.append(rec)

        df_pub = pd.DataFrame(records)
        if not df_pub.empty:
            df_pub = df_pub.sort_values(by="pearson_r", ascending=False)
            self.publisher_correlations = df_pub

            if save:
                out_file = self.output_dir / "data" / f"publisher_market_influence_{self.ticker}.csv"
                df_pub.to_csv(out_file, index=False)
                print(f"✓ Publisher market influence saved to: {out_file}")
                self.plot_publisher_influence(target_col, save=True)

        return df_pub

    def plot_publisher_influence(self, target_col: str, top_n: int = 12, save: bool = True) -> None:
        """Plot publisher correlation ranking."""
        if self.publisher_correlations is None or self.publisher_correlations.empty:
            return

        top_df = self.publisher_correlations.head(top_n).sort_values("pearson_r", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#16a085" if r > 0 else "#c0392b" for r in top_df["pearson_r"]]

        ax.barh(top_df["publisher"], top_df["pearson_r"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Publisher Sentiment Correlation with {target_col} ({self.ticker})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Pearson Correlation (r)", fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()

        if save:
            fig_path = self.output_dir / "figures" / f"publisher_influence_comparison_{self.ticker}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"✓ Publisher influence figure saved to: {fig_path}")
        plt.close()

    # =========================================================================
    # 4. Market Session Predictive Power (Pre-Market vs Regular vs Post)
    # =========================================================================
    def analyze_session_predictive_power(
        self,
        target_col: str = "target_next_day_return",
        save: bool = True
    ) -> pd.DataFrame:
        """
        Compare predictive power of Pre-Market, Regular-Hours, and Post-Market sentiment.

        Returns:
            DataFrame comparing session correlations.
        """
        session_cols = [c for c in self.features_df.columns if c.startswith("sentiment_") and c in ["sentiment_pre_market", "sentiment_regular_hours", "sentiment_post_market"]]

        if not session_cols:
            return pd.DataFrame()

        print(f"Analyzing market session predictive power against {target_col}...")

        records = []
        for col in session_cols:
            metric = self.calculate_stat_correlation(
                self.features_df[col],
                self.features_df[target_col],
                feature_name=col,
                target_name=target_col
            )
            records.append(asdict(metric))

        df_session = pd.DataFrame(records).sort_values(by="pearson_r", ascending=False)
        self.session_correlations = df_session

        if save:
            out_file = self.output_dir / "data" / f"session_predictive_power_{self.ticker}.csv"
            df_session.to_csv(out_file, index=False)
            print(f"✓ Session predictive power saved to: {out_file}")
            self.plot_session_predictive_power(target_col, save=True)

        return df_session

    def plot_session_predictive_power(self, target_col: str, save: bool = True) -> None:
        """Plot comparison of session sentiment correlations."""
        if self.session_correlations is None or self.session_correlations.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        df = self.session_correlations

        sns.barplot(
            data=df,
            x="feature_name",
            y="pearson_r",
            palette="Blues_r",
            ax=ax,
            hue="feature_name",
            legend=False
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Predictive Power by Market Session ({self.ticker})", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Pearson r with {target_col}", fontsize=11)
        ax.set_xlabel("Session Sentiment Feature", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        if save:
            fig_path = self.output_dir / "figures" / f"session_predictive_power_{self.ticker}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"✓ Session predictive power figure saved to: {fig_path}")
        plt.close()

    # =========================================================================
    # 5. Top Feature Scorecard & Summary
    # =========================================================================
    def build_top_features_scorecard(
        self,
        target_col: str = "target_next_day_return",
        top_n: int = 25,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Rank all numeric engineered features by absolute correlation strength
        with the prediction target to prepare for Step 4 (Model Feature Selection).

        Returns:
            Ranked DataFrame of top predictive features.
        """
        if target_col not in self.features_df.columns:
            target_col = "daily_return"

        # Candidate numeric features (excluding target columns and IDs)
        exclude_cols = {
            "Date", "effective_date", "date", "stock", "target_next_day_return",
            "target_next_day_direction", "target_next_day_class", "target_3d_return", "target_5d_return"
        }
        numeric_cols = [c for c in self.features_df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        print(f"Screening {len(numeric_cols)} features for top correlation with {target_col}...")

        records = []
        for col in numeric_cols:
            metric = self.calculate_stat_correlation(
                self.features_df[col],
                self.features_df[target_col],
                feature_name=col,
                target_name=target_col
            )
            rec = asdict(metric)
            rec["abs_pearson_r"] = abs(rec["pearson_r"])
            rec["abs_spearman_rho"] = abs(rec["spearman_rho"])
            records.append(rec)

        scorecard = pd.DataFrame(records).sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)
        self.top_features_scorecard = scorecard

        if save:
            out_file = self.output_dir / "data" / f"top_predictive_features_{self.ticker}.csv"
            scorecard.head(top_n).to_csv(out_file, index=False)
            print(f"✓ Top {top_n} predictive features saved to: {out_file}")
            self._save_correlation_summary(scorecard, target_col)

        return scorecard

    def _save_correlation_summary(self, scorecard: pd.DataFrame, target_col: str) -> Path:
        """Generate comprehensive JSON summary report for Step 3."""
        summary_path = self.output_dir / "data" / f"correlation_summary_{self.ticker}.json"

        sig_features = scorecard[scorecard["is_significant"]]
        top_10 = scorecard.head(10)[["feature_name", "pearson_r", "pearson_pvalue", "spearman_rho", "is_significant"]].to_dict(orient="records")

        summary = {
            "ticker": self.ticker,
            "target_variable": target_col,
            "total_features_evaluated": len(scorecard),
            "statistically_significant_count": len(sig_features),
            "top_10_features": top_10,
            "best_lag_feature": scorecard[scorecard["feature_name"].str.contains("lag|velocity|sma")].head(1)["feature_name"].iloc[0] if not scorecard.empty else None,
            "best_topic_feature": scorecard[scorecard["feature_name"].str.contains("topic|cat_")].head(1)["feature_name"].iloc[0] if not scorecard.empty else None,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Correlation summary report saved to: {summary_path}")
        return summary_path

    # =========================================================================
    # 6. Master Step 3 Pipeline Execution
    # =========================================================================
    def run_full_correlation_analysis(
        self,
        target_col: str = "target_next_day_return",
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the full Step 3 Correlation Analysis:
        1. Multi-lag & horizon correlation analysis
        2. Topic-specific price movement drivers
        3. Publisher market influence & alpha
        4. Market session predictive power
        5. Top predictive feature scorecard

        Returns:
            Dictionary with all analyzed tables and scorecards.
        """
        print("=" * 80)
        print(f"STARTING STEP 3: ADVANCED CORRELATION ANALYSIS ({self.ticker})")
        print("=" * 80)

        # 1. Multi-lag analysis
        lags = self.analyze_lagged_correlations(save=save)

        # 2. Topic drivers
        topics = self.analyze_topic_price_drivers(target_col=target_col, save=save)

        # 3. Publisher influence
        publishers = self.analyze_publisher_market_influence(target_col=target_col, save=save)

        # 4. Session predictive power
        sessions = self.analyze_session_predictive_power(target_col=target_col, save=save)

        # 5. Top feature scorecard
        scorecard = self.build_top_features_scorecard(target_col=target_col, save=save)

        print("\n" + "=" * 80)
        print("STEP 3: CORRELATION ANALYSIS COMPLETED SUCCESSFULLY ✓")
        print("=" * 80)

        return {
            "lags": lags,
            "topics": topics,
            "publishers": publishers,
            "sessions": sessions,
            "scorecard": scorecard,
        }

