"""
Sentiment Analysis Preparation Module
Step 1 of the predictive modeling pipeline for financial news and stock prices.

This module prepares rich sentiment signals from financial news headlines:
1. Topic-Based Sentiment Features:
   - Evaluates sentiment across extracted LDA / BERTopic clusters and financial categories
     (earnings, mergers, FDA approvals, price targets, product launches).
   - Generates daily topic-level sentiment aggregations and pivoted feature matrices.

2. Publisher-Specific Sentiment Patterns:
   - Analyzes sentiment bias, optimism/pessimism skew, and variance across publishers.
   - Computes publisher-adjusted sentiment (demeaned against publisher bias) to neutralize
     editorial bias.

3. News Spike Sentiment Dynamics:
   - Compares sentiment distribution on high-volume news spike days vs normal baseline days.
   - Analyzes sentiment trajectory across multi-day event windows (T-2, T-1, T0, T+1, T+2).

4. Time-Based Sentiment Features:
   - Partitions news timestamps into financial market sessions:
     * Pre-Market (04:00 - 09:29 ET)
     * Regular Trading Hours (09:30 - 16:00 ET)
     * Post-Market / After-Hours (16:01 - 20:00 ET)
     * Overnight (20:01 - 03:59 ET)
   - Computes intraday hourly and day-of-week sentiment profiles.
   - Attributes after-hours sentiment to effective trading dates.

Author: Nova Financial Solutions / Quantitative Research
Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Sentiment engines
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis preparation engine for financial news headlines.
    """

    FINANCIAL_TOPIC_KEYWORDS: Dict[str, List[str]] = {
        "earnings": ["earnings", "revenue", "profit", "loss", "quarter", "eps", "guidance", "sales", "beat", "miss"],
        "mergers_acquisitions": ["merger", "acquisition", "deal", "buyout", "takeover", "merge", "acquired"],
        "fda_regulatory": ["fda", "approval", "drug", "clinical", "trial", "regulatory", "sec", "investigation"],
        "price_target": ["target", "price", "upgrade", "downgrade", "rating", "analyst", "outperform", "underperform"],
        "product_launch": ["launch", "product", "release", "announce", "unveil", "introduced"],
        "market_movement": ["surge", "plunge", "rally", "crash", "highs", "lows", "volatility", "gain", "drop"],
    }

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path] = "output",
        news_timezone: str = "America/New_York",
        market_timezone: str = "America/New_York",
    ) -> None:
        """
        Initialize the SentimentAnalyzer.

        Args:
            df: DataFrame containing at least 'headline', 'date', and optionally 'publisher', 'stock', 'lda_topic'.
            output_dir: Directory where processed CSVs and figures will be saved.
            news_timezone: Source timezone of the news timestamp.
            market_timezone: Target timezone of the equity market.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)

        self.news_timezone = news_timezone
        self.market_timezone = market_timezone

        # Prepare working copy of data
        self.df = df.copy()
        self._ensure_datetime_and_timezone()

        # VADER analyzer instance
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Results caches
        self.publisher_patterns: Optional[pd.DataFrame] = None
        self.spike_analysis: Optional[Dict[str, Any]] = None
        self.time_based_features: Optional[pd.DataFrame] = None
        self.topic_daily_sentiment: Optional[pd.DataFrame] = None

    def _ensure_datetime_and_timezone(self) -> None:
        """Standardize date parsing, timezone localization, and trading date attribution."""
        if "date" not in self.df.columns:
            raise KeyError("Input DataFrame must contain a 'date' column.")

        # Convert to datetime
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")

        # Timezone localization/conversion
        if self.df["date"].dt.tz is None:
            self.df["date"] = self.df["date"].dt.tz_localize(self.news_timezone)
        else:
            self.df["date"] = self.df["date"].dt.tz_convert(self.market_timezone)

        # Date components
        self.df["date_market_tz"] = self.df["date"]
        self.df["hour_market"] = self.df["date"].dt.hour
        self.df["minute_market"] = self.df["date"].dt.minute
        self.df["day_of_week"] = self.df["date"].dt.day_name()
        self.df["date_only"] = pd.to_datetime(self.df["date"].dt.date)

        # Post-market cutoff: News published at or after 16:00 ET (4 PM) impacts the NEXT trading day
        self.df["effective_date"] = np.where(
            self.df["hour_market"] >= 16,
            self.df["date_only"] + pd.Timedelta(days=1),
            self.df["date_only"]
        )
        self.df["effective_date"] = pd.to_datetime(self.df["effective_date"])

        # Market session segmentation
        self.df["market_session"] = self.df.apply(self._classify_market_session, axis=1)

    @staticmethod
    def _classify_market_session(row: pd.Series) -> str:
        """
        Classify news timestamp into trading sessions:
        - Pre-Market: 04:00 - 09:29 ET
        - Regular Market: 09:30 - 16:00 ET
        - Post-Market / After-Hours: 16:01 - 20:00 ET
        - Overnight: 20:01 - 03:59 ET
        """
        hour = row["hour_market"]
        minute = row.get("minute_market", 0)

        # Pre-Market (4:00 AM - 9:29 AM)
        if 4 <= hour < 9 or (hour == 9 and minute < 30):
            return "Pre-Market"
        # Regular Market (9:30 AM - 4:00 PM)
        elif (hour == 9 and minute >= 30) or (10 <= hour < 16) or (hour == 16 and minute == 0):
            return "Regular-Hours"
        # Post-Market (4:01 PM - 8:00 PM)
        elif (hour == 16 and minute > 0) or (17 <= hour < 20):
            return "Post-Market"
        # Overnight (8:01 PM - 3:59 AM)
        else:
            return "Overnight"

    # =========================================================================
    # 1. Headline Sentiment Scoring
    # =========================================================================
    def score_sentiment(
        self,
        headline_col: str = "headline",
        include_textblob: bool = True,
        include_vader: bool = True,
    ) -> pd.DataFrame:
        """
        Compute sentiment scores for each headline.

        Args:
            headline_col: Column containing headline text.
            include_textblob: Whether to calculate TextBlob polarity and subjectivity.
            include_vader: Whether to calculate VADER compound, pos, neg, neu scores.

        Returns:
            DataFrame with enriched sentiment columns.
        """
        print("Scoring headline sentiments...")
        headlines = self.df[headline_col].fillna("").astype(str)

        if include_vader:
            vader_scores = headlines.apply(lambda text: self.vader_analyzer.polarity_scores(text))
            self.df["sentiment_vader_compound"] = [score["compound"] for score in vader_scores]
            self.df["sentiment_vader_pos"] = [score["pos"] for score in vader_scores]
            self.df["sentiment_vader_neg"] = [score["neg"] for score in vader_scores]
            self.df["sentiment_vader_neu"] = [score["neu"] for score in vader_scores]
            # Primary sentiment column
            self.df["sentiment"] = self.df["sentiment_vader_compound"]

        if include_textblob:
            tb_polarities = []
            tb_subjectivities = []
            for text in headlines:
                blob = TextBlob(text)
                tb_polarities.append(blob.sentiment.polarity)
                tb_subjectivities.append(blob.sentiment.subjectivity)

            self.df["sentiment_textblob_polarity"] = tb_polarities
            self.df["sentiment_textblob_subjectivity"] = tb_subjectivities

            if not include_vader:
                self.df["sentiment"] = self.df["sentiment_textblob_polarity"]

        # Classification into sentiment classes: Positive (>0.05), Negative (<-0.05), Neutral
        self.df["sentiment_class"] = pd.cut(
            self.df["sentiment"],
            bins=[-np.inf, -0.05, 0.05, np.inf],
            labels=["Negative", "Neutral", "Positive"]
        )

        print(f"✓ Scored {len(self.df):,} headlines. Positive: {(self.df['sentiment_class'] == 'Positive').sum():,}, "
              f"Negative: {(self.df['sentiment_class'] == 'Negative').sum():,}, "
              f"Neutral: {(self.df['sentiment_class'] == 'Neutral').sum():,}")
        return self.df

    # =========================================================================
    # 2. Topic-Based Sentiment Features
    # =========================================================================
    def map_financial_topic_categories(self, headline_col: str = "headline") -> pd.DataFrame:
        """
        Classify headlines into financial domain categories (earnings, mergers, etc.) using keywords.

        Returns:
            DataFrame with 'financial_category' column.
        """
        def match_category(headline: str) -> str:
            lower = str(headline).lower()
            for cat, keywords in self.FINANCIAL_TOPIC_KEYWORDS.items():
                pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
                if re.search(pattern, lower):
                    return cat
            return "general_market"

        self.df["financial_category"] = self.df[headline_col].apply(match_category)
        return self.df

    def create_topic_sentiment_features(
        self,
        topic_col: Optional[str] = None,
        date_col: str = "effective_date",
        save: bool = True
    ) -> pd.DataFrame:
        """
        Create topic-based sentiment features aggregated per date.

        Args:
            topic_col: Topic column name ('lda_topic', 'bertopic_label', or 'financial_category').
                       If None, defaults to 'lda_topic' if available, else 'financial_category'.
            date_col: Date column for temporal grouping ('effective_date' or 'date_only').
            save: Whether to save results to output directory.

        Returns:
            Pivoted DataFrame where each row is a date and columns represent topic-level sentiments.
        """
        if "sentiment" not in self.df.columns:
            self.score_sentiment()

        # Determine topic column
        if topic_col is None:
            if "lda_topic" in self.df.columns:
                topic_col = "lda_topic"
            else:
                self.map_financial_topic_categories()
                topic_col = "financial_category"

        if topic_col == "financial_category" and "financial_category" not in self.df.columns:
            self.map_financial_topic_categories()

        print(f"Creating topic-based sentiment features using '{topic_col}' grouped by '{date_col}'...")

        # Aggregate sentiment by date and topic
        topic_daily = (
            self.df.groupby([date_col, topic_col])["sentiment"]
            .agg(mean_sentiment="mean", article_count="count", std_sentiment="std")
            .reset_index()
        )
        self.topic_daily_sentiment = topic_daily

        # Pivot to create feature columns per topic (e.g., sentiment_topic_0, count_topic_0)
        pivot_sentiment = topic_daily.pivot(
            index=date_col,
            columns=topic_col,
            values="mean_sentiment"
        ).add_prefix("sentiment_topic_")

        pivot_count = topic_daily.pivot(
            index=date_col,
            columns=topic_col,
            values="article_count"
        ).fillna(0).add_prefix("count_topic_")

        topic_features = pd.concat([pivot_sentiment, pivot_count], axis=1).reset_index()

        if save:
            out_file = self.output_dir / "data" / "topic_daily_sentiment.csv"
            topic_features.to_csv(out_file, index=False)
            print(f"✓ Topic-based sentiment features saved to {out_file}")

        return topic_features

    # =========================================================================
    # 3. Publisher-Specific Sentiment Patterns
    # =========================================================================
    def analyze_publisher_sentiment(
        self,
        min_articles: int = 10,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Analyze sentiment distribution, optimism/pessimism bias, and variance across publishers.

        Args:
            min_articles: Minimum number of articles required to profile a publisher.
            save: Whether to save results to CSV and plot figures.

        Returns:
            DataFrame with publisher sentiment statistics and bias metrics.
        """
        if "sentiment" not in self.df.columns:
            self.score_sentiment()

        if "publisher" not in self.df.columns:
            raise KeyError("Publisher column missing from dataset.")

        print("Analyzing publisher-specific sentiment patterns...")
        global_mean = self.df["sentiment"].mean()
        global_std = self.df["sentiment"].std()

        grouped = self.df.groupby("publisher")

        stats_records = []
        for pub, group in grouped:
            count = len(group)
            if count < min_articles:
                continue

            sentiments = group["sentiment"]
            mean_sent = sentiments.mean()
            median_sent = sentiments.median()
            std_sent = sentiments.std() if count > 1 else 0.0
            pos_ratio = (sentiments > 0.05).sum() / count
            neg_ratio = (sentiments < -0.05).sum() / count
            neu_ratio = 1.0 - (pos_ratio + neg_ratio)
            # Bias = difference from overall dataset baseline
            bias = mean_sent - global_mean

            stats_records.append({
                "publisher": pub,
                "article_count": count,
                "mean_sentiment": round(mean_sent, 4),
                "median_sentiment": round(median_sent, 4),
                "std_sentiment": round(std_sent, 4),
                "positivity_ratio": round(pos_ratio, 4),
                "negativity_ratio": round(neg_ratio, 4),
                "neutral_ratio": round(neu_ratio, 4),
                "sentiment_bias": round(bias, 4),
                "bullish_lean": "Bullish" if bias > 0.03 else ("Bearish" if bias < -0.03 else "Neutral"),
            })

        pub_df = pd.DataFrame(stats_records).sort_values("article_count", ascending=False)
        self.publisher_patterns = pub_df

        # Add publisher-adjusted sentiment to main DataFrame: sentiment - publisher_mean_bias
        bias_map = pub_df.set_index("publisher")["sentiment_bias"].to_dict()
        self.df["publisher_sentiment_bias"] = self.df["publisher"].map(bias_map).fillna(0.0)
        self.df["sentiment_publisher_adjusted"] = self.df["sentiment"] - self.df["publisher_sentiment_bias"]

        if save:
            out_file = self.output_dir / "data" / "publisher_sentiment_patterns.csv"
            pub_df.to_csv(out_file, index=False)
            print(f"✓ Publisher sentiment patterns saved to {out_file}")
            self.plot_publisher_sentiment(save=True)

        return pub_df

    def plot_publisher_sentiment(self, top_n: int = 15, save: bool = True) -> None:
        """Plot sentiment bias for top publishers."""
        if self.publisher_patterns is None or self.publisher_patterns.empty:
            return

        top_pubs = self.publisher_patterns.head(top_n).sort_values("sentiment_bias", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2ecc71" if b > 0 else "#e74c3c" for b in top_pubs["sentiment_bias"]]

        ax.barh(top_pubs["publisher"], top_pubs["sentiment_bias"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Top {top_n} Publishers: Sentiment Bias vs Market Average", fontsize=14, fontweight="bold")
        ax.set_xlabel("Sentiment Bias (Publisher Mean - Global Mean)", fontsize=11)
        ax.set_ylabel("Publisher", fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()

        if save:
            fig_path = self.output_dir / "figures" / "publisher_sentiment_bias.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"✓ Publisher bias plot saved to {fig_path}")
        plt.close()

    # =========================================================================
    # 4. Sentiment Trends Around News Spikes
    # =========================================================================
    def analyze_spike_sentiment(
        self,
        spikes_df: Optional[pd.DataFrame] = None,
        threshold_std: float = 2.0,
        window_days: int = 2,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze sentiment characteristics and multi-day dynamics during news activity spikes.

        Args:
            spikes_df: Optional DataFrame with 'date' column indicating spike dates.
                       If None, detects spikes automatically based on volume threshold.
            threshold_std: Standard deviation threshold above mean for detecting volume spikes.
            window_days: Days before and after spike to include in the event trajectory window.
            save: Whether to save figures and CSV tables.

        Returns:
            Dictionary with statistical comparison and window trajectories.
        """
        if "sentiment" not in self.df.columns:
            self.score_sentiment()

        print("Analyzing sentiment dynamics around news spikes...")

        # Step A: Identify spike dates
        daily_counts = self.df.groupby("date_only")["sentiment"].agg(
            count="count", mean_sentiment="mean", std_sentiment="std"
        ).reset_index()

        if spikes_df is not None and "date" in spikes_df.columns:
            spike_dates = set(pd.to_datetime(spikes_df["date"]).dt.date)
        else:
            mean_vol = daily_counts["count"].mean()
            std_vol = daily_counts["count"].std()
            spike_thresh = mean_vol + (threshold_std * std_vol)
            spike_dates = set(daily_counts[daily_counts["count"] >= spike_thresh]["date_only"].dt.date)

        daily_counts["is_spike_day"] = daily_counts["date_only"].dt.date.isin(spike_dates)

        # Step B: Compare spike days vs normal days
        spike_sentiments = self.df[self.df["date_only"].dt.date.isin(spike_dates)]["sentiment"].dropna()
        normal_sentiments = self.df[~self.df["date_only"].dt.date.isin(spike_dates)]["sentiment"].dropna()

        t_stat, p_val = stats.ttest_ind(spike_sentiments, normal_sentiments, equal_var=False)

        comparison_stats = {
            "spike_days_count": len(spike_dates),
            "spike_articles_count": len(spike_sentiments),
            "normal_articles_count": len(normal_sentiments),
            "spike_mean_sentiment": round(float(spike_sentiments.mean()), 4),
            "normal_mean_sentiment": round(float(normal_sentiments.mean()), 4),
            "spike_std_sentiment": round(float(spike_sentiments.std()), 4),
            "normal_std_sentiment": round(float(normal_sentiments.std()), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "statistically_significant": bool(p_val < 0.05),
        }

        # Step C: Event study window trajectory (-window to +window)
        trajectory_records = []
        daily_lookup = daily_counts.set_index(daily_counts["date_only"].dt.date)["mean_sentiment"].to_dict()

        for spike_date in spike_dates:
            for offset in range(-window_days, window_days + 1):
                target_date = spike_date + pd.Timedelta(days=offset)
                if target_date in daily_lookup:
                    trajectory_records.append({
                        "spike_date": spike_date,
                        "day_offset": offset,
                        "mean_sentiment": daily_lookup[target_date]
                    })

        traj_df = pd.DataFrame(trajectory_records)
        avg_trajectory = (
            traj_df.groupby("day_offset")["mean_sentiment"]
            .agg(avg_sentiment="mean", count="count")
            .reset_index()
        ) if not traj_df.empty else pd.DataFrame()

        self.spike_analysis = {
            "comparison": comparison_stats,
            "trajectory": avg_trajectory,
            "spike_dates": sorted(list(spike_dates))
        }

        if save:
            out_file = self.output_dir / "data" / "spike_sentiment_analysis.csv"
            avg_trajectory.to_csv(out_file, index=False)
            print(f"✓ Spike sentiment trajectory saved to {out_file}")
            self.plot_spike_sentiment(save=True)

        return self.spike_analysis

    def plot_spike_sentiment(self, save: bool = True) -> None:
        """Plot sentiment comparison between spike days and normal days, and event trajectory."""
        if not self.spike_analysis:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Bar chart comparing spike vs normal mean sentiment
        comp = self.spike_analysis["comparison"]
        labels = ["Normal Days", "Spike Days"]
        values = [comp["normal_mean_sentiment"], comp["spike_mean_sentiment"]]
        errors = [comp["normal_std_sentiment"] / 5, comp["spike_std_sentiment"] / 5]

        bars = ax1.bar(labels, values, yerr=errors, color=["#3498db", "#e67e22"], capsize=5, alpha=0.85)
        ax1.set_ylabel("Mean Sentiment Polarity", fontsize=11)
        ax1.set_title(f"Sentiment on Spike Days vs Normal Days\n(p-value: {comp['p_value']:.4e})", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height / 2, f"{height:.4f}", ha="center", va="center", color="white", fontweight="bold")

        # Panel 2: Event window trajectory
        traj = self.spike_analysis["trajectory"]
        if not traj.empty:
            ax2.plot(traj["day_offset"], traj["avg_sentiment"], marker="o", linewidth=2.5, color="#8e44ad")
            ax2.axvline(0, color="red", linestyle="--", alpha=0.8, label="Spike Day (T0)")
            ax2.set_xlabel("Day Offset relative to Spike", fontsize=11)
            ax2.set_ylabel("Average Sentiment", fontsize=11)
            ax2.set_title("Sentiment Trajectory Around News Spikes", fontsize=12, fontweight="bold")
            ax2.set_xticks(traj["day_offset"])
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            fig_path = self.output_dir / "figures" / "spike_sentiment_analysis.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"✓ Spike sentiment plot saved to {fig_path}")
        plt.close()

    # =========================================================================
    # 5. Time-Based Sentiment Features
    # =========================================================================
    def create_time_based_sentiment_features(self, save: bool = True) -> pd.DataFrame:
        """
        Generate time-based sentiment features across trading sessions and day-of-week patterns.

        Features created:
        - Daily average sentiment by session (Pre-Market, Regular-Hours, Post-Market, Overnight)
        - Day of week categorical and indicator features
        - Intraday sentiment progression metrics

        Returns:
            DataFrame indexed by effective_date with session-specific sentiment columns.
        """
        if "sentiment" not in self.df.columns:
            self.score_sentiment()

        print("Generating time-based and session-specific sentiment features...")

        # Step A: Daily aggregation by market session
        session_daily = (
            self.df.groupby(["effective_date", "market_session"])["sentiment"]
            .agg(mean="mean", count="count")
            .reset_index()
        )

        pivot_session_mean = session_daily.pivot(
            index="effective_date", columns="market_session", values="mean"
        ).add_prefix("sentiment_session_")

        pivot_session_count = session_daily.pivot(
            index="effective_date", columns="market_session", values="count"
        ).fillna(0).add_prefix("count_session_")

        daily_time_features = pd.concat([pivot_session_mean, pivot_session_count], axis=1).reset_index()

        # Step B: Add Day-of-Week features
        daily_time_features["day_of_week"] = daily_time_features["effective_date"].dt.day_name()
        daily_time_features["is_monday"] = (daily_time_features["day_of_week"] == "Monday").astype(int)
        daily_time_features["is_friday"] = (daily_time_features["day_of_week"] == "Friday").astype(int)

        self.time_based_features = daily_time_features

        if save:
            out_file = self.output_dir / "data" / "time_based_sentiment_features.csv"
            daily_time_features.to_csv(out_file, index=False)
            print(f"✓ Time-based sentiment features saved to {out_file}")
            self.plot_time_based_sentiment(save=True)

        return daily_time_features

    def plot_time_based_sentiment(self, save: bool = True) -> None:
        """Plot intraday session and day-of-week sentiment distributions."""
        if "market_session" not in self.df.columns or "sentiment" not in self.df.columns:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Sentiment by Market Session
        session_order = ["Pre-Market", "Regular-Hours", "Post-Market", "Overnight"]
        available_sessions = [s for s in session_order if s in self.df["market_session"].unique()]
        sns.barplot(
            data=self.df,
            x="market_session",
            y="sentiment",
            hue="market_session",
            legend=False,
            order=available_sessions,
            palette="Blues_d",
            ax=ax1,
            errorbar=None
        )
        ax1.set_title("Mean Sentiment by Market Session", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Mean Sentiment Polarity", fontsize=11)
        ax1.set_xlabel("Market Session", fontsize=11)
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: Sentiment by Day of Week
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        available_days = [d for d in day_order if d in self.df["day_of_week"].unique()]
        sns.barplot(
            data=self.df,
            x="day_of_week",
            y="sentiment",
            hue="day_of_week",
            legend=False,
            order=available_days,
            palette="Purples_d",
            ax=ax2,
            errorbar=None
        )
        ax2.set_title("Mean Sentiment by Day of Week", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Mean Sentiment Polarity", fontsize=11)
        ax2.set_xlabel("Day of Week", fontsize=11)
        ax2.tick_params(axis="x", rotation=30)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if save:
            fig_path = self.output_dir / "figures" / "time_based_sentiment_distributions.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"✓ Time-based sentiment plot saved to {fig_path}")
        plt.close()

    # =========================================================================
    # 6. Master Preparation Pipeline Runner
    # =========================================================================
    def run_full_preparation(self, save_enriched_articles: bool = True) -> Dict[str, Any]:
        """
        Execute all 4 steps of the Sentiment Analysis Preparation:
        1. Score headlines (VADER & TextBlob)
        2. Create topic-based sentiment features
        3. Analyze publisher sentiment patterns & editorial bias
        4. Analyze sentiment trends around news spikes
        5. Generate session- and day-based temporal sentiment features

        Returns:
            Dictionary containing processed feature tables and summary metrics.
        """
        print("=" * 80)
        print("STARTING SENTIMENT ANALYSIS PREPARATION PIPELINE")
        print("=" * 80)

        # 1. Score sentiment
        self.score_sentiment()

        # 2. Topic sentiment features
        topic_features = self.create_topic_sentiment_features(save=True)

        # 3. Publisher sentiment patterns
        publisher_patterns = self.analyze_publisher_sentiment(save=True)

        # 4. Spike sentiment dynamics
        spike_stats = self.analyze_spike_sentiment(save=True)

        # 5. Time-based features
        time_features = self.create_time_based_sentiment_features(save=True)

        # Save enriched full dataset
        if save_enriched_articles:
            out_file = self.output_dir / "data" / "articles_with_sentiment.csv"
            self.df.to_csv(out_file, index=False)
            print(f"✓ Enriched article dataset with sentiment saved to {out_file}")

        # Generate summary text report
        self._generate_preparation_report()

        print("\n" + "=" * 80)
        print("SENTIMENT ANALYSIS PREPARATION COMPLETED SUCCESSFULLY ✓")
        print("=" * 80)

        return {
            "df": self.df,
            "topic_features": topic_features,
            "publisher_patterns": publisher_patterns,
            "spike_stats": spike_stats,
            "time_features": time_features,
        }

    def _generate_preparation_report(self) -> Path:
        """Write summary text report of sentiment preparation findings."""
        report_path = self.output_dir / "sentiment_preparation_report.txt"

        lines = [
            "=" * 80,
            "SENTIMENT ANALYSIS PREPARATION REPORT",
            "Nova Financial Solutions - Quantitative Research",
            "=" * 80,
            "",
            "1. GLOBAL SENTIMENT OVERVIEW",
            "-" * 80,
            f"Total Articles Processed: {len(self.df):,}",
            f"Mean Sentiment Polarity: {self.df['sentiment'].mean():.4f}",
            f"Median Sentiment Polarity: {self.df['sentiment'].median():.4f}",
            f"Sentiment Standard Deviation: {self.df['sentiment'].std():.4f}",
            f"Positive Articles (> 0.05): {(self.df['sentiment_class'] == 'Positive').sum():,} ({(self.df['sentiment_class'] == 'Positive').mean() * 100:.1f}%)",
            f"Negative Articles (< -0.05): {(self.df['sentiment_class'] == 'Negative').sum():,} ({(self.df['sentiment_class'] == 'Negative').mean() * 100:.1f}%)",
            f"Neutral Articles: {(self.df['sentiment_class'] == 'Neutral').sum():,} ({(self.df['sentiment_class'] == 'Neutral').mean() * 100:.1f}%)",
            "",
            "2. PUBLISHER SENTIMENT BIAS",
            "-" * 80,
        ]

        if self.publisher_patterns is not None and not self.publisher_patterns.empty:
            lines.append("Top 5 Most Bullish Publishers (Highest Positive Bias):")
            bullish = self.publisher_patterns.sort_values("sentiment_bias", ascending=False).head(5)
            for _, r in bullish.iterrows():
                lines.append(f"  * {r['publisher']}: mean={r['mean_sentiment']:.4f}, bias={r['sentiment_bias']:+.4f} ({r['article_count']} articles)")

            lines.append("\nTop 5 Most Bearish Publishers (Lowest Negative Bias):")
            bearish = self.publisher_patterns.sort_values("sentiment_bias", ascending=True).head(5)
            for _, r in bearish.iterrows():
                lines.append(f"  * {r['publisher']}: mean={r['mean_sentiment']:.4f}, bias={r['sentiment_bias']:+.4f} ({r['article_count']} articles)")

        if self.spike_analysis:
            comp = self.spike_analysis["comparison"]
            lines.extend([
                "",
                "3. NEWS SPIKE SENTIMENT DYNAMICS",
                "-" * 80,
                f"Spike Days Analyzed: {comp['spike_days_count']}",
                f"Mean Sentiment on Normal Days: {comp['normal_mean_sentiment']:.4f}",
                f"Mean Sentiment on Spike Days:  {comp['spike_mean_sentiment']:.4f}",
                f"Difference: {comp['spike_mean_sentiment'] - comp['normal_mean_sentiment']:+.4f}",
                f"Statistical Significance (t-test): p = {comp['p_value']:.4e} (Significant: {comp['statistically_significant']})",
            ])

        if self.time_based_features is not None:
            session_cols = [c for c in self.time_based_features.columns if c.startswith("sentiment_session_")]
            lines.extend([
                "",
                "4. MARKET SESSION SENTIMENT METRICS",
                "-" * 80,
            ])
            for col in session_cols:
                sess_name = col.replace("sentiment_session_", "")
                avg_val = self.time_based_features[col].mean()
                lines.append(f"  * {sess_name}: Average Daily Sentiment = {avg_val:.4f}")

        lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ])

        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"✓ Summary report saved to {report_path}")
        return report_path
