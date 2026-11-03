"""
Feature Engineering Module
Step 2 of the predictive modeling pipeline for financial news and stock prices.

This module constructs an ML-ready dataset combining:
1. Headline length and word count aggregates (mean, max, std).
2. Topic exposure vectors from LDA/BERTopic and financial category indicators.
3. Publisher credibility and reliability weighting factors (weighted and demeaned sentiment).
4. Temporal and session features (day of week, month cyclical encodings, minutes from market open, pre-market ratio).
5. Stock-specific news volume dynamics (rolling frequency, volume surge z-score, spike flags).
6. Sentiment momentum and lag features (T-1, T-2, T-3, T-5, rolling SMAs, sentiment velocity).
7. Price alignment & technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR) with forward-looking prediction targets.

Author: Nova Financial Solutions / Quantitative Research
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering engine that transforms raw news headlines, topics,
    and stock price series into a unified tabular feature matrix.
    """

    FINANCIAL_CATEGORIES: Dict[str, List[str]] = {
        "earnings": ["earnings", "revenue", "profit", "loss", "quarter", "eps", "guidance", "sales", "beat", "miss"],
        "mergers": ["merger", "acquisition", "deal", "buyout", "takeover", "merge", "acquired"],
        "fda": ["fda", "approval", "drug", "clinical", "trial", "regulatory"],
        "price_target": ["target", "price", "upgrade", "downgrade", "rating", "analyst"],
        "product_launch": ["launch", "product", "release", "announce", "unveil"],
    }

    def __init__(
        self,
        news_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        ticker: Optional[str] = None,
        output_dir: Union[str, Path] = "output",
        market_open_time: str = "09:30",
    ) -> None:
        """
        Initialize FeatureEngineer.

        Args:
            news_df: Enriched news DataFrame (contains headline, date, publisher, sentiment, etc.)
            price_df: Optional OHLCV DataFrame with Date index or column.
            ticker: Stock symbol to filter for (e.g. 'AAPL'). If None, processes all stocks present.
            output_dir: Directory where generated feature tables and charts will be saved.
            market_open_time: Local market opening time in HH:MM format (default: '09:30' ET).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)

        self.ticker = ticker.upper() if ticker else None
        self.market_open_time = market_open_time

        # Prepare news DataFrame
        self.news = news_df.copy()
        self._prepare_news_base()

        # Prepare price DataFrame if provided
        self.price = price_df.copy() if price_df is not None else None
        if self.price is not None:
            self._prepare_price_base()

        # Engineered daily feature container
        self.daily_features: Optional[pd.DataFrame] = None
        self.master_df: Optional[pd.DataFrame] = None

    def _prepare_news_base(self) -> None:
        """Clean dates, filter by ticker if specified, and extract core components."""
        # Ensure date format with utc=True for mixed timezone offsets
        self.news["date"] = pd.to_datetime(self.news["date"], errors="coerce", utc=True)

        if "effective_date" not in self.news.columns:
            # Shift after 4 PM ET news to next calendar day
            date_et = self.news["date"].dt.tz_convert("America/New_York")
            date_only = pd.to_datetime(date_et.dt.date)
            hour = date_et.dt.hour
            self.news["effective_date"] = np.where(hour >= 16, date_only + pd.Timedelta(days=1), date_only)
        else:
            self.news["effective_date"] = pd.to_datetime(self.news["effective_date"], errors="coerce")

        if hasattr(self.news["effective_date"].dt, "tz") and self.news["effective_date"].dt.tz is not None:
            self.news["effective_date"] = self.news["effective_date"].dt.tz_localize(None)
        else:
            self.news["effective_date"] = pd.to_datetime(self.news["effective_date"])

        # Filter ticker if requested
        if self.ticker and "stock" in self.news.columns:
            before_len = len(self.news)
            filtered = self.news[self.news["stock"].str.upper() == self.ticker].copy()
            if len(filtered) > 0:
                self.news = filtered
                print(f"Filtered news for ticker {self.ticker}: {len(self.news):,} articles (from {before_len:,}).")
            else:
                print(f"Warning: No articles found matching ticker '{self.ticker}'. Using all available articles.")

        # Ensure sentiment is present
        if "sentiment" not in self.news.columns:
            if "sentiment_vader_compound" in self.news.columns:
                self.news["sentiment"] = self.news["sentiment_vader_compound"]
            else:
                # Basic TextBlob polarity as fallback
                from textblob import TextBlob
                self.news["sentiment"] = self.news["headline"].fillna("").apply(
                    lambda t: TextBlob(str(t)).sentiment.polarity
                )

        # Ensure headline length and word count
        if "headline_length" not in self.news.columns:
            self.news["headline_length"] = self.news["headline"].astype(str).str.len()
        if "headline_word_count" not in self.news.columns:
            self.news["headline_word_count"] = self.news["headline"].astype(str).str.split().str.len()

    def _prepare_price_base(self) -> None:
        """Standardize price DataFrame columns and date index."""
        df = self.price.copy()
        # Normalize column names
        df.columns = [str(c).strip().title() for c in df.columns]

        # Ensure Date column
        if "Date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df["Date"] = df.index
        elif "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        # Standardize numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.ffill().bfill()
        self.price = df

    # =========================================================================
    # 1. Headline Length and Word Count Features
    # =========================================================================
    def build_text_features(self) -> pd.DataFrame:
        """
        Aggregate daily headline length and word count statistics.

        Returns:
            DataFrame indexed by effective_date with text length metrics.
        """
        text_agg = self.news.groupby("effective_date").agg(
            headline_length_mean=("headline_length", "mean"),
            headline_length_std=("headline_length", lambda x: x.std() if len(x) > 1 else 0.0),
            headline_length_max=("headline_length", "max"),
            headline_word_count_mean=("headline_word_count", "mean"),
            headline_word_count_max=("headline_word_count", "max"),
        ).reset_index()

        text_agg["headline_length_std"] = text_agg["headline_length_std"].fillna(0.0)
        return text_agg

    # =========================================================================
    # 2. Topic Features (LDA/BERTopic & Financial Categories)
    # =========================================================================
    def build_topic_features(self, topic_col: str = "lda_topic") -> pd.DataFrame:
        """
        Construct topic representation features per day:
        - Daily topic distribution / count per LDA topic
        - Binary indicators and counts for financial categories (earnings, mergers, etc.)
        - Category-specific sentiment scores

        Returns:
            DataFrame indexed by effective_date with topic features.
        """
        grouped = self.news.groupby("effective_date")
        dates = []
        records = []

        # Check if topic column exists
        has_lda = topic_col in self.news.columns

        for dt, group in grouped:
            row: Dict[str, Any] = {"effective_date": dt}
            total_articles = len(group)

            # A. LDA Topic distribution (counts and shares)
            if has_lda:
                topic_counts = group[topic_col].value_counts()
                for t_id in range(10):  # Standard 10 topics
                    cnt = topic_counts.get(t_id, 0)
                    row[f"topic_count_{t_id}"] = cnt
                    row[f"topic_share_{t_id}"] = round(cnt / total_articles, 4)

                # Dominant topic of the day
                row["dominant_topic"] = int(group[topic_col].mode().iloc[0]) if not group[topic_col].empty else -1

            # B. Financial category keyword presence and category-specific sentiment
            headlines_text = group["headline"].astype(str).str.lower()
            for cat_name, kw_list in self.FINANCIAL_CATEGORIES.items():
                pattern = r"\b(?:" + "|".join(kw_list) + r")\b"
                mask = headlines_text.str.contains(pattern, regex=True)
                cat_count = mask.sum()
                row[f"cat_count_{cat_name}"] = int(cat_count)
                row[f"has_{cat_name}"] = int(cat_count > 0)

                # Category-specific sentiment
                if cat_count > 0:
                    row[f"sentiment_{cat_name}"] = round(float(group.loc[mask, "sentiment"].mean()), 4)
                else:
                    row[f"sentiment_{cat_name}"] = 0.0

            records.append(row)

        topic_df = pd.DataFrame(records)
        return topic_df

    # =========================================================================
    # 3. Publisher Credibility & Weighting Factors
    # =========================================================================
    def build_publisher_weighting_features(self, min_articles: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute publisher credibility weights and aggregate daily publisher-weighted
        and publisher-adjusted sentiment.

        Formula for data-driven publisher reliability weight:
            w = ln(1 + count) / (1 + std_sentiment)

        Returns:
            Tuple of (daily_publisher_features, publisher_credibility_table)
        """
        # Step A: Compute publisher profile
        pub_stats = self.news.groupby("publisher")["sentiment"].agg(
            count="count",
            mean="mean",
            std=lambda x: x.std() if len(x) > 1 else 0.1,
        ).reset_index()

        global_mean = self.news["sentiment"].mean()
        pub_stats["bias"] = pub_stats["mean"] - global_mean
        pub_stats["std"] = pub_stats["std"].fillna(0.1)

        # Credibility weight: rewards established track record and penalized high erratic variance
        pub_stats["credibility_weight"] = np.log1p(pub_stats["count"]) / (1.0 + pub_stats["std"])
        # Normalize so average weight is 1.0
        mean_w = pub_stats["credibility_weight"].mean()
        pub_stats["credibility_weight"] = (pub_stats["credibility_weight"] / mean_w).clip(0.2, 5.0)

        weight_map = pub_stats.set_index("publisher")["credibility_weight"].to_dict()
        bias_map = pub_stats.set_index("publisher")["bias"].to_dict()

        # Step B: Attach weights and adjusted sentiment to news rows
        news_copy = self.news.copy()
        news_copy["pub_weight"] = news_copy["publisher"].map(weight_map).fillna(1.0)
        news_copy["pub_bias"] = news_copy["publisher"].map(bias_map).fillna(0.0)
        news_copy["sentiment_adjusted"] = news_copy["sentiment"] - news_copy["pub_bias"]
        news_copy["weighted_sent_contrib"] = news_copy["sentiment"] * news_copy["pub_weight"]

        # Step C: Aggregate daily
        daily_pub = news_copy.groupby("effective_date").apply(
            lambda g: pd.Series({
                "sentiment_publisher_weighted": g["weighted_sent_contrib"].sum() / g["pub_weight"].sum(),
                "sentiment_publisher_adjusted": g["sentiment_adjusted"].mean(),
                "publisher_count_unique": g["publisher"].nunique(),
                # Herfindahl-Hirschman index for publisher concentration
                "publisher_concentration_hhi": ((g["publisher"].value_counts() / len(g)) ** 2).sum(),
            })
        ).reset_index()

        return daily_pub, pub_stats

    # =========================================================================
    # 4. Temporal & Session Features (Minutes from Open, Day of Week, Sessions)
    # =========================================================================
    def build_temporal_features(self) -> pd.DataFrame:
        """
        Construct temporal and intraday timing features:
        - Day of week (one-hot flags)
        - Month cyclical features (sin / cos)
        - Average time before / after market open (in minutes)
        - Market session sentiment breakdown (Pre-Market, Regular, Post-Market)
        - Ratio of pre-market news to total news

        Returns:
            DataFrame indexed by effective_date with temporal features.
        """
        # Ensure timestamp has hour and minute
        dt_series = pd.to_datetime(self.news["date"])
        news_copy = self.news.copy()

        # Minutes from market open (09:30 AM = 570 minutes from midnight)
        market_open_minutes = 9 * 60 + 30
        intraday_minutes = dt_series.dt.hour * 60 + dt_series.dt.minute
        news_copy["minutes_from_open"] = intraday_minutes - market_open_minutes

        # Classify market session if not already done
        if "market_session" not in news_copy.columns:
            hour = dt_series.dt.hour
            minute = dt_series.dt.minute
            conditions = [
                (hour < 9) | ((hour == 9) & (minute < 30)),
                ((hour == 9) & (minute >= 30)) | ((hour >= 10) & (hour < 16)),
                (hour >= 16) & (hour < 20),
            ]
            choices = ["Pre-Market", "Regular-Hours", "Post-Market"]
            news_copy["market_session"] = np.select(conditions, choices, default="Overnight")

        # Group by effective date
        grouped = news_copy.groupby("effective_date")
        records = []

        for dt, group in grouped:
            total_n = len(group)
            sess_counts = group["market_session"].value_counts()
            sess_sent = group.groupby("market_session")["sentiment"].mean()

            pre_cnt = sess_counts.get("Pre-Market", 0)
            reg_cnt = sess_counts.get("Regular-Hours", 0)
            post_cnt = sess_counts.get("Post-Market", 0)

            records.append({
                "effective_date": dt,
                "mean_minutes_from_open": round(float(group["minutes_from_open"].mean()), 2),
                "sentiment_pre_market": round(float(sess_sent.get("Pre-Market", 0.0)), 4),
                "sentiment_regular_hours": round(float(sess_sent.get("Regular-Hours", 0.0)), 4),
                "sentiment_post_market": round(float(sess_sent.get("Post-Market", 0.0)), 4),
                "count_pre_market": pre_cnt,
                "count_regular_hours": reg_cnt,
                "count_post_market": post_cnt,
                "ratio_pre_market": round(pre_cnt / total_n, 4) if total_n > 0 else 0.0,
            })

        temp_df = pd.DataFrame(records)

        # Day of week features
        day_names = temp_df["effective_date"].dt.day_name()
        temp_df["is_monday"] = (day_names == "Monday").astype(int)
        temp_df["is_tuesday"] = (day_names == "Tuesday").astype(int)
        temp_df["is_wednesday"] = (day_names == "Wednesday").astype(int)
        temp_df["is_thursday"] = (day_names == "Thursday").astype(int)
        temp_df["is_friday"] = (day_names == "Friday").astype(int)
        temp_df["day_of_week_num"] = temp_df["effective_date"].dt.day_of_week

        # Cyclical month features
        months = temp_df["effective_date"].dt.month
        temp_df["month_sin"] = np.sin(2 * np.pi * months / 12)
        temp_df["month_cos"] = np.cos(2 * np.pi * months / 12)

        return temp_df

    # =========================================================================
    # 5. Stock-Specific News Frequency & Volume Dynamics
    # =========================================================================
    def build_news_volume_features(self) -> pd.DataFrame:
        """
        Construct news frequency, moving average volume, and surge z-score features.

        Returns:
            DataFrame indexed by effective_date with news volume metrics.
        """
        vol_df = self.news.groupby("effective_date").agg(
            news_volume=("headline", "count"),
            sentiment_mean=("sentiment", "mean"),
            sentiment_std=("sentiment", lambda x: x.std() if len(x) > 1 else 0.0),
            sentiment_median=("sentiment", "median"),
            pos_count=("sentiment", lambda x: (x > 0.05).sum()),
            neg_count=("sentiment", lambda x: (x < -0.05).sum()),
        ).reset_index()

        vol_df["sentiment_std"] = vol_df["sentiment_std"].fillna(0.0)
        vol_df["sentiment_pos_ratio"] = vol_df["pos_count"] / vol_df["news_volume"]
        vol_df["sentiment_neg_ratio"] = vol_df["neg_count"] / vol_df["news_volume"]
        vol_df["sentiment_net_ratio"] = vol_df["sentiment_pos_ratio"] - vol_df["sentiment_neg_ratio"]

        # Sort for rolling computations
        vol_df = vol_df.sort_values("effective_date").reset_index(drop=True)

        # News volume moving averages & abnormal volume surge z-score
        vol_df["news_volume_sma_7"] = vol_df["news_volume"].rolling(7, min_periods=1).mean()
        vol_df["news_volume_sma_30"] = vol_df["news_volume"].rolling(30, min_periods=1).mean()
        vol_std_30 = vol_df["news_volume"].rolling(30, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)
        vol_df["news_volume_surge_zscore"] = (vol_df["news_volume"] - vol_df["news_volume_sma_30"]) / vol_std_30
        vol_df["is_news_volume_spike"] = (vol_df["news_volume_surge_zscore"] >= 2.0).astype(int)

        # News gap: days since last news article
        vol_df["days_since_last_news"] = vol_df["effective_date"].diff().dt.days.fillna(0).astype(int)

        return vol_df

    # =========================================================================
    # 6. Sentiment Dynamics & Momentum Features (Lags & Rolling)
    # =========================================================================
    def build_sentiment_momentum_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute multi-day sentiment lags, moving averages, and momentum velocity.

        Args:
            base_df: Daily DataFrame containing 'effective_date' and 'sentiment_mean'.

        Returns:
            DataFrame with added lagged and momentum features.
        """
        df = base_df.sort_values("effective_date").copy()

        # Multi-day lags (1, 2, 3, 5 days)
        for lag in [1, 2, 3, 5]:
            df[f"sentiment_lag_{lag}"] = df["sentiment_mean"].shift(lag)

        # Rolling sentiment moving averages
        df["sentiment_sma_3"] = df["sentiment_mean"].rolling(3, min_periods=1).mean()
        df["sentiment_sma_7"] = df["sentiment_mean"].rolling(7, min_periods=1).mean()
        df["sentiment_sma_14"] = df["sentiment_mean"].rolling(14, min_periods=1).mean()

        # Sentiment velocity (change over 1-day and 3-days)
        df["sentiment_velocity_1d"] = df["sentiment_mean"] - df["sentiment_lag_1"]
        df["sentiment_velocity_3d"] = df["sentiment_mean"] - df["sentiment_lag_3"]

        # Sentiment volatility (7-day rolling std)
        df["sentiment_rolling_std_7"] = df["sentiment_mean"].rolling(7, min_periods=1).std().fillna(0.0)

        return df

    # =========================================================================
    # 7. Technical Indicators & Price Target Alignment
    # =========================================================================
    def align_with_price_and_indicators(self, daily_news_features: pd.DataFrame) -> pd.DataFrame:
        """
        Merge engineered news features with stock OHLCV prices, compute technical indicators,
        and generate forward-looking prediction targets for machine learning.

        Returns:
            Merged Master Feature DataFrame indexed by Date.
        """
        if self.price is None or self.price.empty:
            print("Notice: No stock price dataset provided. Returning news features only.")
            return daily_news_features

        price = self.price.sort_values("Date").copy()
        close = price["Close"]
        high = price["High"]
        low = price["Low"]
        vol = price["Volume"]

        # A. Returns
        price["daily_return"] = close.pct_change()
        price["log_return"] = np.log(close / close.shift(1))

        # B. Moving Averages
        price["sma_10"] = close.rolling(10, min_periods=1).mean()
        price["sma_20"] = close.rolling(20, min_periods=1).mean()
        price["sma_50"] = close.rolling(50, min_periods=1).mean()
        price["price_to_sma_20"] = close / price["sma_20"]
        price["sma_cross_20_50"] = (price["sma_20"] > price["sma_50"]).astype(int)

        # C. RSI (Relative Strength Index, 14-period)
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_gain = pd.Series(gain, index=price.index).rolling(14, min_periods=1).mean()
        roll_loss = pd.Series(loss, index=price.index).rolling(14, min_periods=1).mean().replace(0.0, 1e-9)
        rs = roll_gain / roll_loss
        price["rsi_14"] = 100 - (100 / (1 + rs))

        # D. MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        price["macd"] = ema_12 - ema_26
        price["macd_signal"] = price["macd"].ewm(span=9, adjust=False).mean()
        price["macd_hist"] = price["macd"] - price["macd_signal"]

        # E. Bollinger Bands (20-period, 2-std)
        bb_std = close.rolling(20, min_periods=1).std()
        bb_upper = price["sma_20"] + (2.0 * bb_std)
        bb_lower = price["sma_20"] - (2.0 * bb_std)
        price["bollinger_width"] = (bb_upper - bb_lower) / price["sma_20"]
        price["bollinger_pband"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0.0, 1e-9)

        # F. ATR (Average True Range, 14-period)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        price["atr_14"] = true_range.rolling(14, min_periods=1).mean()

        # G. Historical Volatility (20-day annualized)
        price["volatility_20d"] = price["daily_return"].rolling(20, min_periods=1).std() * np.sqrt(252)

        # H. Machine Learning Target Variables (Forward-looking Next Day / Multi-Day)
        price["target_next_day_return"] = price["daily_return"].shift(-1)
        price["target_next_day_direction"] = (price["target_next_day_return"] > 0).astype(int)

        # 3-Class Target: Up (> +0.5%), Down (< -0.5%), Neutral
        def classify_return(ret: float) -> str:
            if pd.isna(ret):
                return "Unknown"
            if ret > 0.005:
                return "Up"
            elif ret < -0.005:
                return "Down"
            else:
                return "Neutral"

        price["target_next_day_class"] = price["target_next_day_return"].apply(classify_return)
        price["target_3d_return"] = close.pct_change(3).shift(-3)
        price["target_5d_return"] = close.pct_change(5).shift(-5)

        # I. Merge with daily news features
        merged = pd.merge(
            price,
            daily_news_features,
            left_on="Date",
            right_on="effective_date",
            how="left"
        )

        # Fill news metrics for non-news days with 0 and add indicator
        merged["has_news"] = (~merged["news_volume"].isna()).astype(int)
        news_cols_to_fill_zero = [c for c in daily_news_features.columns if c != "effective_date"]
        for col in news_cols_to_fill_zero:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        # Drop rows where effective_date couldn't map or trailing forward targets (last row)
        return merged

    # =========================================================================
    # 8. Master Feature Pipeline Orchestrator
    # =========================================================================
    def run_feature_pipeline(
        self,
        topic_col: str = "lda_topic",
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Execute end-to-end feature engineering pipeline:
        1. Text length features
        2. Topic representation features
        3. Publisher credibility weighting
        4. Temporal and market session features
        5. Stock-specific news volume dynamics
        6. Sentiment momentum & lags
        7. Stock price & technical indicators alignment

        Returns:
            Unified Master Feature DataFrame.
        """
        print("=" * 80)
        print("STARTING FEATURE ENGINEERING PIPELINE (STEP 2)")
        if self.ticker:
            print(f"Target Stock Symbol: {self.ticker}")
        print("=" * 80)

        # 1. Text length features
        print("1. Extracting headline text length features...")
        text_df = self.build_text_features()

        # 2. Topic features
        print("2. Extracting topic representation and financial category features...")
        topic_df = self.build_topic_features(topic_col=topic_col)

        # 3. Publisher weighting
        print("3. Calculating publisher credibility weights and adjusted sentiment...")
        pub_df, pub_weights = self.build_publisher_weighting_features()

        # 4. Temporal features
        print("4. Generating temporal, session, and market-open timing features...")
        temp_df = self.build_temporal_features()

        # 5. News volume features
        print("5. Computing stock-specific news frequency and volume dynamics...")
        vol_df = self.build_news_volume_features()

        # 6. Merge daily news features
        print("6. Synthesizing daily sentiment momentum and lag features...")
        daily = vol_df
        daily = pd.merge(daily, text_df, on="effective_date", how="left")
        daily = pd.merge(daily, topic_df, on="effective_date", how="left")
        daily = pd.merge(daily, pub_df, on="effective_date", how="left")
        daily = pd.merge(daily, temp_df, on="effective_date", how="left")

        # Add sentiment momentum and lags
        daily = self.build_sentiment_momentum_features(daily)
        self.daily_features = daily

        # 7. Merge with stock prices if provided
        if self.price is not None:
            print("7. Aligning with stock price history and calculating technical indicators...")
            master = self.align_with_price_and_indicators(daily)
        else:
            master = daily

        self.master_df = master

        # Save artifacts
        if save:
            ticker_suffix = f"_{self.ticker}" if self.ticker else ""
            out_csv = self.output_dir / "data" / f"engineered_features{ticker_suffix}.csv"
            pub_csv = self.output_dir / "data" / "publisher_credibility_weights.csv"

            master.to_csv(out_csv, index=False)
            pub_weights.to_csv(pub_csv, index=False)
            print(f"✓ Master feature dataset saved to: {out_csv}")
            print(f"✓ Publisher credibility weights saved to: {pub_csv}")

            # Save metadata summary
            self._save_feature_summary(master, ticker_suffix)
            self.plot_feature_correlations(ticker_suffix, save=True)

        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY ✓")
        print(f"Total Rows: {len(master):,} | Total Features Generated: {len(master.columns):,}")
        print("=" * 80)

        return master

    def _save_feature_summary(self, df: pd.DataFrame, ticker_suffix: str) -> Path:
        """Write JSON summary of engineered feature categories and column counts."""
        summary_path = self.output_dir / "data" / f"feature_summary{ticker_suffix}.json"

        sentiment_features = [c for c in df.columns if "sentiment" in c]
        topic_features = [c for c in df.columns if "topic" in c or "cat_" in c or "has_" in c]
        temporal_features = [c for c in df.columns if "day_" in c or "month" in c or "is_" in c or "minutes" in c]
        volume_features = [c for c in df.columns if "volume" in c or "news_" in c]
        technical_features = [c for c in df.columns if c in ["sma_10", "sma_20", "sma_50", "rsi_14", "macd", "bollinger_width", "atr_14", "volatility_20d"]]
        target_features = [c for c in df.columns if c.startswith("target_")]

        summary = {
            "ticker": self.ticker or "ALL",
            "total_samples": len(df),
            "total_features": len(df.columns),
            "feature_categories": {
                "sentiment_and_momentum": len(sentiment_features),
                "topics_and_categories": len(topic_features),
                "temporal_and_sessions": len(temporal_features),
                "news_volume_dynamics": len(volume_features),
                "technical_indicators": len(technical_features),
                "targets": len(target_features),
            },
            "columns": list(df.columns),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Feature engineering summary saved to: {summary_path}")
        return summary_path

    def plot_feature_correlations(self, ticker_suffix: str, save: bool = True) -> None:
        """Plot preview heatmap of key sentiment features vs technical returns."""
        if self.master_df is None:
            return

        cols_to_check = [
            "sentiment_mean",
            "sentiment_lag_1",
            "sentiment_publisher_weighted",
            "sentiment_publisher_adjusted",
            "news_volume",
            "news_volume_surge_zscore",
            "ratio_pre_market",
            "daily_return",
            "rsi_14",
            "macd",
            "target_next_day_return"
        ]
        available_cols = [c for c in cols_to_check if c in self.master_df.columns]

        if len(available_cols) < 3:
            return

        corr_matrix = self.master_df[available_cols].dropna().corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, vmin=-1.0, vmax=1.0)
        plt.title(f"Engineered Features Correlation Matrix ({self.ticker or 'Portfolio'})", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save:
            out_fig = self.output_dir / "figures" / f"feature_correlation_matrix{ticker_suffix}.png"
            plt.savefig(out_fig, dpi=300, bbox_inches="tight")
            print(f"✓ Correlation matrix heatmap saved to: {out_fig}")
        plt.close()
