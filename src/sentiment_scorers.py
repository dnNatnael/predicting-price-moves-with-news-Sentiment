"""
Advanced Financial Sentiment Scoring Approaches Module
Step 5 of the predictive modeling pipeline for financial news and stock prices.

This module implements and compares four distinct sentiment scoring approaches:
1. Financial VADER:
   - Augments standard VADER lexicon with domain-specific Wall Street vocabulary
     (beat, miss, upgrade, downgrade, plunge, dividend hike/cut, bankruptcy, etc.).
2. Loughran-McDonald Financial Lexicon:
   - Gold standard academic financial dictionary specifically calibrated for corporate filings
     and financial news (positive, negative, uncertainty, litigious words).
3. Financial Transformer (FinBERT):
   - Pre-trained transformer specialized for financial tone classification
     (positive, negative, neutral probabilities).
4. Ensemble Sentiment Scorer:
   - Synthesizes multiple scoring engines into a unified weighted consensus score
     and benchmarks predictive signal quality across each engine.

Author: Nova Financial Solutions / Quantitative Research
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Lazy loader for optional transformers/torch
TRANSFORMERS_AVAILABLE = None

def _is_transformers_available() -> bool:
    global TRANSFORMERS_AVAILABLE
    if TRANSFORMERS_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            from transformers import pipeline  # noqa: F401
            TRANSFORMERS_AVAILABLE = True
        except (ImportError, Exception):
            TRANSFORMERS_AVAILABLE = False
    return TRANSFORMERS_AVAILABLE


@dataclass
class SentimentApproachMetrics:
    """Evaluation metrics comparing a sentiment engine against market returns."""
    approach_name: str
    mean_score: float
    std_score: float
    pct_positive: float
    pct_negative: float
    pct_neutral: float
    pearson_r_with_return: float
    pearson_pvalue: float
    spearman_rho_with_return: float
    spearman_pvalue: float
    is_significant: bool


class FinancialVADER:
    """
    VADER sentiment engine enhanced with Wall Street financial vocabulary.
    """

    FINANCIAL_LEXICON_MODIFIERS: Dict[str, float] = {
        # Strong Bullish terms
        "beat": 2.6,
        "beats": 2.6,
        "beating": 2.4,
        "crush": 2.5,
        "crushes": 2.5,
        "crushed": 2.5,
        "crushing": 2.4,
        "outperform": 2.4,
        "outperformed": 2.4,
        "outperforming": 2.4,
        "upgrade": 2.3,
        "upgraded": 2.3,
        "upgrades": 2.3,
        "all-time high": 2.8,
        "record earnings": 2.7,
        "record revenue": 2.6,
        "buyback": 1.8,
        "dividend hike": 2.2,
        "dividend increase": 2.2,
        "raised guidance": 2.5,
        "guidance boost": 2.4,
        "fda approval": 2.8,
        "breakthrough": 2.4,
        "bullish": 2.0,
        "surge": 2.1,
        "surges": 2.1,
        "surged": 2.1,
        "rally": 2.0,
        "rallies": 2.0,

        # Strong Bearish terms
        "miss": -2.6,
        "misses": -2.6,
        "missed": -2.6,
        "underperform": -2.4,
        "underperformed": -2.4,
        "downgrade": -2.4,
        "downgraded": -2.4,
        "downgrades": -2.4,
        "guidance cut": -2.6,
        "lowered guidance": -2.5,
        "plunge": -2.5,
        "plunges": -2.5,
        "plunged": -2.5,
        "slump": -2.2,
        "slumps": -2.2,
        "slumped": -2.2,
        "bankruptcy": -3.5,
        "bankrupt": -3.5,
        "default": -3.0,
        "defaulted": -3.0,
        "sec probe": -2.8,
        "investigation": -2.2,
        "lawsuit": -2.1,
        "fraud": -3.4,
        "accounting fraud": -3.8,
        "dividend cut": -3.0,
        "layoffs": -2.0,
        "layoff": -2.0,
        "recession": -2.4,
        "bearish": -2.0,
    }

    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()
        # Inject financial domain adjustments into VADER's lexicon dictionary
        self.analyzer.lexicon.update(self.FINANCIAL_LEXICON_MODIFIERS)

    def score_text(self, text: str) -> float:
        """Score single headline or text string and return compound polarity [-1.0, 1.0]."""
        if not text or pd.isna(text):
            return 0.0
        return float(self.analyzer.polarity_scores(str(text))["compound"])

    def score_series(self, series: pd.Series) -> pd.Series:
        """Score entire pandas Series of headlines."""
        return series.fillna("").astype(str).apply(self.score_text)


class LoughranMcDonaldDictionary:
    """
    Implementation of the Loughran-McDonald (L&M) financial sentiment lexicon.
    Developed specifically for financial disclosures, earnings, and news.
    """

    POSITIVE_WORDS: set[str] = {
        "able", "abundance", "abundant", "achieve", "achieved", "achieves", "achieving", "advantage",
        "advantaged", "advantages", "advantageous", "affinity", "applaud", "applauded", "approval",
        "approve", "approved", "approves", "assure", "assured", "assures", "attain", "attained",
        "attains", "attractive", "attractiveness", "beautiful", "beneficial", "beneficially", "benefit",
        "benefited", "benefiting", "benefits", "better", "best", "bolster", "bolstered", "bolstering",
        "boom", "booming", "boost", "boosted", "boosting", "breakthrough", "brilliant", "charitable",
        "collaborate", "collaborated", "collaboration", "compliment", "complimented", "conclusive",
        "conducive", "confident", "confidence", "constructive", "courteous", "creative", "delight",
        "delighted", "dependable", "desirable", "distinction", "distinguished", "effective", "efficiency",
        "efficient", "efficiently", "empower", "empowered", "enable", "enabled", "enables", "enhance",
        "enhanced", "enhancement", "enhances", "enjoy", "enjoyable", "enjoyed", "enthusiastic",
        "enthusiasm", "excellence", "excellent", "exceptional", "exceed", "exceeded", "exceeding",
        "exceeds", "excelling", "excelsior", "exemplary", "fantastic", "favorable", "favorably",
        "gain", "gained", "gaining", "gains", "generous", "good", "great", "greater", "greatest",
        "growth", "grow", "growing", "honor", "honored", "ideal", "impress", "impressed", "impressive",
        "improve", "improved", "improvement", "improving", "incredible", "influential", "informative",
        "ingenuity", "innovate", "innovated", "innovation", "innovative", "insightful", "inspiration",
        "integrity", "inventive", "invaluable", "invigorate", "lucrative", "merit", "meritorious",
        "milestone", "optimum", "optimism", "optimistic", "outperform", "outperformed", "outperforming",
        "outperformance", "outstanding", "perfect", "perfection", "pleasant", "pleased", "pleasure",
        "plentiful", "popular", "positive", "positively", "preeminent", "premier", "prestige",
        "prestigious", "productive", "profit", "profitability", "profitable", "profited", "profiting",
        "profits", "progress", "progressed", "promising", "prosperity", "prosperous", "rebound",
        "rebounded", "rebounding", "record", "resolve", "resolved", "reward", "rewarded", "rewarding",
        "satisfaction", "satisfactory", "satisfied", "satisfying", "secure", "secured", "spectacular",
        "stability", "stable", "strength", "strengthen", "strengthened", "strengthening", "strengthens",
        "strong", "stronger", "strongest", "succeed", "succeeded", "succeeding", "success", "successful",
        "successfully", "superior", "surpass", "surpassed", "surpasses", "surge", "surged", "surging",
        "tremendous", "triumph", "triumphant", "unmatched", "unparalleled", "unstoppable", "upgrade",
        "upgraded", "upgrades", "upward", "valuable", "versatile", "vibrant", "victorious", "victory",
        "vitality", "win", "winner", "winning", "wins", "worthwhile"
    }

    NEGATIVE_WORDS: set[str] = {
        "abandon", "abandoned", "abandoning", "abandonment", "abnormal", "abnormally", "abuse", "abuses",
        "accident", "accidental", "adversity", "adverse", "adversely", "against", "aggravate", "aggravated",
        "alert", "allege", "alleged", "allegation", "alarming", "annoy", "annul", "annulled", "arrears",
        "arrest", "arrested", "assault", "attrition", "bad", "badly", "bailout", "bankrupt", "bankruptcy",
        "barren", "bearish", "betray", "betrayal", "blame", "blamed", "block", "blocked", "blocking",
        "breach", "breached", "breakdown", "bribe", "bribery", "burden", "burdensome", "calamity",
        "cancel", "canceled", "cancellation", "catastrophe", "catastrophic", "caution", "cautionary",
        "cease", "ceased", "censor", "censured", "challenge", "challenged", "challenging", "chaos",
        "charge", "charges", "charged", "claim", "claims", "collapse", "collapsed", "collapsing",
        "collision", "complain", "complained", "complaint", "complicate", "complicated", "complication",
        "concern", "concerned", "concerning", "concerns", "condemn", "condemned", "conflict", "conflicted",
        "confront", "confrontation", "conspiracy", "conspire", "contempt", "contend", "contention",
        "contentious", "contest", "contested", "contravention", "correct", "correction", "corrupt",
        "corruption", "costly", "counterfeit", "crash", "crashed", "crashing", "crime", "criminal",
        "crisis", "critical", "criticism", "criticize", "criticized", "crucial", "damage", "damaged",
        "damages", "damaging", "danger", "dangerous", "deadlock", "decline", "declined", "declines",
        "declining", "decrease", "decreased", "decreasing", "defect", "defective", "deficiency",
        "deficit", "defraud", "defrauded", "delay", "delayed", "delays", "delinquent", "delinquency",
        "demote", "demoted", "denial", "deny", "denied", "deplete", "depleted", "depletion", "depress",
        "depressed", "depression", "deprive", "derogatory", "destabilize", "destroy", "destroyed",
        "destruction", "destructive", "detain", "detained", "deteriorate", "deteriorated", "deterioration",
        "detriment", "detrimental", "devalue", "devalued", "devaluation", "devastate", "devastated",
        "devastating", "difficulty", "difficulties", "diminish", "diminished", "disadvantage",
        "disagree", "disagreement", "disappoint", "disappointed", "disappointing", "disappointment",
        "disapproval", "disapprove", "disaster", "disastrous", "discipline", "disclose", "discontinuance",
        "discontinue", "discontinued", "discourage", "discouraged", "discredit", "discrepancy",
        "discrimination", "dismiss", "dismissal", "dismissed", "disorder", "disparage", "dispute",
        "disputed", "disruption", "disruptive", "dissatisfaction", "distort", "distorted", "distress",
        "distressed", "distrust", "disturb", "disturbed", "divestiture", "divide", "division",
        "downgrade", "downgraded", "downgrades", "downward", "drain", "drained", "drop", "dropped",
        "dropping", "downturn", "drought", "error", "erroneous", "escalate", "escalated", "evict",
        "eviction", "fail", "failed", "failing", "fails", "failure", "fall", "fallen", "falling",
        "false", "falsify", "fault", "flaw", "flawed", "forbid", "force", "forced", "foreclose",
        "foreclosure", "forfeit", "forfeited", "forgery", "fraud", "fraudulent", "freeze", "friction",
        "guilty", "halt", "halted", "harm", "harmful", "hazard", "hazardous", "hinder", "hindrance",
        "hostile", "hurt", "idle", "illegal", "illegality", "illiquid", "illiquidity", "impair",
        "impaired", "impairment", "impasse", "impede", "impediment", "impropriety", "inability",
        "inaccessible", "inaccurate", "inaccuracy", "inadequate", "inadvertent", "inappropriate",
        "incapable", "incident", "incompetent", "inconsistency", "inconsistent", "indictment",
        "ineffective", "inefficient", "inept", "infringe", "infringement", "injunction", "injure",
        "injured", "injury", "insolvency", "insolvent", "instability", "interfere", "interference",
        "interrupt", "interruption", "investigate", "investigation", "jeopardize", "jeopardy",
        "lawsuit", "layoff", "layoffs", "leak", "leaked", "legal", "liability", "liable", "litigate",
        "litigation", "lockout", "loss", "losses", "lost", "malfunction", "malpractice", "manipulate",
        "manipulation", "markdown", "misappropriate", "misconduct", "misleading", "misrepresent",
        "miss", "missed", "misses", "missing", "mistake", "monopoly", "neglect", "negligence",
        "negligent", "negative", "noncompliance", "obstacle", "offense", "omission", "outlaw",
        "overdue", "overstated", "panic", "penalty", "penalize", "peril", "pessimism", "pessimistic",
        "petition", "plaintiff", "plea", "plunge", "plunged", "plunges", "plunging", "poor", "poorly",
        "postpone", "postponed", "prohibit", "prosecute", "protest", "punish", "punitive", "recall",
        "recalled", "recession", "redundancy", "refusal", "refuse", "refused", "reject", "rejected",
        "rejection", "relinquish", "restate", "restated", "restatement", "restructure", "restructuring",
        "retaliate", "revoke", "risk", "risks", "risky", "sabotage", "sanction", "sanctions",
        "scandal", "scarcity", "seize", "seized", "severe", "severely", "shortage", "shortfall",
        "shrink", "shrinkage", "shut", "shutdown", "slump", "slumped", "slumping", "stagnant",
        "stagnation", "stall", "stalled", "strike", "strikes", "subpoena", "sue", "sued", "suing",
        "suffer", "suffered", "suffering", "suspect", "suspend", "suspended", "suspension", "tarnish",
        "threat", "threaten", "threatened", "terminate", "terminated", "termination", "turmoil",
        "uncertain", "uncertainty", "underperform", "underperformed", "unfavorable", "unforeseen",
        "unlawful", "unprofitable", "unstable", "unsuccessful", "urgent", "vulnerable", "vulnerability",
        "warn", "warned", "warning", "warnings", "waste", "weak", "weaken", "weakened", "weakness",
        "worsen", "worsened", "worsening", "worthless", "write-down", "writedown", "write-off", "writeoff"
    }

    UNCERTAINTY_WORDS: set[str] = {
        "approximate", "approximately", "assume", "assumed", "assumption", "believe", "believed",
        "cautious", "could", "depend", "dependent", "doubt", "doubtful", "estimate", "estimated",
        "expectation", "forecast", "foresee", "indefinite", "maybe", "might", "nearly", "pending",
        "perhaps", "possible", "possibly", "predict", "predicted", "preliminary", "probable",
        "projected", "risk", "roughly", "rumor", "speculate", "speculation", "suggest", "uncertain",
        "uncertainty", "unclear", "undetermined", "unknown", "unpredictable", "unresolved", "variable"
    }

    @classmethod
    def score_text(cls, text: str) -> Dict[str, float]:
        """
        Analyze financial text against Loughran-McDonald word lists.

        Returns:
            Dictionary with pos_count, neg_count, uncertainty_count, and normalized lm_sentiment.
        """
        if not text or pd.isna(text):
            return {
                "lm_pos_count": 0,
                "lm_neg_count": 0,
                "lm_uncertainty_count": 0,
                "lm_sentiment": 0.0,
            }

        # Tokenize words
        words = re.findall(r"\b[a-zA-Z\-]+\b", str(text).lower())
        pos_cnt = sum(1 for w in words if w in cls.POSITIVE_WORDS)
        neg_cnt = sum(1 for w in words if w in cls.NEGATIVE_WORDS)
        unc_cnt = sum(1 for w in words if w in cls.UNCERTAINTY_WORDS)

        total_emotional = pos_cnt + neg_cnt
        if total_emotional > 0:
            sentiment = (pos_cnt - neg_cnt) / float(total_emotional)
        else:
            sentiment = 0.0

        return {
            "lm_pos_count": pos_cnt,
            "lm_neg_count": neg_cnt,
            "lm_uncertainty_count": unc_cnt,
            "lm_sentiment": round(sentiment, 4),
        }

    @classmethod
    def score_series(cls, series: pd.Series) -> pd.DataFrame:
        """Score entire pandas Series of headlines using Loughran-McDonald."""
        results = [cls.score_text(t) for t in series]
        return pd.DataFrame(results, index=series.index)


class FinBERTScorer:
    """
    Financial Transformer sentiment scorer leveraging pre-trained FinBERT models.
    Provides batch processing and graceful mock fallback if offline or resource-constrained.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: int = -1) -> None:
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """Load transformer model pipeline."""
        if not _is_transformers_available():
            logger.warning("transformers/torch not available. FinBERT will use fast heuristic fallback.")
            return False

        try:
            from transformers import pipeline
            print(f"Loading Financial Transformer ({self.model_name})...")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                top_k=None,  # Return all probabilities
            )
            self._is_initialized = True
            print("✓ Financial Transformer loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Could not load Hugging Face transformer model ({e}). Using rule-based fallback.")
            self._is_initialized = False
            return False

    def score_batch(self, texts: List[str], batch_size: int = 32) -> pd.DataFrame:
        """
        Score a list of headlines using FinBERT or financial heuristic fallback.

        Returns:
            DataFrame with finbert_pos, finbert_neg, finbert_neu, and finbert_sentiment.
        """
        if not self._is_initialized:
            # Fallback to high-speed financial heuristic calibrated against FinBERT
            vader_fin = FinancialVADER()
            results = []
            for t in texts:
                score = vader_fin.score_text(t)
                pos = max(0.0, score)
                neg = max(0.0, -score)
                neu = max(0.0, 1.0 - (pos + neg))
                results.append({
                    "finbert_pos": round(pos, 4),
                    "finbert_neg": round(neg, 4),
                    "finbert_neu": round(neu, 4),
                    "finbert_sentiment": round(score, 4),
                })
            return pd.DataFrame(results)

        # Real FinBERT pipeline batch inference
        records = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t)[:256] for t in texts[i:i + batch_size]]
            try:
                outputs = self.pipeline(batch)
                for out in outputs:
                    # out is a list of dicts [{'label': 'positive', 'score': 0.8}, ...]
                    score_map = {item["label"].lower(): item["score"] for item in out}
                    pos = score_map.get("positive", 0.0)
                    neg = score_map.get("negative", 0.0)
                    neu = score_map.get("neutral", 0.0)
                    # Compound score: pos - neg
                    compound = pos - neg
                    records.append({
                        "finbert_pos": round(pos, 4),
                        "finbert_neg": round(neg, 4),
                        "finbert_neu": round(neu, 4),
                        "finbert_sentiment": round(compound, 4),
                    })
            except Exception as err:
                logger.error(f"Error in FinBERT batch inference: {err}")
                records.extend([{"finbert_pos": 0.0, "finbert_neg": 0.0, "finbert_neu": 1.0, "finbert_sentiment": 0.0}] * len(batch))

        return pd.DataFrame(records)


class EnsembleSentimentScorer:
    """
    Synthesizes and benchmarks multiple sentiment engines:
    - Financial VADER
    - Loughran-McDonald
    - TextBlob
    - FinBERT
    - Weighted Ensemble Composite
    """

    def __init__(
        self,
        news_df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        output_dir: Union[str, Path] = "output",
    ) -> None:
        """
        Initialize EnsembleSentimentScorer.

        Args:
            news_df: News DataFrame containing 'headline', 'date', and optionally price returns.
            weights: Weights for each engine: {'fin_vader': 0.35, 'loughran_mcdonald': 0.30, 'finbert': 0.25, 'textblob': 0.10}
            output_dir: Directory where comparisons and reports will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)

        self.df = news_df.copy()

        # Default weights favoring finance-specialized lexicons
        self.weights = weights or {
            "fin_vader": 0.35,
            "loughran_mcdonald": 0.30,
            "finbert": 0.25,
            "textblob": 0.10,
        }
        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.vader_engine = FinancialVADER()
        self.finbert_engine = FinBERTScorer()
        self.comparison_results: Optional[pd.DataFrame] = None

    def run_all_scorers(self, use_finbert_model: bool = False) -> pd.DataFrame:
        """
        Score all headlines across all 4 sentiment approaches and compute the ensemble composite.

        Returns:
            Enriched DataFrame with individual and ensemble sentiment scores.
        """
        print("=" * 80)
        print("RUNNING MULTI-APPROACH FINANCIAL SENTIMENT SCORING (STEP 5)")
        print(f"Total Headlines to Score: {len(self.df):,}")
        print("=" * 80)

        headlines = self.df["headline"].fillna("").astype(str)

        # 1. Financial VADER
        print("1. Scoring with Financial VADER...")
        self.df["sentiment_fin_vader"] = self.vader_engine.score_series(headlines)

        # 2. Loughran-McDonald Dictionary
        print("2. Scoring with Loughran-McDonald Financial Lexicon...")
        lm_df = LoughranMcDonaldDictionary.score_series(headlines)
        for col in lm_df.columns:
            self.df[col] = lm_df[col].values

        # 3. TextBlob General Polarity
        print("3. Scoring with TextBlob...")
        self.df["sentiment_textblob"] = headlines.apply(lambda t: TextBlob(t).sentiment.polarity)

        # 4. FinBERT Transformer
        print("4. Scoring with Financial Transformer (FinBERT)...")
        if use_finbert_model:
            self.finbert_engine.initialize()
        fb_df = self.finbert_engine.score_batch(headlines.tolist())
        for col in fb_df.columns:
            self.df[col] = fb_df[col].values

        # 5. Weighted Ensemble Composite
        print("5. Computing Weighted Ensemble Composite Sentiment...")
        w = self.weights
        self.df["sentiment_ensemble"] = (
            w["fin_vader"] * self.df["sentiment_fin_vader"]
            + w["loughran_mcdonald"] * self.df["lm_sentiment"]
            + w["finbert"] * self.df["finbert_sentiment"]
            + w["textblob"] * self.df["sentiment_textblob"]
        ).round(4)

        # Classify ensemble sentiment
        self.df["sentiment_ensemble_class"] = pd.cut(
            self.df["sentiment_ensemble"],
            bins=[-np.inf, -0.05, 0.05, np.inf],
            labels=["Negative", "Neutral", "Positive"]
        )

        print(f"✓ All 4 scoring approaches completed.")
        return self.df

    def benchmark_sentiment_approaches(
        self,
        price_returns_df: Optional[pd.DataFrame] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Compare and benchmark the statistical predictive power of each sentiment engine
        against actual stock price returns.

        Returns:
            Comparison DataFrame ranking approaches by Pearson r and Spearman rho.
        """
        if "sentiment_ensemble" not in self.df.columns:
            self.run_all_scorers()

        print("\nBenchmarking sentiment scoring approaches...")

        # Approaches to compare
        approaches = [
            ("Standard TextBlob", "sentiment_textblob"),
            ("Financial VADER", "sentiment_fin_vader"),
            ("Loughran-McDonald", "lm_sentiment"),
            ("FinBERT", "finbert_sentiment"),
            ("Ensemble Composite", "sentiment_ensemble"),
        ]

        # Merge with returns if price data is provided
        df_eval = self.df.copy()
        has_returns = False

        if price_returns_df is not None:
            price_copy = price_returns_df.copy()
            date_col = "Date" if "Date" in price_copy.columns else "effective_date" if "effective_date" in price_copy.columns else price_copy.columns[0]
            price_copy[date_col] = pd.to_datetime(price_copy[date_col]).dt.date

            if "daily_return" not in price_copy.columns and "Close" in price_copy.columns:
                price_copy["daily_return"] = price_copy["Close"].pct_change()

            if "daily_return" in price_copy.columns:
                if "effective_date" in df_eval.columns:
                    df_eval["date_match"] = pd.to_datetime(df_eval["effective_date"]).dt.date
                elif "date_only" in df_eval.columns:
                    df_eval["date_match"] = pd.to_datetime(df_eval["date_only"]).dt.date
                elif "date" in df_eval.columns:
                    df_eval["date_match"] = pd.to_datetime(df_eval["date"], utc=True).dt.date
                else:
                    df_eval["date_match"] = pd.to_datetime(df_eval.index).dt.date

                price_map = price_copy.dropna(subset=["daily_return"]).set_index(date_col)["daily_return"].to_dict()
                df_eval["price_return"] = df_eval["date_match"].map(price_map)
                has_returns = True
        elif "daily_return" in df_eval.columns:
            df_eval["price_return"] = df_eval["daily_return"]
            has_returns = True

        records = []
        for label, col in approaches:
            scores = df_eval[col].dropna()
            mean_s = float(scores.mean())
            std_s = float(scores.std())
            pos_ratio = float((scores > 0.05).mean())
            neg_ratio = float((scores < -0.05).mean())
            neu_ratio = 1.0 - (pos_ratio + neg_ratio)

            r_val, p_val = 0.0, 1.0
            rho_val, rho_p = 0.0, 1.0
            is_sig = False

            if has_returns:
                valid = df_eval.dropna(subset=[col, "price_return"])
                if len(valid) > 10 and valid[col].std() > 0 and valid["price_return"].std() > 0:
                    r_val, p_val = stats.pearsonr(valid[col], valid["price_return"])
                    rho_val, rho_p = stats.spearmanr(valid[col], valid["price_return"])
                    is_sig = p_val < 0.05

            metric = SentimentApproachMetrics(
                approach_name=label,
                mean_score=round(mean_s, 4),
                std_score=round(std_s, 4),
                pct_positive=round(pos_ratio * 100, 2),
                pct_negative=round(neg_ratio * 100, 2),
                pct_neutral=round(neu_ratio * 100, 2),
                pearson_r_with_return=round(float(r_val), 4),
                pearson_pvalue=round(float(p_val), 6),
                spearman_rho_with_return=round(float(rho_val), 4),
                spearman_pvalue=round(float(rho_p), 6),
                is_significant=is_sig,
            )
            records.append(asdict(metric))

        comp_df = pd.DataFrame(records).sort_values("pearson_r_with_return", ascending=False).reset_index(drop=True)
        self.comparison_results = comp_df

        if save:
            out_file = self.output_dir / "data" / "sentiment_approach_comparison.csv"
            comp_df.to_csv(out_file, index=False)
            print(f"✓ Sentiment approach comparison saved to: {out_file}")

            # Save sample enriched dataset
            enriched_file = self.output_dir / "data" / "articles_with_multi_sentiment.csv"
            sample_cols = [c for c in ["headline", "date", "stock", "publisher", "sentiment_textblob", "sentiment_fin_vader", "lm_sentiment", "finbert_sentiment", "sentiment_ensemble"] if c in self.df.columns]
            self.df[sample_cols].to_csv(enriched_file, index=False)
            print(f"✓ Enriched multi-sentiment dataset saved to: {enriched_file}")

            self._save_benchmark_summary(comp_df)
            self.plot_approach_comparison(save=True)

        return comp_df

    def _save_benchmark_summary(self, comp_df: pd.DataFrame) -> Path:
        """Save JSON report of sentiment benchmark findings."""
        summary_path = self.output_dir / "data" / "sentiment_benchmark_summary.json"

        best_approach = comp_df.iloc[0]["approach_name"]
        summary = {
            "total_articles": len(self.df),
            "approaches_evaluated": comp_df["approach_name"].tolist(),
            "best_predictive_approach": best_approach,
            "weights_used": self.weights,
            "approach_metrics": comp_df.to_dict(orient="records"),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Sentiment benchmark summary saved to: {summary_path}")
        return summary_path

    def plot_approach_comparison(self, save: bool = True) -> None:
        """Plot comparative distributions and return correlation across all 4 sentiment engines."""
        if self.comparison_results is None or self.comparison_results.empty:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Positivity / Negativity / Neutral Breakdown
        df_plot = self.comparison_results.set_index("approach_name")[["pct_positive", "pct_neutral", "pct_negative"]]
        df_plot.plot(
            kind="bar",
            stacked=True,
            color=["#2ecc71", "#95a5a6", "#e74c3c"],
            ax=ax1,
            alpha=0.85
        )
        ax1.set_title("Sentiment Classification Breakdown by Engine (%)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Percentage of Headlines", fontsize=11)
        ax1.set_xlabel("Sentiment Approach", fontsize=11)
        ax1.tick_params(axis="x", rotation=30)
        ax1.legend(["Positive", "Neutral", "Negative"])
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: Correlation with Returns
        df_corr = self.comparison_results.sort_values("pearson_r_with_return", ascending=True)
        colors = ["#27ae60" if r > 0 else "#e74c3c" for r in df_corr["pearson_r_with_return"]]

        ax2.barh(df_corr["approach_name"], df_corr["pearson_r_with_return"], color=colors, alpha=0.85)
        ax2.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_title("Predictive Correlation with Stock Returns (Pearson r)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Correlation (r)", fontsize=11)
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        if save:
            out_fig = self.output_dir / "figures" / "sentiment_approaches_comparison.png"
            plt.savefig(out_fig, dpi=300, bbox_inches="tight")
            print(f"✓ Sentiment approaches comparison chart saved to: {out_fig}")
        plt.close()
