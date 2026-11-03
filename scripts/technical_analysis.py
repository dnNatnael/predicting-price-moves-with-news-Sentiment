import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import pynance as pn
except ImportError:
    pn = None

try:
    import talib
except ImportError:
    talib = None


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "AAPL.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "technical_analysis"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots"

plt.style.use("seaborn-v0_8-darkgrid")


# Configuration with CLI support
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for technical analysis"""
    parser = argparse.ArgumentParser(
        description="Technical Analysis of Stock Data with TA-Lib and PyNance"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to CSV file with OHLCV data. If not provided, uses data/{TICKER}.csv"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for analysis (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for analysis (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--sma-window",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="SMA window periods (default: 20 50 100)"
    )
    parser.add_argument(
        "--rsi-window",
        type=int,
        default=14,
        help="RSI period (default: 14)"
    )
    parser.add_argument(
        "--bb-window",
        type=int,
        default=20,
        help="Bollinger Bands window (default: 20)"
    )
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=30,
        help="Rolling volatility window (default: 30)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    return parser.parse_args()


def validate_data_sufficiency(df: pd.DataFrame, min_periods: int = 100) -> bool:
    """
    Validate that dataframe has sufficient history for indicators
    
    Args:
        df: DataFrame with price data
        min_periods: Minimum number of rows required
        
    Returns:
        True if sufficient data, raises ValueError otherwise
    """
    if len(df) < min_periods:
        raise ValueError(
            f"Insufficient data: {len(df)} rows. "
            f"Minimum {min_periods} rows required for reliable indicator calculation. "
            f"Consider using a longer date range or different ticker."
        )
    
    # Check for missing values in key columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        missing_pct = df[col].isna().sum() / len(df) * 100
        if missing_pct > 10:
            print(f"Warning: Column '{col}' has {missing_pct:.1f}% missing values")
    
    return True


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().title() for col in df.columns]
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("Date").set_index("Date")
    df = df.replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def add_ta_indicators(df: pd.DataFrame, 
                      sma_windows: list = None,
                      rsi_window: int = 14,
                      bb_window: int = 20) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe
    
    Args:
        df: DataFrame with OHLCV data
        sma_windows: List of SMA window periods (default: [20, 50, 100])
        rsi_window: RSI period (default: 14)
        bb_window: Bollinger Bands window (default: 20)
        
    Returns:
        DataFrame with added technical indicators
    """
    if sma_windows is None:
        sma_windows = [20, 50, 100]
    
    if talib is None:
        raise ImportError("TA-Lib is required for technical indicator calculations. Install TA-Lib.")
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    
    # SMA indicators
    for window in sma_windows:
        df[f"SMA_{window}"] = talib.SMA(close, timeperiod=window)
    
    # EMA indicators (using first two SMA windows)
    df["EMA_20"] = talib.EMA(close, timeperiod=20)
    if 50 in sma_windows:
        df["EMA_50"] = talib.EMA(close, timeperiod=50)
    
    # RSI
    df[f"RSI_{rsi_window}"] = talib.RSI(close, timeperiod=rsi_window)
    
    # Stochastic
    slowk, slowd = talib.STOCH(
        high,
        low,
        close,
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    df["STOCH_%K"] = slowk
    df["STOCH_%D"] = slowd
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist
    
    # ATR
    df["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(
        close, timeperiod=bb_window, nbdevup=2, nbdevdn=2, matype=0
    )
    df["BB_Upper"] = upper
    df["BB_Middle"] = middle
    df["BB_Lower"] = lower
    return df


def compute_pynance_metrics(df: pd.DataFrame, volatility_window: int = 30) -> dict:
    """
    Compute metrics using PyNance
    
    Args:
        df: DataFrame with Close and Volume columns
        volatility_window: Rolling window for volatility calculation
        
    Returns:
        Dictionary with computed metrics
    """
    if pn is None:
        raise ImportError("PyNance is required for metric calculations. Install pynance.")
    metrics = {}
    close_frame = df[["Close"]].copy()

    daily_ret = pn.tech.ret(df, selection="Close", outputcol="Daily_Return")
    df["Daily_Return"] = daily_ret["Daily_Return"]
    df["Daily_Return"] = df["Daily_Return"].fillna(0.0)

    log_ret = pn.tech.ln_growth(df, selection="Close", outputcol="Log_Return")
    df["Log_Return"] = log_ret["Log_Return"]
    df["Log_Return"] = df["Log_Return"].fillna(0.0)

    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

    rolling_vol = pn.tech.volatility(
        close_frame, selection="Close", window=volatility_window, outputcol="Rolling_Volatility"
    )
    df[f"Rolling_Volatility_{volatility_window}"] = rolling_vol[f"Rolling_Volatility"]

    returns_frame = df[["Daily_Return"]].copy()
    returns_frame.columns = ["Daily_Return"]
    rolling_return_vol = pn.tech.volatility(
        returns_frame, selection="Daily_Return", window=volatility_window, outputcol="Return_Vol"
    )
    df[f"Return_Vol_{volatility_window}"] = rolling_return_vol[f"Return_Vol"]
    
    # Use the correct column name for the last value
    vol_col = f"Return_Vol_{volatility_window}"
    latest_vol = df[vol_col].dropna().iloc[-1] if vol_col in df.columns else df["Return_Vol_30"].dropna().iloc[-1]
    annualized_vol = latest_vol * math.sqrt(252)
    metrics["Annualized_Volatility"] = annualized_vol

    risk_free_daily = 0.02 / 252
    excess_returns = df["Daily_Return"] - risk_free_daily
    sharpe = (
        excess_returns.mean()
        / excess_returns.std(ddof=0)
        * math.sqrt(252)
        if excess_returns.std(ddof=0) != 0
        else np.nan
    )
    metrics["Sharpe_Ratio"] = sharpe

    metrics["Return_Summary"] = {
        "avg_daily_return": df["Daily_Return"].mean(),
        "avg_log_return": df["Log_Return"].mean(),
        "last_cumulative_return": df["Cumulative_Return"].iloc[-1],
    }

    metrics["Correlation_Matrix"] = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].corr()

    metrics["Autocorrelation_Returns_lag1"] = df["Daily_Return"].autocorr(lag=1)
    metrics["Autocorrelation_Returns_lag5"] = df["Daily_Return"].autocorr(lag=5)
    return metrics


def ensure_output_dirs(output_dir: Path = DEFAULT_OUTPUT_DIR, 
                     plots_dir: Path = DEFAULT_PLOTS_DIR):
    """Create output directories"""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)


def _save_plot(fig, name: str, plots_dir: Path = DEFAULT_PLOTS_DIR):
    fig.tight_layout()
    fig.savefig(plots_dir / f"{name}.png", dpi=200)
    plt.close(fig)


def create_plots(df: pd.DataFrame, plots_dir: Path = DEFAULT_PLOTS_DIR):
    last_year = df.index >= (df.index.max() - pd.DateOffset(years=1))
    plot_df = df.loc[last_year]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Close"], label="Close", color="#1f77b4")
    ax.set_title("Closing Price")
    ax.set_ylabel("Price ($)")
    ax.legend()
    _save_plot(fig, "closing_price")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Close"], label="Close", color="#1f77b4")
    ax.plot(plot_df.index, plot_df["SMA_20"], label="SMA 20", color="#ff7f0e")
    ax.plot(plot_df.index, plot_df["SMA_50"], label="SMA 50", color="#2ca02c")
    ax.plot(plot_df.index, plot_df["EMA_20"], label="EMA 20", color="#d62728")
    ax.plot(plot_df.index, plot_df["EMA_50"], label="EMA 50", color="#9467bd")
    ax.set_title("Price with SMA and EMA Overlays")
    ax.set_ylabel("Price ($)")
    ax.legend()
    _save_plot(fig, "price_with_sma_ema")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(plot_df.index, plot_df["RSI_14"], label="RSI", color="#9467bd")
    ax.axhline(70, color="red", linestyle="--", linewidth=1)
    ax.axhline(30, color="green", linestyle="--", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_title("RSI (14)")
    ax.set_ylabel("RSI")
    ax.legend()
    _save_plot(fig, "rsi")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["MACD"], label="MACD", color="#1f77b4")
    ax.plot(plot_df.index, plot_df["MACD_Signal"], label="Signal", color="#ff7f0e")
    ax.bar(plot_df.index, plot_df["MACD_Hist"], label="Histogram", color="#2ca02c")
    ax.set_title("MACD")
    ax.set_ylabel("Value")
    ax.legend()
    _save_plot(fig, "macd")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(plot_df.index, plot_df["Volume"] / 1e6, color="#17becf")
    ax.set_title("Volume (Millions)")
    ax.set_ylabel("Volume (M)")
    _save_plot(fig, "volume")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Close"], label="Close", color="#1f77b4")
    ax.fill_between(
        plot_df.index,
        plot_df["BB_Lower"],
        plot_df["BB_Upper"],
        color="#c6dbef",
        alpha=0.4,
        label="Bollinger Bands",
    )
    ax.plot(plot_df.index, plot_df["BB_Middle"], label="BB Middle", color="#2ca02c")
    ax.set_title("Bollinger Bands")
    ax.set_ylabel("Price ($)")
    ax.legend()
    _save_plot(fig, "bollinger_bands")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(plot_df["Daily_Return"].dropna(), bins=30, color="#ff7f0e", alpha=0.7)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    _save_plot(fig, "daily_returns_hist")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Cumulative_Return"], color="#2ca02c")
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Cumulative Return")
    _save_plot(fig, "cumulative_returns")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Rolling_Volatility_30"], color="#d62728")
    ax.set_title("30-Day Rolling Volatility (Price)")
    ax.set_ylabel("Volatility")
    _save_plot(fig, "rolling_volatility")


def main(args: argparse.Namespace = None):
    """Main execution function with CLI support"""
    
    # Parse arguments if not provided
    if args is None:
        args = parse_arguments()
    
    # Set up paths
    OUTPUT_DIR = args.output_dir
    PLOTS_DIR = OUTPUT_DIR / "plots"
    
    # Resolve data path
    if args.data_path:
        DATA_PATH = args.data_path
    else:
        DATA_PATH = BASE_DIR / "data" / f"{args.ticker.upper()}.csv"
    
    ensure_output_dirs(OUTPUT_DIR, PLOTS_DIR)
    
    # Load data
    print(f"Loading data from {DATA_PATH}")
    df = load_and_clean(DATA_PATH)
    print(f"Loaded {len(df)} rows of data")
    
    # Filter by date range if specified
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
        df = df[df.index >= start_date]
        print(f"Filtered to {len(df)} rows from {args.start_date}")
    
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
        df = df[df.index <= end_date]
        print(f"Filtered to {len(df)} rows until {args.end_date}")
    
    # Validate data sufficiency
    validate_data_sufficiency(df)
    
    # Add technical indicators
    print(f"Adding technical indicators (SMA: {args.sma_window}, RSI: {args.rsi_window}, BB: {args.bb_window})")
    df = add_ta_indicators(df, sma_windows=args.sma_window, rsi_window=args.rsi_window, bb_window=args.bb_window)
    
    # Compute PyNance metrics
    print(f"Computing metrics (volatility window: {args.volatility_window})")
    metrics = compute_pynance_metrics(df, volatility_window=args.volatility_window)
    
    # Create plots
    if not args.no_plots:
        print("Generating plots...")
        create_plots(df, PLOTS_DIR)
    else:
        print("Skipping plots")
    
    # Save outputs
    snapshot_path = OUTPUT_DIR / "dataframe_snapshot.csv"
    df.tail(10).to_csv(snapshot_path)
    corr_path = OUTPUT_DIR / "correlation_matrix.csv"
    metrics["Correlation_Matrix"].to_csv(corr_path)
    
    summary = {
        "ticker": args.ticker,
        "data_rows": len(df),
        "date_range": f"{df.index.min()} to {df.index.max()}",
        "annualized_volatility": metrics["Annualized_Volatility"],
        "sharpe_ratio": metrics["Sharpe_Ratio"],
        "last_cumulative_return": metrics["Return_Summary"]["last_cumulative_return"],
        "autocorr_lag1": metrics["Autocorrelation_Returns_lag1"],
        "autocorr_lag5": metrics["Autocorrelation_Returns_lag5"],
        "sma_windows": args.sma_window,
        "rsi_window": args.rsi_window,
        "volatility_window": args.volatility_window,
        "snapshot_path": snapshot_path.as_posix(),
        "plots_dir": PLOTS_DIR.as_posix(),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    pd.Series(summary).to_json(summary_path, indent=2)
    print("Analysis complete. Summary saved to", summary_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

