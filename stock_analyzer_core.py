"""
stock_analyzer_core.py
Core engine for stock analysis (no email, no web).
We will import run_analysis() and run_market_scan() from our Flask app.
"""

import warnings, os, time
from datetime import timedelta

import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Use non-GUI backend
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

analysis_cache = {}
CACHE_TTL = 600  # 10 minutes

warnings.filterwarnings("ignore", message="YF.download")

TZ = "Asia/Kolkata"


def get_today_india():
    return pd.Timestamp.now(tz=pytz.timezone(TZ)).normalize()


def compute_obv(df):
    close = pd.to_numeric(df["Close"], errors="coerce").ffill()
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    obv = [0]
    for i in range(1, len(close)):
        prev_close = float(close.iloc[i - 1])
        curr_close = float(close.iloc[i])
        curr_vol = float(volume.iloc[i])
        if curr_close > prev_close:
            obv.append(obv[-1] + curr_vol)
        elif curr_close < prev_close:
            obv.append(obv[-1] - curr_vol)
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)


def cleanup_old_files(folder, hours=6):
    """Delete files older than `hours` in given folder."""
    if not os.path.isdir(folder):
        return  # nothing to clean yet

    cutoff = time.time() - (hours * 3600)

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
            try:
                os.remove(path)
            except:
                pass


def run_analysis(ticker="SOLARINDS.NS", show_plot=False):
    """
    Main analysis function.
    Returns: (df, summary_dict, csv_path)
    """
    # ---------- Cache check ----------
    now = time.time()
    if ticker in analysis_cache:
        cached_df, cached_summary, cached_path, timestamp = analysis_cache[ticker]
        if now - timestamp < CACHE_TTL:
            print(f"Using cached result for {ticker}")
            return cached_df, cached_summary, cached_path

    today = get_today_india().date()
    start_7d = today - timedelta(days=6)
    fetch_start = start_7d - timedelta(days=180)  # 6 months history
    fetch_end = today

    print(f"Fetching {ticker} data from {fetch_start} to {fetch_end} ({TZ}) ...")
    df = yf.download(
        ticker,
        start=str(fetch_start),
        end=str(fetch_end + timedelta(days=1)),
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Handle MultiIndex columns from yfinance if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] != "" else col[1] for col in df.columns]

    df = df[df.index.date <= today]

    # ------------- Indicators -------------
    df["VMA_5"] = df["Volume"].rolling(window=5, min_periods=1).mean()
    df["VMA_20"] = df["Volume"].rolling(window=20, min_periods=1).mean()
    df["OBV"] = compute_obv(df)

    df["Price_Change_%"] = df["Close"].pct_change() * 100
    df["Volume_Change_%"] = df["Volume"].pct_change() * 100
    df["Rel_Volume"] = df["Volume"] / df["VMA_20"]
    df["Strength_Score"] = np.sign(df["Price_Change_%"]) * df["Rel_Volume"]
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["VWAP_Diff_%"] = (df["Close"] - df["VWAP"]) / df["VWAP"] * 100

    # ------------- Spike detection (kept for completeness) -------------
    df["Spike_ruleA"] = df["Volume"] > 1.5 * df["VMA_20"]
    roll_mean = df["Volume"].rolling(window=60, min_periods=10).mean()
    roll_std = df["Volume"].rolling(window=60, min_periods=10).std()
    df["Spike_ruleB"] = df["Volume"] > (roll_mean + 2 * roll_std)

    if len(df) > 30:
        iso = IsolationForest(contamination=0.03, random_state=42)
        df["IF_anomaly"] = iso.fit_predict(df[["Volume"]]) == -1
    else:
        df["IF_anomaly"] = False

    # ------------- Sanitize for ML -------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(
        subset=[
            "Price_Change_%", "Rel_Volume",
            "VWAP_Diff_%", "Strength_Score"
        ],
        inplace=True
    )

    # ------------- Predictive ML Model -------------
    df["Target_Up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["Price_Change_%", "Volume_Change_%", "Rel_Volume",
                "VWAP_Diff_%", "Strength_Score"]
    df = df.dropna(subset=features)

    prob_up = None
    if len(df) > 50:
        X = df[features]
        y = df["Target_Up"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        prob_up = float(model.predict_proba(X_test)[-1, 1])
        print(f"\nML-predicted probability of next-day rise: {prob_up:.2%}")
    else:
        print("\nNot enough data for predictive model.")

    # ------------- Optional plot for local testing -------------
    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["Close"], label="Close Price")
        plt.plot(df.index, df["VWAP"], label="VWAP", linestyle="--")
        plt.title(f"{ticker} - Price vs VWAP with Volume Intelligence")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------- Ensure folders + cleanup -------------
    reports_folder = "static/reports"
    plots_folder = "static/plots"
    os.makedirs(reports_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    cleanup_old_files(reports_folder, hours=6)
    cleanup_old_files(plots_folder, hours=6)

    # ------------- Save report -------------
    report_filename = f"intelligent_volume_report_{ticker.replace('.', '_')}_{today}.csv"
    report_path = os.path.join(reports_folder, report_filename)
    df.to_csv(report_path)

    # ------------- Build summary for UI -------------
    last_row = df.iloc[-1]
    strength_val = float(last_row["Strength_Score"])
    strength_direction = "Bullish" if strength_val > 0 else "Bearish"
    rel_vol = float(last_row["Rel_Volume"])
    vwap_diff = float(last_row["VWAP_Diff_%"])

    summary = {
        "ticker": ticker,
        "date": str(today),
        "strength_score": round(strength_val, 2),
        "strength_direction": strength_direction,
        "relative_volume": round(rel_vol, 2),
        "vwap_deviation_pct": round(vwap_diff, 2),
        "ml_prob_up": round(prob_up, 4) if prob_up is not None else None,
    }

    # ------------------- Generate Plots -------------------
    # Plot 1: Price vs VWAP
    price_plot_path = os.path.join(plots_folder, f"price_vwap_{ticker.replace('.', '_')}.png")

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["Close"], label="Close Price", linewidth=1.5)
    plt.plot(df.index, df["VWAP"], label="VWAP", linestyle="--", linewidth=1.2)
    plt.title(f"{ticker} — Price vs VWAP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(price_plot_path)
    plt.close()

    # Plot 2: Volume with spikes
    volume_plot_path = os.path.join(plots_folder, f"volume_spikes_{ticker.replace('.', '_')}.png")

    plt.figure(figsize=(10, 4))
    plt.bar(df.index, df["Volume"], label="Volume", width=0.6)

    spikes = df[df["IF_anomaly"] == True]
    if not spikes.empty:
        plt.bar(spikes.index, spikes["Volume"], color="red", label="Spike/Anomaly")

    plt.title(f"{ticker} — Volume & Spike Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(volume_plot_path)
    plt.close()

    # Add plot paths to summary (relative to /static)
    summary["price_plot"] = price_plot_path.replace("static/", "")
    summary["volume_plot"] = volume_plot_path.replace("static/", "")

    # ---------- Save to cache ----------
    analysis_cache[ticker] = (df, summary, report_path, time.time())

    return df, summary, report_path


# ======================================================================
# New: Market Scan for NIFTY 50 + Sensex 30
# - Uses full RandomForest pipeline (same quality as single-stock)
# - On-demand only when /market-scan is visited
# - Deduplicates STRICTLY by stock name (before the .NS / .BO)
#   and keeps the highest probability per stock (Method 1).
# ======================================================================

NIFTY50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS",
    "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
    "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
]

SENSEX30_TICKERS = [
    "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
    "BAJAJFINSV.NS", "BHARTIARTL.NS", "DRREDDY.NS", "HCLTECH.NS",
    "HDFC.NS", "HDFCBANK.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATAMOTORS.NS", "TATASTEEL.NS"
]


def run_market_scan():
    """
    Runs a full RandomForest-based scan across NIFTY 50 + Sensex 30.

    - Uses run_analysis() for each ticker (same quality as single-stock view).
    - Skips tickers where ML probability is None.
    - Deduplicates by *stock name only* (before the suffix like .NS / .BO)
      and keeps the entry with highest probability.
    - Returns two lists for Flask:
        bullish_rows: list of dicts with prob > 0.70
        bearish_rows: list of dicts with prob < 0.30
    Each dict: { 'name', 'ticker', 'prob' }
    """
    all_tickers = NIFTY50_TICKERS + SENSEX30_TICKERS

    raw_rows = []

    for ticker in all_tickers:
        try:
            _, summary, _ = run_analysis(ticker)
            prob_up = summary.get("ml_prob_up")

            # Skip if we don't have a valid probability
            if prob_up is None:
                continue

            base_name = ticker.split(".")[0].upper()

            raw_rows.append({
                "name": base_name,   # e.g. RELIANCE
                "ticker": ticker,    # e.g. RELIANCE.NS
                "prob": float(prob_up)
            })

        except Exception as e:
            # Avoid crashing scan for one bad ticker
            print(f"[SCAN] Error for {ticker}: {e}")
            continue

    # ---------------- Deduplicate by stock name (Method 1) ----------------
    best_by_name = {}
    for row in raw_rows:
        name = row["name"]
        if name not in best_by_name:
            best_by_name[name] = row
        else:
            # Keep the row with the higher probability
            if row["prob"] > best_by_name[name]["prob"]:
                best_by_name[name] = row

    unique_rows = list(best_by_name.values())

    # ---------------- Split into Bullish / Bearish ----------------
    bullish = [r for r in unique_rows if r["prob"] > 0.70]
    bearish = [r for r in unique_rows if r["prob"] < 0.30]

    # Sort: bullish highest → lowest, bearish lowest → highest
    bullish.sort(key=lambda r: r["prob"], reverse=True)
    bearish.sort(key=lambda r: r["prob"])

    # Add rank based on sorted order (1, 2, 3...)
    for idx, row in enumerate(bullish, start=1):
        row["rank"] = idx

    for idx, row in enumerate(bearish, start=1):
        row["rank"] = idx

    return bullish, bearish


if __name__ == "__main__":
    # Quick manual test (no GUI windows because Agg backend)
    df, summary, path = run_analysis("SOLARINDS.NS", show_plot=False)
    print("\nSummary for UI:")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nCSV saved at: {path}")

    # Optional: test market scan locally
    # bullish, bearish = run_market_scan()
    # print("\nBullish picks:")
    # for r in bullish:
    #     print(r)
    # print("\nBearish picks:")
    # for r in bearish:
    #     print(r)
