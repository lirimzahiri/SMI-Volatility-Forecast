import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz

# Global Settings
TZ = "Europe/Zurich"
DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_hourly_multivariate(ticker: str, days: int = 60, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetches 5-minute data and aggregates it into hourly multivariate features:
    - log_rv: Log of Realized Volatility (from 5m returns)
    - log_vol: Log of Total Volume
    - returns: Hourly Log Returns
    
    Returns a DataFrame indexed by hourly timestamp (end of hour).
    """
    cache_file = DATA_DIR / f"{ticker.replace('.', '_')}_hourly_multi.csv"
    
    # 1. Try Cache
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                last_ts = df.index[-1]
                if (datetime.now(pytz.timezone(TZ)) - last_ts.tz_convert(TZ)) < timedelta(minutes=65):
                    return df
        except Exception:
            pass # Cache invalid, ignore

    # 2. Fetch 5m Data (max 60 days for 5m in yfinance)
    try:
        raw = yf.download(ticker, interval="5m", period=f"{days}d", progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

    # Clean and localize
    # Handle MultiIndex if present (yfinance > 0.2.40 often returns MultiIndex even for single ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    
    raw = raw.rename(columns=str.lower)
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC").tz_convert(TZ)
    else:
        raw.index = raw.index.tz_convert(TZ)
    
    # Filter market hours (approximate, to avoid noise)
    raw = raw.between_time("09:00", "17:30")
    
    # 3. Compute 5m Log Returns for RV
    raw["log_ret_5m"] = np.log(raw["close"]).diff()
    
    # 4. Resample to 1 Hour
    # We want the aggregation to represent the "past hour".
    # Resample '1h' usually labels with the LEFT bin edge. We can adjust later.
    
    def calc_rv(x):
        # RV = sqrt(sum(r^2))
        return np.sqrt((x**2).sum())

    hourly = raw.resample("1h").agg({
        "log_ret_5m": calc_rv,      # Realized Volatility
        "volume": "sum",            # Total Volume
        "close": "last",            # Closing Price
        "open": "first"             # Opening Price
    })
    
    # Rename and clean
    hourly = hourly.rename(columns={"log_ret_5m": "rv"})
    hourly = hourly.dropna()
    
    # 5. Compute Features
    # Target: Log RV
    hourly["log_rv"] = np.log(hourly["rv"].replace(0, np.nan)).fillna(method='ffill')
    
    # Covariate: Log Volume
    hourly["log_vol"] = np.log(hourly["volume"].replace(0, np.nan)).fillna(0)
    
    # Covariate: Hourly Returns
    hourly["returns"] = np.log(hourly["close"] / hourly["open"])
    
    # Final Selection
    final_df = hourly[["log_rv", "log_vol", "returns"]].dropna()
    
    # Save Cache
    final_df.to_csv(cache_file)
    
    return final_df

def get_latest_market_data(ticker: str) -> dict:
    """Helper to get current price and change for UI."""
    try:
        t = yf.Ticker(ticker)
        # Fast info is often fastest
        fi = t.fast_info
        price = fi.last_price
        prev = fi.previous_close
        if price and prev:
            return {
                "price": price,
                "change_pct": (price - prev) / prev * 100.0,
                "status": "Live"
            }
    except Exception:
        pass
    return {"price": None, "change_pct": None, "status": "Error"}
