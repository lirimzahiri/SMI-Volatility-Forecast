# app.py — TimesFM 2.5 (PyTorch) + Gradio + yfinance Cache + robuste Past-RV
# Empfohlene Umgebung (Windows, Python 3.12):
#   pip install --upgrade pip
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
#   pip install gradio yfinance pandas numpy matplotlib pytz requests

import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import timesfm  # TimesFM 2.5 (GitHub-Version mit Torch-Backend)

# =========================
# Global settings / folders
# =========================
TZ = "Europe/Zurich"
DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# yfinance download (robust) + CSV cache
# ==========================================
def _yf_download(
    ticker: str,
    interval: str,
    period: str,
    auto_adjust: bool = True,
    max_retries: int = 3,
    sleep_sec: float = 2.0,
) -> pd.DataFrame:
    """
    Robust yfinance download with retries and backoff.
    IMPORTANT: do NOT pass a custom session (let yfinance manage curl_cffi internally).
    """
    last_err = None

    def _try(period_try: str):
        try:
            df = yf.download(
                ticker,
                interval=interval,
                period=period_try,
                auto_adjust=auto_adjust,
                progress=False,
                threads=True,
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:
            return e
        return None

    # Attempt with requested period
    for i in range(max_retries):
        res = _try(period)
        if isinstance(res, pd.DataFrame):
            return res
        if isinstance(res, Exception):
            last_err = res
        time.sleep(sleep_sec * (i + 1))

    # Fallback to shorter periods if blocked
    for fb in ["30d", "7d"]:
        res = _try(fb)
        if isinstance(res, pd.DataFrame):
            return res
        if isinstance(res, Exception):
            last_err = res

    raise RuntimeError(f"yfinance empty for {ticker} ({interval}, {period}). Last error: {last_err}")


def fetch_intraday_cached(
    ticker: str,
    interval: str = "5m",
    period: str = "60d",
    tz: str = TZ,
    auto_adjust: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Pull intraday data, cache to data/{TICKER}_{INTERVAL}.csv, and merge new rows on subsequent runs.
    Falls back to cache when fresh download fails.
    """
    cache_fp = DATA_DIR / f"{ticker.replace('.', '_')}_{interval}.csv"

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(tz)
        else:
            df.index = df.index.tz_convert(tz)
        return df.rename(columns=str.lower)[["close"]].dropna()

    fresh = None
    err = None
    try:
        fresh = _yf_download(ticker, interval=interval, period=period, auto_adjust=auto_adjust)
        fresh = _normalize(fresh)
    except Exception as e:
        err = e

    if use_cache and cache_fp.exists():
        cached = pd.read_csv(cache_fp)
        tcol = next((c for c in ["Datetime", "datetime", "Date", "date", "Time", "time"] if c in cached.columns), None) or cached.columns[0]
        cached[tcol] = pd.to_datetime(cached[tcol], utc=True, errors="coerce")
        cached = cached.dropna(subset=[tcol]).set_index(tcol).sort_index()
        cached.index = cached.index.tz_convert(TZ)
        ccol = next((c for c in ["close", "Close", "Adj Close", "adj_close", "adj close"] if c in cached.columns), None) or cached.columns[-1]
        cached = cached[[ccol]].rename(columns={ccol: "close"}).dropna()

        if fresh is None:
            if cached.empty:
                raise RuntimeError(f"Cache empty and download failed: {err}")
            return cached

        merged = pd.concat([cached, fresh[~fresh.index.isin(cached.index)]], axis=0).sort_index()
        merged = merged[merged.index >= (merged.index.max() - pd.Timedelta(days=120))]
        merged.rename_axis("Datetime").reset_index().to_csv(cache_fp, index=False)
        return merged

    if fresh is None:
        raise RuntimeError(f"Download failed: {err}")

    if use_cache:
        fresh.rename_axis("Datetime").reset_index().to_csv(cache_fp, index=False)
    return fresh


def load_from_csv(csv_path: str, tz: str = TZ) -> pd.DataFrame:
    """
    Load from arbitrary CSV with a datetime column + close/adj close column.
    """
    df = pd.read_csv(csv_path)
    tcol = next((c for c in ["Datetime", "datetime", "Date", "date", "Time", "time"] if c in df.columns), None) or df.columns[0]
    ccol = next((c for c in ["close", "Close", "Adj Close", "adj_close", "adj close"] if c in df.columns), None)
    if ccol is None:
        raise ValueError("CSV needs a Close/Adj Close column.")
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    df.index = df.index.tz_convert(tz)
    return df[[ccol]].rename(columns={ccol: "close"}).dropna()


# ==========================================
# Target: Past Realized Volatility (RV) — robust, pro Handelstag
# ==========================================
def realized_vol_series_past(close_df: pd.DataFrame, step_minutes: int = 5, horizon_minutes: int = 60) -> pd.Series:
    """
    Past-RV innerhalb der SIX-Handelszeiten (09:00–17:30 CET).
    - Gleichmäßiger step_minutes-Raster je Tag (resample last + ffill(limit=2))
    - Rolling NUR innerhalb des Tages (kein Overnight-Gap in Fenstern)
    - Fallback auf globale Rolling-Variante, falls Session-RV leer
    """
    if "close" not in close_df.columns:
        raise ValueError("close_df must contain 'close'")

    # 1) sortieren + Duplikate entfernen
    df = close_df.sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    # 2) Nur Handelszeiten (SIX)
    df = df.tz_convert(TZ).between_time("09:00", "17:30")

    if df.empty:
        return pd.Series(dtype=float)

    # 3) Takt prüfen und ggf. resamplen
    s = df.index.to_series().diff().dropna()
    need_resample = (len(s) == 0) or (abs(s.median() - pd.Timedelta(minutes=step_minutes)) > pd.Timedelta(seconds=60))
    if need_resample:
        df = df.resample(f"{step_minutes}min").last()

    # Kleine Lücken füllen (nicht über Nacht, da pro Tag betrachtet)
    df["close"] = df["close"].ffill(limit=2)
    df = df.dropna(subset=["close"])

    # 4) Log-Returns
    r = np.log(df["close"]).diff()
    k = max(1, horizon_minutes // step_minutes)

    # 5) Rolling NUR innerhalb des Tages
    day_key = df.index.date
    rv = (
        r.groupby(day_key)
         .apply(lambda x: x.pow(2).rolling(window=k, min_periods=k).sum().pow(0.5))
         .reset_index(level=0, drop=True)
         .dropna()
    )

    # 6) Fallback, falls leer
    if rv.empty:
        r2 = np.log(df["close"]).diff()
        rv = r2.pow(2).rolling(window=k, min_periods=k).sum().pow(0.5).dropna()

    # Debug
    try:
        print(f"[DEBUG] RV(len)={len(rv)}, k={k}, step={step_minutes}m, need_resample={need_resample}")
    except Exception:
        pass

    return rv


def to_log_series(s: pd.Series) -> pd.Series:
    return np.log(s.replace(0, np.nan)).dropna()


# ==========================================
# TimesFM 2.5 (PyTorch) — load & forecast
# ==========================================
def load_timesfm25_torch(max_context: int = 256, max_horizon: int = 1, torch_compile: bool = False):
    """
    Load TimesFM 2.5 (200M, PyTorch) and compile with a ForecastConfig.
    """
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=torch_compile,
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=int(max_context),
            max_horizon=int(max_horizon),
            normalize_inputs=True,
            use_continuous_quantile_head=False,  # set True if you want quantiles
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def forecast_logrv_one_step_v25(logrv: pd.Series, model, context_len: int = 256):
    """
    One-step point forecast on log(RV) using TimesFM 2.5 (PyTorch).
    Returns (timestamp_of_forecast, logRV_hat_float)
    """
    if len(logrv) < context_len:
        raise ValueError(f"Not enough points ({len(logrv)}) for context_len={context_len}.")
    ctx = logrv.values[-context_len:].astype(np.float32)
    point_fcst, _q = model.forecast(horizon=1, inputs=[ctx])  # shapes: (1, 1), (1, 1, 10)
    logrv_hat = float(point_fcst.squeeze())
    ts_pred = pd.Timestamp(logrv.index[-1])
    return ts_pred, logrv_hat


# ==========================================
# CLI pipeline (per ticker)
# ==========================================
def run_pipeline(
    ticker: Optional[str] = None,
    csv_path: Optional[str] = None,
    interval: str = "5m",
    period: str = "60d",
    step_minutes: int = 5,
    horizons_min: Tuple[int, int] = (60, 360),
    context_len: int = 96,
    use_cache: bool = True,
    plot: bool = True,
) -> pd.DataFrame:
    assert (ticker is not None) ^ (csv_path is not None), "Provide either ticker OR csv_path."

    # Load data
    if csv_path:
        df = load_from_csv(csv_path, tz=TZ)
        name = Path(csv_path).stem
    else:
        df = fetch_intraday_cached(ticker=ticker, interval=interval, period=period, tz=TZ, use_cache=use_cache)
        name = ticker

    print(f"\nData: {name} | Points: {len(df)} | Range: {df.index.min()} .. {df.index.max()}")
    print(f"Interval={interval}, Period={period}, Step={step_minutes}m")

    # Load model (v2.5 torch)
    model = load_timesfm25_torch(max_context=context_len, max_horizon=1, torch_compile=False)

    rows = []
    for H in horizons_min:
        rv = realized_vol_series_past(df, step_minutes=step_minutes, horizon_minutes=H)
        logrv = to_log_series(rv)
        if len(logrv) < context_len:
            print(f"⚠️  Not enough history for context {context_len} @ {H} min (only {len(logrv)})")
            continue

        ts_pred, logrv_hat = forecast_logrv_one_step_v25(logrv, model, context_len=context_len)
        rv_hat = float(np.exp(logrv_hat))
        rv_last = float(np.exp(logrv.iloc[-1]))

        rows.append({"horizon_min": H, "time_pred": ts_pred, "rv_hat": rv_hat, "rv_last": rv_last})
        print(f"🟩 {name} — RV(next {H:>3} min) @ {ts_pred}: {rv_hat:.6f}  |  last RV: {rv_last:.6f}  | backend=torch v2.5")

        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(logrv.index[-context_len:], logrv.values[-context_len:], label="log(RV) context")
            plt.scatter([ts_pred], [logrv_hat], marker="x", s=80, label="Forecast log(RV)")
            plt.title(f"{name} — Forecast log(RV) next {H} min (context={context_len}, backend=torch v2.5)")
            plt.xlabel("Time"); plt.ylabel("log(RV)"); plt.legend(); plt.tight_layout()
            plt.show()

    return pd.DataFrame(rows)


# ==========================================
# Gradio UI
# ==========================================
def build_ui():
    import gradio as gr

    DEFAULT_TICKER = "NESN.SW"
    ALLOWED_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
    ALLOWED_PERIODS = ["7d", "30d", "60d", "90d", "180d", "730d"]
    HORIZONS = [60, 360]

    with gr.Blocks(title="TimesFM 2.5 — SMI Volatility (local)") as demo:
        gr.Markdown("## TimesFM 2.5 — SMI Volatility Forecast\n*Past realized volatility (RV) series → 1-step forecast of **next** 60 min / 6 h.*")

        with gr.Row():
            ticker = gr.Textbox(label="Ticker (e.g., NESN.SW, NOVN.SW, ROG.SW)", value=DEFAULT_TICKER)
            csv = gr.Textbox(label="CSV path (optional; overrides ticker)", value="")

        with gr.Row():
            interval = gr.Dropdown(choices=ALLOWED_INTERVALS, value="5m", label="Yahoo interval")
            period = gr.Dropdown(choices=ALLOWED_PERIODS, value="60d", label="Yahoo period")
            step = gr.Dropdown(choices=[1, 5], value=5, label="Step for RV (minutes)")
            horizons = gr.CheckboxGroup(choices=HORIZONS, value=HORIZONS, label="Horizons (minutes)")

        with gr.Row():
            context = gr.Slider(64, 1024, value=96, step=16, label="Context length (TimesFM 2.5)")
            use_cache = gr.Checkbox(value=True, label="Use cache (./data)")

        run_btn = gr.Button("Forecast")
        status = gr.Textbox(label="Status / Results", lines=12)
        plot = gr.Plot(label="log(RV) context & forecast (first selected horizon)")

        def _run_ui(tic, csv_path, interval_sel, period_sel, step_min, horizons_list, ctx, cache_on):
            try:
                if csv_path.strip():
                    df = load_from_csv(csv_path.strip(), tz=TZ)
                    name = Path(csv_path.strip()).stem
                else:
                    df = fetch_intraday_cached(
                        ticker=tic.strip(),
                        interval=interval_sel,
                        period=period_sel,
                        tz=TZ,
                        use_cache=bool(cache_on),
                    )
                    name = tic.strip()

                model = load_timesfm25_torch(max_context=int(ctx), max_horizon=1, torch_compile=False)

                lines: List[str] = []
                plot_fig = None

                for H in horizons_list:
                    rv = realized_vol_series_past(df, step_minutes=int(step_min), horizon_minutes=int(H))
                    logrv = to_log_series(rv)
                    if len(logrv) < int(ctx):
                        lines.append(f"⚠️  Not enough history for context {ctx} @ {H} min (only {len(logrv)})")
                        continue

                    ts_pred, logrv_hat = forecast_logrv_one_step_v25(logrv, model, context_len=int(ctx))
                    rv_hat = float(np.exp(logrv_hat))
                    rv_last = float(np.exp(logrv.iloc[-1]))
                    lines.append(f"🟩 {name} — RV(next {int(H):>3} min) @ {ts_pred}: {rv_hat:.6f}  |  last RV: {rv_last:.6f}  | backend=torch v2.5")

                # simple plot for first selected horizon
                if horizons_list:
                    H0 = int(horizons_list[0])
                    rv0 = realized_vol_series_past(df, step_minutes=int(step_min), horizon_minutes=H0)
                    logrv0 = to_log_series(rv0)
                    fig = plt.figure(figsize=(10, 4))
                    plt.plot(logrv0.index[-int(ctx):], logrv0.values[-int(ctx):], label="log(RV) context")
                    if len(logrv0) >= int(ctx):
                        ts_pred0, logrv_hat0 = forecast_logrv_one_step_v25(logrv0, model, context_len=int(ctx))
                        plt.scatter([ts_pred0], [logrv_hat0], marker="x", s=80, label="Forecast log(RV)")
                    plt.title(f"{name} — log(RV) (H={H0} min, context={int(ctx)}, backend=torch v2.5)")
                    plt.xlabel("Time"); plt.ylabel("log(RV)"); plt.legend(); plt.tight_layout()
                    plot_fig = fig

                head = f"Data: {name} | Points: {len(df)} | Range: {df.index.min()} .. {df.index.max()}\nInterval={interval_sel}, Period={period_sel}, Step={step_min}m"
                return head + "\n" + "\n".join(lines), plot_fig

            except Exception as e:
                return f"❌ Error: {e}", None

        run_btn.click(
            _run_ui,
            inputs=[ticker, csv, interval, period, step, horizons, context, use_cache],
            outputs=[status, plot],
        )

    return demo


# ==========================================
# CLI entry
# ==========================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="TimesFM 2.5 — SMI volatility (local)")
    parser.add_argument("--ticker", type=str, default="NESN.SW", help="Yahoo ticker, e.g., NESN.SW")
    parser.add_argument("--csv", type=str, default="", help="Alternative: CSV path instead of Yahoo")
    parser.add_argument("--interval", type=str, default="5m", choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"])
    parser.add_argument("--period", type=str, default="60d", help="Yahoo period: 7d, 30d, 60d, 730d ...")
    parser.add_argument("--step", type=int, default=5, choices=[1, 5], help="Step minutes for RV")
    parser.add_argument("--context", type=int, default=96, help="Context length for TimesFM 2.5")
    parser.add_argument("--no-cache", action="store_true", help="Disable local cache")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI")
    args = parser.parse_args()

    if args.ui:
        demo = build_ui()
        demo.launch(server_name="127.0.0.1", server_port=7860)
    else:
        if args.csv.strip():
            run_pipeline(
                ticker=None,
                csv_path=args.csv.strip(),
                interval=args.interval,
                period=args.period,
                step_minutes=args.step,
                context_len=args.context,
                use_cache=not args.no_cache,
                plot=True,
            )
        else:
            run_pipeline(
                ticker=args.ticker,
                csv_path=None,
                interval=args.interval,
                period=args.period,
                step_minutes=args.step,
                context_len=args.context,
                use_cache=not args.no_cache,
                plot=True,
            )


if __name__ == "__main__":
    main()
