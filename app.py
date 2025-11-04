# app.py — SMI 60-min Volatility (Variante B, Ampel, robuste Fallbacks + Delayed-Status)
# Plot: letzte X Stunden + Streuung (Quantile) farbig
# Dashboard: zusätzlich σ60m in %
#
# Setup (Windows, Python 3.12 empfohlen):
#   pip install --upgrade pip
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
#   pip install gradio yfinance pandas numpy matplotlib pytz requests

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import timesfm  # TimesFM 2.5 (Torch backend; HF/Git)

# ---------------- Global ----------------
TZ = "Europe/Zurich"
ZRH = pytz.timezone(TZ)
DATA_DIR = Path("./data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

# Plotfenster (letzte X Stunden)
PLOT_WINDOW_HOURS = 4

SMI_TICKERS = [
    "NESN.SW","NOVN.SW","ROG.SW","ABBN.SW","SIKA.SW","UBSG.SW","ZURN.SW",
    "GIVN.SW","CFR.SW","UHR.SW","SCMN.SW","ADEN.SW","SGSN.SW"
]
SMI_NAMES: Dict[str, str] = {
    "NESN.SW": "Nestlé", "NOVN.SW": "Novartis", "ROG.SW": "Roche", "ABBN.SW": "ABB",
    "SIKA.SW": "Sika", "UBSG.SW": "UBS", "ZURN.SW": "Zurich", "GIVN.SW": "Givaudan",
    "CFR.SW": "Richemont", "UHR.SW": "Swatch", "SCMN.SW": "Swisscom", "ADEN.SW": "Adecco", "SGSN.SW": "SGS",
}

# ---- Yahoo limits ----
MAX_PERIOD_BY_INTERVAL = {"1m":"7d","2m":"60d","5m":"60d","15m":"730d","30m":"730d","60m":"730d","90m":"730d","1h":"730d"}
FALLBACK_PERIODS = ["60d","30d","7d"]

def _clamp_period(interval: str, period: str) -> str:
    limit = MAX_PERIOD_BY_INTERVAL.get(interval, "60d")
    def _days(p):
        if p.endswith("d"): return int(p[:-1])
        if p.endswith("y"): return int(p[:-1]) * 365
        return 9999
    return period if _days(period) <= _days(limit) else limit

# ---------------- Market status ----------------
def is_business_day(dt: datetime) -> bool: return dt.weekday() < 5
def next_open_datetime(now: datetime) -> datetime:
    d=0
    while True:
        d+=1
        cand=(now+timedelta(days=d)).replace(hour=9,minute=0,second=0,microsecond=0)
        if is_business_day(cand): return cand

def market_status_now() -> Tuple[bool, Optional[datetime], Optional[datetime]]:
    now = datetime.now(ZRH)
    open_t  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    close_t = now.replace(hour=17, minute=30, second=0, microsecond=0)
    if not is_business_day(now): return False, None, next_open_datetime(now)
    if now < open_t:            return False, None, open_t
    if now > close_t:           return False, close_t, next_open_datetime(now)
    return True, close_t, None

# ---------------- yfinance + Cache ----------------
def _yf_download_strict(ticker: str, interval="5m", period="60d", auto_adjust=True) -> pd.DataFrame:
    period = _clamp_period(interval, period)
    ladder = [period] + [p for p in FALLBACK_PERIODS if _clamp_period(interval,p)==p and p!=period]
    last_err=None
    for p in ladder:
        try:
            df = yf.download(ticker, interval=interval, period=p, auto_adjust=auto_adjust,
                             progress=False, threads=True)
            if df is not None and not df.empty:
                df.attrs["yf_period_used"]=p
                return df
        except Exception as e:
            last_err=e
    raise RuntimeError(f"yfinance empty for {ticker} (interval={interval}, tried={ladder}). Last error: {last_err}")

def _norm_close(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC").tz_convert(tz)
    else: idx = idx.tz_convert(tz)
    out = df.copy(); out.index = idx
    return out.rename(columns=str.lower)[["close"]].dropna()

def fetch_intraday_cached(ticker: str, interval="5m", period="60d",
                          tz: str=TZ, use_cache: bool=True,
                          min_fresh_minutes: int = 10,
                          force_no_cache: bool = False) -> pd.DataFrame:
    """
    Holt Intraday-Daten; während Markt offen wird der Cache verworfen,
    wenn die letzte Zeitmarke älter als min_fresh_minutes ist.
    """
    cache_fp = DATA_DIR / f"{ticker.replace('.','_')}_{interval}.csv"

    def _read_cache() -> Optional[pd.DataFrame]:
        if not cache_fp.exists(): return None
        cached = pd.read_csv(cache_fp)
        tcol = next((c for c in ["Datetime","datetime","Date","date","Time","time"] if c in cached.columns), cached.columns[0])
        cached[tcol] = pd.to_datetime(cached[tcol], utc=True, errors="coerce")
        cached = cached.dropna(subset=[tcol]).set_index(tcol).sort_index()
        cached.index = cached.index.tz_convert(tz)
        ccol = next((c for c in ["close","Close","Adj Close","adj_close","adj close"] if c in cached.columns), cached.columns[-1])
        return cached[[ccol]].rename(columns={ccol:"close"}).dropna()

    # optional: Cache skippen
    if force_no_cache:
        fresh = _norm_close(_yf_download_strict(ticker, interval=interval, period=period), tz)
        fresh.rename_axis("Datetime").reset_index().to_csv(cache_fp, index=False)
        return fresh

    # Cache lesen
    cached = _read_cache() if use_cache else None

    # Wenn Markt offen: ist Cache frisch genug?
    open_now, _, _ = market_status_now()
    if open_now and cached is not None and len(cached) > 0:
        age = datetime.now(ZRH) - cached.index.max()
        if age > timedelta(minutes=min_fresh_minutes):
            cached = None  # erzwinge frische Ladung

    # Fresh laden
    fresh, err = None, None
    try:
        fresh = _norm_close(_yf_download_strict(ticker, interval=interval, period=period), tz)
    except Exception as e:
        err = e

    # Mergen/Fallback
    if cached is not None and (fresh is None or fresh.empty):
        return cached
    if cached is None and (fresh is not None and not fresh.empty):
        if use_cache:
            fresh.rename_axis("Datetime").reset_index().to_csv(cache_fp, index=False)
        return fresh
    if cached is not None and fresh is not None and not fresh.empty:
        merged = pd.concat([cached, fresh[~fresh.index.isin(cached.index)]], axis=0).sort_index()
        merged = merged[merged.index >= (merged.index.max()-pd.Timedelta(days=180))]
        merged.rename_axis("Datetime").reset_index().to_csv(cache_fp, index=False)
        return merged

    raise RuntimeError(f"Download failed: {err}")

# ---------------- RV Kernfunktionen ----------------
def _prep_base(close_df: pd.DataFrame) -> pd.DataFrame:
    df = close_df.sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["close"])

def _maybe_resample(df: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    diffs = df.index.to_series().diff().dropna()
    need_resample = (len(diffs)==0) or (abs(diffs.median()-pd.Timedelta(minutes=step_minutes))>pd.Timedelta(seconds=60))
    return df.resample(f"{step_minutes}min").last() if need_resample else df

def realized_vol_series(df_in: pd.DataFrame, step_minutes=5, horizon_minutes=60,
                        session_only=True, ffill_limit=12, min_periods_fac: float=0.75) -> pd.Series:
    df = _prep_base(df_in)
    if session_only:
        df = df.tz_convert(TZ).between_time("09:00","17:30")
        if df.empty: return pd.Series(dtype=float)
    df = _maybe_resample(df, step_minutes)
    df["close"] = df["close"].ffill(limit=ffill_limit)
    df = df.dropna(subset=["close"])
    r = np.log(df["close"]).diff()
    k = max(1, horizon_minutes // step_minutes)
    minp = max(2, int(round(k * min_periods_fac)))
    if session_only:
        day_key = df.index.date
        rv = (r.groupby(day_key)
                .apply(lambda x: x.pow(2).rolling(window=k, min_periods=minp).sum().pow(0.5))
                .reset_index(level=0, drop=True)
                .dropna())
    else:
        rv = r.pow(2).rolling(window=k, min_periods=minp).sum().pow(0.5).dropna()
    return rv

def realized_vol_series_std_fallback(df_in: pd.DataFrame, step_minutes=5, horizon_minutes=60,
                                     session_only=False, ffill_limit: int=24, min_periods_fac: float=0.6) -> pd.Series:
    df = _prep_base(df_in)
    if session_only:
        df = df.tz_convert(TZ).between_time("09:00","17:30")
        if df.empty: return pd.Series(dtype=float)
    df = _maybe_resample(df, step_minutes)
    df["close"] = df["close"].ffill(limit=ffill_limit)
    df = df.dropna(subset=["close"])
    r = np.log(df["close"]).diff()
    k = max(1, horizon_minutes // step_minutes)
    minp = max(2, int(round(k * min_periods_fac)))
    rv = r.rolling(window=k, min_periods=minp).std(ddof=1) * np.sqrt(k)
    return rv.replace([np.inf,-np.inf],np.nan).dropna()

# -------- Ultra-Regrid-Fallback --------
def regrid_logrv_ultra(df_in: pd.DataFrame, step_minutes=5, horizon_minutes=60,
                       session_only: bool=False, ffill_limit:int=24) -> pd.Series:
    df = _prep_base(df_in)
    if df.empty: return pd.Series(dtype=float)
    if session_only:
        df = df.tz_convert(TZ).between_time("09:00","17:30")
        if df.empty: return pd.Series(dtype=float)
    start = df.index.min().floor(f"{step_minutes}min")
    end   = df.index.max().ceil(f"{step_minutes}min")
    full_index = pd.date_range(start, end, freq=f"{step_minutes}min", tz=df.index.tz)
    df = df.reindex(full_index)
    df["close"] = df["close"].ffill(limit=ffill_limit)
    df = df.dropna(subset=["close"])
    if df.empty: return pd.Series(dtype=float)
    r = np.log(df["close"]).diff()
    k = max(1, horizon_minutes // step_minutes)
    rv = r.pow(2).rolling(window=k, min_periods=max(2, k//2)).sum().pow(0.5).dropna()
    return rv

# -------- Daily-Proxy --------
def proxy_sigma_60m_series_from_daily(ticker: str, days:int=120) -> Tuple[pd.Series, Optional[float]]:
    try:
        df = yf.download(ticker, interval="1d", period=f"{days}d", auto_adjust=True, progress=False, threads=True)
        if df is None or df.empty: return pd.Series(dtype=float), None
        ret = np.log(df["Close"]).diff().dropna()
        if ret.empty: return pd.Series(dtype=float), None
        scale = 1.0 / np.sqrt(8.5)  # 60min von ~8.5h
        sigma60_series = ret.abs() * scale
        proxy = float(sigma60_series.iloc[-1]) if len(sigma60_series)>0 else None
        sigma60_series.index = pd.to_datetime(sigma60_series.index).tz_localize("UTC").tz_convert(TZ)
        return sigma60_series, proxy
    except Exception:
        return pd.Series(dtype=float), None

# ---------------- Auto-Compute logRV (mit erweiterten Fallbacks) ----------------
def compute_logrv_auto(df: pd.DataFrame, horizons_minutes=60) -> Tuple[pd.Series, dict]:
    tries = []
    for step in [5, 15, 30]:
        k = max(1, horizons_minutes // step)

        rv = realized_vol_series(df, step_minutes=step, horizon_minutes=horizons_minutes,
                                 session_only=True, ffill_limit=12, min_periods_fac=0.75)
        logrv = np.log(pd.Series(rv).replace([np.inf, -np.inf, 0], np.nan)).dropna()
        tries.append((f"{step}m-session", len(logrv)))
        if len(logrv) >= 16:
            return logrv, {"step": step, "mode": "session-soft", "k": k, "tries": tries}

        rv = realized_vol_series(df, step_minutes=step, horizon_minutes=horizons_minutes,
                                 session_only=False, ffill_limit=12, min_periods_fac=0.75)
        logrv = np.log(pd.Series(rv).replace([np.inf, -np.inf, 0], np.nan)).dropna()
        tries.append((f"{step}m-global", len(logrv)))
        if len(logrv) >= 16:
            return logrv, {"step": step, "mode": "global", "k": k, "tries": tries}

        rv = realized_vol_series_std_fallback(df, step_minutes=step, horizon_minutes=horizons_minutes,
                                              session_only=False, ffill_limit=24, min_periods_fac=0.6)
        logrv = np.log(pd.Series(rv).replace([np.inf, -np.inf, 0], np.nan)).dropna()
        tries.append((f"{step}m-stdFallback", len(logrv)))
        if len(logrv) >= 16:
            return logrv, {"step": step, "mode": "std-fallback", "k": k, "tries": tries}

        rv = regrid_logrv_ultra(df, step_minutes=step, horizon_minutes=horizons_minutes,
                                session_only=True, ffill_limit=24)
        logrv = np.log(pd.Series(rv).replace([np.inf,-np.inf,0], np.nan)).dropna()
        tries.append((f"{step}m-ultraRegridSession", len(logrv)))
        if len(logrv) >= 16:
            return logrv, {"step": step, "mode": "ultra-regrid-session", "k": k, "tries": tries}

        rv = regrid_logrv_ultra(df, step_minutes=step, horizon_minutes=horizons_minutes,
                                session_only=False, ffill_limit=24)
        logrv = np.log(pd.Series(rv).replace([np.inf,-np.inf,0], np.nan)).dropna()
        tries.append((f"{step}m-ultraRegrid", len(logrv)))
        if len(logrv) >= 16:
            return logrv, {"step": step, "mode": "ultra-regrid", "k": k, "tries": tries}

    meta = {"step": None, "mode": "none", "k": None, "tries": tries}
    return pd.Series(dtype=float), meta

# -------- Chart-spezifisch --------
def logrv_for_chart(df: pd.DataFrame, horizon=60):
    rv = realized_vol_series(df, step_minutes=5, horizon_minutes=horizon,
                             session_only=True, ffill_limit=12, min_periods_fac=0.75)
    logrv = np.log(rv.replace([np.inf,-np.inf,0], np.nan)).dropna()
    if len(logrv) >= 16:
        return logrv, {"mode":"session-soft", "step":5}
    rv2 = regrid_logrv_ultra(df, step_minutes=5, horizon_minutes=horizon,
                             session_only=True, ffill_limit=12)
    logrv2 = np.log(rv2.replace([np.inf,-np.inf,0], np.nan)).dropna()
    if len(logrv2) >= 16:
        return logrv2, {"mode":"ultra-regrid-session", "step":5}
    return pd.Series(dtype=float), {"mode":"none", "step":None}

# ---------------- TimesFM (Torch) ----------------
def load_timesfm25_torch(max_context: int=512, quantiles: bool=True):
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False
    )
    model.compile(timesfm.ForecastConfig(
        max_context=int(max_context), max_horizon=1,
        normalize_inputs=True,
        use_continuous_quantile_head=bool(quantiles),
        force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
    ))
    return model

def forecast_logrv_one_step(logrv: pd.Series, model, desired_ctx:int=128):
    ctx = max(16, min(int(desired_ctx), len(logrv)))
    arr = logrv.values[-ctx:].astype(np.float32)
    point_fcst, q_fcst = model.forecast(horizon=1, inputs=[arr])
    ts_pred = pd.Timestamp(logrv.index[-1])
    y_hat = float(point_fcst.squeeze())
    q = q_fcst.squeeze() if q_fcst is not None else None
    return ts_pred, y_hat, q, ctx

# ---------------- Preis & Tages-% (robust) ----------------
_FASTINFO_KEYS = [
    ("regular_market_price", "previous_close"),
    ("last_price", "previous_close"),
    ("last_price", "prev_close"),
]

def _daily_last_and_prev(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        d = yf.Ticker(ticker).history(period="6d", interval="1d", prepost=False, auto_adjust=False)
        if isinstance(d, pd.DataFrame) and len(d) >= 2:
            close = d["Close"].dropna()
            return float(close.iloc[-1]), float(close.iloc[-2])
    except Exception:
        pass
    return None, None

def latest_price_and_change(_: pd.DataFrame, ticker: Optional[str]=None) -> Tuple[Optional[float], Optional[float], str]:
    """
    Reihenfolge:
      1) fast_info (mehrere Key-Varianten)
      2) Intraday 1m -> Daily prev_close
      3) Intraday 5m -> Daily prev_close
      4) Daily (zeigt Schlusskurs gestern, change='—')
    Rückgabe: (price, pct_change, label)
    """
    label = ""
    if ticker:
        # 1) fast_info
        try:
            tk = yf.Ticker(ticker)
            fi = getattr(tk, "fast_info", {}) or {}
            for k_last, k_prev in _FASTINFO_KEYS:
                last = fi.get(k_last, None)
                prev = fi.get(k_prev, None)
                if last is not None:
                    last = float(last)
                    if prev is not None and float(prev) > 0:
                        return last, (last - float(prev)) / float(prev) * 100.0, "Live/Real-time (fast_info)"
                    return last, None, "Last price (fast_info)"
        except Exception:
            pass

        # 2/3) Intraday
        for intr in ("1m", "5m"):
            try:
                h = yf.Ticker(ticker).history(period="2d", interval=intr, prepost=False, auto_adjust=False)
                if isinstance(h, pd.DataFrame) and len(h) > 0:
                    last = float(h["Close"].dropna().iloc[-1])
                    _, prev = _daily_last_and_prev(ticker)
                    if prev is not None and prev > 0:
                        return last, (last - prev) / prev * 100.0, f"Intraday {intr}"
                    return last, None, f"Intraday {intr}"
            except Exception:
                continue

        # 4) Daily Fallback → Schlusskurs gestern
        last, prev = _daily_last_and_prev(ticker)
        if last is not None:
            return last, None, "Schluss gestern (Daily)"

    return None, None, ""

def sigma_to_pct(sigma: float) -> Optional[float]:
    if sigma is None or not np.isfinite(sigma): return None
    return float(np.expm1(sigma) * 100.0)  # ≈ %-Bewegung für 1σ in 60 Min

def classify_vola_text(rv_series: pd.Series, rv_hat: float) -> Tuple[str, str]:
    s = pd.Series(rv_series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        if rv_hat is None or not np.isfinite(rv_hat):
            return "Unbekannt", "Es liegen derzeit nicht genug Daten vor."
        return "Normal", "Erwartete marktübliche Schwankungen."
    if rv_hat is None or not np.isfinite(rv_hat): rv_hat = float(s.iloc[-1])
    q33, q66 = np.nanpercentile(s.values, [33, 66])
    if rv_hat <= q33:   return "Ruhig",  "Erwartete geringe Kursbewegung."
    elif rv_hat <= q66: return "Normal", "Erwartete marktübliche Schwankungen."
    else:               return "Erhöht", "Größere Kursbewegungen möglich — höhere Unsicherheit."

# ---------------- Status-Erkennung ----------------
def session_last_timestamp(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty: return None
    sidx = df.tz_convert(TZ).between_time("09:00","17:30").index
    return None if len(sidx)==0 else pd.Timestamp(sidx.max())

def _has_today_intraday(df: pd.DataFrame) -> bool:
    ts = session_last_timestamp(df)
    if ts is None: return False
    return ts.tz_convert(ZRH).date() == datetime.now(ZRH).date()

def detect_status(df: pd.DataFrame, logrv: pd.Series) -> Dict[str, Optional[str]]:
    open_now, _, _ = market_status_now()
    last_sess_ts = session_last_timestamp(df)
    ts_label = last_sess_ts.tz_convert(TZ).strftime("%Y-%m-%d %H:%M %Z") if last_sess_ts is not None else None

    if open_now:
        if _has_today_intraday(df) and len(logrv) >= 16:
            return {"status":"live","msg":"Markt offen — Prognose für die nächsten 60 Minuten (laufende Handelssitzung).","ts_label":None}
        # Markt offen, aber keine heutigen Ticks → Vendor-Delay
        if last_sess_ts is not None:
            return {"status":"delayed","msg":"Datenanbieter liefert noch keine heutigen Intraday-Daten. Anzeige basiert auf der letzten Session.","ts_label":ts_label}
        return {"status":"data_error","msg":"Es liegen derzeit keine verwendbaren Intraday-Daten vor.","ts_label":None}
    else:
        if ts_label:
            return {"status":"closed","msg":"Markt geschlossen — Prognose bezieht sich auf die erste Handelsstunde nach Wiedereröffnung.","ts_label":ts_label}
        return {"status":"data_error","msg":"Es liegen derzeit keine vollständigen Intraday-Daten vor. Bitte später erneut versuchen.","ts_label":None}

# ---------------- UI ----------------
def build_ui(force_no_cache_cli: bool = False):
    import gradio as gr

    CSS = """
    <style>
      .cards { display:grid; grid-template-columns: 1fr; gap:14px; }
      @media(min-width:720px){ .cards{ grid-template-columns: 1fr 1fr; } }
      @media(min-width:1100px){ .cards{ grid-template-columns: 1fr 1fr 1fr; } }
      .card{ background:#0b0f18; color:#e6e9ef; border-radius:16px; padding:16px 18px; box-shadow: 0 3px 14px rgba(0,0,0,.25); border:1px solid #1c2233;}
      .hdr{ display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;}
      .title{ font-weight:800; letter-spacing:.5px; }
      .sub{ opacity:.75; font-size:.9rem; }
      .price{ font-size:2rem; font-weight:800; margin-top:8px;}
      .row{ display:flex; align-items:center; gap:10px; margin-top:6px;}
      .chg.up{ color:#65d38a; } .chg.down{ color:#ff6b6b; } .muted{ opacity:.8; }
      .badge{ padding:4px 10px; border-radius:18px; font-size:.85rem; font-weight:800; }
      .b-low{ background:#133e20; color:#9af7be; border:1px solid #1f5a32;}
      .b-mid{ background:#4e3a10; color:#ffd88f; border:1px solid #6b4c14;}
      .b-high{ background:#4a1212; color:#ffb0b0; border:1px solid #7a2323;}
      .pred{ font-weight:700; }
      .sep{ height:1px; background:#222a3e; margin:10px 0; }
      .statusbar{ display:flex; gap:10px; align-items:center; margin:6px 0 14px 0;}
      .status-badge{ padding:4px 10px; border-radius:14px; font-weight:700; border:1px solid #2a334a; }
      .open{ background:#122a1a; color:#7ee2a6; border-color:#1f5331;}
      .closed{ background:#2a1a1a; color:#ffb1b1; border-color:#5c2a2a;}
      .delayed{ background:#2a1f12; color:#ffd9a0; border-color:#6b521f;}
      .error{ background:#3c1a1a; color:#ffbdbd; border-color:#6b2a2a;}
      .hint{ opacity:.8; font-size:.9rem;}
      .kline{ opacity:.9; font-size:.9rem; margin-top:6px; }
      .proxy { font-size:.85rem; opacity:.85; }
      .labelgray { opacity:.7; font-size:.9rem; }
    </style>
    """

    def render_cards(rows: List[dict], top_status: dict) -> str:
        html = [CSS]
        if top_status["status"] == "live":
            status_html = '<span class="status-badge open">Markt offen</span><span class="hint">Prognose für die nächsten 60 Minuten (laufende Handelssitzung).</span>'
        elif top_status["status"] == "closed":
            ts = top_status.get("ts_label") or ""
            status_html = f'<span class="status-badge closed">Markt geschlossen</span><span class="hint">Prognose für die erste Handelsstunde nach Wiedereröffnung. Letzte verfügbare Daten: {ts}</span>'
        elif top_status["status"] == "delayed":
            ts = top_status.get("ts_label") or ""
            status_html = f'<span class="status-badge delayed">Daten verzögert</span><span class="hint">Keine heutigen Intraday-Daten von Yahoo. Anzeige basiert auf letzter Session: {ts}</span>'
        else:
            status_html = '<span class="status-badge error">Datenproblem</span><span class="hint">Es liegen derzeit keine vollständigen Intraday-Daten vor.</span>'
        html.append(f'<div class="statusbar">{status_html}</div>')

        html.append('<div class="cards">')
        for r in rows:
            level = r["level"]
            badge_cls = "b-low" if level=="Ruhig" else ("b-mid" if level=="Normal" else ("b-high" if level=="Erhöht" else ""))
            chg_cls = "chg up" if (r["change_pct"] is not None and np.isfinite(r["change_pct"]) and r["change_pct"] >= 0) else "chg down"
            chg_txt = f"{r['change_pct']:+.2f}%" if r["change_pct"] is not None and np.isfinite(r["change_pct"]) else "—"
            price_main = "—"
            if r["price"] is not None and np.isfinite(r["price"]):
                price_main = f"CHF {r['price']:.2f}"
                if r.get("price_label"):
                    price_main += f'  <span class="labelgray">({r["price_label"]})</span>'
            note = r.get("note") or ""
            st = r.get("status_info") or {}
            if st.get("status") == "live":
                kline = "Live — Prognose für die nächsten 60 Minuten."
            elif st.get("status") == "closed":
                kline = f"Markt geschlossen — Prognose für die erste Handelsstunde nach Wiedereröffnung • Letzte Daten @ {st.get('ts_label','')}"
            elif st.get("status") == "delayed":
                kline = f"Markt offen — Daten verzögert. Anzeige basiert auf letzter Session @ {st.get('ts_label','')}"
            else:
                kline = f"Datenproblem — {st.get('msg','')}"
            proxy_badge = '<div class="proxy">(Proxy aus Tagesdaten)</div>' if r.get("is_proxy") else ""
            html.append(f"""
              <div class="card">
                <div class="hdr">
                  <div>
                    <div class="title">{r['ticker'].split('.')[0]}</div>
                    <div class="sub">{r['name']}</div>
                  </div>
                  <div class="badge {badge_cls}">{level}</div>
                </div>
                <div class="price">{price_main}</div>
                <div class="row"><div class="{chg_cls}">{chg_txt}</div></div>
                <div class="sep"></div>
                <div class="muted">Volatilität (nächste Handelsstunde):</div>
                <div class="pred">{note}</div>
                {proxy_badge}
                <div class="kline">{kline}</div>
              </div>
            """)
        html.append("</div>")
        return "".join(html)

    import gradio as gr
    with gr.Blocks(title="SMI 60-Minuten-Volatilität") as demo:
        gr.Markdown("## Stonks AI — SMI 60-Minuten-Volatilität\n*Ampel (Ruhig / Normal / Erhöht) + σ₆₀min in % — mit klaren Statusmeldungen.*")

        # --------- Dashboard ---------
        with gr.Tab("Dashboard"):
            dd_multi = gr.Dropdown(choices=SMI_TICKERS,
                                   value=["NESN.SW","NOVN.SW","ROG.SW","ABBN.SW","UBSG.SW","ZURN.SW"],
                                   multiselect=True, label="Wähle Aktien (Multi-Select)", allow_custom_value=False)
            btn = gr.Button("Aktualisieren", variant="primary")
            cards = gr.HTML()
            table = gr.Dataframe(
                headers=["Ticker","Name","Preis","Tages-%","Level","σ60m (%)","RV_hat","RV_last","Modus","Step","Kontext","Tries","Status","Letzte Session","Quelle","Preisquelle"],
                interactive=False
            )

            def _run_dashboard(tickers: List[str]):
                # Top-Status anhand eines Referenz-Tickers
                try:
                    ref_df = fetch_intraday_cached(tickers[0], interval="5m", period="60d", tz=TZ, use_cache=True,
                                                   force_no_cache=force_no_cache_cli)
                except Exception:
                    ref_df = pd.DataFrame()
                ref_logrv, _ = compute_logrv_auto(ref_df, horizons_minutes=60)
                ref_status = detect_status(ref_df, ref_logrv)

                out_rows, df_rows = [], []
                model = load_timesfm25_torch(max_context=512, quantiles=False)

                for t in tickers:
                    try:
                        df = fetch_intraday_cached(t, interval="5m", period="60d", tz=TZ, use_cache=True,
                                                   force_no_cache=force_no_cache_cli)
                        price, chg, price_label = latest_price_and_change(df, t)
                        logrv, meta = compute_logrv_auto(df, horizons_minutes=60)
                        s_info = detect_status(df, logrv)

                        is_proxy = False
                        rv_hat, rv_last, used_ctx = np.nan, np.nan, len(logrv)
                        rv_pct = None
                        src = "none"
                        ref_series = pd.Series(dtype=float)

                        if len(logrv) >= 16 and _has_today_intraday(df):
                            _, yhat_log, _, used_ctx = forecast_logrv_one_step(logrv, model, desired_ctx=128)
                            rv_hat = float(np.exp(yhat_log))
                            rv_last = float(np.exp(logrv.iloc[-1]))
                            rv_pct = sigma_to_pct(rv_hat)
                            src = meta.get("mode")
                            ref_series = np.exp(logrv)
                        else:
                            daily_series, proxy = proxy_sigma_60m_series_from_daily(t, days=120)
                            if proxy is not None and np.isfinite(proxy):
                                is_proxy = True
                                rv_hat = float(proxy)
                                rv_pct = sigma_to_pct(rv_hat)
                                rv_last = np.nan
                                src = "proxy-daily"
                                ref_series = daily_series

                        level, hint = classify_vola_text(ref_series, rv_hat) if np.isfinite(rv_hat) else ("Unbekannt","Es liegen derzeit nicht genug Daten vor.")
                        if rv_pct is not None:
                            hint = f"{hint} (σ₆₀min ≈ {rv_pct:.2f} %)"
                        if is_proxy:
                            hint = f"{hint} — Proxy aus Tagesdaten."

                        out_rows.append({
                            "ticker": t, "name": SMI_NAMES.get(t, t),
                            "price": price if (price is not None and np.isfinite(price)) else None,
                            "price_label": price_label,
                            "change_pct": chg if (chg is not None and np.isfinite(chg)) else None,
                            "level": level, "note": hint,
                            "status_info": s_info, "is_proxy": is_proxy
                        })
                        df_rows.append([
                            t, SMI_NAMES.get(t, t),
                            None if (price is None or not np.isfinite(price)) else round(price, 4),
                            None if (chg is None or not np.isfinite(chg)) else round(chg, 3),
                            level,
                            None if rv_pct is None else round(rv_pct, 2),
                            None if not np.isfinite(rv_hat) else round(rv_hat, 6),
                            None if not np.isfinite(rv_last) else round(rv_last, 6),
                            meta.get("mode"), meta.get("step"), used_ctx, str(meta.get("tries")),
                            s_info.get("status"), s_info.get("ts_label"), src, price_label
                        ])

                    except Exception as e:
                        s_info = {"status":"data_error","msg":f"Fehler: {e}","ts_label":None}
                        out_rows.append({"ticker": t, "name": SMI_NAMES.get(t, t),
                                         "price": None, "change_pct": None, "price_label": "",
                                         "level": "Unbekannt", "note": "Datenabruf fehlgeschlagen.", "status_info": s_info, "is_proxy": False})
                        df_rows.append([t, SMI_NAMES.get(t, t), None, None, "Unbekannt", None, None, None, "Error", None, None, None, "data_error", None, "error",""])

                return render_cards(out_rows, ref_status), pd.DataFrame(
                    df_rows,
                    columns=["Ticker","Name","Preis","Tages-%","Level","σ60m (%)","RV_hat","RV_last","Modus","Step","Kontext","Tries","Status","Letzte Session","Quelle","Preisquelle"]
                )

            btn.click(_run_dashboard, inputs=[dd_multi], outputs=[cards, table])

        # --------- Forecast (Chart) ---------
        with gr.Tab("Forecast (Chart)"):
            dd_single = gr.Dropdown(choices=SMI_TICKERS, value="NESN.SW", multiselect=False, label="Aktie wählen", allow_custom_value=False)
            btn2 = gr.Button("Berechnen", variant="primary")
            status = gr.Textbox(label="Status", lines=14)
            plot = gr.Plot(label="log(RV) — Kontext (letzte Stunden) & Prognose (60 Min)")

            def _last_hours(ix, hours=PLOT_WINDOW_HOURS):
                if len(ix) == 0: return slice(None)
                cutoff = ix[-1] - pd.Timedelta(hours=hours)
                return ix >= cutoff

            def _chart(tic: str):
                try:
                    df = fetch_intraday_cached(tic.strip(), interval="5m", period="60d", tz=TZ, use_cache=True,
                                               force_no_cache=force_no_cache_cli)
                    logrv, meta = logrv_for_chart(df, horizon=60)  # CHART: nur Session-Kontext
                    s_info = detect_status(df, logrv)
                    lines = [f"Data: {tic} | logRV len={len(logrv)} | Modus={meta.get('mode')} | step={meta.get('step')}"]

                    if s_info["status"] == "live":
                        lines.append("Markt offen — Prognose für die nächsten 60 Minuten (laufende Handelssitzung).")
                    elif s_info["status"] == "closed":
                        lines.append(f"Markt geschlossen — Prognose bezieht sich auf die erste Handelsstunde nach Wiedereröffnung. Letzte verfügbare Daten: {s_info.get('ts_label')}.")
                    elif s_info["status"] == "delayed":
                        lines.append(f"Markt offen — Datenanbieter verzögert. Anzeige basiert auf der letzten Session: {s_info.get('ts_label')}.")

                    # Proxy falls zu kurz
                    if len(logrv) < 16:
                        daily_series, proxy = proxy_sigma_60m_series_from_daily(tic.strip(), days=120)
                        if proxy is None or not np.isfinite(proxy):
                            lines.append("⚠️ Keine Intraday- und keine Daily-Proxy-Daten verfügbar.")
                            return "\n".join(lines), None
                        level, hint = classify_vola_text(daily_series, proxy)
                        pct = sigma_to_pct(proxy)
                        more = f" (σ₆₀min ≈ {pct:.2f} %)" if pct is not None else ""
                        lines.append(f"Volatilität nächste Handelsstunde (Proxy): {level} — {hint}{more}")
                        lines.append("Hinweis: Proxy aus Tagesdaten (Skalierung auf 60 Min).")
                        return "\n".join(lines), None

                    # Normale Intraday-Prognose
                    model = load_timesfm25_torch(max_context=512, quantiles=True)
                    ts_pred, yhat_log, qvec, used_ctx = forecast_logrv_one_step(logrv, model, desired_ctx=128)

                    used_ctx_plot = min(used_ctx, len(logrv))
                    fig = plt.figure(figsize=(10, 4))
                    ctx_x_all = logrv.index[-used_ctx_plot:]
                    ctx_y_all = logrv.values[-used_ctx_plot:]

                    mask = _last_hours(ctx_x_all, hours=PLOT_WINDOW_HOURS)
                    ctx_x = ctx_x_all[mask]; ctx_y = ctx_y_all[mask]

                    plt.plot(ctx_x, ctx_y, label="log(RV) Kontext")
                    plt.scatter([ts_pred], [yhat_log], marker="x", s=90, label="Forecast log(RV)")
                    plt.axvline(ts_pred, linestyle="--", alpha=0.6)

                    if qvec is not None and len(qvec) == 10:
                        q_lo, q_hi = float(qvec[4]), float(qvec[5])   # 25/75
                        q05, q95 = float(qvec[0]), float(qvec[9])     # 5/95
                        plt.vlines(ts_pred, q_lo, q_hi, linewidth=6, alpha=0.3, label="Streuung (25–75%)")
                        plt.vlines(ts_pred, q05, q95, linestyle="--", alpha=0.5, label="5–95%")

                    # Deutliches Datumsformat
                    ax = plt.gca()
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M', tz=ZRH))
                    plt.title(f"{tic} — log(RV) Kontext & 60-Min Prognose ({meta.get('mode')}, step={meta.get('step')}, ctx={used_ctx_plot})")
                    plt.xlabel("Zeit (CET)"); plt.ylabel("log(RV)"); plt.legend(); plt.tight_layout()

                    rv_hat = float(np.exp(yhat_log))
                    pct = sigma_to_pct(rv_hat)
                    level, hint = classify_vola_text(np.exp(logrv), rv_hat)
                    more = f" (σ₆₀min ≈ {pct:.2f} %)" if pct is not None else ""
                    lines.append(f"Volatilität nächste Handelsstunde: {level} — {hint}{more}")
                    if s_info.get("ts_label"):
                        lines.append(f"Letzter verwendeter Session-Zeitstempel: {s_info.get('ts_label')}.")

                    return "\n".join(lines), fig

                except Exception as e:
                    return f"❌ Error: {e}", None

            btn2.click(_chart, inputs=[dd_single], outputs=[status, plot])

    return demo

# ---------------- Entry ----------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="SMI 60-Minuten-Volatilität (Variante B, Ampel, robuste Fallbacks + Delayed-Status)")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for UI")
    parser.add_argument("--no-cache", action="store_true", help="Ignoriere Cache vollständig (frische Downloads).")
    args = parser.parse_args()

    if args.ui:
        demo = build_ui(force_no_cache_cli=args.no_cache)
        demo.launch(server_name="127.0.0.1", server_port=args.server_port)
    else:
        df = fetch_intraday_cached("NESN.SW", interval="5m", period="60d",
                                   force_no_cache=args.no_cache)
        logrv, meta = compute_logrv_auto(df, horizons_minutes=60)
        s_info = detect_status(df, logrv)
        if len(logrv) >= 16 and _has_today_intraday(df):
            m = load_timesfm25_torch()
            _, yhat_log, _, ctx = forecast_logrv_one_step(logrv, m, 128)
            print("RV_hat (60m):", float(np.exp(yhat_log)), "meta:", meta, "ctx:", ctx, "status:", s_info)
        else:
            daily_series, proxy = proxy_sigma_60m_series_from_daily("NESN.SW", days=120)
            print("Kein heutiges Intraday — Proxy 60m:", proxy, "| status:", s_info, "| meta:", meta)

if __name__ == "__main__":
    main()
