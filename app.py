# app.py — SMI 60-min Volatility (Variante B, farbige Ampel, adaptiver Step-Fallback)
# Mit klaren Statusmeldungen: Markt offen / geschlossen (mit Zeitstempel) / Datenfehler
# Setup (Windows, Python 3.12 empfohlen):
#   pip install --upgrade pip
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
#   pip install gradio yfinance pandas numpy matplotlib pytz requests

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import matplotlib.pyplot as plt
import timesfm  # TimesFM 2.5 (Torch backend; HF/Git)

# ---------------- Global ----------------
TZ = "Europe/Zurich"
ZRH = pytz.timezone(TZ)
DATA_DIR = Path("./data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

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
MAX_PERIOD_BY_INTERVAL = {
    "1m":  "7d",
    "2m":  "60d",
    "5m":  "60d",
    "15m": "730d",
    "30m": "730d",
    "60m": "730d",
    "90m": "730d",
    "1h":  "730d",
}
FALLBACK_PERIODS = ["60d", "30d", "7d"]

def _clamp_period(interval: str, period: str) -> str:
    limit = MAX_PERIOD_BY_INTERVAL.get(interval, "60d")
    def _days(p):
        if p.endswith("d"): return int(p[:-1])
        if p.endswith("y"): return int(p[:-1]) * 365
        return 9999
    return period if _days(period) <= _days(limit) else limit

# ---------------- yfinance + Cache ----------------
def _yf_download_strict(ticker: str, interval="5m", period="60d", auto_adjust=True) -> pd.DataFrame:
    period = _clamp_period(interval, period)
    ladder = [period] + [p for p in FALLBACK_PERIODS if _clamp_period(interval, p) == p and p != period]
    last_err = None
    for p in ladder:
        try:
            df = yf.download(ticker, interval=interval, period=p, auto_adjust=auto_adjust,
                             progress=False, threads=True)
            if df is not None and not df.empty:
                df.attrs["yf_period_used"] = p
                return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"yfinance empty for {ticker} (interval={interval}, tried={ladder}). Last error: {last_err}")

def fetch_intraday_cached(ticker: str, interval="5m", period="60d",
                          tz: str = TZ, use_cache: bool = True) -> pd.DataFrame:
    cache_fp = DATA_DIR / f"{ticker.replace('.', '_')}_{interval}.csv"

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC").tz_convert(tz)
        else:
            idx = idx.tz_convert(tz)
        out = df.copy(); out.index = idx
        return out.rename(columns=str.lower)[["close"]].dropna()

    fresh, err = None, None
    try:
        fresh = _norm(_yf_download_strict(ticker, interval=interval, period=period))
    except Exception as e:
        err = e

    if use_cache and cache_fp.exists():
        cached = pd.read_csv(cache_fp)
        tcol = next((c for c in ["Datetime","datetime","Date","date","Time","time"] if c in cached.columns), cached.columns[0])
        cached[tcol] = pd.to_datetime(cached[tcol], utc=True, errors="coerce")
        cached = cached.dropna(subset=[tcol]).set_index(tcol).sort_index()
        cached.index = cached.index.tz_convert(tz)
        ccol = next((c for c in ["close","Close","Adj Close","adj_close","adj close"] if c in cached.columns), cached.columns[-1])
        cached = cached[[ccol]].rename(columns={ccol: "close"}).dropna()

        if fresh is None:
            if cached.empty: raise RuntimeError(f"Cache empty and download failed: {err}")
            return cached

        merged = pd.concat([cached, fresh[~fresh.index.isin(cached.index)]], axis=0).sort_index()
        merged = merged[merged.index >= (merged.index.max() - pd.Timedelta(days=180))]
        merged.rename_axis("Datetime").reset_index().to_csv(cache_fp, index=False)
        return merged

    if fresh is None:
        raise RuntimeError(f"Download failed: {err}")
    if use_cache:
        fresh.rename_axis("Datetime").reset_index().to_csv(cache_fp, index=False)
    return fresh

# ---------------- Market status ----------------
def is_business_day(dt: datetime) -> bool:
    return dt.weekday() < 5

def next_open_datetime(now: datetime) -> datetime:
    d = 0
    while True:
        d += 1
        cand = (now + timedelta(days=d)).replace(hour=9, minute=0, second=0, microsecond=0)
        if is_business_day(cand):
            return cand

def market_status_now() -> Tuple[bool, Optional[datetime], Optional[datetime]]:
    now = datetime.now(ZRH)
    open_t  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    close_t = now.replace(hour=17, minute=30, second=0, microsecond=0)
    if not is_business_day(now): return False, None, next_open_datetime(now)
    if now < open_t:            return False, None, open_t
    if now > close_t:           return False, close_t, next_open_datetime(now)
    return True, close_t, None

# ---------------- RV (mit adaptivem Step & stärkeren Fallbacks) ----------------
def _prep_base(close_df: pd.DataFrame) -> pd.DataFrame:
    df = close_df.sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["close"])

def _maybe_resample(df: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    diffs = df.index.to_series().diff().dropna()
    need_resample = (len(diffs) == 0) or (abs(diffs.median() - pd.Timedelta(minutes=step_minutes)) > pd.Timedelta(seconds=60))
    return df.resample(f"{step_minutes}min").last() if need_resample else df

def realized_vol_series(df_in: pd.DataFrame, step_minutes=5, horizon_minutes=60,
                        session_only=True, ffill_limit=8, min_periods_fac: float = 1.0) -> pd.Series:
    df = _prep_base(df_in)
    if session_only:
        df = df.tz_convert(TZ).between_time("09:00", "17:30")
        if df.empty: return pd.Series(dtype=float)
    df = _maybe_resample(df, step_minutes).dropna()
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
                                     session_only=False, ffill_limit: int = 20, min_periods_fac: float = 0.5) -> pd.Series:
    df = _prep_base(df_in)
    if session_only:
        df = df.tz_convert(TZ).between_time("09:00", "17:30")
        if df.empty: return pd.Series(dtype=float)
    df = _maybe_resample(df, step_minutes).dropna()
    df["close"] = df["close"].ffill(limit=ffill_limit)
    df = df.dropna(subset=["close"])
    r = np.log(df["close"]).diff()
    k = max(1, horizon_minutes // step_minutes)
    minp = max(2, int(round(k * min_periods_fac)))
    rv = r.rolling(window=k, min_periods=minp).std(ddof=1) * np.sqrt(k)
    return rv.replace([np.inf, -np.inf], np.nan).dropna()

def compute_logrv_auto(df: pd.DataFrame, horizons_minutes=60) -> Tuple[pd.Series, dict]:
    """
    Adaptiver Plan:
      steps: 5 -> 15 -> 30
      je step:
        1) session_only=True,  ffill=8,  minp=k
        2) session_only=False, ffill=8,  minp=k
        3) std_fallback       (global), ffill=20, minp=max(2, k//2)
    """
    tries = []
    for step in [5, 15, 30]:
        k = max(1, horizons_minutes // step)

        rv = realized_vol_series(df, step_minutes=step, horizon_minutes=horizons_minutes,
                                 session_only=True, ffill_limit=8, min_periods_fac=1.0)
        logrv = np.log(pd.Series(rv).replace([np.inf, -np.inf, 0], np.nan)).dropna()
        tries.append((f"{step}m-session", len(logrv)))
        if len(logrv) >= 16:
            return logrv, {"step": step, "mode": "session-soft", "k": k, "tries": tries}

        rv = realized_vol_series(df, step_minutes=step, horizon_minutes=horizons_minutes,
                                 session_only=False, ffill_limit=8, min_periods_fac=1.0)
        logrv = np.log(pd.Series(rv).replace([np.inf, -np.inf, 0], np.nan)).dropna()
        tries.append((f"{step}m-global", len(logrv)))
        if len(logrv) >= 16:
            return logrv, {"step": step, "mode": "global", "k": k, "tries": tries}

        rv = realized_vol_series_std_fallback(df, step_minutes=step, horizon_minutes=horizons_minutes,
                                              session_only=False, ffill_limit=20, min_periods_fac=0.5)
        logrv = np.log(pd.Series(rv).replace([np.inf, -np.inf, 0], np.nan)).dropna()
        tries.append((f"{step}m-stdFallback", len(logrv)))
        if len(logrv) >= 16:
            return logrv, {"step": step, "mode": "std-fallback", "k": k, "tries": tries}

    # alles leer
    return pd.Series(dtype=float), {"step": None, "mode": "none", "k": None, "tries": tries}

# ---------------- TimesFM (Torch) ----------------
def load_timesfm25_torch(max_context: int = 512, quantiles: bool = True):
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

def forecast_logrv_one_step(logrv: pd.Series, model, desired_ctx: int = 128):
    ctx = max(16, min(int(desired_ctx), len(logrv)))
    arr = logrv.values[-ctx:].astype(np.float32)
    point_fcst, q_fcst = model.forecast(horizon=1, inputs=[arr])
    ts_pred = pd.Timestamp(logrv.index[-1])
    y_hat = float(point_fcst.squeeze())
    q = q_fcst.squeeze() if q_fcst is not None else None
    return ts_pred, y_hat, q, ctx

# ---------------- Price, Change, Ampel ----------------
def latest_price_and_change(df_close_5m: pd.DataFrame) -> Tuple[float, float]:
    if df_close_5m.empty: return float("nan"), float("nan")
    sess = df_close_5m.tz_convert(TZ).between_time("09:00", "17:30")["close"]
    if sess.empty: return float("nan"), float("nan")
    byday_last = sess.groupby(sess.index.date).last()
    last_price = float(byday_last.iloc[-1])
    if len(byday_last) >= 2:
        prev_close = float(byday_last.iloc[-2])
        chg = (last_price / prev_close - 1.0) * 100.0 if prev_close > 0 else float("nan")
    else:
        chg = float("nan")
    return last_price, chg

def classify_vola_text(rv_series: pd.Series, rv_hat: float) -> Tuple[str, str]:
    s = pd.Series(rv_series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty: return "Unbekannt", "Nicht genug Daten."
    if rv_hat is None or not np.isfinite(rv_hat): rv_hat = float(s.iloc[-1])
    q33, q66 = np.nanpercentile(s.values, [33, 66])
    if rv_hat <= q33:   return "Ruhig", "Kleine Schwankungen erwartet."
    elif rv_hat <= q66: return "Normal aktiv", "Übliche Marktdynamik."
    else:               return "Hohe Bewegung möglich", "Preis kann stark schwanken (höheres Risiko)."

# ---------------- Status-Erkennung & Zeitstempel ----------------
def session_last_timestamp(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Letzter Zeitstempel innerhalb der SIX-Session (CET)."""
    if df is None or df.empty: return None
    sidx = df.tz_convert(TZ).between_time("09:00","17:30").index
    return None if len(sidx)==0 else pd.Timestamp(sidx.max())

def detect_status(df: pd.DataFrame, logrv: pd.Series) -> Dict[str, Optional[str]]:
    """
    Liefert:
      status: 'live', 'closed', 'data_error'
      msg: kurzer Text
      ts_label: 'YYYY-MM-DD HH:MM CET' oder None
    """
    open_now, close_t, next_open = market_status_now()
    # letzter Session-Timestamp (falls vorhanden)
    last_sess_ts = session_last_timestamp(df)
    ts_label = last_sess_ts.tz_convert(TZ).strftime("%Y-%m-%d %H:%M %Z") if last_sess_ts is not None else None

    if open_now:
        if len(logrv) >= 16:
            return {"status": "live", "msg": "Live (Session)", "ts_label": None}
        else:
            # Markt ist offen, aber RV leer → Datenlücke/Yahoo
            return {"status": "data_error", "msg": "Datenfehler (unvollständige Intraday-Daten von Yahoo).", "ts_label": ts_label}
    else:
        # Markt geschlossen → wenn wir eine letzte Sessionzeit haben, zeigen wir die an
        if ts_label:
            return {"status": "closed", "msg": "Markt geschlossen — basierend auf letzter Session.", "ts_label": ts_label}
        # kein ts_label → auch geschlossen, aber keine Sessiondaten (Feiertag/Wochenende/OHNE Historie)
        return {"status": "data_error", "msg": "Keine Sessiondaten verfügbar (Feiertag/Wochenende oder Yahoo-Lücke).", "ts_label": None}

# ---------------- UI ----------------
def build_ui():
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
      .error{ background:#3c1a1a; color:#ffbdbd; border-color:#6b2a2a;}
      .hint{ opacity:.8; font-size:.9rem;}
      .kline{ opacity:.9; font-size:.9rem; margin-top:6px; }
    </style>
    """

    def render_cards(rows: List[dict], top_status: dict) -> str:
        html = [CSS]
        # Top-Statusleiste
        if top_status["status"] == "live":
            status_html = f'<span class="status-badge open">Markt offen</span><span class="hint">Session 09:00–17:30 CET</span>'
        elif top_status["status"] == "closed":
            ts = top_status.get("ts_label") or ""
            status_html = f'<span class="status-badge closed">Markt geschlossen</span><span class="hint">Letzte Session: {ts}</span>'
        else:
            status_html = f'<span class="status-badge error">Datenfehler</span><span class="hint">{top_status.get("msg","")}</span>'
        html.append(f'<div class="statusbar">{status_html}</div>')

        # Karten
        html.append('<div class="cards">')
        for r in rows:
            level = r["level"]
            badge_cls = "b-low" if level=="Ruhig" else ("b-mid" if level=="Normal aktiv" else ("b-high" if level=="Hohe Bewegung möglich" else ""))
            chg_cls = "chg up" if (r["change_pct"] is not None and np.isfinite(r["change_pct"]) and r["change_pct"] >= 0) else "chg down"
            chg_txt = f"{r['change_pct']:+.2f}%" if r["change_pct"] is not None and np.isfinite(r["change_pct"]) else "—"
            price_txt = f"CHF {r['price']:.2f}" if r["price"] is not None and np.isfinite(r["price"]) else "—"
            note = r.get("note") or ""
            st = r.get("status_info") or {}
            # per-Karte Statuszeile:
            if st.get("status") == "live":
                kline = "Live (Session)"
            elif st.get("status") == "closed":
                kline = f"Letzte Session @ {st.get('ts_label','')}"
            else:
                kline = f"Datenfehler — {st.get('msg','')}"
            html.append(f"""
              <div class="card">
                <div class="hdr">
                  <div>
                    <div class="title">{r['ticker'].split('.')[0]}</div>
                    <div class="sub">{r['name']}</div>
                  </div>
                  <div class="badge {badge_cls}">{level}</div>
                </div>
                <div class="price">{price_txt}</div>
                <div class="row"><div class="{chg_cls}">{chg_txt}</div></div>
                <div class="sep"></div>
                <div class="muted">Volatilität (nächste 60 Min):</div>
                <div class="pred">{note}</div>
                <div class="kline">{kline}</div>
              </div>
            """)
        html.append("</div>")
        return "".join(html)

    import gradio as gr
    with gr.Blocks(title="SMI 60-Minuten-Volatilität") as demo:
        gr.Markdown("## Stonks AI — SMI 60-Minuten-Volatilität\n*Klarer Status: Markt offen / geschlossen (mit Zeitstempel) / Datenfehler.*")

        # --------- Dashboard ---------
        with gr.Tab("Dashboard"):
            dd_multi = gr.Dropdown(
                choices=SMI_TICKERS,
                value=["NESN.SW","NOVN.SW","ROG.SW","ABBN.SW","UBSG.SW","ZURN.SW"],
                multiselect=True,
                label="Wähle Aktien (Multi-Select)",
                allow_custom_value=False
            )
            btn = gr.Button("Aktualisieren", variant="primary")
            cards = gr.HTML()
            table = gr.Dataframe(
                headers=["Ticker","Name","Preis","Tages-%","Level","RV_hat","RV_last","Modus","Step","Kontext","Tries","Status","Zeitstempel"],
                interactive=False
            )

            def _run_dashboard(tickers: List[str]):
                open_now, close_t, next_open = market_status_now()
                # Top-Status mit globalem Zeitstempel (vom ersten Ticker abgeleitet)
                # Wir lesen den ersten Ticker, um einen sinnvollen ts_label zu zeigen, wenn geschlossen
                top_status = {"status":"live","msg":"Live (Session)","ts_label":None}
                model = load_timesfm25_torch(max_context=512, quantiles=False)
                out_rows, df_rows = [], []

                # für top_status einen Referenz-df holen (erster ticker) – robust gegen Fehler
                ref_df, ref_ts = None, None
                try:
                    ref_df = fetch_intraday_cached(tickers[0], interval="5m", period="60d", tz=TZ, use_cache=True)
                    ref_ts = session_last_timestamp(ref_df)
                except Exception:
                    ref_df, ref_ts = None, None

                if not open_now:
                    top_status = {"status":"closed","msg":"Markt geschlossen","ts_label": ref_ts.tz_convert(TZ).strftime("%Y-%m-%d %H:%M %Z") if ref_ts is not None else None}

                for t in tickers:
                    try:
                        df = fetch_intraday_cached(t, interval="5m", period="60d", tz=TZ, use_cache=True)
                        price, chg = latest_price_and_change(df)
                        logrv, meta = compute_logrv_auto(df, horizons_minutes=60)

                        # Status pro Ticker (mit letztem Session-Timestamp)
                        s_info = detect_status(df, logrv)

                        rv_hat, rv_last, used_ctx = np.nan, np.nan, len(logrv)
                        if len(logrv) >= 16:
                            _, yhat_log, _, used_ctx = forecast_logrv_one_step(logrv, model, desired_ctx=128)
                            rv_hat = float(np.exp(yhat_log))
                            rv_last = float(np.exp(logrv.iloc[-1]))

                        level, hint = classify_vola_text(np.exp(logrv), rv_hat) if len(logrv) > 0 else ("Unbekannt","Nicht genug Daten.")
                        out_rows.append({
                            "ticker": t, "name": SMI_NAMES.get(t, t),
                            "price": price if np.isfinite(price) else None,
                            "change_pct": chg if np.isfinite(chg) else None,
                            "level": level, "note": hint,
                            "status_info": s_info
                        })
                        df_rows.append([
                            t, SMI_NAMES.get(t, t),
                            None if not np.isfinite(price) else round(price, 4),
                            None if not np.isfinite(chg) else round(chg, 3),
                            level,
                            None if not np.isfinite(rv_hat) else round(rv_hat, 6),
                            None if not np.isfinite(rv_last) else round(rv_last, 6),
                            meta.get("mode"), meta.get("step"), used_ctx, str(meta.get("tries")),
                            s_info.get("status"), s_info.get("ts_label")
                        ])
                    except Exception as e:
                        s_info = {"status":"data_error","msg":f"Fehler: {e}","ts_label":None}
                        out_rows.append({"ticker": t, "name": SMI_NAMES.get(t, t),
                                         "price": None, "change_pct": None,
                                         "level": "Unbekannt", "note": "Datenabruf fehlgeschlagen.", "status_info": s_info})
                        df_rows.append([t, SMI_NAMES.get(t, t), None, None, "Unbekannt", None, None, "Error", None, None, None, "data_error", None])

                return render_cards(out_rows, top_status), pd.DataFrame(
                    df_rows,
                    columns=["Ticker","Name","Preis","Tages-%","Level","RV_hat","RV_last","Modus","Step","Kontext","Tries","Status","Zeitstempel"]
                )

            btn.click(_run_dashboard, inputs=[dd_multi], outputs=[cards, table])

        # --------- Forecast (Chart) ---------
        with gr.Tab("Forecast (Chart)"):
            dd_single = gr.Dropdown(
                choices=SMI_TICKERS, value="NESN.SW", multiselect=False,
                label="Aktie wählen", allow_custom_value=False
            )
            btn2 = gr.Button("Berechnen", variant="primary")
            status = gr.Textbox(label="Status", lines=12)
            plot = gr.Plot(label="log(RV) — Kontext & Prognose (60 Min)")

            def _chart(tic: str):
                try:
                    df = fetch_intraday_cached(tic.strip(), interval="5m", period="60d", tz=TZ, use_cache=True)
                    logrv, meta = compute_logrv_auto(df, horizons_minutes=60)

                    s_info = detect_status(df, logrv)
                    lines = [
                        f"Data: {tic} | logRV len={len(logrv)} | Modus={meta.get('mode')} | step={meta.get('step')} | k={meta.get('k')}",
                        f"DEBUG tries: {meta.get('tries')}"
                    ]

                    # Klarer Status
                    if s_info["status"] == "live":
                        lines.append("Status: Markt offen — Live (Session).")
                    elif s_info["status"] == "closed":
                        lines.append(f"Status: Markt geschlossen — Letzte Session: {s_info.get('ts_label')}.")
                    else:
                        lines.append(f"Status: Datenfehler — {s_info.get('msg')}")
                        if len(logrv) < 16:
                            return "\n".join(lines), None

                    if len(logrv) < 16:
                        # Markt geschlossen, aber zu wenig Punkte → trotzdem informieren
                        lines.append("⚠️ logRV zu kurz (zu wenige Intraday-Ticks). Intern 5→15→30m versucht.")
                        return "\n".join(lines), None

                    model = load_timesfm25_torch(max_context=512, quantiles=True)
                    ts_pred, yhat_log, qvec, used_ctx = forecast_logrv_one_step(logrv, model, desired_ctx=128)

                    used_ctx_plot = min(used_ctx, len(logrv))
                    fig = plt.figure(figsize=(10, 4))
                    ctx_x = logrv.index[-used_ctx_plot:]
                    ctx_y = logrv.values[-used_ctx_plot:]
                    plt.plot(ctx_x, ctx_y, label="log(RV) Kontext")
                    plt.scatter([ts_pred], [yhat_log], marker="x", s=80, label="Forecast log(RV)")

                    if qvec is not None and len(qvec) == 10:
                        q05, q25, q75, q95 = qvec[0], qvec[4], qvec[5], qvec[9]
                        plt.hlines([q25, q75], xmin=ctx_x[-1], xmax=ctx_x[-1], linestyles="dashed")
                        plt.scatter([ts_pred, ts_pred, ts_pred, ts_pred], [q05, q25, q75, q95], marker="|", s=200, label="Quantile")

                    plt.title(f"{tic} — log(RV) Kontext & 60-Min Prognose ({meta.get('mode')}, step={meta.get('step')}, ctx={used_ctx_plot})")
                    plt.xlabel("Zeit"); plt.ylabel("log(RV)"); plt.legend(); plt.tight_layout()

                    rv_hat = float(np.exp(yhat_log))
                    rv_last = float(np.exp(logrv.iloc[-1]))
                    level, hint = classify_vola_text(np.exp(logrv), rv_hat)
                    lines.append(f"RV_hat(60m)={rv_hat:.6f}  |  last RV={rv_last:.6f}  |  Level={level} — {hint}")
                    if s_info["status"] == "closed" and s_info.get("ts_label"):
                        lines.append(f"Hinweis: Prognose bezieht sich auf die letzte verfügbare Session @ {s_info.get('ts_label')}.")

                    return "\n".join(lines), fig
                except Exception as e:
                    return f"❌ Error: {e}", None

            btn2.click(_chart, inputs=[dd_single], outputs=[status, plot])

    return demo

# ---------------- Entry ----------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="SMI 60-Minuten-Volatilität (Variante B, Statusklarheit)")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for UI")
    args = parser.parse_args()
    if args.ui:
        demo = build_ui()
        demo.launch(server_name="127.0.0.1", server_port=args.server_port)
    else:
        df = fetch_intraday_cached("NESN.SW", interval="5m", period="60d")
        logrv, meta = compute_logrv_auto(df, horizons_minutes=60)
        s_info = detect_status(df, logrv)
        if len(logrv) >= 16:
            m = load_timesfm25_torch()
            _, yhat_log, _, ctx = forecast_logrv_one_step(logrv, m, 128)
            print("RV_hat (60m):", float(np.exp(yhat_log)), "meta:", meta, "ctx:", ctx, "status:", s_info)
        else:
            print("logRV too short:", len(logrv), "meta:", meta, "status:", s_info)

if __name__ == "__main__":
    main()
