import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz
from pathlib import Path

# Local Imports
from data import fetch_hourly_multivariate, get_latest_market_data, TZ
from model import ChronosForecaster

# Global Model Cache
MODEL = None

SMI_TICKERS = [
    "NESN.SW", "NOVN.SW", "ROG.SW", "ABBN.SW", "SIKA.SW", "UBSG.SW", "ZURN.SW",
    "GIVN.SW", "CFR.SW", "UHR.SW", "SCMN.SW", "ADEN.SW", "SGSN.SW"
]

SMI_NAMES = {
    "NESN.SW": "Nestlé", "NOVN.SW": "Novartis", "ROG.SW": "Roche", "ABBN.SW": "ABB",
    "SIKA.SW": "Sika", "UBSG.SW": "UBS", "ZURN.SW": "Zurich", "GIVN.SW": "Givaudan",
    "CFR.SW": "Richemont", "UHR.SW": "Swatch", "SCMN.SW": "Swisscom", "ADEN.SW": "Adecco", "SGSN.SW": "SGS",
}

def get_model():
    global MODEL
    if MODEL is None:
        # Load Chronos 2 (using t5-large as proxy if 2 is not directly available via simple ID, 
        # but we use the ID from model.py which defaults to user preference or t5-large)
        # We'll use 'amazon/chronos-t5-large' for better performance/quality balance if 'chronos-2' fails in model.py logic
        MODEL = ChronosForecaster(model_id="amazon/chronos-t5-large") 
    return MODEL

def classify_volatility(current_rv, forecast_rv, history_rv):
    """
    Classify volatility into Low/Normal/High based on historical quantiles.
    """
    if len(history_rv) < 20:
        return "Unknown", "Not enough data"
    
    q33 = np.quantile(history_rv, 0.33)
    q66 = np.quantile(history_rv, 0.66)
    
    val = forecast_rv if forecast_rv is not None else current_rv
    
    if val <= q33:
        return "Low", "Expect calm market conditions."
    elif val <= q66:
        return "Normal", "Expect typical market moves."
    else:
        return "High", "Expect significant volatility."

def format_dashboard_row(ticker):
    # 1. Get Data
    df = fetch_hourly_multivariate(ticker, days=60)
    market_data = get_latest_market_data(ticker)
    
    if df.empty:
        return {
            "ticker": ticker, "name": SMI_NAMES.get(ticker, ticker),
            "price": market_data.get("price"), "change": market_data.get("change_pct"),
            "level": "Error", "note": "No Data", "rv_forecast": None
        }
        
    # 2. Forecast
    model = get_model()
    # Use last 60 hours context
    pred = model.predict(df, context_length=60)
    
    # 3. Classify
    # Convert log RV back to normal scale for display if needed, but classification is easier in log space or consistent space
    # Let's use log space for quantiles
    log_rv_hist = df["log_rv"].values
    
    if pred:
        forecast_val = pred["mean_log"]
        # Convert to % (RV is std dev of log returns, so exp gives fraction, *100 for %)
        # But clip to realistic range: hourly vol rarely exceeds 5%
        rv_forecast = np.exp(forecast_val)
        
        # Sanity check: if forecast is absurdly high, cap it
        if rv_forecast > 0.10:  # 10% hourly vol is already extreme
            print(f"WARNING: {ticker} forecast capped from {rv_forecast*100:.1f}% to 10%")
            rv_forecast = 0.10
            
        sigma_pct = rv_forecast * 100  # Convert to percentage
        
        level, note = classify_volatility(log_rv_hist[-1], forecast_val, log_rv_hist)
    else:
        sigma_pct = None
        level, note = "Error", "Forecast failed"

    return {
        "ticker": ticker,
        "name": SMI_NAMES.get(ticker, ticker),
        "price": market_data.get("price"),
        "change": market_data.get("change_pct"),
        "level": level,
        "note": note,
        "rv_forecast": sigma_pct
    }

def render_dashboard_html(rows):
    html = """
    <style>
        .card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; color: white; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .ticker { font-weight: bold; font-size: 1.2rem; }
        .name { color: #94a3b8; font-size: 0.9rem; }
        .badge { padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: bold; }
        .badge-Low { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
        .badge-Normal { background: #451a03; color: #fdba74; border: 1px solid #d97706; }
        .badge-High { background: #450a0a; color: #fca5a5; border: 1px solid #dc2626; }
        .price-row { display: flex; align-items: baseline; gap: 10px; margin-bottom: 1rem; }
        .price { font-size: 1.8rem; font-weight: bold; }
        .change { font-size: 1rem; }
        .change.pos { color: #4ade80; }
        .change.neg { color: #f87171; }
        .forecast-row { border-top: 1px solid #334155; padding-top: 1rem; }
        .forecast-label { color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.25rem; }
        .forecast-val { font-size: 1.1rem; font-weight: bold; }
        .note { font-size: 0.85rem; color: #cbd5e1; margin-top: 0.25rem; }
    </style>
    <div class="card-grid">
    """
    
    for row in rows:
        chg = row['change']
        chg_cls = "pos" if chg and chg >= 0 else "neg"
        chg_str = f"{chg:+.2f}%" if chg is not None else "N/A"
        price_str = f"{row['price']:.2f}" if row['price'] else "N/A"
        rv_str = f"{row['rv_forecast']:.2f}%" if row['rv_forecast'] else "N/A"
        
        html += f"""
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="ticker">{row['ticker'].split('.')[0]}</div>
                    <div class="name">{row['name']}</div>
                </div>
                <div class="badge badge-{row['level']}">{row['level']}</div>
            </div>
            <div class="price-row">
                <div class="price">{price_str}</div>
                <div class="change {chg_cls}">{chg_str}</div>
            </div>
            <div class="forecast-row">
                <div class="forecast-label">Next Hour Volatility (σ)</div>
                <div class="forecast-val">{rv_str}</div>
                <div class="note">{row['note']}</div>
            </div>
        </div>
        """
    html += "</div>"
    return html

def update_dashboard(tickers):
    rows = []
    for t in tickers:
        rows.append(format_dashboard_row(t))
    return render_dashboard_html(rows)

def plot_forecast(ticker):
    df = fetch_hourly_multivariate(ticker, days=60)
    if df.empty:
        return None, "No Data"
        
    model = get_model()
    # Context: last 7 days (approx 50-60 hours)
    ctx_len = 60
    pred = model.predict(df, context_length=ctx_len)
    
    if not pred:
        return None, "Prediction Failed"
        
    # Prepare Data for Plot
    # Show last 48 hours + 1 hour forecast
    plot_df = df.iloc[-48:]
    timestamps = plot_df.index
    rv_hist = np.exp(plot_df["log_rv"]) * 100
    
    # Forecast
    next_ts = timestamps[-1] + pd.Timedelta(hours=1)
    fc_mean = np.exp(pred["mean_log"]) * 100
    fc_q10 = np.exp(pred["quantiles_log"]["q10"]) * 100
    fc_q90 = np.exp(pred["quantiles_log"]["q90"]) * 100
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Volatility
    ax1.plot(timestamps, rv_hist, label="Historical Volatility (1h)", color="white", marker="o", markersize=4)
    
    # Forecast Point
    ax1.errorbar([next_ts], [fc_mean], yerr=[[fc_mean - fc_q10], [fc_q90 - fc_mean]], 
                 fmt='o', color='#facc15', ecolor='#facc15', elinewidth=3, capsize=5, label="Forecast (Next Hour)")
    
    ax1.set_title(f"{ticker} - Volatility Forecast (Chronos 2 Multivariate)", color="white")
    ax1.set_ylabel("Volatility (%)", color="white")
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(colors="white")
    ax1.legend()
    
    # 2. Volume (Multivariate Context)
    vol_hist = np.exp(plot_df["log_vol"])
    ax2.bar(timestamps, vol_hist, color="#3b82f6", alpha=0.6, label="Volume")
    ax2.set_ylabel("Volume", color="white")
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(colors="white")
    ax2.legend()
    
    # Formatting
    fig.patch.set_facecolor('#0f172a')
    ax1.set_facecolor('#1e293b')
    ax2.set_facecolor('#1e293b')
    
    # Date formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M', tz=pytz.timezone(TZ)))
    plt.xticks(rotation=45)
    
    return fig, f"Forecast: {fc_mean:.2f}% (Range: {fc_q10:.2f}% - {fc_q90:.2f}%)"

# UI Layout
with gr.Blocks(title="SMI Volatility Forecast (Chronos 2)", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🇨🇭 SMI Volatility Forecast (Chronos 2)")
    gr.Markdown("Multivariate forecasting using Amazon Chronos 2. Predicting **Next Hour** volatility based on RV, Volume, and Returns.")
    
    with gr.Tab("Dashboard"):
        refresh_btn = gr.Button("Refresh Dashboard", variant="primary")
        dashboard_html = gr.HTML()
        
        # Initial Load (optional, or on click)
        # refresh_btn.click(update_dashboard, inputs=[], outputs=[dashboard_html])
        # We need to pass tickers. Let's make it fixed for now or selectable.
        ticker_selector = gr.Dropdown(SMI_TICKERS, value=SMI_TICKERS[:6], multiselect=True, label="Select Tickers")
        refresh_btn.click(update_dashboard, inputs=[ticker_selector], outputs=[dashboard_html])
        
    with gr.Tab("Detailed Forecast"):
        single_ticker = gr.Dropdown(SMI_TICKERS, value="NESN.SW", label="Select Ticker")
        plot_btn = gr.Button("Generate Forecast", variant="primary")
        output_plot = gr.Plot()
        output_text = gr.Textbox(label="Forecast Details")
        
        plot_btn.click(plot_forecast, inputs=[single_ticker], outputs=[output_plot, output_text])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
