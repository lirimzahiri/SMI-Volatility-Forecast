# model_eval.py — Verbesserte Evaluierung mit individuellen Plots pro Aktie
# Läuft mit der aktuellen app.py (robustere Cache-Spaltenbereinigung).
# Usage-Beispiele:
#   python model_eval.py --ticker NESN.SW
#   python model_eval.py --multi
#   python model_eval.py --multi --all --output ./evaluation_out --test-size 0.25

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import json
import traceback
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---- Imports aus deiner App ----
from app import (
    fetch_intraday_cached, compute_logrv_auto, load_timesfm25_torch,
    forecast_logrv_one_step, SMI_TICKERS, SMI_NAMES, TZ
)

# --------------------------
# Hilfsfunktionen
# --------------------------

def _ensure_series(x) -> pd.Series:
    """Sichere Rückgabe als Series (kein DataFrame)."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        # nimm erste numerische Spalte
        num_cols = x.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            return x[num_cols[0]]
        return x.iloc[:, 0]
    x = pd.Series(x)
    return x

def _drop_non_finite(s: pd.Series) -> pd.Series:
    s = _ensure_series(s)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s

def _quantile_columns_from_timesfm(qvec: Optional[np.ndarray]) -> Dict[str, float]:
    """
    TimesFM liefert i.d.R. 10 Quantile in aufsteigender Reihenfolge.
    Wir mappen auf gängige Prozentpunkte, wenn genug Werte vorhanden sind.
    """
    out = {}
    if qvec is None:
        return out
    q = np.array(qvec).flatten()
    if q.size >= 10:
        # typische Indizes (0..9): ~[0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
        # wir nutzen 5/10/25/50/75/90/95:
        out["q5"]  = float(q[0])
        out["q10"] = float(q[1])
        out["q25"] = float(q[3])  # leicht näherungsweise
        out["q50"] = float(np.median(q))
        out["q75"] = float(q[6])
        out["q90"] = float(q[8])
        out["q95"] = float(q[9])
    elif q.size >= 4:
        # Minimal: nutze min, 25, 75, max
        out["q25"] = float(q[int(0.25*(q.size-1))])
        out["q50"] = float(np.median(q))
        out["q75"] = float(q[int(0.75*(q.size-1))])
        out["q5"]  = float(q[0])
        out["q95"] = float(q[-1])
    return out

# --------------------------
# Evaluator-Klasse
# --------------------------

class VolatilityModelEvaluator:
    """Evaluiert TimesFM-Modell für Volatilitätsprognosen (log(RV) → One-Step)"""

    def __init__(self, ticker: str, model=None, output_dir: str = "./evaluation"):
        self.ticker = ticker
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict = {}

    # ---------- Daten laden ----------
    def load_data(self, interval="5m", period="60d") -> Tuple[pd.Series, Dict]:
        """Lade Intraday-Daten und berechne log(RV)-Serie via compute_logrv_auto."""
        print(f"\n[{self.ticker}] Lade Daten...")
        df = fetch_intraday_cached(self.ticker, interval=interval, period=period, tz=TZ, use_cache=True)
        logrv, meta = compute_logrv_auto(df, horizons_minutes=60)

        logrv = _drop_non_finite(logrv)
        if len(logrv) < 50:
            raise ValueError(f"Zu wenig logRV-Punkte: {len(logrv)} (min. 50 benötigt)")

        print(f"[{self.ticker}] {len(logrv)} log(RV)-Punkte | Modus: {meta.get('mode')} | Step: {meta.get('step')}m")
        return logrv, meta

    # ---------- Train/Test ----------
    def train_test_split(self, logrv: pd.Series, test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
        n = len(logrv)
        split_idx = int(n * (1 - test_size))
        train = logrv.iloc[:split_idx]
        test = logrv.iloc[split_idx:]
        print(f"[{self.ticker}] Split: Train={len(train)} | Test={len(test)} ({test_size*100:.0f}%)")
        return train, test

    # ---------- Rolling-Window Prognosen ----------
    def rolling_window_forecast(
        self, logrv: pd.Series,
        initial_train_size: Optional[int] = None,
        step_size: int = 1,
        context_len: int = 128
    ) -> pd.DataFrame:

        if self.model is None:
            print(f"[{self.ticker}] Lade TimesFM-Modell...")
            self.model = load_timesfm25_torch(max_context=512, quantiles=True)

        logrv = _drop_non_finite(logrv)
        n = len(logrv)
        if initial_train_size is None:
            initial_train_size = int(n * 0.8)
        initial_train_size = max(20, min(initial_train_size, n-1))

        predictions, actuals, timestamps, contexts_used = [], [], [], []
        qcols = {k: [] for k in ["q5","q10","q25","q50","q75","q90","q95"]}

        print(f"[{self.ticker}] Rolling-Window (init={initial_train_size}, step={step_size})...")

        success = 0
        for i in range(initial_train_size, n, step_size):
            train_data = logrv.iloc[:i]
            actual_val = float(logrv.iloc[i])

            try:
                _, yhat_log, qvec, ctx = forecast_logrv_one_step(train_data, self.model, desired_ctx=context_len)
                predictions.append(float(yhat_log))
                actuals.append(actual_val)
                timestamps.append(logrv.index[i])
                contexts_used.append(int(ctx))

                qd = _quantile_columns_from_timesfm(qvec)
                for k in qcols:
                    qcols[k].append(float(qd[k]) if k in qd else np.nan)

                success += 1
                if success % 20 == 0:
                    print(f"[{self.ticker}] ... {success}/{n-initial_train_size} OK")

            except Exception as e:
                print(f"[{self.ticker}] Warnung @ idx {i}: {e}")
                continue

        if success == 0:
            raise RuntimeError(f"Keine erfolgreichen Prognosen für {self.ticker}")

        results_df = pd.DataFrame({
            "timestamp": timestamps,
            "actual": actuals,
            "predicted": predictions,
            "context_used": contexts_used
        })
        for k, arr in qcols.items():
            results_df[k] = arr

        print(f"[{self.ticker}] {len(results_df)} Prognosen erstellt")
        return results_df

    # ---------- Metriken ----------
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        actual = results_df["actual"].values
        pred   = results_df["predicted"].values

        # Log-Space
        mae  = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        mape = float(np.mean(np.abs((actual - pred) / np.where(actual != 0, actual, 1))) * 100)

        corr = float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 1 else 0.0
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

        if len(actual) > 1:
            ad = np.diff(actual)
            pdiff = np.diff(pred)
            directional_accuracy = float(np.sum((ad * pdiff) > 0) / len(ad) * 100)
        else:
            directional_accuracy = 0.0

        # RV-Space
        actual_rv = np.exp(actual)
        pred_rv   = np.exp(pred)
        mae_rv  = float(np.mean(np.abs(actual_rv - pred_rv)))
        rmse_rv = float(np.sqrt(np.mean((actual_rv - pred_rv) ** 2)))
        mape_rv = float(np.mean(np.abs((actual_rv - pred_rv) / np.where(actual_rv != 0, actual_rv, 1))) * 100)

        metrics: Dict = {
            "log_space": {
                "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2,
                "Correlation": corr, "Directional_Accuracy": directional_accuracy
            },
            "rv_space": {
                "MAE": mae_rv, "RMSE": rmse_rv, "MAPE": mape_rv
            }
        }

        # Quantile-Coverage (sofern vorhanden)
        if {"q25","q75"}.issubset(results_df.columns):
            mask50 = results_df["q25"].notna() & results_df["q75"].notna()
            if mask50.any():
                a = results_df.loc[mask50, "actual"].values
                c50 = np.mean((a >= results_df.loc[mask50, "q25"].values) &
                              (a <= results_df.loc[mask50, "q75"].values)) * 100
                metrics["quantiles"] = {"Coverage_50pct": float(c50)}

                if {"q5","q95"}.issubset(results_df.columns):
                    mask90 = results_df["q5"].notna() & results_df["q95"].notna()
                    if mask90.any():
                        a90 = results_df.loc[mask90, "actual"].values
                        c90 = np.mean((a90 >= results_df.loc[mask90, "q5"].values) &
                                      (a90 <= results_df.loc[mask90, "q95"].values)) * 100
                        metrics["quantiles"]["Coverage_90pct"] = float(c90)

        return metrics

    # ---------- Plot ----------
    def plot_comprehensive(self, results_df: pd.DataFrame, metrics: Dict, save: bool = True):
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ticker_name = f"{self.ticker} ({SMI_NAMES.get(self.ticker, self.ticker)})"
        fig.suptitle(f"Modell-Evaluierung: {ticker_name}", fontsize=16, fontweight="bold")

        # 1) Zeitreihe
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(results_df["timestamp"], results_df["actual"], label="Tatsächliche log(RV)", alpha=0.85, lw=2, color="#2c3e50")
        ax1.plot(results_df["timestamp"], results_df["predicted"], label="Vorhergesagte log(RV)", alpha=0.85, lw=2, color="#e74c3c")
        if {"q25","q75"}.issubset(results_df.columns):
            ax1.fill_between(results_df["timestamp"], results_df["q25"], results_df["q75"], alpha=0.2, color="#3498db", label="50% Intervall")
        ax1.set_title("log(RV): Prognose vs. Ist", fontweight="bold")
        ax1.set_xlabel("Zeit"); ax1.set_ylabel("log(RV)")
        ax1.grid(alpha=0.3); ax1.legend(loc="best")

        # 2) Metrikbox
        ax2 = fig.add_subplot(gs[0, 2]); ax2.axis("off")
        txt = [
            "METRIKEN",
            "=========================",
            "Log-Space:",
            f"  R²   = {metrics['log_space']['R2']:.4f}",
            f"  MAE  = {metrics['log_space']['MAE']:.4f}",
            f"  RMSE = {metrics['log_space']['RMSE']:.4f}",
            f"  Korr = {metrics['log_space']['Correlation']:.4f}",
            f"  DirA = {metrics['log_space']['Directional_Accuracy']:.1f}%",
            "",
            "RV-Space:",
            f"  MAE  = {metrics['rv_space']['MAE']:.6f}",
            f"  RMSE = {metrics['rv_space']['RMSE']:.6f}",
            f"  MAPE = {metrics['rv_space']['MAPE']:.2f}%",
        ]
        if "quantiles" in metrics:
            txt += ["", "Quantile Coverage:"]
            for k, v in metrics["quantiles"].items():
                txt += [f"  {k} = {v:.1f}%"]
        ax2.text(0.1, 0.95, "\n".join(txt), transform=ax2.transAxes, fontsize=10,
                 va="top", family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

        # 3) Scatter
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(results_df["actual"], results_df["predicted"], alpha=0.55, s=30, color="#3498db")
        mn = min(results_df["actual"].min(), results_df["predicted"].min())
        mx = max(results_df["actual"].max(), results_df["predicted"].max())
        ax3.plot([mn, mx], [mn, mx], "r--", lw=2, alpha=0.7, label="Perfekt")
        ax3.set_title("Scatter: Prognose vs. Ist", fontweight="bold")
        ax3.set_xlabel("Ist"); ax3.set_ylabel("Prognose"); ax3.grid(alpha=0.3); ax3.legend()

        # 4) Fehlerverteilung
        ax4 = fig.add_subplot(gs[1, 1])
        errors = results_df["actual"] - results_df["predicted"]
        ax4.hist(errors, bins=30, alpha=0.8, edgecolor="black", color="#95a5a6")
        ax4.axvline(0, color="r", ls="--", lw=2, label="0")
        ax4.axvline(errors.mean(), color="g", ls="--", lw=1.5, label=f"μ={errors.mean():.4f}")
        ax4.set_title(f"Fehlerverteilung (μ={errors.mean():.4f}, σ={errors.std():.4f})", fontweight="bold")
        ax4.set_xlabel("Fehler"); ax4.set_ylabel("Häufigkeit"); ax4.grid(alpha=0.3); ax4.legend()

        # 5) Residuen über Zeit
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.scatter(results_df["timestamp"], errors, alpha=0.5, s=18, color="#e67e22")
        ax5.axhline(0, color="r", ls="--", alpha=0.7, lw=2)
        ax5.axhline(errors.mean(), color="g", ls="--", alpha=0.5, lw=1)
        ax5.set_title("Residuen über Zeit", fontweight="bold")
        ax5.set_xlabel("Zeit"); ax5.set_ylabel("Fehler"); ax5.grid(alpha=0.3)

        # 6) RV-Space (letzte 50)
        ax6 = fig.add_subplot(gs[2, 0])
        actual_rv = np.exp(results_df["actual"])
        pred_rv   = np.exp(results_df["predicted"])
        sl = slice(-min(50, len(results_df)), None)
        ax6.plot(results_df["timestamp"].iloc[sl], actual_rv.iloc[sl], lw=2, marker="o", ms=3, label="Ist")
        ax6.plot(results_df["timestamp"].iloc[sl], pred_rv.iloc[sl], lw=2, marker="s", ms=3, label="Prog")
        ax6.set_title("RV-Space: Letzte 50 Werte", fontweight="bold")
        ax6.set_xlabel("Zeit"); ax6.set_ylabel("Realized Volatility"); ax6.legend(); ax6.grid(alpha=0.3)

        # 7) |Fehler|
        ax7 = fig.add_subplot(gs[2, 1])
        abs_err = np.abs(errors)
        ax7.plot(results_df["timestamp"], abs_err, lw=1.2, alpha=0.7, color="#9b59b6")
        ax7.axhline(abs_err.mean(), color="r", ls="--", lw=2, label=f"μ={abs_err.mean():.4f}")
        ax7.set_title("Absolute Fehler über Zeit", fontweight="bold")
        ax7.set_xlabel("Zeit"); ax7.set_ylabel("|Fehler|"); ax7.legend(); ax7.grid(alpha=0.3)

        # 8) QQ-Plot
        ax8 = fig.add_subplot(gs[2, 2])
        try:
            from scipy import stats
            stats.probplot(errors, dist="norm", plot=ax8)
        except Exception:
            ax8.text(0.5, 0.5, "scipy nicht verfügbar", ha="center", va="center")
        ax8.set_title("Q-Q Plot (Normalität der Fehler)", fontweight="bold")
        ax8.grid(alpha=0.3)

        if save:
            path = self.output_dir / f"{self.ticker.replace('.', '_')}_complete_analysis.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[{self.ticker}] Plot gespeichert: {path}")
            plt.close(fig)
        return fig

    # ---------- Run ----------
    def run_evaluation(self, test_size: float = 0.2, rolling_window: bool = True, context_len: int = 128) -> Optional[Dict]:
        print("\n" + "="*70)
        print(f"MODELL-EVALUIERUNG: {self.ticker} ({SMI_NAMES.get(self.ticker, self.ticker)})")
        print("="*70)

        try:
            logrv, meta = self.load_data()
            self.results["meta"] = meta

            if rolling_window:
                results_df = self.rolling_window_forecast(
                    logrv,
                    initial_train_size=int(len(logrv) * (1 - test_size)),
                    step_size=1,
                    context_len=context_len
                )
            else:
                train, test = self.train_test_split(logrv, test_size)
                if self.model is None:
                    self.model = load_timesfm25_torch(max_context=512, quantiles=True)
                preds = []
                for i in range(len(test)):
                    _, yhat_log, _, _ = forecast_logrv_one_step(train, self.model, desired_ctx=context_len)
                    preds.append(float(yhat_log))
                    train = pd.concat([train, pd.Series([test.iloc[i]], index=[test.index[i]])])
                results_df = pd.DataFrame({"timestamp": test.index, "actual": test.values, "predicted": preds})

            print(f"[{self.ticker}] Berechne Metriken...")
            metrics = self.calculate_metrics(results_df)
            self.results["metrics"] = metrics
            self.results["predictions"] = results_df

            # stdout Zusammenfassung
            print("\n" + "="*70)
            print(f"ERGEBNISSE: {self.ticker}")
            print("="*70)
            print("\nLog-Space:")
            for k, v in metrics["log_space"].items():
                print(f"  {k:25s}: {v:.4f}")
            print("\nRV-Space:")
            for k, v in metrics["rv_space"].items():
                print(f"  {k:25s}: {v:.4f}")
            if "quantiles" in metrics:
                print("\nQuantile Coverage:")
                for k, v in metrics["quantiles"].items():
                    print(f"  {k:25s}: {v:.2f}%")

            print(f"\n[{self.ticker}] Erstelle Visualisierungen...")
            self.plot_comprehensive(results_df, metrics)

            self.save_results()
            print(f"\n[{self.ticker}] Evaluierung abgeschlossen! Ergebnisse: {self.output_dir}")
            return self.results

        except Exception as e:
            print(f"\n[{self.ticker}] FEHLER: {e}")
            traceback.print_exc()
            return None

    # ---------- Speichern ----------
    def save_results(self):
        try:
            metrics_file = self.output_dir / f"{self.ticker.replace('.', '_')}_metrics.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(self.results["metrics"], f, indent=2, ensure_ascii=False)

            csv_file = self.output_dir / f"{self.ticker.replace('.', '_')}_predictions.csv"
            self.results["predictions"].to_csv(csv_file, index=False, encoding="utf-8")

            print(f"[{self.ticker}] Dateien gespeichert: {metrics_file.name}, {csv_file.name}")
        except Exception as e:
            print(f"[{self.ticker}] Fehler beim Speichern: {e}")


# --------------------------
# Multi-Evaluierung
# --------------------------

def evaluate_multiple_tickers(tickers: List[str], output_dir: str = "./evaluation_multi") -> Dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Optional[Dict]] = {}

    print("\n" + "="*70)
    print("INITIALISIERUNG")
    print("="*70)
    try:
        model = load_timesfm25_torch(max_context=512, quantiles=True)
        print("TimesFM-Modell erfolgreich geladen")
    except Exception as e:
        print(f"Warnung: TimesFM konnte nicht geladen werden: {e}")
        print("Fahre ohne vorab geladenes Modell fort...")
        model = None

    for i, t in enumerate(tickers, 1):
        print("\n" + "#"*70)
        print(f"# [{i}/{len(tickers)}] {t} ({SMI_NAMES.get(t, t)})")
        print("#"*70)

        try:
            evaluator = VolatilityModelEvaluator(t, model=model, output_dir=out / t.replace(".", "_"))
            res = evaluator.run_evaluation(test_size=0.2, rolling_window=True)
            all_results[t] = res["metrics"] if res is not None else None
        except Exception as e:
            print(f"[{t}] KRITISCHER FEHLER: {e}")
            traceback.print_exc()
            all_results[t] = None

    print("\n" + "="*70)
    print("ZUSAMMENFASSUNG ALLER AKTIEN")
    print("="*70)

    ok = {k: v for k, v in all_results.items() if v is not None}
    if not ok:
        print("\nKeine erfolgreichen Evaluierungen! Abbruch.")
        return all_results

    print(f"\nErfolgreich evaluiert: {len(ok)}/{len(tickers)} Aktien")

    summary_rows = []
    for t, m in ok.items():
        summary_rows.append({
            "Ticker": t,
            "Name": SMI_NAMES.get(t, t),
            "R²": m["log_space"]["R2"],
            "RMSE": m["log_space"]["RMSE"],
            "MAE": m["log_space"]["MAE"],
            "Korr.": m["log_space"]["Correlation"],
            "Dir.Acc.%": m["log_space"]["Directional_Accuracy"],
            "MAPE_RV%": m["rv_space"]["MAPE"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = out / "summary_all_tickers.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8")

    print("\n" + summary_df.to_string(index=False))
    print(f"\nSummary gespeichert: {summary_file}")

    if len(ok) >= 2:
        create_comparison_plots(ok, out)

    return all_results


def create_comparison_plots(results: Dict, output_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Modell-Performance: Vergleich aller SMI-Aktien", fontsize=16, fontweight="bold")

    metrics_cfg = [
        ("R2", "log_space", "R² (Bestimmtheitsmaß)", 0),
        ("RMSE", "log_space", "RMSE (Log-Space)", 1),
        ("MAE", "log_space", "MAE (Log-Space)", 2),
        ("Correlation", "log_space", "Korrelation", 3),
        ("Directional_Accuracy", "log_space", "Direktionale Genauigkeit (%)", 4),
        ("MAPE", "rv_space", "MAPE (RV-Space, %)", 5),
    ]

    for mkey, space, title, idx in metrics_cfg:
        ax = axes[idx // 3, idx % 3]
        labels, values = [], []
        for t, m in results.items():
            if m is not None and space in m and mkey in m[space]:
                labels.append(t.split(".")[0])
                values.append(m[space][mkey])

        if values:
            median_val = float(np.median(values))
            colors = ["#2ecc71" if ((mkey != "RMSE" and v >= median_val) or (mkey == "RMSE" and v <= median_val))
                      else "#e74c3c" for v in values]
            bars = ax.barh(labels, values, color=colors, alpha=0.8)
            ax.set_xlabel("Wert"); ax.set_title(title, fontweight="bold"); ax.grid(axis="x", alpha=0.3)
            ax.axvline(median_val, color="blue", linestyle="--", linewidth=2, alpha=0.5, label=f"Median={median_val:.3f}")
            ax.legend()

    plt.tight_layout()
    out_file = output_path / "comparison_all_tickers.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Vergleichs-Plot gespeichert: {out_file}")
    plt.close(fig)


# ================== MAIN ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimesFM Volatilitätsmodell Evaluierung")
    parser.add_argument("--ticker", type=str, default="NESN.SW", help="Ticker für Einzel-Evaluierung")
    parser.add_argument("--multi", action="store_true", help="Mehrere SMI-Ticker evaluieren")
    parser.add_argument("--all", action="store_true", help="Bei --multi alle SMI-Ticker statt nur Top 5")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test-Set Größe (0..1)")
    parser.add_argument("--output", type=str, default="./evaluation", help="Output-Verzeichnis")
    args = parser.parse_args()

    if args.multi:
        selected = SMI_TICKERS if args.all else SMI_TICKERS[:5]
        print(f"\n🎯 Evaluiere {len(selected)} Aktien: {', '.join(selected)}\n")
        evaluate_multiple_tickers(selected, output_dir=args.output)
    else:
        evaluator = VolatilityModelEvaluator(args.ticker, output_dir=args.output)
        evaluator.run_evaluation(test_size=args.test_size, rolling_window=True)
