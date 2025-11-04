# model_evaluation.py — Umfassendes Testing für TimesFM Volatilitätsmodell
# Enthält: Train/Test-Split, Rolling-Window-Validierung, Metriken, Visualisierungen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import json

# Deine existierenden Imports
from app import (
    fetch_intraday_cached, compute_logrv_auto, load_timesfm25_torch,
    forecast_logrv_one_step, SMI_TICKERS, SMI_NAMES, TZ
)


class VolatilityModelEvaluator:
    """Evaluiert TimesFM-Modell für Volatilitätsprognosen"""
    
    def __init__(self, ticker: str, model=None, output_dir: str = "./evaluation"):
        self.ticker = ticker
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def load_data(self, interval="5m", period="60d"):
        """Lade und bereite Daten vor"""
        print(f"📊 Lade Daten für {self.ticker}...")
        df = fetch_intraday_cached(self.ticker, interval=interval, period=period, tz=TZ)
        logrv, meta = compute_logrv_auto(df, horizons_minutes=60)
        
        if len(logrv) < 50:
            raise ValueError(f"Zu wenig Datenpunkte: {len(logrv)} (min. 50 benötigt)")
        
        print(f"✓ {len(logrv)} log(RV) Punkte geladen | Modus: {meta.get('mode')} | Step: {meta.get('step')}m")
        return logrv, meta
    
    def train_test_split(self, logrv: pd.Series, test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
        """Einfacher Train/Test-Split"""
        n = len(logrv)
        split_idx = int(n * (1 - test_size))
        train = logrv.iloc[:split_idx]
        test = logrv.iloc[split_idx:]
        
        print(f"📈 Split: Train={len(train)} | Test={len(test)} ({test_size*100:.0f}%)")
        return train, test
    
    def rolling_window_forecast(self, logrv: pd.Series, 
                                 initial_train_size: int = None,
                                 step_size: int = 1,
                                 context_len: int = 128) -> pd.DataFrame:
        """
        Rolling-Window One-Step-Ahead Prognosen
        
        Args:
            logrv: Vollständige Zeitreihe
            initial_train_size: Initiale Trainingsgröße (None = 80% der Daten)
            step_size: Schrittweite für Rolling Window
            context_len: Kontextlänge für Modell
        """
        if self.model is None:
            print("🔄 Lade TimesFM-Modell...")
            self.model = load_timesfm25_torch(max_context=512, quantiles=True)
        
        n = len(logrv)
        if initial_train_size is None:
            initial_train_size = int(n * 0.8)
        
        predictions = []
        actuals = []
        timestamps = []
        contexts_used = []
        quantiles_list = []
        
        print(f"🔮 Starte Rolling-Window-Prognosen (init={initial_train_size}, step={step_size})...")
        
        for i in range(initial_train_size, n, step_size):
            train_data = logrv.iloc[:i]
            actual_val = logrv.iloc[i]
            
            try:
                _, yhat_log, qvec, ctx = forecast_logrv_one_step(
                    train_data, self.model, desired_ctx=context_len
                )
                
                predictions.append(yhat_log)
                actuals.append(actual_val)
                timestamps.append(logrv.index[i])
                contexts_used.append(ctx)
                quantiles_list.append(qvec if qvec is not None else None)
                
                if (i - initial_train_size + 1) % 10 == 0:
                    print(f"  ... Prognose {i - initial_train_size + 1}/{n - initial_train_size}")
                    
            except Exception as e:
                print(f"⚠️ Fehler bei Index {i}: {e}")
                continue
        
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'actual': actuals,
            'predicted': predictions,
            'context_used': contexts_used
        })
        
        # Quantile als separate Spalten
        if quantiles_list and quantiles_list[0] is not None:
            for i, q in enumerate([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
                results_df[f'q{int(q*100)}'] = [
                    float(qv[i]) if qv is not None and len(qv) > i else np.nan 
                    for qv in quantiles_list
                ]
        
        print(f"✓ {len(results_df)} erfolgreiche Prognosen erstellt")
        return results_df
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Berechne Evaluierungsmetriken"""
        actual = results_df['actual'].values
        pred = results_df['predicted'].values
        
        # Basis-Metriken (log-Space)
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        # Korrelation
        corr = np.corrcoef(actual, pred)[0, 1]
        
        # R² Score
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Direktionale Genauigkeit (Trend richtig vorhergesagt?)
        if len(actual) > 1:
            actual_diff = np.diff(actual)
            pred_diff = np.diff(pred)
            direction_correct = np.sum((actual_diff * pred_diff) > 0)
            directional_accuracy = direction_correct / len(actual_diff) * 100
        else:
            directional_accuracy = np.nan
        
        # In RV-Space (exp transformiert)
        actual_rv = np.exp(actual)
        pred_rv = np.exp(pred)
        
        mae_rv = np.mean(np.abs(actual_rv - pred_rv))
        rmse_rv = np.sqrt(np.mean((actual_rv - pred_rv) ** 2))
        mape_rv = np.mean(np.abs((actual_rv - pred_rv) / actual_rv)) * 100
        
        metrics = {
            'log_space': {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2,
                'Correlation': corr,
                'Directional_Accuracy': directional_accuracy
            },
            'rv_space': {
                'MAE': mae_rv,
                'RMSE': rmse_rv,
                'MAPE': mape_rv
            }
        }
        
        # Quantile Coverage (falls vorhanden)
        if 'q25' in results_df.columns and 'q75' in results_df.columns:
            in_50pct = np.sum((actual >= results_df['q25']) & (actual <= results_df['q75']))
            coverage_50 = in_50pct / len(actual) * 100
            metrics['quantiles'] = {'Coverage_50pct': coverage_50}
            
            if 'q5' in results_df.columns and 'q95' in results_df.columns:
                in_90pct = np.sum((actual >= results_df['q5']) & (actual <= results_df['q95']))
                coverage_90 = in_90pct / len(actual) * 100
                metrics['quantiles']['Coverage_90pct'] = coverage_90
        
        return metrics
    
    def plot_predictions(self, results_df: pd.DataFrame, save: bool = True):
        """Visualisiere Prognosen vs. Ist-Werte"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Zeitreihe: Prognose vs. Ist
        ax = axes[0, 0]
        ax.plot(results_df['timestamp'], results_df['actual'], 
                label='Ist-Werte', alpha=0.7, linewidth=1.5)
        ax.plot(results_df['timestamp'], results_df['predicted'], 
                label='Prognosen', alpha=0.7, linewidth=1.5)
        
        # Konfidenzintervall (falls vorhanden)
        if 'q25' in results_df.columns:
            ax.fill_between(results_df['timestamp'], 
                           results_df['q25'], results_df['q75'],
                           alpha=0.2, label='50% PI')
        
        ax.set_title('log(RV): Prognose vs. Ist-Werte')
        ax.set_xlabel('Zeit')
        ax.set_ylabel('log(RV)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Scatter: Prognose vs. Ist
        ax = axes[0, 1]
        ax.scatter(results_df['actual'], results_df['predicted'], 
                  alpha=0.5, s=20)
        
        # Perfekte Linie
        min_val = min(results_df['actual'].min(), results_df['predicted'].min())
        max_val = max(results_df['actual'].max(), results_df['predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', label='Perfekte Prognose', alpha=0.7)
        
        ax.set_title('Scatter: Prognose vs. Ist')
        ax.set_xlabel('Ist-Werte')
        ax.set_ylabel('Prognosen')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Fehlerverteilung
        ax = axes[1, 0]
        errors = results_df['actual'] - results_df['predicted']
        ax.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', label='Null-Fehler')
        ax.set_title(f'Fehlerverteilung (μ={errors.mean():.4f}, σ={errors.std():.4f})')
        ax.set_xlabel('Fehler (Ist - Prognose)')
        ax.set_ylabel('Häufigkeit')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Residuen über Zeit
        ax = axes[1, 1]
        ax.scatter(results_df['timestamp'], errors, alpha=0.5, s=20)
        ax.axhline(0, color='r', linestyle='--', alpha=0.7)
        ax.set_title('Residuen über Zeit')
        ax.set_xlabel('Zeit')
        ax.set_ylabel('Fehler')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"{self.ticker.replace('.', '_')}_predictions.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"💾 Plot gespeichert: {filepath}")
        
        return fig
    
    def plot_metrics_comparison(self, metrics: Dict, save: bool = True):
        """Visualisiere Metriken"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Log-Space Metriken
        ax = axes[0]
        log_metrics = metrics['log_space']
        names = ['MAE', 'RMSE', 'MAPE', 'R²', 'Korr.', 'Dir.Acc.']
        values = [
            log_metrics['MAE'],
            log_metrics['RMSE'],
            log_metrics['MAPE'] / 10,  # Skalierung für Visualisierung
            log_metrics['R2'],
            log_metrics['Correlation'],
            log_metrics['Directional_Accuracy'] / 100
        ]
        
        colors = ['#2ecc71' if v > 0.7 else '#e74c3c' if v < 0.3 else '#f39c12' 
                 for v in values]
        
        ax.barh(names, values, color=colors, alpha=0.7)
        ax.set_xlabel('Wert')
        ax.set_title('Metriken (log-Space)')
        ax.grid(axis='x', alpha=0.3)
        
        # RV-Space Metriken
        ax = axes[1]
        rv_metrics = metrics['rv_space']
        names = ['MAE', 'RMSE', 'MAPE']
        values = [rv_metrics['MAE'], rv_metrics['RMSE'], rv_metrics['MAPE'] / 10]
        
        ax.barh(names, values, color='#3498db', alpha=0.7)
        ax.set_xlabel('Wert')
        ax.set_title('Metriken (RV-Space)')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"{self.ticker.replace('.', '_')}_metrics.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"💾 Metriken-Plot gespeichert: {filepath}")
        
        return fig
    
    def run_evaluation(self, test_size: float = 0.2, 
                      rolling_window: bool = True,
                      context_len: int = 128) -> Dict:
        """Führe komplette Evaluierung durch"""
        print(f"\n{'='*60}")
        print(f"🔬 MODELL-EVALUIERUNG: {self.ticker}")
        print(f"{'='*60}\n")
        
        # Lade Daten
        logrv, meta = self.load_data()
        self.results['meta'] = meta
        
        if rolling_window:
            # Rolling-Window Validierung
            results_df = self.rolling_window_forecast(
                logrv, 
                initial_train_size=int(len(logrv) * (1 - test_size)),
                step_size=1,
                context_len=context_len
            )
        else:
            # Einfacher Train/Test-Split
            train, test = self.train_test_split(logrv, test_size)
            
            if self.model is None:
                self.model = load_timesfm25_torch(max_context=512, quantiles=True)
            
            predictions = []
            for i in range(len(test)):
                _, yhat_log, _, _ = forecast_logrv_one_step(
                    train, self.model, desired_ctx=context_len
                )
                predictions.append(yhat_log)
                train = pd.concat([train, pd.Series([test.iloc[i]], index=[test.index[i]])])
            
            results_df = pd.DataFrame({
                'timestamp': test.index,
                'actual': test.values,
                'predicted': predictions
            })
        
        # Berechne Metriken
        print("\n📊 Berechne Metriken...")
        metrics = self.calculate_metrics(results_df)
        self.results['metrics'] = metrics
        self.results['predictions'] = results_df
        
        # Ausgabe
        print(f"\n{'='*60}")
        print("📈 ERGEBNISSE (log-Space):")
        print(f"{'='*60}")
        for key, val in metrics['log_space'].items():
            print(f"  {key:25s}: {val:.4f}")
        
        print(f"\n{'='*60}")
        print("📈 ERGEBNISSE (RV-Space):")
        print(f"{'='*60}")
        for key, val in metrics['rv_space'].items():
            print(f"  {key:25s}: {val:.4f}")
        
        if 'quantiles' in metrics:
            print(f"\n{'='*60}")
            print("📈 QUANTILE COVERAGE:")
            print(f"{'='*60}")
            for key, val in metrics['quantiles'].items():
                print(f"  {key:25s}: {val:.2f}%")
        
        # Visualisierungen
        print("\n🎨 Erstelle Visualisierungen...")
        self.plot_predictions(results_df)
        self.plot_metrics_comparison(metrics)
        
        # Speichere Ergebnisse
        self.save_results()
        
        print(f"\n✅ Evaluierung abgeschlossen!")
        print(f"📁 Ergebnisse gespeichert in: {self.output_dir}")
        
        return self.results
    
    def save_results(self):
        """Speichere Ergebnisse als JSON und CSV"""
        # Metriken als JSON
        metrics_file = self.output_dir / f"{self.ticker.replace('.', '_')}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
        
        # Prognosen als CSV
        csv_file = self.output_dir / f"{self.ticker.replace('.', '_')}_predictions.csv"
        self.results['predictions'].to_csv(csv_file, index=False)
        
        print(f"💾 Metriken: {metrics_file}")
        print(f"💾 Prognosen: {csv_file}")


def evaluate_multiple_tickers(tickers: List[str], output_dir: str = "./evaluation_multi"):
    """Evaluiere mehrere Ticker und vergleiche"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    model = load_timesfm25_torch(max_context=512, quantiles=True)
    
    for ticker in tickers:
        print(f"\n{'#'*60}")
        print(f"# Evaluiere: {ticker} ({SMI_NAMES.get(ticker, ticker)})")
        print(f"{'#'*60}")
        
        try:
            evaluator = VolatilityModelEvaluator(
                ticker, 
                model=model,
                output_dir=output_path / ticker.replace('.', '_')
            )
            results = evaluator.run_evaluation(test_size=0.2, rolling_window=True)
            all_results[ticker] = results['metrics']
        except Exception as e:
            print(f"❌ Fehler bei {ticker}: {e}")
            all_results[ticker] = None
    
    # Vergleichs-Plot
    print("\n🎨 Erstelle Vergleichs-Visualisierung...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('R2', 'log_space', 'R²'),
        ('RMSE', 'log_space', 'RMSE'),
        ('Directional_Accuracy', 'log_space', 'Direktionale Genauigkeit (%)'),
        ('MAPE', 'rv_space', 'MAPE (RV-Space)')
    ]
    
    for idx, (metric_key, space, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        tickers_valid = []
        values = []
        
        for ticker, result in all_results.items():
            if result is not None and space in result and metric_key in result[space]:
                tickers_valid.append(ticker.split('.')[0])
                val = result[space][metric_key]
                if metric_key == 'Directional_Accuracy':
                    val = val  # Already in percentage
                values.append(val)
        
        colors = ['#2ecc71' if v > np.median(values) else '#e74c3c' 
                 for v in values]
        
        ax.barh(tickers_valid, values, color=colors, alpha=0.7)
        ax.set_xlabel('Wert')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    comparison_file = output_path / "comparison_all_tickers.png"
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    print(f"💾 Vergleichs-Plot: {comparison_file}")
    
    # Summary-Tabelle
    summary_data = []
    for ticker, result in all_results.items():
        if result is not None:
            summary_data.append({
                'Ticker': ticker,
                'Name': SMI_NAMES.get(ticker, ticker),
                'R²': result['log_space']['R2'],
                'RMSE': result['log_space']['RMSE'],
                'MAE': result['log_space']['MAE'],
                'Dir.Acc.': result['log_space']['Directional_Accuracy'],
                'MAPE_RV': result['rv_space']['MAPE']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / "summary_all_tickers.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Summary-Tabelle: {summary_file}")
    print("\n" + "="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    
    return all_results


# ============ MAIN ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TimesFM Volatilitätsmodell Evaluierung")
    parser.add_argument("--ticker", type=str, default="NESN.SW", 
                       help="Ticker für Evaluierung (default: NESN.SW)")
    parser.add_argument("--multi", action="store_true",
                       help="Evaluiere alle SMI-Ticker")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test-Set Größe (default: 0.2)")
    parser.add_argument("--output", type=str, default="./evaluation",
                       help="Output-Verzeichnis (default: ./evaluation)")
    
    args = parser.parse_args()
    
    if args.multi:
        # Evaluiere mehrere SMI-Ticker
        # Default: Erste 5 (oder mit --all für alle 13)
        if hasattr(args, 'all') and args.all:
            selected_tickers = SMI_TICKERS
        else:
            selected_tickers = SMI_TICKERS[:5]
        
        print(f"\n🎯 Evaluiere {len(selected_tickers)} Aktien: {', '.join(selected_tickers)}\n")
        evaluate_multiple_tickers(selected_tickers, output_dir=args.output)
    else:
        # Einzelne Aktie
        evaluator = VolatilityModelEvaluator(args.ticker, output_dir=args.output)
        evaluator.run_evaluation(test_size=args.test_size, rolling_window=True)
