import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from data import fetch_hourly_multivariate, SMI_TICKERS, SMI_NAMES
from model import ChronosForecaster

# Settings
OUTPUT_DIR = Path("./evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ChronosEvaluator:
    def __init__(self, ticker, model=None):
        self.ticker = ticker
        self.model = model if model else ChronosForecaster(model_id="amazon/chronos-t5-large")
        
    def load_data(self):
        print(f"[{self.ticker}] Loading data...")
        # Fetch more days for evaluation to have a good test set
        return fetch_hourly_multivariate(self.ticker, days=120, use_cache=True)
        
    def evaluate(self, test_size=0.2):
        df = self.load_data()
        if len(df) < 50:
            print(f"[{self.ticker}] Not enough data ({len(df)} rows). Skipping.")
            return None
            
        # Split
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"[{self.ticker}] Evaluating on {len(test_df)} hours...")
        
        results = []
        
        # Rolling Forecast
        # We iterate through the test set, feeding the growing history
        # For efficiency, we might step every N hours, but let's do step=1 for accuracy
        
        history = train_df.copy()
        
        for i in range(len(test_df)):
            # Current ground truth to predict is test_df.iloc[i]
            # Context is history
            
            # Predict next step (which corresponds to test_df.iloc[i])
            # Wait, Chronos predicts the *future*.
            # So if we have history up to T, we predict T+1.
            # test_df[i] is T+1.
            
            target_log_rv = test_df.iloc[i]["log_rv"]
            
            # Predict
            pred = self.model.predict(history, prediction_length=1, context_length=60)
            
            if pred:
                results.append({
                    "timestamp": test_df.index[i],
                    "actual_log_rv": target_log_rv,
                    "pred_log_rv": pred["mean_log"],
                    "q10": pred["quantiles_log"]["q10"],
                    "q90": pred["quantiles_log"]["q90"]
                })
            
            # Update history with the *actual* observation to simulate real-time
            # We append the row we just predicted (or rather, the actual value of it)
            # so it becomes context for the next step.
            history = pd.concat([history, test_df.iloc[[i]]])
            
            if i % 10 == 0:
                print(f"[{self.ticker}] Step {i}/{len(test_df)}")
                
        results_df = pd.DataFrame(results)
        return results_df

    def calculate_metrics(self, df):
        if df is None or df.empty:
            return {}
            
        actual = df["actual_log_rv"]
        pred = df["pred_log_rv"]
        
        mse = np.mean((actual - pred)**2)
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(mse)
        
        # Directional Accuracy
        actual_diff = np.diff(actual)
        pred_diff = np.diff(pred)
        # Align lengths
        if len(actual_diff) > 0:
            da = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100
        else:
            da = 0
            
        return {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "Directional_Accuracy": da
        }

    def plot_results(self, df, metrics):
        if df is None or df.empty:
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(df["timestamp"], df["actual_log_rv"], label="Actual log(RV)", color="black", alpha=0.7)
        plt.plot(df["timestamp"], df["pred_log_rv"], label="Predicted log(RV)", color="blue", alpha=0.7)
        
        # Quantiles
        plt.fill_between(df["timestamp"], df["q10"], df["q90"], color="blue", alpha=0.1, label="10-90% Confidence")
        
        plt.title(f"Evaluation: {self.ticker} (RMSE: {metrics['RMSE']:.4f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = OUTPUT_DIR / f"{self.ticker}_eval.png"
        plt.savefig(out_path)
        print(f"[{self.ticker}] Plot saved to {out_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="NESN.SW")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    tickers = SMI_TICKERS if args.all else [args.ticker]
    
    # Load model once
    model = ChronosForecaster(model_id="amazon/chronos-t5-large")
    
    summary = []
    
    for t in tickers:
        evaluator = ChronosEvaluator(t, model)
        res_df = evaluator.evaluate()
        metrics = evaluator.calculate_metrics(res_df)
        
        if metrics:
            evaluator.plot_results(res_df, metrics)
            metrics["ticker"] = t
            summary.append(metrics)
            print(f"[{t}] Metrics: {metrics}")
            
    if summary:
        sum_df = pd.DataFrame(summary)
        print("\nSummary:")
        print(sum_df)
        sum_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

if __name__ == "__main__":
    main()
