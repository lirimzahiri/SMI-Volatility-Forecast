from model import ChronosForecaster
from data import fetch_hourly_multivariate
import pandas as pd
import numpy as np

def verify():
    print("1. Initializing Model...")
    try:
        # Use tiny for faster verification if possible, but let's stick to what we used in app
        model = ChronosForecaster(model_id="amazon/chronos-t5-tiny")
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return

    print("\n2. Fetching Data...")
    try:
        df = fetch_hourly_multivariate("NESN.SW", days=10)
        print(f"Data fetched. Shape: {df.shape}")
        print(df.head())
        if df.empty:
            print("Data is empty! Verification failed.")
            return
    except Exception as e:
        print(f"Data fetching failed: {e}")
        return

    print("\n3. Running Prediction...")
    try:
        pred = model.predict(df, prediction_length=1)
        print("Prediction result:")
        print(pred)
        
        if pred and "mean_log" in pred:
            print(f"Forecast (log): {pred['mean_log']:.4f}")
            print(f"Forecast (RV %): {np.exp(pred['mean_log'])*100:.2f}%")
        else:
            print("Prediction returned empty or invalid format.")
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    print("\nVerification Complete!")

if __name__ == "__main__":
    verify()
