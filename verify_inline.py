import torch
import pandas as pd
import numpy as np
from chronos import ChronosPipeline
import sys

print("Starting inline verification...", flush=True)

try:
    print("Loading model...", flush=True)
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print("Model loaded.", flush=True)

    # Mock Data
    print("Preparing data...", flush=True)
    df = pd.DataFrame({
        "log_rv": np.random.randn(60).astype(np.float32),
        "log_vol": np.random.randn(60).astype(np.float32),
        "returns": np.random.randn(60).astype(np.float32)
    })
    
    context_tensor = torch.tensor(df.values)
    print(f"Context shape: {context_tensor.shape}, Dtype: {context_tensor.dtype}", flush=True)
    
    print("Predicting...", flush=True)
    forecast = pipeline.predict(
        context_tensor,
        prediction_length=1,
        num_samples=20,
    )
    print(f"Forecast shape: {forecast.shape}", flush=True)
    print("Success!", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
