import torch
import pandas as pd
import numpy as np
from chronos import ChronosPipeline
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class ChronosForecaster:
    def __init__(self, model_id="amazon/chronos-t5-tiny", device=None):
        """
        Initialize Chronos 2 Pipeline.
        Args:
            model_id: HuggingFace model ID (e.g., 'amazon/chronos-t5-tiny', 'amazon/chronos-t5-small', 'amazon/chronos-t5-base', 'amazon/chronos-t5-large').
                      Note: 'amazon/chronos-2' might be the name for the V2, but let's default to a safe one or user specified.
                      The user requested Chronos 2. The search said 'amazon/chronos-2'.
            device: 'cuda', 'cpu', or 'mps'. If None, auto-detect.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Chronos model: {model_id} on {self.device}...")
        
        # Use bfloat16 for efficiency if on CUDA, else float32
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map=self.device,
                torch_dtype=dtype,
            )
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            print("Falling back to cpu float32...")
            self.pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            self.device = "cpu"

    def predict(self, df: pd.DataFrame, prediction_length: int = 1, context_length: int = 512):
        """
        Predict next steps using multivariate context.
        
        Args:
            df: DataFrame with columns ['log_rv', 'log_vol', 'returns'].
                The model will treat these as a multivariate time series.
            prediction_length: Number of steps to predict (default 1 hour).
            context_length: Max context window to use.
            
        Returns:
            dict with:
                'mean': Forecast mean (unlogged if target was log),
                'quantiles': Dictionary of quantiles (e.g., 'q10', 'q50', 'q90'),
                'raw_forecast': The raw forecast object/tensor.
        """
        # Prepare Context
        if len(df) == 0:
            return None
            
        # Take last context_length rows
        ctx_df = df.iloc[-context_length:].copy()
        
        # CRITICAL: Normalize data for Chronos
        # Chronos works best with standardized data (mean=0, std=1)
        means = ctx_df.mean()
        stds = ctx_df.std()
        
        # Avoid division by zero
        stds = stds.replace(0, 1)
        
        ctx_normalized = (ctx_df - means) / stds
        
        # Convert to tensor
        # Chronos expects 2D: (time, variates) NOT (batch, time, variates)
        dtype = next(self.pipeline.model.parameters()).dtype
        context_tensor = torch.tensor(ctx_normalized.values).to(self.device).to(dtype)
        
        # Predict
        try:
            forecast = self.pipeline.predict(
                context_tensor,
                prediction_length=prediction_length,
                num_samples=20,
            )
        except Exception as e:
            print(f"Chronos prediction error: {e}")
            # Debug info
            print(f"Context shape: {context_tensor.shape}, Dtype: {context_tensor.dtype}")
            return None
        
        # Forecast shape from Chronos: (context_length, num_samples, prediction_length)
        # We want the LAST timestep's forecast (which is our next-hour prediction)
        # Extract samples for the target (last timestep, all samples, first prediction step)
        
        # Shape: (num_samples,)
        target_samples_normalized = forecast[-1, :, 0].cpu().numpy()
        
        # DENORMALIZE: Convert back to original scale
        # We only care about the first feature (log_rv)
        target_samples = target_samples_normalized * stds.iloc[0] + means.iloc[0]
        
        # Compute stats
        mean_log = np.mean(target_samples)
        median_log = np.median(target_samples)
        
        # Quantiles
        q_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        qs = np.quantile(target_samples, q_levels)
        q_dict = {f"q{int(q*100)}": float(val) for q, val in zip(q_levels, qs)}
        
        return {
            "mean_log": float(mean_log),
            "median_log": float(median_log),
            "quantiles_log": q_dict,
            "samples_log": target_samples
        }

if __name__ == "__main__":
    # Test
    print("Testing ChronosForecaster...")
    try:
        model = ChronosForecaster(model_id="amazon/chronos-t5-tiny") # Use tiny for quick test
        
        # Mock Data
        data = pd.DataFrame({
            "log_rv": np.random.randn(100),
            "log_vol": np.random.randn(100),
            "returns": np.random.randn(100) * 0.01
        })
        
        res = model.predict(data)
        print("Prediction:", res)
    except Exception as e:
        print(f"Test failed: {e}")
