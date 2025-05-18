import torch
from torch.utils.data import Dataset
import polars as pl
from typing import List, Dict, Tuple, Any
import numpy as np

class EscapeRoomTimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for creating time series sequences from a Polars DataFrame.
    Each item consists of context windows for features and targets, and a forecast horizon for targets.
    """
    def __init__(
        self,
        data_df: pl.DataFrame,
        target_cols: List[str],
        feature_cols: List[str],
        context_length: int,
        forecast_horizon: int,
        time_col: str = "timestamp" # A generic name, will be timestamp_hourly or timestamp_daily
    ):
        """
        Args:
            data_df: Polars DataFrame containing the time series data.
                     Assumed to be sorted by time.
            target_cols: List of column names to be used as targets.
            feature_cols: List of column names to be used as exogenous features.
            context_length: Length of the input sequence (context window).
            forecast_horizon: Length of the output sequence to predict (forecast horizon).
            time_col: Name of the timestamp column in data_df.
        """
        super().__init__()
        self.data_df = data_df
        self.target_cols = target_cols
        self.feature_cols = feature_cols
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.time_col = time_col # Store for potential debugging or advanced indexing

        # Pre-convert to NumPy for faster slicing in __getitem__ if DataFrame is not too large.
        # For very large datasets, direct Polars slicing might be better, but involves more pl->torch conversion per item.
        # Let's compromise: convert interested columns to numpy.
        
        self.target_data_np = data_df.select(target_cols).to_numpy().astype(np.float32)
        if feature_cols:
            self.feature_data_np = data_df.select(feature_cols).to_numpy().astype(np.float32)
        else:
            self.feature_data_np = np.array([], dtype=np.float32).reshape(len(data_df), 0) # Empty features

        # Validate lengths
        if len(self.data_df) < self.context_length + self.forecast_horizon:
            # Not enough data to form even one sequence
            self.valid_indices_len = 0
        else:
            self.valid_indices_len = len(self.data_df) - self.context_length - self.forecast_horizon + 1
            
    def __len__(self) -> int:
        return self.valid_indices_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.valid_indices_len:
            raise IndexError(f"Index {idx} out of bounds for dataset with length {self.valid_indices_len}")

        start_idx = idx
        context_end_idx = start_idx + self.context_length
        horizon_end_idx = context_end_idx + self.forecast_horizon

        # Context window for features (exogenous)
        # Shape: (context_length, num_feature_cols)
        ctx_features = torch.from_numpy(self.feature_data_np[start_idx:context_end_idx, :])

        # Context window for targets (autoregressive part)
        # Shape: (context_length, num_target_cols)
        ctx_targets = torch.from_numpy(self.target_data_np[start_idx:context_end_idx, :])
        
        # Forecast horizon for targets (what we want to predict)
        # Shape: (forecast_horizon, num_target_cols)
        fwd_targets = torch.from_numpy(self.target_data_np[context_end_idx:horizon_end_idx, :])
        
        return {
            "x_features": ctx_features, # Exogenous features for context window
            "x_targets": ctx_targets,   # Target values for context window
            "y_targets": fwd_targets    # Target values for forecast horizon
        }

if __name__ == '__main__':
    # --- Test Example ---
    # Create a dummy Polars DataFrame
    n_rows = 100
    rng = pl.datetime_range(
        start=pl.datetime(2023, 1, 1, 0, 0, 0), 
        end=pl.datetime(2023, 1, 1, 23, 0, 0), # Shortened for n_rows=100, this makes ~1 day hourly data
        interval="1h", 
        eager=True
    )[:n_rows].alias("timestamp")
    
    data = {
        "timestamp": rng,
        "target1": np.arange(n_rows, dtype=float),
        "target2": np.arange(n_rows, n_rows * 2, dtype=float),
        "feature1": np.random.rand(n_rows),
        "feature2": np.random.rand(n_rows) + 10,
        "hour_of_day": rng.dt.hour().cast(pl.Float32) # Example time feature
    }
    dummy_df = pl.DataFrame(data)

    target_cols = ["target1", "target2"]
    # Feature cols should be things known in advance or derived time features
    feature_cols = ["feature1", "feature2", "hour_of_day"] 
    context_len = 24
    forecast_hor = 12

    dataset = EscapeRoomTimeSeriesDataset(
        data_df=dummy_df,
        target_cols=target_cols,
        feature_cols=feature_cols,
        context_length=context_len,
        forecast_horizon=forecast_hor,
        time_col="timestamp"
    )

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample 0:")
        print(f"  x_features shape: {sample['x_features'].shape}") # Expected: (context_len, num_feature_cols)
        print(f"  x_targets shape: {sample['x_targets'].shape}")   # Expected: (context_len, num_target_cols)
        print(f"  y_targets shape: {sample['y_targets'].shape}")   # Expected: (forecast_hor, num_target_cols)

        assert sample['x_features'].shape == (context_len, len(feature_cols))
        assert sample['x_targets'].shape == (context_len, len(target_cols))
        assert sample['y_targets'].shape == (forecast_hor, len(target_cols))
        
        # Check first value of x_targets[0,0] should be target1[0]
        assert np.isclose(sample['x_targets'][0,0].item(), dummy_df["target1"][0])
        # Check first value of y_targets[0,0] should be target1[context_len]
        assert np.isclose(sample['y_targets'][0,0].item(), dummy_df["target1"][context_len])

        sample_last = dataset[len(dataset)-1]
        print(f"\nSample {len(dataset)-1} (last):")
        print(f"  x_features shape: {sample_last['x_features'].shape}")
        # Check first value of y_targets for the last sample
        expected_last_y_target_start_idx = (len(dataset)-1) + context_len
        assert np.isclose(sample_last['y_targets'][0,0].item(), dummy_df["target1"][expected_last_y_target_start_idx])


    print("\nTest with no features:")
    dataset_no_feats = EscapeRoomTimeSeriesDataset(
        data_df=dummy_df,
        target_cols=target_cols,
        feature_cols=[], # No feature columns
        context_length=context_len,
        forecast_horizon=forecast_hor,
        time_col="timestamp"
    )
    print(f"Dataset (no features) length: {len(dataset_no_feats)}")
    if len(dataset_no_feats) > 0:
        sample_nf = dataset_no_feats[0]
        print(f"  x_features shape (no features): {sample_nf['x_features'].shape}") # Expected: (context_len, 0)
        assert sample_nf['x_features'].shape == (context_len, 0)

    print("\nTest with insufficient data:")
    short_df = dummy_df.head(context_len + forecast_hor - 5)
    dataset_short = EscapeRoomTimeSeriesDataset(
        data_df=short_df,
        target_cols=target_cols,
        feature_cols=feature_cols,
        context_length=context_len,
        forecast_horizon=forecast_hor,
        time_col="timestamp"
    )
    print(f"Dataset (short data) length: {len(dataset_short)}") # Expected: 0
    assert len(dataset_short) == 0
    
    print("\nAll tests passed for EscapeRoomTimeSeriesDataset!") 