import lightning.pytorch as pl
import polars as pl_polars 
from typing import List, Dict, Optional, Any
import numpy as np

from forecast_cli.data.processing.escape_room_processor import preprocess_escape_room_data

# _CombinedEscapeRoomDataset and MMRS_collate_fn are removed as they are no longer needed.

class EscapeRoomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        target_column: str = "participants",
        freq: str = "H", # Frequency of the timeseries data (e.g., "H" for hourly, "D" for daily)
        prediction_length: int = 24, # Forecast horizon for AutoGluon
        train_split_ratio: float = 0.7,
        val_split_ratio: float = 0.15,
        # test_split_ratio is implied (1 - train - val)
        status_filter_include: Optional[List[str]] = None, # Default is ["CONFIRMED", "PAID"] in processor
        known_covariates_static: Optional[List[str]] = None, # e.g. ["channel"] if static per item_id
        known_covariates_dynamic: Optional[List[str]] = None, # e.g. ["is_weekend"] if time-varying
        random_seed: int = 42 # For reproducible splits
    ):
        super().__init__()
        self.csv_path = csv_path
        self.target_column = target_column
        self.freq = freq
        self.prediction_length = prediction_length
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.status_filter_include = status_filter_include 
        self.known_covariates_static = known_covariates_static if known_covariates_static else []
        self.known_covariates_dynamic = known_covariates_dynamic if known_covariates_dynamic else [] 
        self.random_seed = random_seed

        self.train_df: Optional[pl_polars.DataFrame] = None
        self.val_df: Optional[pl_polars.DataFrame] = None
        self.test_df: Optional[pl_polars.DataFrame] = None
        self.full_df: Optional[pl_polars.DataFrame] = None
        self.all_known_covariates: List[str] = []

    def prepare_data(self) -> None:
        # Called only on 1 GPU. Good for download, tokenize, etc.
        # Preprocessing is done in setup for this module as it's relatively fast.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data, preprocess, and split into train, validation, and test sets."""
        if self.full_df is not None:
            # Avoid reprocessing if already done
            return

        self.full_df = preprocess_escape_room_data(
            csv_path=self.csv_path,
            status_filter_include=self.status_filter_include,
            target_column=self.target_column
        )

        if self.full_df.is_empty():
            raise ValueError("Preprocessing returned an empty DataFrame. Cannot proceed.")

        # Determine known covariates from the processed dataframe
        # Common covariates produced by the processor are "is_weekend", "channel"
        potential_covariates = ["is_weekend", "channel"]
        self.all_known_covariates = [col for col in potential_covariates if col in self.full_df.columns]

        # Chronological split based on timestamp for each item_id
        # AutoGluon TimeSeriesPredictor can also handle this if given a single DataFrame,
        # but providing explicit splits is also an option.
        # Here, we do a global chronological split.

        # Ensure data is sorted by item_id then timestamp for consistent splitting
        # Though a global chronological split doesn't strictly need per-item sorting first,
        # it's good practice if any per-item operations were to be done.
        # For global split, just sort by timestamp.
        self.full_df = self.full_df.sort("timestamp")

        n = len(self.full_df)
        train_end_idx = int(self.train_split_ratio * n)
        val_end_idx = train_end_idx + int(self.val_split_ratio * n)

        self.train_df = self.full_df.slice(0, train_end_idx)
        self.val_df = self.full_df.slice(train_end_idx, val_end_idx - train_end_idx)
        self.test_df = self.full_df.slice(val_end_idx, n - val_end_idx)

        print(f"Data loaded and split: Train ({len(self.train_df)}), Val ({len(self.val_df)}), Test ({len(self.test_df)})")
        if self.train_df.is_empty() or self.val_df.is_empty():
            print("Warning: Train or Validation DataFrame is empty after splitting. Check split ratios and data size.")

    # The following dataloader methods are typically used by PyTorch Lightning Trainer.
    # If using AutoGluon's TimeSeriesPredictor.fit() directly with DataFrames,
    # these might not be strictly necessary in their current form.
    # They are kept here as stubs or if a future Lightning integration needs them.
    # AutoGluon usually creates its own dataloaders internally from pandas DataFrames.
    # For now, these will return None, as the primary way to use this DataModule
    # with AutoGluon will be to access self.train_df, self.val_df, self.test_df.

    def train_dataloader(self) -> Optional[Any]: # DataLoader
        # AutoGluon's TimeSeriesPredictor.fit() typically takes the DataFrame directly.
        # If a custom PyTorch Lightning loop is built around an AutoGluon model, 
        # this would need to provide a compatible DataLoader.
        # For direct use: print("Access self.train_df for training data with AutoGluon.")
        return None 

    def val_dataloader(self) -> Optional[Any]: # DataLoader
        # print("Access self.val_df for validation data with AutoGluon.")
        return None

    def test_dataloader(self) -> Optional[Any]: # DataLoader
        # print("Access self.test_df for test data with AutoGluon.")
        return None


if __name__ == "__main__":
    # Create a dummy CSV for testing
    dummy_data = {
        "Booking number": ["B001", "B002", "B003", "B004", "B005"],
        "Start": [
            "01/01/2023 14:00 PM", "01/01/2023 14:30 PM", "01/01/2023 15:00 PM", 
            "01/02/2023 10:00 AM", "01/02/2023 10:30 AM"
        ],
        "Game": [
            "Indoor Game Alpha", "Indoor Game Alpha ONLINE", "Indoor Game Beta", 
            "Indoor Game Alpha", "Indoor Game Beta ONLINE"
        ],
        "Participants": [2, 3, 4, 5, 1],
        "Total Gross": ["$100.00", "$150.00", "$200.00", "$250.00", "$50.00"],
        "Status": ["CONFIRMED", "PAID", "CONFIRMED", "PAID", "CONFIRMED"],
        "Created": ["30/12/2022 10:00 AM"] * 5 # Simplified
    }
    dummy_csv_path = "./dummy_escape_room_data_long.csv"
    df_dummy = pl_polars.DataFrame(dummy_data)
    df_dummy.write_csv(dummy_csv_path)

    print(f"Created dummy CSV for DataModule test: {dummy_csv_path}")

    dm = EscapeRoomDataModule(
        csv_path=dummy_csv_path,
        target_column="participants",
        freq="H",
        prediction_length=12, # Predict 12 hours ahead
        status_filter_include=["CONFIRMED", "PAID"]
    )
    dm.setup()

    print("\n--- Train DataFrame ---")
    if dm.train_df is not None and not dm.train_df.is_empty():
        print(dm.train_df.head())
        print(f"Shape: {dm.train_df.shape}")
    else:
        print("Train DataFrame is empty or None.")

    print("\n--- Validation DataFrame ---")
    if dm.val_df is not None and not dm.val_df.is_empty():
        print(dm.val_df.head())
        print(f"Shape: {dm.val_df.shape}")
    else:
        print("Validation DataFrame is empty or None.")

    print("\n--- Test DataFrame ---")
    if dm.test_df is not None and not dm.test_df.is_empty():
        print(dm.test_df.head())
        print(f"Shape: {dm.test_df.shape}")
    else:
        print("Test DataFrame is empty or None.")
    
    print(f"\nTarget column: {dm.target_column}")
    print(f"All known covariates: {dm.all_known_covariates}")

    # Clean up dummy file
    import os
    os.remove(dummy_csv_path)
    print(f"Cleaned up dummy CSV: {dummy_csv_path}") 