import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Iterator, Tuple

class ExpandingWindowSplitter:
    """
    A splitter that uses an expanding window for training and a fixed-size window for validation.
    Wraps sklearn's TimeSeriesSplit to control validation set size.
    """
    def __init__(self, n_splits: int = 5, val_horizon: int = 1, gap: int = 0):
        """
        Args:
            n_splits: Number of splits.
            val_horizon: The number of time steps to include in the validation set.
            gap: Number of time steps to skip between train and validation set.
        """
        if val_horizon <= 0:
            raise ValueError("val_horizon must be positive.")
        
        self.n_splits = n_splits
        self.val_horizon = val_horizon
        self.gap = gap
        # Underlying TimeSeriesSplit will be configured to give test sets of at least val_horizon size.
        # We will then take the first `val_horizon` samples from the test set it provides.
        self.base_splitter = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.val_horizon + self.gap, gap=0)
        # Note: The `gap` in TimeSeriesSplit is between train and test. We apply our own `gap` logic.
        # The `test_size` for base_splitter ensures its test set is large enough.

    def split(self, X: np.ndarray | pd.DataFrame | None, y: np.ndarray | pd.Series | None = None, groups: np.ndarray | None = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and validation set.

        Args:
            X: Array-like of shape (n_samples, n_features) or (n_samples,).
               The time series data to split.
            y: Always ignored, exists for compatibility.
            groups: Always ignored, exists for compatibility.

        Yields:
            tuple: (train_indices, val_indices)
        """
        if X is None:
            raise ValueError("X cannot be None for splitting.")
        
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)

        if n_samples < self.val_horizon + (self.n_splits -1) * self.val_horizon + self.gap + 1 : # Simplified check
             raise ValueError(
                f"Not enough samples ({n_samples}) to perform {self.n_splits} splits "
                f"with val_horizon={self.val_horizon} and gap={self.gap}."
            )

        # TimeSeriesSplit internally works on indices from 0 to n_samples-1
        # We need to adjust the test_indices from TimeSeriesSplit to respect our val_horizon and gap
        for train_idx_base, test_idx_base in self.base_splitter.split(X):
            # train_idx_base is good as is (expanding window)
            
            # The test_idx_base from TimeSeriesSplit is a block after train_idx_base.
            # We need to apply our gap and then take val_horizon.
            val_start_idx = test_idx_base[0] + self.gap
            val_end_idx = val_start_idx + self.val_horizon
            
            if val_end_idx > n_samples:
                # This can happen if the last fold's validation set would exceed available data.
                # TimeSeriesSplit with test_size might already prevent this, but good to check.
                # Or, could simply truncate: val_indices = np.arange(val_start_idx, n_samples)
                # For fixed validation horizon, better to raise or ensure data is sufficient.
                # The initial check for n_samples should mostly cover this.
                # If test_idx_base from TimeSeriesSplit is already too short, this indicates an issue.
                # For now, let's assume n_samples check is sufficient.
                 pass # If it occurs, the slice will handle it.
            
            val_indices = np.arange(val_start_idx, min(val_end_idx, n_samples))

            # Ensure train and validation indices don't overlap after applying gap.
            # train_idx_base[-1] should be < val_indices[0] if gap >=0
            if self.gap >= 0 and train_idx_base.size > 0 and val_indices.size > 0:
                 if train_idx_base[-1] >= val_indices[0]:
                    # This shouldn't happen with TimeSeriesSplit default behavior and our gap logic
                    # if TimeSeriesSplit's own gap is 0 and test_idx_base[0] = train_idx_base[-1] + 1
                    raise ValueError(
                        f"Train and validation indices overlap or are not separated by gap. "
                        f"Max train index: {train_idx_base[-1]}, Min val index: {val_indices[0]}, Gap: {self.gap}"
                    )
            
            if len(val_indices) < self.val_horizon and val_end_idx <= n_samples :
                # This can happen if gap pushes validation beyond available data for the *required* horizon
                # Or if TimeSeriesSplit's last split is too small
                print(f"Warning: Fold produced validation set of size {len(val_indices)},"
                      f" smaller than requested val_horizon {self.val_horizon}.")
                # Depending on strictness, could raise error or continue with smaller set.
                # For now, we allow it, but it's a sign the data might be too short for the setup.

            if len(val_indices) > 0: # Only yield if validation set is not empty
                yield train_idx_base, val_indices
            else:
                print(f"Warning: Skipping fold due to empty validation set after applying gap and horizon.")


if __name__ == '__main__':
    import pandas as pd
    print("Testing ExpandingWindowSplitter...")

    # Create dummy data
    data = np.arange(1, 101) # 100 samples
    df = pd.DataFrame({'feature': data, 'time': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))})
    df = df.set_index('time')

    n_splits_test = 5
    val_horizon_test = 7
    gap_test = 1 # 1 day gap

    splitter = ExpandingWindowSplitter(n_splits=n_splits_test, val_horizon=val_horizon_test, gap=gap_test)
    
    fold_count = 0
    for train_indices, val_indices in splitter.split(df):
        fold_count += 1
        print(f"--- Fold {fold_count} ---")
        print(f"Train indices: {train_indices[0]}...{train_indices[-1]} (len: {len(train_indices)})")
        print(f"Val indices:   {val_indices[0]}...{val_indices[-1]} (len: {len(val_indices)})")
        
        train_data = df.iloc[train_indices]
        val_data = df.iloc[val_indices]
        
        print(f"Train data: {len(train_data)} samples, from {train_data.index.min()} to {train_data.index.max()}")
        print(f"Val data:   {len(val_data)} samples, from {val_data.index.min()} to {val_data.index.max()}")

        assert len(val_data) <= val_horizon_test # Can be less for last fold if data runs out
        if fold_count > 1 and len(val_data) > 0 and len(train_indices)>0 and val_indices.size > 0: # Avoid checking on first fold where prev_val_indices is not set
            assert train_indices[-1] < val_indices[0] - gap_test # Check gap
            assert val_indices[0] == prev_val_indices[0] + val_horizon_test # Check rolling window for val set
        
        if len(val_data) > 0: # only if val_data exists
             prev_val_indices = val_indices.copy()


    print(f"Total folds generated: {fold_count}")
    assert fold_count <= n_splits_test # May be less if data too short

    # Test with insufficient data
    short_data = np.arange(1, 21) # 20 samples
    df_short = pd.DataFrame({'feature': short_data})
    splitter_short = ExpandingWindowSplitter(n_splits=5, val_horizon=5, gap=0)
    print("\nTesting with insufficient data (expect ValueError or fewer folds):")
    try:
        short_fold_count = 0
        for tr_idx, v_idx in splitter_short.split(df_short):
            short_fold_count +=1
            print(f"Short Fold {short_fold_count}: Train {len(tr_idx)}, Val {len(v_idx)}")
        print(f"Short data folds: {short_fold_count}")
    except ValueError as e:
        print(f"Caught expected error for short data: {e}")
    
    # Test case where val_horizon cannot be met on last fold
    data_edge = np.arange(1, 25) # 24 samples
    splitter_edge = ExpandingWindowSplitter(n_splits=3, val_horizon=10, gap=0) # 3 folds, 10 val horizon
    # Fold 1: train up to sample for 1st val set, val=10
    # min train = 1. test_size = 10. So 1st train has 24 - 2*10 = 4 samples. train [0,3], val [4,13]
    # Fold 2: train [0, 13], val [14, 23]
    # Fold 3: train [0, ?], val should be smaller
    print("\nTesting edge case for val_horizon on last fold:")
    edge_fold_count = 0
    for tr_idx, v_idx in splitter_edge.split(data_edge):
        edge_fold_count += 1
        print(f"Edge Fold {edge_fold_count}: Train {len(tr_idx)} (indices {tr_idx[0]}-{tr_idx[-1]}), Val {len(v_idx)} (indices {v_idx[0]}-{v_idx[-1] if len(v_idx)>0 else 'empty'})")
    print(f"Edge data folds: {edge_fold_count}")

    # Test with gap pushing validation out of bounds
    data_gap_test = np.arange(1, 30) # 29 samples
    # n_splits=3, val_horizon=5, gap=5
    # min_train_size for TSS is n_splits for some reason, so if n_splits = 3, min_train is 3.
    # test_size for base_splitter = val_horizon + gap = 10
    # 1st split from TSS: train indices [0,1,2], test indices [3..12]
    #   Our val_indices: test_idx_base[0]+gap = 3+5=8. val_start=8, val_end=8+5=13. -> val[8,12]
    # 2nd split from TSS: train indices [0..12], test indices [13..22]
    #   Our val_indices: test_idx_base[0]+gap = 13+5=18. val_start=18, val_end=18+5=23. -> val[18,22]
    # 3rd split from TSS: train indices [0..22], test indices [23..29] (len 7, not 10)
    #   Our val_indices: test_idx_base[0]+gap = 23+5=28. val_start=28, val_end=28+5=33. -> val[28,28] (len 1 if data has 29 samples, indices up to 28) -> val[28]
    # If data is length 29 (indices 0-28), then last val_indices would be np.arange(28, min(33, 29)) = np.arange(28,29) = [28] (len 1)

    splitter_gap_test = ExpandingWindowSplitter(n_splits=3, val_horizon=5, gap=5)
    print("\nTesting with gap that might shorten last validation set:")
    gap_fold_count = 0
    for tr_idx, v_idx in splitter_gap_test.split(data_gap_test):
        gap_fold_count += 1
        print(f"Gap Test Fold {gap_fold_count}: Train {len(tr_idx)} (indices {tr_idx[0]}-{tr_idx[-1]}), Val {len(v_idx)} (indices {v_idx[0] if len(v_idx)>0 else 'N/A'}-{v_idx[-1] if len(v_idx)>0 else 'N/A'})")
    print(f"Gap test folds: {gap_fold_count}") 