import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from typing import Any, Tuple, List, Dict, Callable

# Data types for clarity (can be more specific, e.g. np.ndarray or torch.Tensor)
# For a single sample from IndividualResolutionDataset:
# ( { "x_cont": ..., "x_cat": ... }, ( target_tensor, weight_tensor_or_None ) )
SampleInputFeatures = Dict[str, torch.Tensor]
SampleTarget = Tuple[torch.Tensor, Any] # (target_tensor, weight_or_metadata)
IndividualSample = Tuple[SampleInputFeatures, SampleTarget]

# For a single sample from PairedMultiResolutionDataset:
# ( daily_IndividualSample, hourly_IndividualSample )
PairedSample = Tuple[IndividualSample, IndividualSample]


class IndividualResolutionDataset(Dataset):
    """
    A PyTorch Dataset for a single resolution (e.g., daily or hourly).
    Assumes data is pre-processed into tensors for x_cont, x_cat, and y.
    """
    def __init__(self, 
                 x_cont: torch.Tensor, 
                 x_cat: torch.Tensor | None, 
                 y: torch.Tensor, 
                 weights: torch.Tensor | None = None):
        """
        Args:
            x_cont (torch.Tensor): Continuous features, shape (num_samples, sequence_length, num_cont_features).
            x_cat (torch.Tensor | None): Categorical features, shape (num_samples, sequence_length, num_cat_features).
            y (torch.Tensor): Target variable(s), shape (num_samples, prediction_length, num_targets).
            weights (torch.Tensor | None): Optional sample weights, shape (num_samples,).
        """
        self.x_cont = x_cont
        self.x_cat = x_cat
        self.y = y
        self.weights = weights

        if not (self.x_cont.size(0) == self.y.size(0) and \
                (self.x_cat is None or self.x_cat.size(0) == self.x_cont.size(0)) and \
                (self.weights is None or self.weights.size(0) == self.x_cont.size(0))):
            raise ValueError("Number of samples must be consistent across x_cont, x_cat, y, and weights.")

    def __len__(self) -> int:
        return self.x_cont.size(0)

    def __getitem__(self, idx: int) -> IndividualSample:
        x_features: SampleInputFeatures = {"x_cont": self.x_cont[idx]}
        if self.x_cat is not None:
            x_features["x_cat"] = self.x_cat[idx]
        
        target_tuple: SampleTarget = (self.y[idx], self.weights[idx] if self.weights is not None else None)
        
        return x_features, target_tuple


class PairedMultiResolutionDataset(Dataset):
    """
    A PyTorch Dataset that pairs samples from a daily and an hourly dataset.
    Assumes a one-to-one correspondence between daily and hourly samples for simplicity.
    For example, each daily sequence corresponds to a set of hourly sequences that are aggregated.
    The model structure then expects these to be batched together.
    """
    def __init__(self, daily_dataset: IndividualResolutionDataset, hourly_dataset: IndividualResolutionDataset):
        self.daily_dataset = daily_dataset
        self.hourly_dataset = hourly_dataset

        # Critical assumption: The number of "series" or "items" must match.
        # The LightningModule handles batch size alignment if the underlying datasets are conceptually paired.
        if len(self.daily_dataset) != len(self.hourly_dataset):
            # This could be relaxed if one daily sample maps to N hourly samples explicitly,
            # requiring more complex indexing or a different dataset design.
            # For now, enforce they are of same conceptual length for direct pairing.
            print(
                f"Warning: Daily dataset length ({len(self.daily_dataset)}) and "
                f"hourly dataset length ({len(self.hourly_dataset)}) differ. "
                "This PairedMultiResolutionDataset assumes a 1-to-1 correspondence. "
                "Ensure this is intended or adapt dataset/datamodule logic."
            )
            # Depending on strategy, one might take min(len(daily), len(hourly))
            # For now, will proceed with min_len to avoid index errors, but this needs careful data prep.
        
        self._len = min(len(self.daily_dataset), len(self.hourly_dataset))
        if self._len == 0 and (len(self.daily_dataset) > 0 or len(self.hourly_dataset) > 0):
            print("Warning: One of the datasets is empty, so the paired dataset will be empty.")


    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> PairedSample:
        if idx >= self._len:
            raise IndexError("Index out of bounds for paired dataset.")
        daily_sample = self.daily_dataset[idx]
        hourly_sample = self.hourly_dataset[idx]
        return daily_sample, hourly_sample


def multi_res_collate_fn(batch: List[PairedSample]) -> Tuple[IndividualSample, IndividualSample]:
    """
    Custom collate function for PairedMultiResolutionDataset.
    Takes a list of PairedSample and collates them into batches suitable for MultiresMultiTarget.
    
    Output structure:
    ( 
      (batched_x_daily_dict, (batched_y_daily_tensor, batched_daily_weights_or_None)),  # Daily batch
      (batched_x_hourly_dict, (batched_y_hourly_tensor, batched_hourly_weights_or_None)) # Hourly batch
    )
    """
    daily_samples, hourly_samples = zip(*batch) # Unzip list of pairs

    # Collate daily samples
    daily_x_features_list, daily_y_tuples_list = zip(*daily_samples)
    batched_daily_x_cont = torch.stack([s["x_cont"] for s in daily_x_features_list], dim=0)
    batched_daily_x_cat = None
    if daily_x_features_list and "x_cat" in daily_x_features_list[0] and daily_x_features_list[0]["x_cat"] is not None:
        batched_daily_x_cat = torch.stack([s["x_cat"] for s in daily_x_features_list], dim=0)
    
    batched_daily_y = torch.stack([t[0] for t in daily_y_tuples_list], dim=0)
    batched_daily_weights = None
    if daily_y_tuples_list and daily_y_tuples_list[0][1] is not None:
        batched_daily_weights = torch.stack([t[1] for t in daily_y_tuples_list], dim=0)

    batched_daily_x_dict: SampleInputFeatures = {"x_cont": batched_daily_x_cont}
    if batched_daily_x_cat is not None:
        batched_daily_x_dict["x_cat"] = batched_daily_x_cat
    daily_batch: IndividualSample = (batched_daily_x_dict, (batched_daily_y, batched_daily_weights))

    # Collate hourly samples
    hourly_x_features_list, hourly_y_tuples_list = zip(*hourly_samples)
    batched_hourly_x_cont = torch.stack([s["x_cont"] for s in hourly_x_features_list], dim=0)
    batched_hourly_x_cat = None
    if hourly_x_features_list and "x_cat" in hourly_x_features_list[0] and hourly_x_features_list[0]["x_cat"] is not None:
        batched_hourly_x_cat = torch.stack([s["x_cat"] for s in hourly_x_features_list], dim=0)

    batched_hourly_y = torch.stack([t[0] for t in hourly_y_tuples_list], dim=0)
    batched_hourly_weights = None
    if hourly_y_tuples_list and hourly_y_tuples_list[0][1] is not None:
        batched_hourly_weights = torch.stack([t[1] for t in hourly_y_tuples_list], dim=0)
        
    batched_hourly_x_dict: SampleInputFeatures = {"x_cont": batched_hourly_x_cont}
    if batched_hourly_x_cat is not None:
        batched_hourly_x_dict["x_cat"] = batched_hourly_x_cat
    hourly_batch: IndividualSample = (batched_hourly_x_dict, (batched_hourly_y, batched_hourly_weights))

    return daily_batch, hourly_batch


class MultiResolutionDataModule(pl.LightningDataModule):
    def __init__(self,
                 # Daily data components
                 train_daily_x_cont: torch.Tensor | None = None, train_daily_x_cat: torch.Tensor | None = None, train_daily_y: torch.Tensor | None = None, train_daily_weights: torch.Tensor | None = None,
                 val_daily_x_cont: torch.Tensor | None = None, val_daily_x_cat: torch.Tensor | None = None, val_daily_y: torch.Tensor | None = None, val_daily_weights: torch.Tensor | None = None,
                 test_daily_x_cont: torch.Tensor | None = None, test_daily_x_cat: torch.Tensor | None = None, test_daily_y: torch.Tensor | None = None, test_daily_weights: torch.Tensor | None = None,
                 # Hourly data components
                 train_hourly_x_cont: torch.Tensor | None = None, train_hourly_x_cat: torch.Tensor | None = None, train_hourly_y: torch.Tensor | None = None, train_hourly_weights: torch.Tensor | None = None,
                 val_hourly_x_cont: torch.Tensor | None = None, val_hourly_x_cat: torch.Tensor | None = None, val_hourly_y: torch.Tensor | None = None, val_hourly_weights: torch.Tensor | None = None,
                 test_hourly_x_cont: torch.Tensor | None = None, test_hourly_x_cat: torch.Tensor | None = None, test_hourly_y: torch.Tensor | None = None, test_hourly_weights: torch.Tensor | None = None,
                 batch_size: int = 32,
                 num_workers: int = 0, # Adjust based on system
                 pin_memory: bool = False # Adjust based on system (True if using GPU usually)
                ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "train_daily_x_cont", "train_daily_x_cat", "train_daily_y", "train_daily_weights",
            "val_daily_x_cont", "val_daily_x_cat", "val_daily_y", "val_daily_weights",
            "test_daily_x_cont", "test_daily_x_cat", "test_daily_y", "test_daily_weights",
            "train_hourly_x_cont", "train_hourly_x_cat", "train_hourly_y", "train_hourly_weights",
            "val_hourly_x_cont", "val_hourly_x_cat", "val_hourly_y", "val_hourly_weights",
            "test_hourly_x_cont", "test_hourly_x_cat", "test_hourly_y", "test_hourly_weights",
        ]) # Saves batch_size, num_workers, pin_memory
        
        # Store raw data tensors - these would be prepared by some external data loading/processing step
        # and passed to the DataModule constructor.
        self.raw_data = {
            "train_daily": (train_daily_x_cont, train_daily_x_cat, train_daily_y, train_daily_weights),
            "val_daily": (val_daily_x_cont, val_daily_x_cat, val_daily_y, val_daily_weights),
            "test_daily": (test_daily_x_cont, test_daily_x_cat, test_daily_y, test_daily_weights),
            "train_hourly": (train_hourly_x_cont, train_hourly_x_cat, train_hourly_y, train_hourly_weights),
            "val_hourly": (val_hourly_x_cont, val_hourly_x_cat, val_hourly_y, val_hourly_weights),
            "test_hourly": (test_hourly_x_cont, test_hourly_x_cat, test_hourly_y, test_hourly_weights),
        }
        
        self.train_dataset: PairedMultiResolutionDataset | None = None
        self.val_dataset: PairedMultiResolutionDataset | None = None
        self.test_dataset: PairedMultiResolutionDataset | None = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            # Daily datasets
            train_d_xc, train_d_xcat, train_d_y, train_d_w = self.raw_data["train_daily"]
            val_d_xc, val_d_xcat, val_d_y, val_d_w = self.raw_data["val_daily"]
            
            if train_d_xc is None or val_d_xc is None: # Add check for required data
                raise ValueError("Training and validation data (x_cont) must be provided for daily resolution in fit stage.")

            train_daily_ds = IndividualResolutionDataset(train_d_xc, train_d_xcat, train_d_y, train_d_w)
            val_daily_ds = IndividualResolutionDataset(val_d_xc, val_d_xcat, val_d_y, val_d_w)

            # Hourly datasets
            train_h_xc, train_h_xcat, train_h_y, train_h_w = self.raw_data["train_hourly"]
            val_h_xc, val_h_xcat, val_h_y, val_h_w = self.raw_data["val_hourly"]

            if train_h_xc is None or val_h_xc is None: # Add check for required data
                raise ValueError("Training and validation data (x_cont) must be provided for hourly resolution in fit stage.")

            train_hourly_ds = IndividualResolutionDataset(train_h_xc, train_h_xcat, train_h_y, train_h_w)
            val_hourly_ds = IndividualResolutionDataset(val_h_xc, val_h_xcat, val_h_y, val_h_w)

            self.train_dataset = PairedMultiResolutionDataset(train_daily_ds, train_hourly_ds)
            self.val_dataset = PairedMultiResolutionDataset(val_daily_ds, val_hourly_ds)
            print(f"Fit stage: Train dataset size: {len(self.train_dataset)}, Val dataset size: {len(self.val_dataset)}")


        if stage == "test" or stage is None:
            # Test data might not always be available, handle None
            test_d_xc, test_d_xcat, test_d_y, test_d_w = self.raw_data["test_daily"]
            test_h_xc, test_h_xcat, test_h_y, test_h_w = self.raw_data["test_hourly"]

            if test_d_xc is not None and test_h_xc is not None: # Basic check
                test_daily_ds = IndividualResolutionDataset(test_d_xc, test_d_xcat, test_d_y, test_d_w)
                test_hourly_ds = IndividualResolutionDataset(test_h_xc, test_h_xcat, test_h_y, test_h_w)
                self.test_dataset = PairedMultiResolutionDataset(test_daily_ds, test_hourly_ds)
                print(f"Test stage: Test dataset size: {len(self.test_dataset)}")
            else:
                print("Test stage: No test data provided or one of the resolutions is missing for test.")
                self.test_dataset = None


    def train_dataloader(self) -> DataLoader:
        if not self.train_dataset:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        return DataLoader(self.train_dataset, 
                          batch_size=self.hparams.batch_size, 
                          shuffle=True, 
                          collate_fn=multi_res_collate_fn,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def val_dataloader(self) -> DataLoader:
        if not self.val_dataset:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        return DataLoader(self.val_dataset, 
                          batch_size=self.hparams.batch_size, 
                          shuffle=False, 
                          collate_fn=multi_res_collate_fn,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def test_dataloader(self) -> DataLoader | None:
        if not self.test_dataset:
            # raise RuntimeError("Test dataset not initialized. Call setup() first or ensure test data is provided.")
            print("Test dataloader requested but no test dataset is available.")
            return None # Or an empty DataLoader
        return DataLoader(self.test_dataset, 
                          batch_size=self.hparams.batch_size, 
                          shuffle=False, 
                          collate_fn=multi_res_collate_fn,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def predict_dataloader(self) -> DataLoader | None:
        # For prediction, the input format might be different (no y_true)
        # This DataModule is primarily for training/validation/testing with labels.
        # A separate dataloader or predict_step handling might be needed for raw inference.
        # For now, can reuse test_dataloader if prediction data is structured similarly.
        print("Predict dataloader using test_dataset structure. Adapt if prediction input differs.")
        return self.test_dataloader()


if __name__ == '__main__':
    # Example Usage
    print("Testing MultiResolutionDataModule...")

    # Dummy data parameters
    num_train_samples = 100
    num_val_samples = 20
    daily_seq_len, daily_pred_len, daily_cont_feats, daily_cat_feats, daily_targets = 60, 30, 5, 2, 3
    hourly_seq_len, hourly_pred_len, hourly_cont_feats, hourly_cat_feats, hourly_targets = 168, 72, 8, 3, 3 # hourly_cont_feats matches wrapper expectation for x_cont + daily_pred_stub
    
    # Create dummy tensors (replace with actual data loading and preprocessing)
    # Training data
    train_d_xc = torch.randn(num_train_samples, daily_seq_len, daily_cont_feats)
    train_d_xcat = torch.randint(0, 5, (num_train_samples, daily_seq_len, daily_cat_feats))
    train_d_y = torch.randn(num_train_samples, daily_pred_len, daily_targets)
    train_d_w = torch.rand(num_train_samples)

    train_h_xc = torch.randn(num_train_samples, hourly_seq_len, hourly_cont_feats)
    train_h_xcat = torch.randint(0, 3, (num_train_samples, hourly_seq_len, hourly_cat_feats))
    train_h_y = torch.randn(num_train_samples, hourly_pred_len, hourly_targets)
    # train_h_w = None # Example: hourly might not have weights

    # Validation data - ensure all required args for IndividualResolutionDataset are non-None
    val_d_xc = torch.randn(num_val_samples, daily_seq_len, daily_cont_feats)
    val_d_xcat = torch.randint(0, 5, (num_val_samples, daily_seq_len, daily_cat_feats))
    val_d_y = torch.randn(num_val_samples, daily_pred_len, daily_targets)
    # val_d_w is None by default in __init__ call below, which is fine for IndividualResolutionDataset
    
    val_h_xc = torch.randn(num_val_samples, hourly_seq_len, hourly_cont_feats)
    val_h_xcat = torch.randint(0, 3, (num_val_samples, hourly_seq_len, hourly_cat_feats))
    val_h_y = torch.randn(num_val_samples, hourly_pred_len, hourly_targets)
    # val_h_w is None by default in __init__ call below

    dm = MultiResolutionDataModule(
        # Provide all required training data for IndividualResolutionDataset
        train_daily_x_cont=train_d_xc, train_daily_x_cat=train_d_xcat, train_daily_y=train_d_y, train_daily_weights=train_d_w,
        val_daily_x_cont=val_d_xc, val_daily_x_cat=val_d_xcat, val_daily_y=val_d_y, # val_daily_weights default to None
        train_hourly_x_cont=train_h_xc, train_hourly_x_cat=train_h_xcat, train_hourly_y=train_h_y, # train_hourly_weights default to None
        val_hourly_x_cont=val_h_xc, val_hourly_x_cat=val_h_xcat, val_hourly_y=val_h_y, # val_hourly_weights default to None
        # Test data defaults to None
        batch_size=32
    )

    dm.setup(stage="fit")
    
    print("\nChecking train_dataloader...")
    train_loader = dm.train_dataloader()
    for i, batch in enumerate(train_loader):
        daily_batch, hourly_batch = batch
        (x_d, (y_d, w_d)), (x_h, (y_h, w_h)) = daily_batch, hourly_batch
        print(f"Train Batch {i+1}:")
        print(f"  Daily x_cont: {x_d['x_cont'].shape}, x_cat: {x_d.get('x_cat', 'None') if x_d.get('x_cat') is None else x_d.get('x_cat').shape}, y: {y_d.shape}, weights: {w_d.shape if w_d is not None else 'None'}")
        print(f"  Hourly x_cont: {x_h['x_cont'].shape}, x_cat: {x_h.get('x_cat', 'None') if x_h.get('x_cat') is None else x_h.get('x_cat').shape}, y: {y_h.shape}, weights: {w_h.shape if w_h is not None else 'None'}")
        if i == 0: break # Just check one batch

    print("\nChecking val_dataloader...")
    val_loader = dm.val_dataloader()
    for i, batch in enumerate(val_loader):
        daily_batch, hourly_batch = batch
        (x_d, (y_d, w_d)), (x_h, (y_h, w_h)) = daily_batch, hourly_batch
        print(f"Val Batch {i+1}:")
        print(f"  Daily x_cont: {x_d['x_cont'].shape}, x_cat: {x_d.get('x_cat', 'None') if x_d.get('x_cat') is None else x_d.get('x_cat').shape}, y: {y_d.shape}, weights: {w_d.shape if w_d is not None else 'None'}")
        print(f"  Hourly x_cont: {x_h['x_cont'].shape}, x_cat: {x_h.get('x_cat', 'None') if x_h.get('x_cat') is None else x_h.get('x_cat').shape}, y: {y_h.shape}, weights: {w_h.shape if w_h is not None else 'None'}")
        if i == 0: break
    
    # Example for test dataloader (assuming test data is provided)
    # dm.setup(stage="test") 
    # test_loader = dm.test_dataloader()
    # if test_loader:
    #     for batch in test_loader: ...
    
    print("\nMultiResolutionDataModule test finished.")
