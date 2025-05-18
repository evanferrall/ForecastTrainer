import optuna
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping # For Optuna pruning
from optuna.integration import PyTorchLightningPruningCallback # Import Pruning Callback

# Updated import for the new DataModule
from forecast_cli.datamodules.datasets.multires_dataset import MultiResolutionDataModule 
from forecast_cli.models.wrappers.multitarget_wrapper import MultiresMultiTarget # Updated path
from forecast_cli.training.trainer import get_trainer # Updated path

# --- Placeholder for Backbone Instantiation ---
# In a real scenario, this would load backbone config from YAML or trial suggestions
def get_dummy_backbones():
    daily_backbone = torch.nn.LSTM(input_size=5, hidden_size=20, batch_first=True, num_layers=1) # Dummy
    hourly_backbone = torch.nn.LSTM(input_size=8 + 3, hidden_size=20, batch_first=True, num_layers=1) # Dummy, 8 x_cont_hourly + 3 daily_pred_stub

    # Need to make them output in expected format (batch, seq, features_or_quantiles, targets)
    # This is a gross simplification for the dummy to run.
    # Real backbones from PyTorch Forecasting have specific output formats.
    class DummyWrapper(torch.nn.Module):
        def __init__(self, backbone, output_features, num_targets):
            super().__init__()
            self.backbone = backbone
            self.output_features = output_features # e.g. num_quantiles
            self.num_targets = num_targets
            self.fc = torch.nn.Linear(20, output_features * num_targets)

        def forward(self, x_cont, x_cat=None): # Match expected input
            # x_cat is ignored in this dummy
            out, _ = self.backbone(x_cont)
            out = self.fc(out) # (batch, seq, output_features * num_targets)
            # Reshape to (batch, seq, output_features, num_targets)
            # This assumes output_features is typically num_quantiles and it's the second to last dim
            return out.reshape(out.size(0), out.size(1), self.output_features, self.num_targets)


    # Assuming 3 targets, and 7 quantiles for pytorch_forecasting.QuantileLoss
    num_targets = 3
    num_quantiles = 7 # Typical for QuantileLoss (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    
    return DummyWrapper(daily_backbone, num_quantiles, num_targets), \
           DummyWrapper(hourly_backbone, num_quantiles, num_targets)

# --- Loss Function Placeholder ---
from pytorch_forecasting.metrics import QuantileLoss # As an example

def objective(trial: optuna.trial.Trial, config: dict, data_for_trial: dict) -> float:
    """
    Optuna objective function for hyperparameter tuning.
    Args:
        trial: Optuna trial object.
        config: General configuration for the trial.
        data_for_trial: A dictionary containing pre-loaded tensor data for the trial 
                        (e.g., train_daily_x_cont, val_daily_x_cont, etc.).
    """
    # --- 1. Suggest Hyperparameters ---
    learning_rate = trial.suggest_float("learning_rate", 3e-4, 3e-2, log=True)
    lambda_hier = trial.suggest_float("lambda_hier", 0.0, 5.0)
    
    # Backbone hyperparameter suggestions (example for NHITS & PatchTST as per roadmap)
    # This part needs to be adapted based on the actual backbone chosen for the tuning run.
    # For simplicity, we assume these are passed via `config` or a global setting for the study.
    # Or, one could have nested trials or separate studies per backbone.

    # Example: if config["daily_backbone_type"] == "nhits":
    #     n_stacks = trial.suggest_categorical("nhits_n_stacks", [2, 3, 4])
    #     n_layers = trial.suggest_categorical("nhits_n_layers", [1, 2, 3])
    #     # ... and other NHITS specific params
    # if config["hourly_backbone_type"] == "patchtst":
    #     d_model = trial.suggest_categorical("patchtst_d_model", [64, 128, 256])
    #     # ... and other PatchTST specific params

    # --- 2. Instantiate Model, Data, Trainer ---
    daily_backbone, hourly_backbone = get_dummy_backbones() 
    loss_function = QuantileLoss() 

    model = MultiresMultiTarget(
        daily_backbone=daily_backbone,
        hourly_backbone=hourly_backbone,
        loss_function=loss_function,
        lambda_hier=lambda_hier,
        learning_rate=learning_rate,
        target_names=["target1", "target2", "target3"] # Placeholder
    )

    # Data - Use MultiResolutionDataModule with pre-loaded data for the trial
    # The data_for_trial dict should contain all necessary tensor components.
    data_module = MultiResolutionDataModule(
        train_daily_x_cont=data_for_trial["train_daily_x_cont"], 
        train_daily_x_cat=data_for_trial.get("train_daily_x_cat"), 
        train_daily_y=data_for_trial["train_daily_y"],
        train_daily_weights=data_for_trial.get("train_daily_weights"),
        val_daily_x_cont=data_for_trial["val_daily_x_cont"], 
        val_daily_x_cat=data_for_trial.get("val_daily_x_cat"), 
        val_daily_y=data_for_trial["val_daily_y"],
        val_daily_weights=data_for_trial.get("val_daily_weights"),
        # Hourly data
        train_hourly_x_cont=data_for_trial["train_hourly_x_cont"], 
        train_hourly_x_cat=data_for_trial.get("train_hourly_x_cat"), 
        train_hourly_y=data_for_trial["train_hourly_y"],
        train_hourly_weights=data_for_trial.get("train_hourly_weights"),
        val_hourly_x_cont=data_for_trial["val_hourly_x_cont"], 
        val_hourly_x_cat=data_for_trial.get("val_hourly_x_cat"), 
        val_hourly_y=data_for_trial["val_hourly_y"],
        val_hourly_weights=data_for_trial.get("val_hourly_weights"),
        batch_size=trial.suggest_categorical("batch_size", [config.get("trial_batch_size", 32), config.get("trial_batch_size", 64)]), # Example batch size suggestion
        num_workers=config.get("trial_num_workers", 0)
    )

    # Trainer
    # Optuna Pruning: Use PyTorchLightningPruningCallback.
    # EarlyStopping can still be used as a safeguard or for different conditions.
    trainer_epochs_for_trial = config.get("trainer_epochs_for_trial", 10) 
    
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/wape_avg")
    
    trial_specific_early_stopping = EarlyStopping(
        monitor="val/wape_avg", 
        patience=config.get("patience_early_stopping_trial", 3),
        verbose=False,
        mode="min"
    )

    trial_callbacks = [pruning_callback, trial_specific_early_stopping]
    
    trainer = get_trainer(
        experiment_name=f"optuna_study_{config.get('study_name', 'default_study')}",
        run_name=f"trial_{trial.number}",
        max_epochs=trainer_epochs_for_trial,
        accumulate_grad_batches=config.get("accumulate_grad_batches_trial", 1),
        enable_swa=False,
        monitor_metric="val/wape_avg", # Still useful for ModelCheckpoint if enabled
        monitor_mode="min",
        additional_callbacks=trial_callbacks, # Pass trial-specific callbacks
        enable_default_early_stopping=False # Disable default ES from get_trainer
    )

    # --- 3. Run Training & Validation ---
    try:
        trainer.fit(model, datamodule=data_module)
    except optuna.exceptions.TrialPruned as e:
        # If PyTorchLightningPruningCallback is used and prunes.
        print(f"Trial {trial.number} pruned by Optuna callback: {e}")
        raise
    except Exception as e:
        print(f"Exception during trial {trial.number}: {e}")
        # Consider how to handle other exceptions: re-raise, or return a high error value
        # For now, let Optuna handle it as a failed trial if it re-raises.
        # If we return a high value, Optuna might still consider it.
        # Returning a very large number if it failed, to mark as bad trial
        return float('inf') 


    # --- 4. Return Metric to Optimize ---
    # Fetch the metric that EarlyStopping was monitoring.
    # The metric should be available in callback_metrics after fit.
    metric_value = trainer.callback_metrics.get("val/wape_avg")

    if metric_value is None:
        # This can happen if training failed before the first validation epoch completed
        # or if the metric name is incorrect.
        print(f"Warning: Metric 'val/wape_avg' not found for trial {trial.number}. Returning high error.")
        return float('inf') # Return a large value if metric not found

    return metric_value.item()


def run_tuning(
    n_trials: int = 50, 
    study_name: str = "multires_model_tuning",
    storage_url: str | None = None, # e.g., "sqlite:///optuna_study.db"
    config_overrides: dict | None = None,
    pruner: optuna.pruners.BasePruner | None = optuna.pruners.MedianPruner(),
    # Add a way to pass data for trials
    trial_data_provider: Callable[[], dict] | None = None 
):
    """
    Runs the Optuna hyperparameter tuning study.
    Args:
        ...
        trial_data_provider: A callable that returns a dictionary of tensor data for a single trial.
                             This data should be small and fixed for all trials to ensure consistency.
    """
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage_url,
        load_if_exists=True,
        pruner=pruner # Use the pruner (e.g., MedianPruner)
    )

    current_config = config_overrides if config_overrides is not None else {}
    # Add default values if not in overrides
    current_config.setdefault('study_name', study_name)
    current_config.setdefault('trainer_epochs_for_trial', 10) # Default epochs for each trial
    current_config.setdefault('patience_early_stopping_trial', 3) 
    current_config.setdefault('accumulate_grad_batches_trial', 1)

    if trial_data_provider is None:
        raise ValueError("trial_data_provider must be provided to supply data for Optuna trials.")

    # Load data for trials once
    data_for_all_trials = trial_data_provider()

    study.optimize(lambda trial: objective(trial, current_config, data_for_all_trials), n_trials=n_trials, show_progress_bar=True)

    print(f"Study {study_name} completed.")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"  Value (min val/wape_avg): {study.best_trial.value}")
    print(f"  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()

    return study


if __name__ == "__main__":
    print("Starting Optuna tuning example with MultiResolutionDataModule...")

    # Define a dummy trial_data_provider for the example
    def get_dummy_trial_data() -> dict:
        num_train_samples = 50 # Smaller dataset for trials
        num_val_samples = 10
        daily_seq_len, daily_pred_len, daily_cont_feats, daily_cat_feats, daily_targets = 30, 7, 5, 2, 1
        hourly_seq_len, hourly_pred_len, hourly_cont_feats, hourly_cat_feats, hourly_targets = 72, 24, 8, 3, 1
        return {
            "train_daily_x_cont": torch.randn(num_train_samples, daily_seq_len, daily_cont_feats),
            "train_daily_x_cat": torch.randint(0, 2, (num_train_samples, daily_seq_len, daily_cat_feats)),
            "train_daily_y": torch.randn(num_train_samples, daily_pred_len, daily_targets),
            "val_daily_x_cont": torch.randn(num_val_samples, daily_seq_len, daily_cont_feats),
            "val_daily_x_cat": torch.randint(0, 2, (num_val_samples, daily_seq_len, daily_cat_feats)),
            "val_daily_y": torch.randn(num_val_samples, daily_pred_len, daily_targets),
            "train_hourly_x_cont": torch.randn(num_train_samples, hourly_seq_len, hourly_cont_feats),
            "train_hourly_x_cat": torch.randint(0, 2, (num_train_samples, hourly_seq_len, hourly_cat_feats)),
            "train_hourly_y": torch.randn(num_train_samples, hourly_pred_len, hourly_targets),
            "val_hourly_x_cont": torch.randn(num_val_samples, hourly_seq_len, hourly_cont_feats),
            "val_hourly_x_cat": torch.randint(0, 2, (num_val_samples, hourly_seq_len, hourly_cat_feats)),
            "val_hourly_y": torch.randn(num_val_samples, hourly_pred_len, hourly_targets),
            # Weights can be None
        }

    example_config = {
        "study_name": "my_dummy_study_new_dm",
        "trial_batch_size": 16 # Example for Optuna trials
    }
    run_tuning(n_trials=3, 
                 study_name="dummy_tuning_run_new_dm", 
                 config_overrides=example_config,
                 trial_data_provider=get_dummy_trial_data
                )
    print("Optuna tuning example finished.")
