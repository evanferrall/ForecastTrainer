import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import pandas as pd # AutoGluon TimeSeriesPredictor expects pandas DataFrames
from typing import Optional, Tuple # For type hinting

# import torch # Keep if other parts of your CLI might use it, or for version printing
# import neuralforecast # For version print, replace/augment with autogluon

try:
    # from autogluon.common.utils.utils import setup_outputdir # Not used directly
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    from autogluon.timeseries.version import __version__ as ag_version # Correct way to get version
except ImportError as e:
    logging.error(
        "Autogluon is not installed or a sub-module is missing. "
        "Please install it with `poetry install` or `pip install autogluon.timeseries`."
    )
    logging.error(f"Import error: {e}")
    sys.exit(1)

from forecast_cli.utils.config import load_config, print_config_as_yaml, AppConfig
from forecast_cli.datamodules.escape_room_datamodule import EscapeRoomDataModule
from forecast_cli.utils.app_logging import setup_rich_logger as setup_logging # Corrected import
# Removed: from forecast_cli.utils.utils import setup_logging, create_run_artifact_dirs, get_run_name

# Removed: MultiresMultiTarget, DummyBackbone, CustomNHITS, PatchTST
# Removed: get_trainer
# Removed: LOSS_REGISTRY, BACKBONE_REGISTRY

# --- Helper functions for run management (moved here from the problematic utils.py import) ---
def get_run_name(experiment_name: str) -> str:
    """Generates a unique run name with a timestamp."""
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{current_time_str}_{experiment_name}"

def create_run_artifact_dirs(base_run_dir: str, run_name: str) -> str:
    """Creates directories for the current run and returns the run-specific directory path."""
    run_specific_dir = os.path.join(base_run_dir, run_name)
    os.makedirs(run_specific_dir, exist_ok=True)
    # Optionally, create subdirectories like 'logs', 'models', 'plots' within run_specific_dir
    # os.makedirs(os.path.join(run_specific_dir, "logs"), exist_ok=True)
    # os.makedirs(os.path.join(run_specific_dir, "models"), exist_ok=True)
    return run_specific_dir
# --- End Helper functions ---

def train(config_path: Optional[str] = None):
    """Main training loop for AutoGluon TimeSeries.

    Args:
        config_path: Path to the YAML configuration file.
    """
    cfg: AppConfig = load_config(config_path)
    setup_logging(level=logging.INFO) # Basic logging setup
    print_config_as_yaml(cfg)

    logging.info(f"Using AutoGluon-TimeSeries version: {ag_version}")

    # --- Setup Run Artifacts Directory ---
    run_name = get_run_name(cfg.training.experiment_name)
    run_dir = create_run_artifact_dirs(str(cfg.training.run_artefacts_dir), run_name)
    logging.info(f"Run artifacts will be saved to: {run_dir}")

    # --- Data Loading and Preparation ---
    logging.info("Initializing EscapeRoomDataModule...")
    dm = EscapeRoomDataModule(
        csv_path=str(cfg.escape_room_datamodule.csv_path), # Ensure path is string
        target_column=cfg.model_autogluon.target_column,
        freq=cfg.model_autogluon.freq,
        prediction_length=cfg.model_autogluon.prediction_length,
        status_filter_include=cfg.escape_room_datamodule.status_filter_include,
        train_split_ratio=cfg.escape_room_datamodule.train_split_ratio,
        val_split_ratio=cfg.escape_room_datamodule.val_split_ratio,
    )
    dm.prepare_data() # Download/process (if any)
    dm.setup()      # Splits and assigns dataframes

    # Convert Polars DataFrames from DataModule to Pandas for AutoGluon
    # AutoGluon expects item_id, timestamp, target, and known_covariates
    train_pd_df = dm.train_df.to_pandas()
    val_pd_df = dm.val_df.to_pandas()
    # test_pd_df = dm.test_df.to_pandas() # Keep test set for later evaluation

    # Ensure correct 'ds' column name if DataModule used something else internally
    # Assuming dm.train_df etc., already have 'ds' as the timestamp column name
    # and cfg.model_autogluon.target_column is the target column name.
    # Item ID is also crucial, assuming 'item_id' is present.

    logging.info(f"Training data shape (Pandas): {train_pd_df.shape}")
    logging.info(f"Validation data shape (Pandas): {val_pd_df.shape}")
    if train_pd_df.empty:
        logging.error("Training DataFrame is empty. Cannot proceed.")
        sys.exit(1)

    # --- TimeSeriesPredictor Initialization ---
    predictor_path = os.path.join(run_dir, cfg.model_autogluon.predictor_path_suffix)
    # Create the directory for the predictor if it doesn't exist
    os.makedirs(predictor_path, exist_ok=True)
    logging.info(f"TimeSeriesPredictor will be saved to: {predictor_path}")

    # Known covariates (features known in advance for the forecast horizon)
    known_covariates_names = dm.all_known_covariates
    logging.info(f"Using known_covariates: {known_covariates_names}")

    # Hierarchy configuration from YAML (if present)
    hierarchy_config = cfg.model_autogluon.hierarchy # This is already a Dict or None
    if hierarchy_config:
        logging.info(f"Hierarchy config found: {hierarchy_config}")
        # Ensure item_id is structured to support these levels or data has these columns
        # Example: item_id could be 'game1_locationA_daily'
        # AutoGluon's TimeSeriesPredictor might take this directly or need specific data format.
        # For now, passing it as an argument to the constructor.

    predictor_init_kwargs = {
        "target": cfg.model_autogluon.target_column,
        "prediction_length": cfg.model_autogluon.prediction_length,
        "path": predictor_path,
        "eval_metric": cfg.model_autogluon.eval_metric,
        "freq": cfg.model_autogluon.freq,
        "known_covariates_names": known_covariates_names if known_covariates_names else None,
        "verbosity": 3
    }

    predictor = TimeSeriesPredictor(**predictor_init_kwargs)

    # --- Model Training ---
    logging.info(f"Fitting TimeSeriesPredictor with preset: {cfg.model_autogluon.preset}...")
    fit_hyperparameters = cfg.model_autogluon.fit_hyperparameters or {}
    logging.info(f"Using fit_hyperparameters: {fit_hyperparameters}")

    fit_kwargs = {
        "tuning_data": val_pd_df,
        "time_limit": cfg.model_autogluon.time_limit_fit,
        "presets": cfg.model_autogluon.preset,
        "hyperparameters": fit_hyperparameters,
        "random_seed": cfg.training.random_seed
    }
    if cfg.model_autogluon.trainer_kwargs:
        fit_kwargs["trainer_kwargs"] = cfg.model_autogluon.trainer_kwargs
        logging.info(f"Passing trainer_kwargs to predictor.fit: {cfg.model_autogluon.trainer_kwargs}")

    predictor.fit(
        train_pd_df,
        **fit_kwargs
    )

    # --- Post-Training Actions ---
    logging.info("--- AutoGluon TimeSeriesPredictor Fit Summary ---")
    try:
        fit_summary = predictor.fit_summary(verbosity=3) # verbosity controls detail
        # logging.info(fit_summary) # fit_summary() prints to console, also returns dict
    except Exception as e:
        logging.warning(f"Could not generate fit_summary: {e}")

    predictor_save_path = predictor.path # Path where predictor was saved
    logging.info(f"Predictor saved to {predictor_save_path}")

    # Optionally, show leaderboard
    if cfg.model_autogluon.show_leaderboard:
        logging.info("--- AutoGluon TimeSeries Score Leaderboard (on validation data) ---")
        leaderboard = predictor.leaderboard(val_pd_df, silent=False) # Pass val_df for leaderboard
        # The leaderboard is a pandas DataFrame, so it will print nicely by default
        # logging.info(leaderboard.to_string()) # Example if direct print is not ideal

    # Save model card if available (new in AG >=1.4 typically)
    try:
        model_card_path = os.path.join(run_dir, "model_card.md")
        predictor.save_model_card(model_card_path)
        logging.info(f"Model card saved to {model_card_path}")
    except AttributeError:
        logging.info("TimeSeriesPredictor.save_model_card() not available in this AutoGluon version.")
    except Exception as e:
        logging.warning(f"Could not save model card: {e}")


    logging.info(f"Training finished. Experiment run: {run_name}")
    logging.info(f"Predictor and artifacts are in: {run_dir}")

def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train the forecasting model using AutoGluon.")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Path to the YAML configuration file. If not provided, uses default search paths."
    )
    args = parser.parse_args()

    try:
        train(config_path=args.config)
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Failed to train: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 