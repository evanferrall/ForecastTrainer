# Main training script will be implemented here.
# (e.g., for 'poetry run python -m forecast_cli.training.train ...')

import argparse
import yaml
import os
import datetime
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import logging
import traceback
from pathlib import Path
import copy # Added for deepcopy

# Import the new DataPreprocessor
from forecast_cli.prep.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

def train(config_path: str):
    """
    Trains multiple TimeSeriesPredictor models (one per KPI family) using AutoGluon.
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    project_name = config.get('project_name', 'DefaultProject')
    experiment_name = config.get('experiment_name', 'DefaultExperiment')
    
    # Get data module config
    datamodule_config = config.get('escape_room_datamodule', {})
    csv_path_str = datamodule_config.get('csv_path')
    if not csv_path_str:
        print("ERROR: csv_path not found in datamodule_config. Exiting.")
        logger.error("csv_path not found in datamodule_config.")
        return
    csv_path = Path(csv_path_str) # Convert to Path object for DataPreprocessor

    # Timestamp columns from config if available, otherwise default to 'Start'/'End'
    # These are the *raw* column names from the input CSV
    raw_ts_col = datamodule_config.get('raw_timestamp_column', 'Start')
    raw_end_col = datamodule_config.get('raw_end_timestamp_column', 'End')
    history_cutoff = datamodule_config.get('history_cutoff') # Get history_cutoff string, will be None if not present

    # Get configurable raw column names for DataPreprocessor
    raw_created_col = datamodule_config.get('raw_created_col_name', 'created')
    raw_promo_title_col = datamodule_config.get('raw_promo_title_col_name', 'promotion')
    raw_coupon_code_col = datamodule_config.get('raw_coupon_code_col_name', 'coupons')
    raw_flat_rate_col = datamodule_config.get('raw_flat_rate_col_name', 'flat_rate')
    raw_addl_player_fee_col = datamodule_config.get('raw_addl_player_fee_col_name', 'additional_player_fee')
    raw_participants_col = datamodule_config.get('raw_participants_col_name', 'participants') # Get new raw col name

    # Load KPI configurations for the preprocessor
    kpi_config_dict = config.get('kpi_configs', {}) # Load kpi_configs from main config

    batch_size_from_config = datamodule_config.get('batch_size', 32) # This is a general config, Chronos will override
    
    model_config = config.get('model_autogluon', {})
    prediction_length = model_config.get('prediction_length', 24)
    global_eval_metric = model_config.get('eval_metric', 'WAPE')
    time_limit_per_family = model_config.get('time_limit_per_family', 3600)
    
    chronos_hyperparameters = model_config.get('fit_hyperparameters', {}).get('Chronos', {})
    if not chronos_hyperparameters or 'model_path' not in chronos_hyperparameters:
        print("ERROR: Chronos hyperparameters (especially model_path) not found in config. Exiting.")
        logger.error("Chronos hyperparameters (especially model_path) not found in config.")
        return
    if 'batch_size' not in chronos_hyperparameters:
        chronos_hyperparameters['batch_size'] = batch_size_from_config

    print(f"\n--- Starting Multi-KPI Training --- ")
    print(f"Project: {project_name}, Experiment: {experiment_name}")
    print(f"Dataset CSV: {csv_path}")
    print(f"Raw timestamp column: {raw_ts_col}, Raw end timestamp column: {raw_end_col}")
    print(f"Prediction length: {prediction_length}, Evaluation Metric: {global_eval_metric}")
    print(f"Chronos Hyperparameters for each family: {chronos_hyperparameters}")
    print(f"Raw column names passed to Preprocessor: created='{raw_created_col}', promo_title='{raw_promo_title_col}', coupon_code='{raw_coupon_code_col}', flat_rate='{raw_flat_rate_col}', addl_player_fee='{raw_addl_player_fee_col}', participants='{raw_participants_col}'") # Added participants to log

    try:
        # Instantiate DataPreprocessor
        logger.info(f"Initializing DataPreprocessor with csv_path: {csv_path}, ts_col: {raw_ts_col}, end_col: {raw_end_col}")
        # logger.info(f"History cutoff from config (if any): {history_cutoff}") # Log it separately
        
        preprocessor = DataPreprocessor(
            csv_path=csv_path,
            kpi_configs=kpi_config_dict, # Pass kpi_configs
            ts_col=raw_ts_col, 
            end_col=raw_end_col,
            # history_cutoff_str=history_cutoff, # Removed: history_cutoff_str is not an init arg
            # Pass configured raw column names (corrected keys)
            raw_created_col=raw_created_col,
            raw_promo_col=raw_promo_title_col,
            raw_coupon_col=raw_coupon_code_col,
            raw_flat_rate_col=raw_flat_rate_col,
            raw_addl_player_fee_col=raw_addl_player_fee_col,
            raw_participants_col=raw_participants_col # Pass to DataPreprocessor
        )
        
        # If history_cutoff is intended to be used by DataPreprocessor's current internal logic:
        if history_cutoff:
            preprocessor.history_cutoff_str = history_cutoff # Set it as an attribute
            logger.info(f"Attribute preprocessor.history_cutoff_str set to: {history_cutoff}")

        # Process data
        logger.info("Starting data preprocessing...")
        full_long_df, static_features_df = preprocessor.process()
        logger.info("Data preprocessing complete.")
        logger.info(f"Shape of full_long_df: {full_long_df.shape}")
        logger.info(f"Shape of static_features_df: {static_features_df.shape if static_features_df is not None else 'None'}")

    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {csv_path}. Please check the path in your config.")
        logger.error(f"CSV file not found at {csv_path}", exc_info=True)
        return
    except ValueError as ve: # Catch specific errors from preprocessor if possible
        print(f"ERROR: A ValueError occurred during data preparation: {ve}")
        logger.error(f"ValueError during data preparation: {ve}", exc_info=True)
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred during data preparation (preprocessor.process()):\\n{traceback.format_exc()}")
        logger.error(f"Unexpected error during preprocessor.process(): {e}", exc_info=True)
        return

    if full_long_df.empty:
        print("No data available after preparation (full_long_df is empty). Aborting training.")
        logger.warning("full_long_df is empty after preprocessing. Aborting training.")
        return
    
    # Handle static_features_df (already done well in DataPreprocessor, but good to check)
    if static_features_df is None:
        logger.warning("static_features_df is None. Initializing to empty DataFrame with 'series_id' index.")
        static_features_df = pd.DataFrame().set_index(pd.Index([], name="series_id"))
    elif static_features_df.empty:
        logger.info("static_features_df is empty. No static features will be added to models.")
    elif static_features_df.index.name != "series_id":
        logger.warning(f"static_features_df.index.name is '{static_features_df.index.name}', expected 'series_id'. Forcing it.")
        static_features_df.index.name = "series_id"


    # Create a unique main run directory
    timestamp_run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure 'runs' directory is at the project root (where pyproject.toml is)
    # The CWD for `poetry run forecast train ...` should be `escape-room-forecast/`
    main_run_dir = Path("runs") / experiment_name / f"run_{timestamp_run}"
    main_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Main run directory for all families: {main_run_dir.resolve()}")
    logger.info(f"Main run directory: {main_run_dir.resolve()}")

    families = {
        # "bookings": {"pattern": r"^bookings_"}, # Old single bookings
        # "minutes": {"pattern": r"^minutes_"}, # Removed
        "bookings_daily":    {"pattern": r"^bookings_daily_"}, # Series ID is kpi_name + _ + game_norm
        "bookings_hourly":   {"pattern": r"^bookings_hourly_"},
        "participants_daily":{"pattern": r"^participants_daily_"},
        "participants_hourly":{"pattern": r"^participants_hourly_"},
        "prob":              {"pattern": r"^prob_"}, # Prob in config, assumes preprocessor makes prob_mygame
        "revenue":           {"pattern": r"^revenue_"} # Revenue in config, assumes preprocessor makes revenue_mygame
    }
    # Note: The preprocessor now creates series_ids like kpi_name_from_config + "_" + game_norm.
    # So, for a kpi_config entry "bookings_daily", it will create "bookings_daily_mygame".
    # The patterns above reflect this. For "prob" and "revenue", if their kpi_config keys remain simple 
    # (e.g., "prob", not "prob_hourly"), then pattern should be r"^prob_" etc.
    # Let's verify kpi_config keys: they are `prob` and `revenue`.
    # The refactored _create_target_series makes series_id = kpi_name + "_" + game_norm.
    # So for kpi_config `prob`, series_id becomes `prob_mygame`. Pattern r"^prob_" is correct.
    # Same for `revenue` -> `revenue_mygame`. Pattern r"^revenue_" is correct.

    # Dynamically determine known_covariate_names from the processed df_long
    # These are columns other than target, timestamp, and item_id
    reserved_columns = ["series_id", "timestamp", "y"]
    known_covariate_names = [col for col in full_long_df.columns if col not in reserved_columns]
    
    if known_covariate_names:
        print(f"Dynamically determined known_covariate_names: {known_covariate_names}")
        logger.info(f"Dynamically determined known_covariate_names: {known_covariate_names}")
    else:
        print("No known_covariate_names found in the processed data.")
        logger.info("No known_covariate_names found in the processed data.")


    for fam_name, fam_config in families.items():
        print(f"\n--- Training Family: {fam_name} ---")
        logger.info(f"Starting training for family: {fam_name}")
        
        fam_df_pd = full_long_df[full_long_df["series_id"].str.contains(fam_config["pattern"], regex=True)]

        if fam_df_pd.empty:
            print(f"No data for family {fam_name} after filtering. Skipping.")
            logger.warning(f"No data for family {fam_name}. Skipping.")
            continue
        
        logger.info(f"Data shape for family {fam_name}: {fam_df_pd.shape}")

        # Determine frequency and other KPI-specific settings from kpi_config_dict
        kpi_specific_config = kpi_config_dict.get(fam_name, {})
        current_freq = kpi_specific_config.get('autogluon_freq', 'H') # Default H, override from config
        current_eval_metric = kpi_specific_config.get('autogluon_eval_metric', global_eval_metric)
        current_pred_len = kpi_specific_config.get('prediction_length', prediction_length)
        current_time_limit = kpi_specific_config.get('time_limit_per_family', time_limit_per_family)

        # Get model hyperparameters for the current KPI family
        # These come from the 'autogluon_models' section in the kpi_configs
        hyperparameters_for_kpi = copy.deepcopy(kpi_specific_config.get('autogluon_models', {}))

        logger.info(f"Using frequency '{current_freq}' for family {fam_name}")
        logger.info(f"Hyperparameters for family {fam_name}: {hyperparameters_for_kpi}")


        # --- Adjust Chronos hyperparameters if Chronos is part of the models for this KPI ---
        chronos_model_key = None
        if "Chronos" in hyperparameters_for_kpi:
            chronos_model_key = "Chronos"
        elif "ChronosModel" in hyperparameters_for_kpi: # AutoGluon might also use this key
            chronos_model_key = "ChronosModel"

        if chronos_model_key:
            # Ensure the specific Chronos config dict exists
            if not isinstance(hyperparameters_for_kpi.get(chronos_model_key), dict):
                hyperparameters_for_kpi[chronos_model_key] = {} # Initialize if not a dict (e.g. if it was True)

            # Merge with global chronos_hyperparameters as a base, allowing kpi-specific overrides
            # kpi-specific settings in autogluon_models[chronos_model_key] take precedence
            base_chronos_settings = copy.deepcopy(chronos_hyperparameters) # Global defaults
            specific_chronos_settings = hyperparameters_for_kpi.get(chronos_model_key, {})
            
            # Merge: specific settings override global defaults
            merged_chronos_hps = {**base_chronos_settings, **specific_chronos_settings}
            hyperparameters_for_kpi[chronos_model_key] = merged_chronos_hps


            # Adjust batch_size based on frequency for Chronos
            # Allow kpi_config to specify 'batch_size_daily' or 'batch_size_hourly' within ChronosModel HPs
            # or fall back to defaults we set here (16 for D, 8 for H)
            # These defaults can also be part of the global chronos_hyperparameters if desired
            default_batch_daily = chronos_hyperparameters.get('batch_size_daily_default', 16)
            default_batch_hourly = chronos_hyperparameters.get('batch_size_hourly_default', 8)

            if current_freq == 'D':
                batch_size_to_set = merged_chronos_hps.get('batch_size_daily', default_batch_daily)
                hyperparameters_for_kpi[chronos_model_key]['batch_size'] = batch_size_to_set
                logger.info(f"Set Chronos batch_size to {batch_size_to_set} for Daily family {fam_name}")
            elif current_freq == 'H':
                batch_size_to_set = merged_chronos_hps.get('batch_size_hourly', default_batch_hourly)
                hyperparameters_for_kpi[chronos_model_key]['batch_size'] = batch_size_to_set
                logger.info(f"Set Chronos batch_size to {batch_size_to_set} for Hourly family {fam_name}")
        else:
                logger.warning(f"Unknown frequency '{current_freq}' for family {fam_name}. Using Chronos batch_size from its merged config if present: {merged_chronos_hps.get('batch_size')}")
        # --- End Chronos specific HP adjustment ---


        # Filter static features for the current family
        fam_static_features_for_autogluon = None # Renamed variable
        if not static_features_df.empty and static_features_df.index.name == "series_id":
            current_family_series_ids = fam_df_pd['series_id'].unique()
            _fam_static_features_indexed = static_features_df[static_features_df.index.isin(current_family_series_ids)] # Keep index for now
            
            if _fam_static_features_indexed is not None and not _fam_static_features_indexed.empty:
                fam_static_features_for_autogluon = _fam_static_features_indexed.reset_index() # Make 'series_id' a column
                logger.debug(f"For family {fam_name}, static features prepared with shape {fam_static_features_for_autogluon.shape} after reset_index.")
            elif len(current_family_series_ids) > 0: # If _fam_static_features_indexed is empty but there were IDs to filter by
                logger.debug(f"For family {fam_name}, fam_static_features is empty after filtering. Series IDs in fam_df_pd: {current_family_series_ids[:5]}")
            # If static_features_df was empty to begin with, fam_static_features_for_autogluon remains None.

        # Ensure all known_covariate_names actually exist in fam_df_pd columns for this specific family's slice
        actual_known_covariates_for_fit = [col for col in known_covariate_names if col in fam_df_pd.columns]
        if len(actual_known_covariates_for_fit) < len(known_covariate_names):
            missing_covs = set(known_covariate_names) - set(actual_known_covariates_for_fit)
            logger.warning(f"For family {fam_name}, some determined known_covariates were not found in the family's data slice and will not be used: {list(missing_covs)}")
        
        logger.info(f"Covariates for family {fam_name}: {actual_known_covariates_for_fit}")

        try:
            ts_fam_data = TimeSeriesDataFrame(
                fam_df_pd, 
                id_column="series_id", 
                timestamp_column="timestamp",
                static_features=fam_static_features_for_autogluon # Use the DataFrame with 'series_id' as a column
            )
            # --- BEGIN DEBUG: Save TimeSeriesDataFrame for specific families ---
            if fam_name in ["participants_daily", "revenue"]:
                debug_save_path = main_run_dir / f"debug_{fam_name}_ts_fam_data.parquet"
                try:
                    # TimeSeriesDataFrame can be saved if converted to pandas first
                    # Keep static features separate if they cause issues with direct save
                    if ts_fam_data.static_features is not None and not ts_fam_data.static_features.empty:
                        static_debug_save_path = main_run_dir / f"debug_{fam_name}_static_features.parquet"
                        ts_fam_data.static_features.to_parquet(static_debug_save_path)
                        logger.info(f"Saved DEBUG static features for {fam_name} to {static_debug_save_path}")
                        # Save main data without static features for simplicity if direct save is problematic
                        ts_fam_data.copy().drop(columns=ts_fam_data.static_features.columns, errors='ignore').to_parquet(debug_save_path)
                    else:
                        ts_fam_data.to_parquet(debug_save_path) # Save the main DataFrame part
                    logger.info(f"Saved DEBUG TimeSeriesDataFrame for {fam_name} to {debug_save_path}")
                except Exception as e_save:
                    logger.error(f"Error saving DEBUG TimeSeriesDataFrame for {fam_name}: {e_save}")
            # --- END DEBUG ---
        except Exception as e:
            print(f"ERROR: Could not create TimeSeriesDataFrame for family {fam_name}. Details: {e}")
            logger.error(f"Could not create TimeSeriesDataFrame for family {fam_name}: {e}", exc_info=True)
            print(f"Problematic data for {fam_name} (first 5 rows):\\n{fam_df_pd.head()}")
            if fam_df_pd['timestamp'].isnull().any(): print(f"NaNs found in timestamp column for {fam_name}")
            if fam_df_pd['series_id'].isnull().any(): print(f"NaNs found in series_id column for {fam_name}")
            if fam_df_pd['y'].isnull().any(): print(f"NaNs found in y column for {fam_name}. This should have been handled by preprocessor.")
            continue

        predictor_path_fam = main_run_dir / f"{fam_name}_predictor"
        print(f"Predictor for {fam_name} will be saved to: {predictor_path_fam.resolve()}")
        logger.info(f"Predictor path for {fam_name}: {predictor_path_fam.resolve()}")

        # current_eval_metric was already set from kpi_specific_config or global_eval_metric
        if fam_name == "prob" and current_eval_metric != "MAE": # Target 'y' for prob is already logit_transformed by DataPreprocessor
            logger.info(f"For probability family, overriding eval_metric to MAE (target is logit-transformed). Original: {current_eval_metric}")
            current_eval_metric = "MAE" 
        
        if not hyperparameters_for_kpi: # If no models were specified for the KPI
            logger.warning(f"No specific models defined in autogluon_models for KPI {fam_name}. AutoGluon will use its default model set.")
            # Let AutoGluon decide, or you could specify a default set here e.g.
            # hyperparameters_for_kpi = {'SimpleFeedForward': {}, 'DeepAR': {}, 'ETS': {}, 'ARIMA': {}}


        predictor = TimeSeriesPredictor(
            target="y",
            prediction_length=current_pred_len, # Use KPI specific prediction length
            path=str(predictor_path_fam), # Convert Path to string
            eval_metric=current_eval_metric, # Use KPI specific eval_metric
            known_covariates_names=actual_known_covariates_for_fit if actual_known_covariates_for_fit else None,
            verbosity=2 # Or from config
        )

        print(f"Fitting predictor for {fam_name}...")
        logger.info(f"Fitting predictor for {fam_name}. Models from config: {list(hyperparameters_for_kpi.keys()) if hyperparameters_for_kpi else 'AutoGluon Defaults'}. Time limit: {current_time_limit}s. Frequency: {current_freq}")

        try:
            predictor.fit(
                ts_fam_data,
                presets=['best_quality'], # Use best_quality preset
                time_limit=current_time_limit, # Use KPI specific time limit
                random_seed=config.get('random_seed', 123) # Global random seed
            )
            logger.info(f"--- Family {fam_name} Training Complete ---")
            if config.get('show_leaderboard', True):
                print(f"\n--- Leaderboard for {fam_name} ---")
                # Ensure ts_fam_data is passed if leaderboard requires it and has not seen it.
                leaderboard = predictor.leaderboard(ts_fam_data, silent=False) # silent=False for printing
                # leaderboard already prints itself when silent=False
                # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                #     print(leaderboard)
        except Exception as e:
            print(f"ERROR during predictor.fit() for family {fam_name}: {e}")
            logger.error(f"Predictor fit error for family {fam_name}: {e}", exc_info=True)
            # Log more details
            try: # Inner try for debug logging
                logger.debug(f"TimeSeriesDataFrame info for {fam_name} before fit:\\\\n{ts_fam_data.info() if ts_fam_data else 'No ts_fam_data'}")
                static_features_content = ts_fam_data.static_features if ts_fam_data else None
                if static_features_content is not None and not static_features_content.empty:
                    logger.debug(f"Static features for {fam_name}:\\\\n{static_features_content.head()}")
                else: # This else is now correctly associated with the `if` inside the try
                    logger.debug(f"No static features for {fam_name}.")
            except Exception as debug_e: # This except is for the inner try block
                logger.error(f"Error during debug logging for fit failure: {debug_e}")
            # The 'continue' for the outer loop remains here
            continue 

    print("\n--- All KPI Family Training Sessions Attempted ---")
    print(f"All artifacts saved under main run directory: {main_run_dir.resolve()}")
    logger.info(f"All training artifacts saved under: {main_run_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-KPI forecasting models using AutoGluon TimeSeriesPredictor.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    
    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() # Log to console
            # Add FileHandler if you want to log to a file as well
        ]
    )
    
    args = parser.parse_args()
    
    # Ensure the config path is resolved correctly if it's relative
    config_file_path = Path(args.config)
    if not config_file_path.is_absolute():
        config_file_path = Path.cwd() / config_file_path # Assuming it's relative to CWD
        
    train(str(config_file_path.resolve())) 