from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath, FilePath, PositiveInt, Field, model_validator, NonNegativeInt, AnyHttpUrl
from typing import Optional, List, Dict, Any, Union
import yaml
import os
import json # Add json import

# --- Default Paths (can be overridden by .env file or environment variables) ---
# Assuming a project structure where conf/ and raw_data/ are at the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Moves up from forecast_cli/utils/ to project root
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "conf", "main_config.yaml") # Default if no specific config provided
DEFAULT_RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
# DEFAULT_PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "forecast_cli", "data", "cache") # Cache not explicitly used in new flow
DEFAULT_RUN_ARTEFACTS_DIR = os.path.join(PROJECT_ROOT, "runs")

# --- Pydantic Models for Configuration Sections ---

class DataPaths(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='DATA_')
    raw_data_dir: DirectoryPath = Field(default=DEFAULT_RAW_DATA_DIR)
    # processed_data_dir: Optional[DirectoryPath] = Field(default=DEFAULT_PROCESSED_DATA_DIR) # Removed as not directly used by new flow

class EscapeRoomDataModuleParams(BaseSettings): # Renamed for clarity
    model_config = SettingsConfigDict(env_prefix='ER_DATAMODULE_')
    csv_path: FilePath # Essential: Path to the input data CSV
    train_split_ratio: float = Field(default=0.7, gt=0, lt=1)
    val_split_ratio: float = Field(default=0.15, gt=0, lt=1)
    # test_split_ratio is implied as 1.0 - train_split_ratio - val_split_ratio
    status_filter_include: Optional[List[str]] = Field(default_factory=lambda: ["CONFIRMED", "PAID"])
    # Removed: batch_size, context_lengths, forecast_horizons, num_workers, collate_fn_type
    # These are handled by AutoGluon or not applicable in the new setup.

    @model_validator(mode='after')
    def check_split_ratios(cls, values):
        # Access fields correctly if they might be None or not present (though current fields are not Optional)
        train_split = getattr(values, 'train_split_ratio', 0.0)
        val_split = getattr(values, 'val_split_ratio', 0.0)
        if train_split + val_split >= 1.0:
            raise ValueError("Sum of train_split_ratio and val_split_ratio must be less than 1.0")
        return values

# Removed GeneralModelParams, BackboneParams, ModelWrapperParams as they are replaced by ModelAutogluonParams or are no longer needed.

class ModelAutogluonParams(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='MODEL_AUTOGLUON_')
    target_column: str = "participants"
    freq: str = "H" # Frequency of the time series (e.g., "H", "D", "M")
    prediction_length: PositiveInt = 24 # Forecast horizon
    eval_metric: str = "WAPE" # Metric for AutoGluon to optimize
    preset: str = "medium_quality" # AutoGluon preset (e.g., "best_quality", "medium_quality", "chronos_base")
    time_limit_fit: Optional[PositiveInt] = Field(default=3600, description="Time limit for predictor.fit() in seconds")
    predictor_path_suffix: str = Field(default="autogluon_predictor", description="Suffix for the predictor save path within the run directory")
    show_leaderboard: bool = True
    fit_hyperparameters: Optional[Dict[str, Any]] = Field(default=None, description="Hyperparameters for specific models within AutoGluon's fit method")
    hierarchy: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for hierarchical forecasting, e.g., {'level_names': ['level1', 'level2']}")
    trainer_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Arguments to pass to the PyTorch Lightning trainer for applicable models, e.g., {'accelerator': 'mps', 'devices': 1}")


class TrainingParams(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='TRAINING_')
    experiment_name: str = "autogluon_forecasting_experiment"
    run_artefacts_dir: DirectoryPath = Field(default=DEFAULT_RUN_ARTEFACTS_DIR)
    random_seed: Optional[int] = Field(default=42, description="Global random seed, can be used by AutoGluon if passed.")
    # Removed: learning_rate, max_epochs, patience_early_stopping, accumulate_grad_batches,
    # enable_swa, swa_lrs, accelerator, devices, etc. These are managed by AutoGluon or not applicable.

class TuneParams(BaseSettings): # Kept for potential future use with AutoGluon HPO, or if Optuna is reintroduced
    model_config = SettingsConfigDict(env_prefix='TUNE_')
    study_name: str = "autogluon_tuning_study"
    # AutoGluon's TimeSeriesPredictor.fit handles HPO internally if a preset enabling HPO is chosen,
    # or if specific model hyperparameters are provided with search spaces.
    # This section might be used if driving Optuna separately for some reason or configuring AG HPO details.
    # n_trials: Optional[PositiveInt] = None # AG handles trials based on time_limit or preset
    # timeout_per_trial_seconds: Optional[PositiveInt] = None # AG handles this

# --- Main Configuration Model ---

class AppConfig(BaseSettings):
    """Main application configuration model, composed of other specific config models."""
    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        # Pydantic settings will try to load from .env file by default if python-dotenv is installed
    )
    
    project_name: str = "DashboardModelForecastCLI"
    data_paths: DataPaths = Field(default_factory=DataPaths)
    escape_room_datamodule: EscapeRoomDataModuleParams # Needs to be provided in YAML or ENV
    
    model_autogluon: ModelAutogluonParams # New model config section
    # Removed: model_general, model_wrapper
    
    training: TrainingParams = Field(default_factory=TrainingParams)
    tuning: Optional[TuneParams] = Field(default_factory=TuneParams) # Making tuning optional for now

# --- Loading and Accessing Configuration ---

_cached_config: Optional[AppConfig] = None

def load_config(config_file_path: Optional[str] = None) -> AppConfig:
    """
    Loads configuration from a YAML file and environment variables.
    """
    global _cached_config
    # If a specific file path is given, always reload. Otherwise, use cache if available.
    if _cached_config and config_file_path is None:
        return _cached_config

    effective_config_path = config_file_path or os.getenv("APP_CONFIG_FILE") or DEFAULT_CONFIG_PATH

    file_settings = {}
    if os.path.exists(effective_config_path):
        print(f"Loading configuration from YAML: {effective_config_path}")
        with open(effective_config_path, 'r') as f:
            file_settings = yaml.safe_load(f) or {}
    else:
        print(f"Warning: Config file not found at {effective_config_path}. Using defaults and env vars if available.")
        # If the default config path doesn't exist, we might be in a test or CI environment
        # relying solely on defaults and environment variables.

    # Pydantic-settings will automatically try to load from environment variables
    # and .env files (if python-dotenv is installed). Settings from YAML act as base values.
    # For nested structures, env vars use `__` e.g. MODEL_AUTOGLUON__PREDICTION_LENGTH=48

    # Ensure mandatory nested sections have at least a placeholder if not in YAML and not fully defined by env vars
    # This helps prevent Pydantic validation errors for missing nested models before env var overlay.
    # `escape_room_datamodule` is mandatory because of `csv_path`.
    if 'escape_room_datamodule' not in file_settings and not os.getenv("ER_DATAMODULE__CSV_PATH"):
        # This path is critical. If not in YAML or ENV, it's a problem.
        # Pydantic will raise an error if csv_path is missing, which is correct.
        print("Warning: 'escape_room_datamodule' section or ER_DATAMODULE__CSV_PATH env var is missing. App will likely fail if CSV path is required.")
        # file_settings.setdefault('escape_room_datamodule', {}) # Pydantic will handle missing required fields.

    # `model_autogluon` is now a required part of AppConfig.
    if 'model_autogluon' not in file_settings:
        print("Info: 'model_autogluon' section not found in YAML. Using defaults or environment variables.")
        file_settings.setdefault('model_autogluon', {})


    try:
        config = AppConfig(**file_settings)
    except Exception as e:
        print(f"Error creating AppConfig: {e}")
        # Provide more detailed error context if possible, e.g., Pydantic validation errors
        # This might involve iterating through e.errors() if it's a ValidationError
        raise

    _cached_config = config
    return config

def get_config() -> AppConfig:
    """
    Returns the global application configuration. Loads it if not already loaded.
    Uses the default config path if no specific path has been provided to load_config yet.
    """
    if _cached_config is None:
        return load_config() # Uses default path logic
    return _cached_config

def print_config_as_yaml(config: AppConfig):
    """
    Prints the configuration model as YAML.
    Uses model_dump_json() for robust serialization of Pydantic types.
    """
    print("--- Current Application Configuration ---")
    # Serialize to JSON string first to handle Pydantic specific types (like FilePath, AnyHttpUrl)
    dump_data_json_str = config.model_dump_json(indent=2) # indent for intermediate readability if needed
    # Convert JSON string to Python dict structure for yaml.dump
    dump_data_for_yaml = json.loads(dump_data_json_str)
    
    print(yaml.dump(dump_data_for_yaml, sort_keys=False, indent=2))
    print("---------------------------------------")


if __name__ == "__main__":
    print(f"Project root determined as: {PROJECT_ROOT}")
    print(f"Default main config path: {DEFAULT_CONFIG_PATH}")
    
    # Example of creating a dummy test_escape_room_config.yaml for testing
    # This should align with the new AppConfig structure
    dummy_test_config_content = {
        "project_name": "MyForecastProjectCLI_AutoGluon",
        "data_paths": {
            "raw_data_dir": DEFAULT_RAW_DATA_DIR,
        },
        "escape_room_datamodule": {
            "csv_path": "/path/to/your/dummy_data.csv", # Needs a valid-looking path for FilePath
            "train_split_ratio": 0.7,
            "val_split_ratio": 0.15,
            "status_filter_include": ["CONFIRMED", "PAID"]
        },
        "model_autogluon": {
            "target_column": "participants",
            "freq": "H",
            "prediction_length": 24,
            "eval_metric": "WAPE",
            "preset": "medium_quality", # Changed from chronos_base for a more general default
            "time_limit_fit": 600, # Shorter time for quick test
            "predictor_path_suffix": "ag_predictor_test",
            "show_leaderboard": False
        },
        "training": {
            "experiment_name": "escape_room_autogluon_test_experiment",
            "run_artefacts_dir": DEFAULT_RUN_ARTEFACTS_DIR, # Ensure this path is valid
            "random_seed": 123
        },
        "tuning": { # Tuning is optional, but can be included
            "study_name": "escape_room_autogluon_test_tuning"
        }
    }
    
    # Create a temporary dummy CSV for FilePath validation
    dummy_csv_for_test_path = os.path.join(PROJECT_ROOT, "dummy_data_for_config_test.csv")
    with open(dummy_csv_for_test_path, 'w') as f:
        f.write("header1,header2\\nvalue1,value2")
    dummy_test_config_content["escape_room_datamodule"]["csv_path"] = dummy_csv_for_test_path

    conf_dir = os.path.join(PROJECT_ROOT, "conf")
    os.makedirs(conf_dir, exist_ok=True)
    
    # Use a specific name for this test config to avoid conflicts
    test_specific_config_filename = "test_autogluon_config_generated.yaml"
    test_config_path = os.path.join(conf_dir, test_specific_config_filename)
    
    with open(test_config_path, 'w') as f:
        yaml.dump(dummy_test_config_content, f, sort_keys=False)
    print(f"Created a dummy test config: {test_config_path}")

    # Ensure necessary directories exist before loading the config that might validate them
    os.makedirs(DEFAULT_RUN_ARTEFACTS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_RAW_DATA_DIR, exist_ok=True) # Ensure raw_data_dir also exists for validation

    print(f"\\n--- Loading {test_specific_config_filename} ---")
    try:
        loaded_app_config = load_config(config_file_path=test_config_path)
        print_config_as_yaml(loaded_app_config)
        print("\\nConfig loaded and printed successfully.")
    except Exception as e:
        print(f"Error during test load_config or print_config_as_yaml: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up the dummy CSV and config file
        if os.path.exists(dummy_csv_for_test_path):
            os.remove(dummy_csv_for_test_path)
        # if os.path.exists(test_config_path): # Keep the generated config for inspection if needed
        #     os.remove(test_config_path)
        #     print(f"Cleaned up {test_config_path}")
        pass


