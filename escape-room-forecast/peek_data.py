import yaml # For loading config
from pathlib import Path
import pandas as pd
from forecast_cli.prep.preprocessor import DataPreprocessor # Assuming src is in PYTHONPATH via poetry
import logging # Add logging import

# Configure logging to show DEBUG messages, using force=True to override other basicConfigs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)

# --- Load relevant parts from config ---
config_file_path = Path("conf/quick_test_config_1min.yaml")
config = {}
if config_file_path.exists():
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
datamodule_config = config.get('escape_room_datamodule', {})

# Configuration (mimicking train.py and using values from quick_test_config_1min.yaml)
csv_path_str = datamodule_config.get('csv_path', "/home/evan/Dropbox/Pass Through Files/HQ Booking Data (For EBF).csv")
raw_ts_col = datamodule_config.get('raw_timestamp_column', 'Start')
raw_end_col = datamodule_config.get('raw_end_timestamp_column', 'End')
history_cutoff = datamodule_config.get('history_cutoff', "2023-01-01")

# Configurable raw column names from the quick_test_config_1min.yaml
raw_created_col = datamodule_config.get('raw_created_col_name', 'created')
raw_promo_title_col = datamodule_config.get('raw_promo_title_col_name', 'promotion')
raw_coupon_code_col = datamodule_config.get('raw_coupon_code_col_name', 'coupons')
raw_flat_rate_col = datamodule_config.get('raw_flat_rate_col_name', 'flat_rate')
raw_addl_player_fee_col = datamodule_config.get('raw_addl_player_fee_col_name', 'additional_player_fee')

# KPI configurations from the main config (passed to DataPreprocessor)
kpi_config_dict = config.get('kpi_configs', {})

print(f"--- Using Configuration from: {config_file_path} ---")
print(f"CSV Path: {csv_path_str}")
print(f"Raw Created Column: {raw_created_col}")
print(f"Raw Promo Title Column: {raw_promo_title_col}")
print(f"Raw Coupon Code Column: {raw_coupon_code_col}")
print(f"Raw Flat Rate Column: {raw_flat_rate_col}")
print(f"Raw Addl Player Fee Column: {raw_addl_player_fee_col}")


print(f"\n--- Initializing DataPreprocessor ---")
# Set a higher display width for pandas outputs
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 50) # Show more columns

try:
    preprocessor = DataPreprocessor(
        csv_path=Path(csv_path_str),
        kpi_configs=kpi_config_dict, # Pass the loaded kpi_configs
        ts_col=raw_ts_col,
        end_col=raw_end_col,
        # Pass the configured raw column names
        raw_created_col=raw_created_col, # Parameter name updated to match __init__
        raw_promo_col=raw_promo_title_col, # Parameter name updated
        raw_coupon_col=raw_coupon_code_col, # Parameter name updated
        raw_flat_rate_col=raw_flat_rate_col, # Parameter name updated
        raw_addl_player_fee_col=raw_addl_player_fee_col # Parameter name updated
    )

    print(f"\n--- Running preprocessor.process() ---")
    full_long_df, static_features_df = preprocessor.process()

    print(f"\n--- Checking Feature Variability ---")
    features_to_check = [
        'lead_time_days',
        'ft_has_promo',
        'ft_uses_coupon_code',
        'ft_flat_rate',
        'ft_addl_player_fee'
    ]
    
    if not full_long_df.empty:
        for feature in features_to_check:
            if feature in full_long_df.columns:
                num_unique = full_long_df[feature].nunique()
                print(f"Feature: '{feature}' - Unique values: {num_unique}")
                if num_unique == 1:
                    print(f"  Value counts for '{feature}':\n{full_long_df[feature].value_counts().head()}")
                elif num_unique < 5: # Also show value counts if very few unique values
                    print(f"  Value counts for '{feature}':\n{full_long_df[feature].value_counts().head()}")

            else:
                print(f"Feature: '{feature}' - NOT FOUND in full_long_df columns.")
    else:
        print("full_long_df is empty, cannot check feature variability.")


    # --- Original detailed printouts (can be uncommented if needed) ---
    # print(f"\n--- full_long_df ---")
    # print("Info:")
    # full_long_df.info(verbose=True, show_counts=True)
    # print("\nHead:")
    # print(full_long_df.head())
    # print(f"\nShape: {full_long_df.shape}")

    # example_prob_series_id = 'prob_hq_a_slice_of_crime_outdoor_adventure'
    # example_bookings_series_id = 'bookings_hq_a_slice_of_crime_outdoor_adventure'
    
    # if not full_long_df.empty and example_prob_series_id in full_long_df['series_id'].unique():
    #     print(f"\nHead for series_id: {example_prob_series_id}")
    #     print(full_long_df[full_long_df['series_id'] == example_prob_series_id].head())
    # elif not full_long_df.empty:
    #     print(f"\nWARNING: Example series_id '{example_prob_series_id}' not found in full_long_df.")
    #     print(f"Sample available series_ids (prob): {full_long_df[full_long_df['series_id'].str.startswith('prob_')]['series_id'].unique()[:3]}")

    # if not full_long_df.empty and example_bookings_series_id in full_long_df['series_id'].unique():
    #     print(f"\nHead for series_id: {example_bookings_series_id}")
    #     print(full_long_df[full_long_df['series_id'] == example_bookings_series_id].head())
    # elif not full_long_df.empty:
    #     print(f"\nWARNING: Example series_id '{example_bookings_series_id}' not found in full_long_df.")
    #     print(f"Sample available series_ids (bookings): {full_long_df[full_long_df['series_id'].str.startswith('bookings_')]['series_id'].unique()[:3]}")

    # print(f"\n--- static_features_df ---")
    # if static_features_df is not None:
    #     print("Info:")
    #     static_features_df.info(verbose=True, show_counts=True)
    #     print("\nHead:")
    #     print(static_features_df.head())
    #     print(f"\nShape: {static_features_df.shape}")
    # else:
    #     print("static_features_df is None")

    # print(f"\n--- Value counts for 'y' in 'prob' family (first 10000 rows of a sample series if available) ---")
    # if not full_long_df.empty:
    #     prob_series_ids = full_long_df[full_long_df['series_id'].str.startswith('prob_')]['series_id'].unique()
    #     if len(prob_series_ids) > 0:
    #         one_prob_series_data = full_long_df[full_long_df['series_id'] == prob_series_ids[0]].head(10000)
    #         if not one_prob_series_data.empty:
    #             print(f"Analyzing 'y' for series: {prob_series_ids[0]}")
    #             print(one_prob_series_data['y'].value_counts(dropna=False).sort_index().head(20))
    #             print("\nDescribe 'y' for this sample prob series data (logit transformed):")
    #             print(one_prob_series_data['y'].describe())
    #         else:
    #             print(f"No data for series {prob_series_ids[0]} sample.")
    #     else:
    #         print("No 'prob_' series found to analyze 'y'.")

    # print("\n--- Describe 'y' for a sample bookings series data (raw counts / log1p transformed):")
    # if not full_long_df.empty:
    #     bookings_series_ids = full_long_df[full_long_df['series_id'].str.startswith('bookings_')]['series_id'].unique()
    #     if len(bookings_series_ids) > 0:
    #         one_bookings_series_data = full_long_df[full_long_df['series_id'] == bookings_series_ids[0]].head(10000)
    #         if not one_bookings_series_data.empty:
    #             print(f"Analyzing 'y' for series: {bookings_series_ids[0]}")
    #             print(one_bookings_series_data['y'].describe())
    #         else:
    #             print(f"No data for series {bookings_series_ids[0]} sample.")
    #     else:
    #         print("No 'bookings_' series found to analyze 'y'.")
    # --- End of original detailed printouts ---

except FileNotFoundError:
    print(f"ERROR: CSV file not found at {csv_path_str}. Please check the path.")
except ImportError as e:
    print(f"ERROR: Could not import DataPreprocessor or other modules: {e}")
    print("Ensure you are running this with 'poetry run python peek_data.py' from the 'escape-room-forecast' directory.")
except Exception as e:
    import traceback
    print(f"An unexpected error occurred: {e}")
    print(traceback.format_exc()) 