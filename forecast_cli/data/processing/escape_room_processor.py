import polars as pl
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime

# Define expected column names based on the dummy CSV, make them available globally
# These are the names as they appear in a header row if present.
DUMMY_CSV_HEADER_RAW = [
    "Booking number", "Start", "End", "First name", "Last name", "Email address", "Phone", 
    "Participants", "Adults", "Flat Rate (1-7 Players)", "Additional players", "Players", 
    "Participants (details)", "Game", "Seven Dwarfs: Mining Mission", "TriWizard Trials", 
    "Neverland Heist on the High Seas", "Jingles", "Café", "Undersea Overthrow", 
    "Cure for the Common Zombie", "Product code", "Private event", "Status", "Promotion", 
    "Number of coupons", "Coupons", "Specific gift voucher", "Prepaid credits", 
    "Prepaid package", "Adjustments", "Total adjustments", "Total net", "HST", 
    "HST included", "Total gross", "Total paid", "Total due", "Participants (names)", 
    "Created", "Created by", "Last changed", "Last changed by", "Canceled", "Canceled by", 
    "Reschedule until", "Source", "IP address", "External ref.", "Alert", "Game Location", 
    "Eastern Daylight Time", "Horror Experience", "In-Real-Life", "Location", "Online", 
    "Beta Test", "Time of game", "Game Requirements", "Online Experience", "Eastern Standard Time"
]
# These are the standardized names we expect to work with internally.
EXPECTED_COLUMN_NAMES_STANDARDIZED = [col.lower().strip() for col in DUMMY_CSV_HEADER_RAW]

# A subset of critical columns that *must* be present after loading and standardization.
CRITICAL_COLUMNS_STANDARDIZED = ['start', 'created', 'game', 'status', 'participants', 'total gross']

# This mapping is used when the CSV has no header row.
# The values should correspond to the raw header names expected, which will then be standardized.
RAW_COLUMN_NAME_MAPPING_NO_HEADER = {
    idx: raw_name for idx, raw_name in enumerate(DUMMY_CSV_HEADER_RAW)
}

CRITICAL_COLUMNS_NORMALIZED = [ # Normalized names we expect after initial processing
    "start", "created", "game", "status", "participants", "total_gross"
]


def _standardize_escape_room_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """Standardizes column names to lowercase and replaces spaces with underscores."""
    rename_map = {col: col.lower().replace(" ", "_").replace("(", "").replace(")", "") 
                  for col in df.columns}
    return df.rename(rename_map)

def _load_and_standardize_csv(csv_path: str) -> pl.DataFrame:
    """Attempts to load the CSV with different header/no-header strategies."""
    strategies = [
        {"name": "header_line_0", "has_header": True, "skip_rows": 0, "rename_map": None},
        {"name": "header_line_1", "has_header": True, "skip_rows": 1, "rename_map": None},
        {"name": "no_header_line_0", "has_header": False, "skip_rows": 0, "rename_map": RAW_COLUMN_NAME_MAPPING_NO_HEADER},
    ]

    loaded_df: Optional[pl.DataFrame] = None
    used_strategy_name: Optional[str] = None

    for strategy in strategies:
        try:
            df_attempt = pl.read_csv(
                csv_path,
                try_parse_dates=False, 
                has_header=strategy["has_header"],
                skip_rows=strategy["skip_rows"],
                infer_schema_length=1000,
                encoding='utf-8'
            )

            if strategy["rename_map"]:
                actual_rename_map = {}
                if not strategy["has_header"]: 
                    for i, col_prefix_name in enumerate(df_attempt.columns):
                        if i in strategy["rename_map"]:
                             actual_rename_map[col_prefix_name] = strategy["rename_map"][i]
                        else:
                            actual_rename_map[col_prefix_name] = col_prefix_name 
                else: 
                    actual_rename_map = strategy["rename_map"]

                if actual_rename_map: 
                    df_attempt = df_attempt.rename(actual_rename_map)
            
            df_standardized = _standardize_escape_room_column_names(df_attempt)
            
            missing_critical = [col for col in CRITICAL_COLUMNS_NORMALIZED if col not in df_standardized.columns]
            if not missing_critical:
                loaded_df = df_standardized
                used_strategy_name = strategy['name']
                break
            else:
                pass 
        except Exception as e:
            pass 

    if loaded_df is None:
        raise ValueError(f"Could not load CSV {csv_path} with any known strategy. Ensure critical columns {CRITICAL_COLUMNS_NORMALIZED} are present after standardization.")
    
    return loaded_df

def preprocess_escape_room_data(
    csv_path: str, 
    status_filter_include: Optional[List[str]] = None,
    target_column: str = "participants"
) -> pl.DataFrame:
    """
    Loads escape room data from a CSV, preprocesses it into a long format
    suitable for time series forecasting with libraries like AutoGluon.

    Args:
        csv_path: Path to the input CSV file.
        status_filter_include: List of booking statuses to include (e.g., ["CONFIRMED", "PAID"]).
                               If None, no status filtering is applied.
        target_column: Name of the column to be used as the target variable (e.g., "participants", "total_gross").

    Returns:
        A Polars DataFrame with columns: "item_id" (game name), "timestamp" (ds), 
        target_column (e.g., "participants"), and covariates ("is_weekend", "channel").
    """
    if status_filter_include is None:
        status_filter_include = ["CONFIRMED", "PAID"] # Default filter

    df = _load_and_standardize_csv(csv_path)

    if df.height == 0:
        print("Warning: Loaded DataFrame is empty after initial load and standardization.")
        # Define schema for empty DataFrame to match expected output
        return pl.DataFrame(schema={
            "item_id": pl.Utf8, 
            "timestamp": pl.Datetime, 
            target_column: pl.Float64, # Assuming target is float
            "is_weekend": pl.Boolean,
            "channel": pl.Utf8
        })

    # Ensure critical columns for processing are present
    required_cols_for_processing = ["start", "game", target_column]
    if "status" not in df.columns and status_filter_include:
         print("Warning: 'status' column not found, cannot apply status filter.")
    elif "status" in df.columns:
        required_cols_for_processing.append("status")


    missing_processing_cols = [col for col in required_cols_for_processing if col not in df.columns]
    if missing_processing_cols:
        raise ValueError(f"Missing critical columns for processing: {missing_processing_cols}. Available: {df.columns}")

    # 1. Filter by status
    if status_filter_include and "status" in df.columns:
        # Make the filtering case-insensitive
        uppercase_status_filter = [s.upper() for s in status_filter_include]
        df = df.filter(pl.col("status").str.to_uppercase().is_in(uppercase_status_filter))
    
    if df.height == 0:
        print(f"Warning: DataFrame is empty after status filter (if applied).")
        return pl.DataFrame(schema={
            "item_id": pl.Utf8, 
            "timestamp": pl.Datetime, 
            target_column: pl.Float64,
            "is_weekend": pl.Boolean,
            "channel": pl.Utf8
        })

    # 2. Parse 'start' column to datetime 'timestamp'
    #    Common date formats: '%d/%m/%Y %I:%M %p', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M'
    parsed_datetime_col = None
    if df["start"].dtype == pl.Object or df["start"].dtype == pl.Utf8:
        fmts = ["%d/%m/%Y %I:%M %p", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%Y-%m-%dT%H:%M:%S%.f%z", "%Y-%m-%dT%H:%M:%S%.f"]
        for fmt in fmts:
            try:
                current_parse_attempt = df.get_column("start").str.to_datetime(fmt, strict=False, time_unit='ns')
                if current_parse_attempt.is_not_null().any():
                    parsed_datetime_col = current_parse_attempt
                    break
            except Exception:
                continue
        
        if parsed_datetime_col is not None:
            df = df.with_columns(parsed_datetime_col.alias("timestamp"))
        else:
            raise ValueError("Could not parse 'start' column to datetime with any attempted format.")
    elif isinstance(df["start"].dtype, pl.Datetime):
        df = df.with_columns(pl.col("start").alias("timestamp"))
    else:
        raise ValueError(f"'start' column is not a string, object or datetime. Type: {df['start'].dtype}")

    # Ensure target column is numeric, fill NaNs with 0 for target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    
    if not isinstance(df[target_column].dtype, tuple(pl.NUMERIC_DTYPES)):
        try:
            df = df.with_columns(pl.col(target_column).cast(pl.Float64, strict=False).fill_null(0.0))
        except Exception as e:
            raise ValueError(f"Could not cast target column '{target_column}': {e}")
    else:
        df = df.with_columns(pl.col(target_column).fill_null(0.0))


    # 3. Add covariates
    df = df.with_columns([
        (pl.col("timestamp").dt.weekday() >= 5).alias("is_weekend"), # Saturday is 5, Sunday is 6 in Polars
        pl.when(pl.col("game").str.to_uppercase().str.contains("ONLINE"))
          .then(pl.lit("online"))
          .otherwise(pl.lit("in-person"))
          .alias("channel")
    ])

    # 4. Select final columns and rename
    #    AutoGluon expects "item_id", "timestamp", and the target column.
    #    Covariates are passed separately.
    final_columns = {
        "game": "item_id",
        "timestamp": "timestamp", # already named "timestamp"
        target_column: target_column, # keep target column name as is
        "is_weekend": "is_weekend",
        "channel": "channel"
    }
    
    # Ensure all selected columns exist before select and rename
    missing_final_cols = [col for col in final_columns.keys() if col not in df.columns and col != target_column] # target_column check is implicit
    if target_column not in df.columns: # explicit check for target
        missing_final_cols.append(target_column)

    if missing_final_cols:
        raise ValueError(f"Missing columns needed for final selection: {missing_final_cols}. Available: {df.columns}")

    df_long = df.select(list(final_columns.keys())).rename(final_columns)
    
    # Drop rows where essential columns might be null after all processing
    # (e.g. if 'game' was null, item_id would be null)
    df_long = df_long.drop_nulls(subset=["item_id", "timestamp", target_column])

    if df_long.height == 0:
        print("Warning: DataFrame is empty after all processing and NaN removal.")
    
    return df_long

# The aggregate_and_pivot function and _clean_game_names are no longer needed 
# for the AutoGluon-based approach and will be removed.

if __name__ == "__main__":
    # ... (main block for testing, no DEBUG prints to remove here generally)
    pass

    # Create a dummy CSV file for testing
    # Note: DUMMY_CSV_HEADER_RAW is now defined globally
    dummy_csv_header_str = ",".join(DUMMY_CSV_HEADER_RAW)
    dummy_csv_rows = [
        "1,01/01/2023 02:00 PM,01/01/2023 03:00 PM,A,User,a@b.com,123,2,2,0,0,2,,Indoor Game Alpha,,,,,,,,,,,FALSE,normal,,,,,,0,,0,100,10,TRUE,110,110,0,,01/01/2023 10:00 AM,,,,,,,,,,,,,,,,,,,",
        "2,01/01/2023 02:30 PM,01/01/2023 03:30 PM,B,User,b@b.com,123,3,3,0,0,3,,Indoor Game Alpha,,,,,,,,,,,FALSE,normal,,,,,,0,,0,150,15,TRUE,165,165,0,,01/01/2023 10:00 AM,,,,,,,,,,,,,,,,,,,",
        "3,01/01/2023 03:00 PM,01/01/2023 04:00 PM,C,User,c@b.com,123,4,4,0,0,4,,Indoor Game Beta,,,,,,,,,,,FALSE,normal,,,,,,0,,0,200,20,TRUE,220,220,0,,01/01/2023 11:00 AM,,,,,,,,,,,,,,,,,,,",
        "4,01/01/2023 03:00 PM,01/01/2023 04:00 PM,D,User,d@b.com,123,1,1,0,0,1,,ONLINE: Game Gamma,,,,,,,,,,,FALSE,normal,,,,,,0,,0,50,5,TRUE,55,55,0,,01/01/2023 11:00 AM,,,,,,,,,,,,,,,,,,,",
        "5,01/02/2023 04:00 PM,01/02/2023 05:00 PM,E,User,e@b.com,123,2,2,0,0,2,,Outdoor Adventure X,,,,,,,,,,,FALSE,normal,,,,,,0,,0,100,10,TRUE,110,110,0,,01/02/2023 12:00 PM,,,,,,,,,,,,,,,,,,,",
        "6,01/02/2023 04:00 PM,01/02/2023 05:00 PM,F,User,f@b.com,123,5,5,0,0,5,,Indoor Game Alpha,,,,,,,,,,,FALSE,canceled,,,,,,0,,0,250,25,TRUE,275,275,0,,01/02/2023 12:00 PM,,,,,,,,,,,,,,,,,,,",
        "7,01/03/2023 05:00 PM,01/03/2023 06:00 PM,G,User,g@b.com,123,2,2,0,0,2,,Indoor Game Alpha,,,,,,,,,,,FALSE,normal,,,,,,0,,0,100,10,TRUE,110,110,0,,01/03/2023 01:00 PM,,,,,,,,,,,,,,,,,,,",
    ]
    # Ensure each dummy row has the same number of fields as the header
    num_header_fields = len(DUMMY_CSV_HEADER_RAW)
    processed_dummy_rows = []
    for row_str in dummy_csv_rows:
        fields = row_str.split(',')
        # Pad with empty strings if too short, truncate if too long
        padded_fields = (fields + [''] * num_header_fields)[:num_header_fields]
        processed_dummy_rows.append(",".join(padded_fields))

    dummy_csv_content = dummy_csv_header_str + "\n" + "\n".join(processed_dummy_rows)

    dummy_csv_path = "dummy_bookings_test.csv"
    with open(dummy_csv_path, "w") as f:
        f.write(dummy_csv_content)

    # print(f"Created dummy CSV: {dummy_csv_path}")
    
    df_long = preprocess_escape_room_data(dummy_csv_path)

    # print("\n--- DataFrame ---")
    if df_long.height > 0:
        # print(df_long.head())
        # print(f"Shape: {df_long.shape}")
        # print(df_long.schema)
        pass # Added pass as the block is otherwise empty
    else:
        # print("DataFrame is empty.")
        pass # Added pass

    # Clean up dummy file
    import os
    os.remove(dummy_csv_path)
    # print(f"Removed dummy CSV: {dummy_csv_path}") 