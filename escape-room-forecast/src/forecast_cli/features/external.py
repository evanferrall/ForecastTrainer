import pandas as pd
import numpy as np # Added for more complex placeholders
import logging

logger = logging.getLogger(__name__)

def add_external_data_placeholders(df: pd.DataFrame, timestamp_col: str = "timestamp", known_covariates_list: list | None = None) -> pd.DataFrame:
    """Adds placeholder columns for external data features with varied values for testing.

    These are 'known only up to T-0', so their future values are not supplied 
    at prediction time by AutoGluon if marked as past_covariates. 
    If they are known future covariates, they'd be handled differently.
    For now, these are just placeholders in the historical data.

    Args:
        df: Input DataFrame.
        timestamp_col: Name of the timestamp column in df (used for generating varied data).
        known_covariates_list: Optional list to append newly created feature names to.

    Returns:
        DataFrame with added placeholder columns.
    """
    new_features = ["ext_temp_c", "ext_precip_mm", "ext_snow_cm", "ext_google_trends", "ext_is_major_event"]

    if df.empty:
        logger.warning("DataFrame is empty. Skipping external data placeholders.")
        # Add columns anyway if df is empty but has columns, so downstream doesn't break
        for col_name in new_features:
             if col_name not in df.columns:
                    df[col_name] = pd.Series(dtype='float64' if 'mm' in col_name or 'temp' in col_name or 'snow' in col_name or 'trends' in col_name else 'int64')
        # Even if empty, if list is provided, assume these columns *would* have been added.
        if known_covariates_list is not None:
            known_covariates_list.extend(new_features)
        return df

    if timestamp_col not in df.columns:
        logger.warning(f"Timestamp column '{timestamp_col}' not found. Adding 0s for external features.")
        df["ext_temp_c"] = 0.0
        df["ext_precip_mm"] = 0.0
        df["ext_snow_cm"] = 0.0
        df["ext_google_trends"] = 0.0
        df["ext_is_major_event"] = 0
        if known_covariates_list is not None:
            known_covariates_list.extend(new_features)
        return df
    
    # Ensure timestamp is datetime for dt accessor
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Temperature: cycle based on hour of day
    temp_cycle = [5, 6, 8, 10, 12, 14, 15, 16, 15, 13, 11, 9] # Simple 12-hour cycle pattern
    df["ext_temp_c"] = df[timestamp_col].dt.hour.apply(lambda h: temp_cycle[h % 12]).astype(float)

    # Precipitation: non-zero for a few specific hours
    df["ext_precip_mm"] = 0.0
    # Example: make it rain between hour 2 and 4, and hour 14 and 16
    precip_hours = [2,3,4, 14,15,16]
    df.loc[df[timestamp_col].dt.hour.isin(precip_hours), "ext_precip_mm"] = np.random.choice([0.5, 1.0, 1.5], size=df[df[timestamp_col].dt.hour.isin(precip_hours)].shape[0])


    # Snow: similar to precipitation, but different hours and potentially different logic (e.g. only in winter months)
    df["ext_snow_cm"] = 0.0
    snow_hours = [22,23,0,1] # Example: snow late night / early morning
    # Only apply snow if month is Dec, Jan, Feb (winter months)
    winter_months = [12, 1, 2]
    df.loc[df[timestamp_col].dt.hour.isin(snow_hours) & df[timestamp_col].dt.month.isin(winter_months), "ext_snow_cm"] = np.random.choice([1.0, 2.0, 3.0], size=df[df[timestamp_col].dt.hour.isin(snow_hours) & df[timestamp_col].dt.month.isin(winter_months)].shape[0])


    # Google Trends: weekly-like pattern, changes every N days (e.g., 7)
    # Using dayofyear to create a somewhat weekly step pattern
    df["ext_google_trends"] = (df[timestamp_col].dt.dayofyear // 7 % 5).astype(float) * 10 # Values 0, 10, 20, 30, 40

    # Major Event: flag for a specific day or days
    df["ext_is_major_event"] = 0
    # Example: A major event on Jan 10th and July 1st of any year in the data
    specific_event_dates = [(1, 10), (7, 1)] # (Month, Day)
    for month, day in specific_event_dates:
        df.loc[(df[timestamp_col].dt.month == month) & (df[timestamp_col].dt.day == day), "ext_is_major_event"] = 1
    
    logger.info("Added varied placeholder columns for external data features.")

    if known_covariates_list is not None:
        # Ensure we only add features that were actually created (all are in this case)
        known_covariates_list.extend(new_features) 
    
    return df 