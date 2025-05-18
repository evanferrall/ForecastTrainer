import pandas as pd
import numpy as np
import logging
from forecast_cli.utils.common import LEAD_TIME_BINS, LEAD_TIME_LABELS

logger = logging.getLogger(__name__)

def add_lead_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp", created_col: str = "created_at", known_covariates_list: list | None = None) -> pd.DataFrame:
    """Adds lead time features to the DataFrame.

    Args:
        df: Input DataFrame with timestamp and creation date columns.
        timestamp_col: Name of the main event timestamp column (e.g., game start time).
        created_col: Name of the booking creation timestamp column (should be processed and exist in df).
        known_covariates_list: Optional list to append newly created feature names to.

    Returns:
        DataFrame with added lead time features.
    """
    if df.empty:
        logger.warning("DataFrame is empty. Skipping lead time features.")
        return df

    new_features = []

    if created_col not in df.columns:
        logger.warning(f"Creation date column '{created_col}' not found. Skipping lead time features.")
        # Add placeholder columns to maintain schema if this feature is expected
        df['lead_time_days'] = np.nan 
        new_features.append('lead_time_days')
        for label in LEAD_TIME_LABELS:
            col_name = f'ft_lead_time_bucket_{label}'
            df[col_name] = 0
            new_features.append(col_name)
        if known_covariates_list is not None:
            known_covariates_list.extend(new_features)
        return df

    if timestamp_col not in df.columns:
        logger.warning(f"Timestamp column '{timestamp_col}' not found. Skipping lead time features.")
        df['lead_time_days'] = np.nan
        new_features.append('lead_time_days')
        for label in LEAD_TIME_LABELS:
            col_name = f'ft_lead_time_bucket_{label}'
            df[col_name] = 0
            new_features.append(col_name)
        if known_covariates_list is not None:
            known_covariates_list.extend(new_features)
        return df

    # Ensure columns are datetime
    try:
        # Assuming created_col is already datetime from DataPreprocessor._load_and_initial_clean
        # df[created_col] = pd.to_datetime(df[created_col], errors='coerce')
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce') 
    except Exception as e:
        logger.error(f"Error converting '{timestamp_col}' to datetime: {e}. Skipping lead time features.") # created_col conversion removed
        df['lead_time_days'] = np.nan
        new_features.append('lead_time_days')
        for label in LEAD_TIME_LABELS:
            col_name = f'ft_lead_time_bucket_{label}'
            df[col_name] = 0
            new_features.append(col_name)
        if known_covariates_list is not None:
            known_covariates_list.extend(new_features)
        return df

    # Calculate lead time in days
    df['lead_time_days'] = (df[timestamp_col] - df[created_col]).dt.total_seconds() / (24 * 60 * 60)
    df['lead_time_days'] = df['lead_time_days'].apply(lambda x: x if x >= 0 else 0)
    # df['lead_time_days'].fillna(0, inplace=True) # fillna(0) is applied in DataPreprocessor._add_all_features now
    new_features.append('lead_time_days')
    logger.info(f"Calculated 'lead_time_days'. Min: {df['lead_time_days'].min()}, Max: {df['lead_time_days'].max()}, Mean: {df['lead_time_days'].mean()}.")

    # Create lead time buckets
    bucket_col_names = [f'ft_lead_time_bucket_{label}' for label in LEAD_TIME_LABELS]
    if not LEAD_TIME_BINS or not LEAD_TIME_LABELS or len(LEAD_TIME_BINS) != len(LEAD_TIME_LABELS) + 1:
        logger.error("LEAD_TIME_BINS or LEAD_TIME_LABELS are not correctly defined. Skipping bucketing.")
        for col_name in bucket_col_names:
            df[col_name] = 0
            new_features.append(col_name)
        if known_covariates_list is not None:
            known_covariates_list.extend(new_features)
        return df
    
    df['lead_time_bucket_cat'] = pd.cut(df['lead_time_days'].fillna(0), bins=LEAD_TIME_BINS, labels=LEAD_TIME_LABELS, right=False)
    bucket_dummies = pd.get_dummies(df['lead_time_bucket_cat'], prefix='ft_lead_time_bucket')
    
    for i, label in enumerate(LEAD_TIME_LABELS):
        expected_col_name = bucket_col_names[i]
        if expected_col_name in bucket_dummies.columns:
            df[expected_col_name] = bucket_dummies[expected_col_name].astype(int)
        else:
            df[expected_col_name] = 0 
        new_features.append(expected_col_name)
            
    df.drop(columns=['lead_time_bucket_cat'], inplace=True, errors='ignore')
    logger.info(f"Added lead time bucket features: {bucket_col_names}")

    if known_covariates_list is not None:
        known_covariates_list.extend(new_features)

    return df 