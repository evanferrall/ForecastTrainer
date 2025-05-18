import pandas as pd
import logging
from forecast_cli.utils.common import LEAD_TIME_BINS, LEAD_TIME_LABELS

logger = logging.getLogger(__name__)

def add_lead_time_features(
    df: pd.DataFrame, 
    timestamp_col: str = "timestamp", 
    created_col: str = "created_at", # Assuming this is the current name after snaking
    add_one_hot_buckets: bool = True
) -> pd.DataFrame:
    """Adds lead time features to the DataFrame.

    Args:
        df: Input DataFrame.
        timestamp_col: Name of the event timestamp column.
        created_col: Name of the creation timestamp column.
        add_one_hot_buckets: Whether to add one-hot encoded lead time buckets.

    Returns:
        DataFrame with added lead time features:
            - lead_time_days
            - ft_lead_time_bucket_0_1d (if add_one_hot_buckets)
            - ft_lead_time_bucket_2_7d (if add_one_hot_buckets)
            - ft_lead_time_bucket_8_30d (if add_one_hot_buckets)
            - ft_lead_time_bucket_gt_30d (if add_one_hot_buckets)
    """
    if df.empty:
        logger.warning("DataFrame is empty. Skipping lead time features.")
        return df

    if created_col not in df.columns or timestamp_col not in df.columns:
        logger.warning(
            f"Required columns ('{created_col}' or '{timestamp_col}') not found. "
            f"Skipping lead time features."
        )
        # Add default columns if they are expected downstream
        df['lead_time_days'] = 0
        if add_one_hot_buckets:
            for label in LEAD_TIME_LABELS:
                df[f'ft_lead_time_bucket_{label}'] = 0
        return df

    try:
        # Ensure columns are datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df[created_col] = pd.to_datetime(df[created_col])

        # Calculate lead time in days
        df['lead_time_days'] = (df[timestamp_col] - df[created_col]).dt.days
        # Handle potential NaNs from invalid 'created' dates or subtraction, clip at 0
        df['lead_time_days'] = df['lead_time_days'].fillna(0).clip(lower=0).astype(int)
        logger.info("Added 'lead_time_days' feature.")

        if add_one_hot_buckets:
            # Create ft_lead_time_bucket and then one-hot encode it.
            df['ft_lead_time_bucket_categorical'] = pd.cut(
                df['lead_time_days'], 
                bins=LEAD_TIME_BINS, 
                labels=LEAD_TIME_LABELS, 
                right=True
            )
            
            # One-hot encode the bucket categories
            lead_time_dummies = pd.get_dummies(
                df['ft_lead_time_bucket_categorical'], 
                prefix='ft_lead_time_bucket', 
                dummy_na=False # Do not create a column for NaN buckets
            )
            df = pd.concat([df, lead_time_dummies], axis=1)
            
            # Ensure all expected one-hot columns exist, filling with 0 if a category was not present
            for label in LEAD_TIME_LABELS:
                col_name = f'ft_lead_time_bucket_{label}'
                if col_name not in df.columns:
                    df[col_name] = 0
            
            df.drop(columns=['ft_lead_time_bucket_categorical'], inplace=True)
            logger.info(f"Added one-hot encoded lead time bucket features: ft_lead_time_bucket_...")

    except Exception as e:
        logger.error(f"Error calculating lead time features: {e}. Adding default columns.")
        df['lead_time_days'] = 0
        if add_one_hot_buckets:
            for label in LEAD_TIME_LABELS:
                df[f'ft_lead_time_bucket_{label}'] = 0
                
    return df 