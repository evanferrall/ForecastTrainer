import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_external_data_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Adds placeholder columns for external data features.

    These are 'known only up to T-0', so their future values are not supplied 
    at prediction time by AutoGluon if marked as past_covariates. 
    If they are known future covariates, they'd be handled differently.
    For now, these are just placeholders in the historical data.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with added placeholder columns:
            - ext_temp_c
            - ext_precip_mm
            - ext_snow_cm
            - ext_google_trends
            - ext_is_major_event
    """
    if df.empty:
        logger.warning("DataFrame is empty. Skipping external data placeholders.")
        return df

    df["ext_temp_c"] = 0.0  # Placeholder for hourly temperature
    df["ext_precip_mm"] = 0.0  # Placeholder for hourly precipitation
    df["ext_snow_cm"] = 0.0  # Placeholder for hourly snow
    df["ext_google_trends"] = 0.0  # Placeholder for weekly Google Trends (upsampled to hourly)
    df["ext_is_major_event"] = 0  # Placeholder for daily major event flag (upsampled to hourly)
    logger.info("Added placeholder columns for external data features (temp, precip, snow, trends, events).")
    
    return df 