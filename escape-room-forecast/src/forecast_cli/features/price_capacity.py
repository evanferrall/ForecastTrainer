import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_price_capacity_features(
    df: pd.DataFrame,
    flat_rate_col: str | None = "flat_rate", # Current name after snaking
    addl_player_fee_col: str | None = "additional_player_fee", # Current name after snaking
    known_covariates_list: list | None = None
) -> pd.DataFrame:
    """Adds price and capacity related features to the DataFrame.

    Args:
        df: Input DataFrame.
        flat_rate_col: Name of the flat rate column.
        addl_player_fee_col: Name of the additional player fee column.
        known_covariates_list: Optional list to append newly created feature names to.

    Returns:
        DataFrame with added features:
            - ft_flat_rate
            - ft_addl_player_fee
            - ft_capacity_left_placeholder (always 0 for now)
    """
    if df.empty:
        logger.warning("DataFrame is empty. Skipping price & capacity features.")
        return df

    new_features = []

    if flat_rate_col and flat_rate_col in df.columns:
        df["ft_flat_rate"] = pd.to_numeric(df[flat_rate_col], errors='coerce').fillna(0)
        logger.info("Added 'ft_flat_rate'.")
    else:
        df["ft_flat_rate"] = 0
        logger.warning(f"Flat rate column ('{flat_rate_col}') not found or not in DataFrame. Defaulting 'ft_flat_rate' to 0.")
    new_features.append("ft_flat_rate")

    if addl_player_fee_col and addl_player_fee_col in df.columns:
        df["ft_addl_player_fee"] = pd.to_numeric(df[addl_player_fee_col], errors='coerce').fillna(0)
        logger.info("Added 'ft_addl_player_fee'.")
    else:
        df["ft_addl_player_fee"] = 0
        logger.warning(f"Additional player fee column ('{addl_player_fee_col}') not found or not in DataFrame. Defaulting 'ft_addl_player_fee' to 0.")
    new_features.append("ft_addl_player_fee")
    
    # Placeholder for capacity_left - would require knowledge of total slots per time-block
    df["ft_capacity_left_placeholder"] = 0 
    new_features.append("ft_capacity_left_placeholder")
    logger.info("Added 'ft_capacity_left_placeholder' (currently always 0).")

    if known_covariates_list is not None:
        known_covariates_list.extend(new_features)

    return df 