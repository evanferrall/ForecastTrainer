import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_static_features(unique_series_ids: list[str], channel_map: dict | None = None) -> pd.DataFrame:
    """Creates static features for each series_id.

    Args:
        unique_series_ids: List of unique series IDs (e.g., 'bookings_game_a').
        channel_map: Optional dictionary mapping series_id to its channel (0 for online, 1 for in-person).
                     This should come from ft_channel_is_online (where 0=online, 1=IRL).
                     So, sf_is_in_real_life = 1 if channel_map value is 1, else 0.

    Returns:
        pd.DataFrame with 'series_id' as index and static features as columns:
            - sf_kpi_family (e.g., 'bookings', 'minutes')
            - sf_game_normalized (e.g., 'game_a')
            - sf_is_bookings
            - sf_is_minutes
            - sf_is_revenue
            - sf_is_probability
            - sf_is_in_real_life (based on channel_map)
    """
    if len(unique_series_ids) == 0:
        logger.warning("No unique series IDs provided to create_static_features. Returning empty DataFrame.")
        # Return empty DataFrame with expected index name and columns for consistency
        return pd.DataFrame(index=pd.Index([], name="series_id"), 
                            columns=['sf_kpi_family', 'sf_game_normalized', 
                                     'sf_is_bookings', 'sf_is_minutes', 'sf_is_revenue', 
                                     'sf_is_probability', 'sf_is_in_real_life'])

    static_data = []
    for sid in unique_series_ids:
        parts = sid.split('_', 1)
        kpi_family = parts[0] if len(parts) > 0 else "unknown"
        game_normalized = parts[1] if len(parts) > 1 else sid # Fallback if no underscore

        is_irl = 0 # Default to not in-real-life (online)
        if channel_map and sid in channel_map:
            # channel_map values: 0 for online, 1 for in-person (physical)
            # sf_is_in_real_life should be 1 if physical, 0 if online.
            is_irl = 1 if channel_map[sid] == 1 else 0 
            
        static_data.append({
            "series_id": sid,
            "sf_kpi_family": kpi_family,
            "sf_game_normalized": game_normalized,
            "sf_is_bookings": int(kpi_family == "bookings"),
            "sf_is_minutes": int(kpi_family == "minutes"),
            "sf_is_revenue": int(kpi_family == "revenue"),
            "sf_is_probability": int(kpi_family == "prob"), # 'prob' is the family name for probability
            "sf_is_in_real_life": is_irl
        })

    static_df = pd.DataFrame(static_data)
    if not static_df.empty:
        static_df = static_df.set_index("series_id")
    else: # Handle case where static_data was empty for some reason (should not happen if unique_series_ids is not empty)
        logger.warning("Static data list was empty despite having unique_series_ids. Returning empty DataFrame with columns.")
        return pd.DataFrame(index=pd.Index([], name="series_id"), 
                            columns=['sf_kpi_family', 'sf_game_normalized', 
                                     'sf_is_bookings', 'sf_is_minutes', 'sf_is_revenue', 
                                     'sf_is_probability', 'sf_is_in_real_life'])
        
    logger.info(f"Created static features DataFrame with shape: {static_df.shape}")
    logger.debug(f"Static features head:\n{static_df.head()}")
    return static_df 