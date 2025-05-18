import pandas as pd
import logging

# Test comment to force recompile / cache refresh
logger = logging.getLogger(__name__)

def create_static_features(
    unique_series_ids: list[str],
    series_id_to_channel_map: dict[str, int] | None = None
) -> pd.DataFrame:
    """Creates a DataFrame of static features for given series IDs.

    Args:
        unique_series_ids: A list of unique series_id strings.
        series_id_to_channel_map: A mapping from series_id to its 'ft_channel_is_online' value.
                                    If None, 'sf_is_in_real_life' defaults to 1 (True).

    Returns:
        A DataFrame with 'series_id' as index and static feature columns:
            - sf_game_norm
            - sf_genre
            - sf_is_in_real_life
            - sf_is_horror
            - sf_room_size_group
            - sf_base_price_group
    """
    static_features_list = []
    expected_sf_columns = [
        "series_id", "sf_game_norm", "sf_genre", "sf_is_in_real_life", 
        "sf_is_horror", "sf_room_size_group", "sf_base_price_group"
    ]

    if not series_id_to_channel_map:
        series_id_to_channel_map = {}

    if not unique_series_ids:
        logger.warning("No unique series IDs provided. Returning empty static features DataFrame.")
        empty_static_df = pd.DataFrame(columns=[col for col in expected_sf_columns if col != "series_id"])
        empty_static_df.index.name = "series_id"
        return empty_static_df

    for sid in unique_series_ids:
        game_norm_parts = sid.split("_", 1)
        game_norm = game_norm_parts[1] if len(game_norm_parts) > 1 else "unknown_game"

        sf_genre = "Misc"
        if any(keyword in game_norm for keyword in ["asylum", "crypt", "zombie", "curse", "horror", "nightmare"]):
            sf_genre = "Horror"
        elif any(keyword in game_norm for keyword in ["quest", "journey", "expedition", "adventure", "mystery"]):
            sf_genre = "Adventure"
        elif any(keyword in game_norm for keyword in ["galaxy", "space", "time", "future", "sci"]):
            sf_genre = "SciFi"
        elif any(keyword in game_norm for keyword in ["kids", "family", "fun", "magic"]):
            sf_genre = "Family"

        sf_is_horror = 1 if sf_genre == "Horror" else 0
        
        series_ft_channel_is_online = series_id_to_channel_map.get(sid, 0) # Default to 0 (not online)
        sf_is_in_real_life = 1 - series_ft_channel_is_online

        sf_room_size_group = "standard"
        if any(keyword in game_norm for keyword in ["large", "asylum", "complex"]):
            sf_room_size_group = "large"
        elif any(keyword in game_norm for keyword in ["small", "kids", "compact"]):
            sf_room_size_group = "small"
        
        sf_base_price_group = "standard"
        if any(keyword in game_norm for keyword in ["premium", "vip", "deluxe"]):
            sf_base_price_group = "high"
        elif any(keyword in game_norm for keyword in ["budget", "basic", "promo"]):
            sf_base_price_group = "low"

        static_features_list.append({
            "series_id": sid,
            "sf_game_norm": game_norm,
            "sf_genre": sf_genre,
            "sf_is_in_real_life": sf_is_in_real_life,
            "sf_is_horror": sf_is_horror,
            "sf_room_size_group": sf_room_size_group,
            "sf_base_price_group": sf_base_price_group
        })
    
    static_features_df = pd.DataFrame(static_features_list)
    if not static_features_df.empty:
        static_features_df = static_features_df.set_index("series_id")
        logger.info(f"Generated static_features_df with shape: {static_features_df.shape}")
    else:
        logger.info("No static features generated (list was empty). Returning empty DataFrame with series_id index.")
        # Ensure it has the correct index name even if empty
        static_features_df = pd.DataFrame(columns=[col for col in expected_sf_columns if col != "series_id"])
        static_features_df.index.name = "series_id"
        
    return static_features_df 