import polars as pl
# from polars import Expr as E # Removed problematic import
from typing import List

# --- Feature: Capacity Saturation ---
def add_capacity_saturation_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Adds capacity saturation feature: rooms_available - rooms_booked_so_far (zero-floor).
    Roadmap Section 7: Accuracy-boost tricks.

    Assumes df has columns: 'rooms_available', 'rooms_booked_so_far'.
    """
    print("Adding capacity saturation features...")
    required_cols = ["rooms_available", "rooms_booked_so_far"]
    for col_name in required_cols:
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found for capacity saturation. Returning original df.")
            return df

    df = df.with_columns(
        (pl.col("rooms_available") - pl.col("rooms_booked_so_far")) # Use pl.col()
        .clip_lower_bound(0) 
        .alias("remaining_capacity")
    )
    return df

# --- Feature: Lead-time Effect ---
def add_lead_time_features(df: pl.LazyFrame, group_by_cols: List[str], time_col: str) -> pl.LazyFrame:
    """
    Adds lead-time effect features: Lagged cumulative bookings at T-7d, -14d as static covariates.
    Roadmap Section 7: Accuracy-boost tricks.

    Assumes df has columns for bookings (e.g., 'booking_count'), a time column (e.g., 'date'),
    and group_by_cols (e.g., ['branch_id']).
    This is intended for daily data to create static covariates for the daily head.
    """
    print("Adding lead-time effect features (placeholder - complex logic)...")
    required_cols_lead = [time_col, "booking_count"] + group_by_cols
    for col_name in required_cols_lead:
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found for lead-time features. Returning original df.")
            return df
    
    print("Placeholder: Lead-time features require complex time-series operations (lagged cumulative sums).")
    df = df.with_columns([
        pl.lit(0).cast(pl.Int64).alias("cum_bookings_t_minus_7d_placeholder"),
        pl.lit(0).cast(pl.Int64).alias("cum_bookings_t_minus_14d_placeholder"),
    ])
    return df

# --- Feature: Google Trends / Meta Ads Spend ---
def add_external_signal_features(df: pl.LazyFrame, time_col: str) -> pl.LazyFrame:
    """
    Adds features from Google Trends / Meta Ads spend.
    Roadmap Section 7: Pull weekly index, forward-fill, feed to both heads.
    Assumes df has a time_col. The external data (trends, ads) needs to be loaded separately
    and joined/merged with df.
    """
    print("Adding external signal features (Google Trends, Ads - placeholder)...")
    if time_col not in df.columns:
        print(f"Warning: Time column '{time_col}' not found for external signals. Returning original df.")
        return df

    print("Placeholder: External signal features require data loading and joining logic.")
    df = df.with_columns([
        pl.lit(0.0).alias("google_trends_placeholder"),
        pl.lit(0.0).alias("meta_ads_spend_placeholder"),
    ])
    return df

# --- Main Feature Building Function ---
def build_features(df: pl.LazyFrame, config=None) -> pl.LazyFrame:
    """
    Main function to orchestrate feature building on a Polars LazyFrame.
    Based on roadmap sections 2 and 7.

    Args:
        df: Input Polars LazyFrame.
        config: Optional configuration object (e.g., AppConfig from utils.config)
                to get column names, paths, etc.

    Returns:
        Polars LazyFrame with added features.
    """
    print(f"Input df columns before feature building: {df.columns}")

    # --- Calendar Features (Roadmap Section 2) ---
    # TODO: Implement calendar features (dow, wom, doy, quarter, month_start)
    # Example: Assuming a 'date' column exists
    # if "date" in df.columns:
    #     df = df.with_columns([
    #         E.col("date").dt.weekday().alias("day_of_week"),
    #         E.col("date").dt.day().alias("day_of_month"),
    #         E.col("date").dt.month().alias("month"),
    #         # ... and more complex ones like week_of_month, quarter etc.
    #     ])
    # else:
    #     print("Warning: 'date' column not found for calendar features.")
    print("Placeholder: Calendar features to be implemented.")

    # --- Holiday/Event Features (Roadmap Section 2) ---
    # TODO: Implement holiday/event features (is_school_holiday, is_local_event)
    # This would involve joining with external holiday/event calendar data.
    print("Placeholder: Holiday/Event features to be implemented.")

    # --- Weather Lags/Leads (Roadmap Section 2) ---
    # TODO: Implement weather lags/leads. Requires weather data and joining.
    print("Placeholder: Weather lag/lead features to be implemented.")

    # --- Accuracy Boost Tricks (Roadmap Section 7) ---
    # 1. Capacity Saturation
    df = add_capacity_saturation_features(df)

    # 2. Lead-time Effect
    # Needs group_by_cols (e.g. branch_id) and time_col (e.g. date for daily data)
    # These should ideally come from config
    group_cols_for_lead_time = ["branch_id"] # Example
    time_col_for_lead_time = "date" # Example, assuming daily granularity for this feature
    # Check if necessary columns exist before calling
    if all(c in df.columns for c in group_cols_for_lead_time + [time_col_for_lead_time]):
        df = add_lead_time_features(df, group_cols_for_lead_time, time_col_for_lead_time)
    else:
        print(f"Skipping lead-time features due to missing columns (requires: {group_cols_for_lead_time + [time_col_for_lead_time]})")

    # 3. Google Trends / Meta Ads Spend
    time_col_for_external = "date" # Example, common time column
    if time_col_for_external in df.columns:
        df = add_external_signal_features(df, time_col_for_external)
    else:
        print(f"Skipping external signal features due to missing column: {time_col_for_external}")

    # 4. Multi-instance transfer (Handled in dataset creation - `group_ids`)
    # Nothing to do in feature_builder.py itself for this, but noted here.

    # 5. Quantile fusion (Post-processing, not in feature_builder.py)

    # Collect at the end of build_features to inspect the schema of the transformed LazyFrame
    # It's good practice to df.schema to check schema during lazy operations too.
    final_df_collected = df.collect() # Collect once after all lazy operations
    print(f"Output df columns after feature building attempt: {final_df_collected.columns}") 
    return final_df_collected.lazy() # Return as LazyFrame again


if __name__ == "__main__":
    print("Testing feature_builder.py...")
    
    # Create a dummy Polars DataFrame for testing
    dummy_data = {
        "branch_id": ["A", "A", "B", "A", "B", "B"],
        "date": pl.date_range(low=pl.datetime(2023,1,1), high=pl.datetime(2023,1,3), interval="1d", eager=True).to_list() * 2,
        "rooms_available": [10, 10, 5, 10, 5, 5],
        "rooms_booked_so_far": [5, 8, 1, 9, 5, 2],
        "booking_count": [3, 6, 1, 7, 4, 2] # Daily bookings
    }
    lazy_df_main = pl.DataFrame(dummy_data).lazy()

    print("Original LazyFrame schema:")
    print(lazy_df_main.schema) # Correct way to print schema for LazyFrame
    print("Original LazyFrame data (collected for display):")
    print(lazy_df_main.collect())

    # Test feature building
    try:
        lazy_df_with_features_main = build_features(lazy_df_main)
        print("\nLazyFrame with features schema:")
        print(lazy_df_with_features_main.schema) # Correct way to print schema
        print("LazyFrame with features data (collected for display):")
        print(lazy_df_with_features_main.collect())
        
        print("\nTesting specific features:")
        # Test capacity saturation
        df_cap_test_lazy = pl.DataFrame({
            "rooms_available": [10, 5, 3],
            "rooms_booked_so_far": [8, 5, 4]
        }).lazy()
        df_cap_test_out_collected = add_capacity_saturation_features(df_cap_test_lazy).collect()
        print("Capacity Saturation Test Output:")
        print(df_cap_test_out_collected)
        assert df_cap_test_out_collected["remaining_capacity"].to_list() == [2,0,0]

    except Exception as e_main:
        print(f"Error during feature_builder test: {e_main}")
        import traceback
        traceback.print_exc()
