import pandas as pd
import numpy as np
import logging
import traceback
from pathlib import Path

from forecast_cli.utils.common import (
    _snake, 
    make_unique_cols, 
    logit_transform,
    ONLINE_GAME_INDICATORS,
    VALID_STATUSES,
    LEAD_TIME_LABELS
)
from forecast_cli.features import (
    add_calendar_features,
    add_lead_time_features,
    add_promotional_features,
    add_price_capacity_features,
    add_external_data_placeholders,
    create_static_features,
    add_weather_features
)

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing for time series forecasting."""

    def __init__(self,
                 csv_path: str | Path,
                 kpi_configs: dict,
                 ts_col: str = "start",
                 end_col: str = "end",
                 raw_created_col: str = 'created',
                 raw_promo_col: str = 'promotion',
                 raw_coupon_col: str = 'coupons',
                 raw_flat_rate_col: str = 'flat_rate',
                 raw_addl_player_fee_col: str = 'additional_player_fee',
                 raw_participants_col: str = 'participants'
                 ):
        """Initialize DataPreprocessor.

        Args:
            csv_path: Path to the CSV data.
            kpi_configs: Dictionary of KPI configurations.
            ts_col: Name of the main timestamp column.
            end_col: Name of the end timestamp column.
            raw_created_col: Original name of the 'created at' column.
            raw_promo_col: Original name of the 'promotion title' column.
            raw_coupon_col: Original name of the 'coupon code used' column.
            raw_flat_rate_col: Original name of the 'flat rate booking' indicator column.
            raw_addl_player_fee_col: Original name of the 'additional player fee' column.
            raw_participants_col: Original name of the 'participants' or player count column.
        """
        self.csv_path = Path(csv_path)
        self.kpi_configs = kpi_configs
        self.raw_ts_col = ts_col
        self.raw_end_col = end_col
        
        # Store configured raw column names
        self.raw_created_col = raw_created_col
        self.raw_promo_col = raw_promo_col
        self.raw_coupon_col = raw_coupon_col
        self.raw_flat_rate_col = raw_flat_rate_col
        self.raw_addl_player_fee_col = raw_addl_player_fee_col
        self.raw_participants_col = raw_participants_col
        
        # These will be updated after initial snaking and uniquing
        self.current_ts_col: str | None = None
        self.current_end_col: str | None = None
        self.current_participants_col: str | None = None
        self.current_total_gross_col: str | None = None
        self.current_status_col: str | None = None
        self.current_game_col: str | None = None
        self.current_created_col: str | None = None
        self.current_game_location_col: str | None = None
        # For promotional features
        self.current_promo_col: str | None = None
        self.current_num_coupons_col: str | None = None
        self.current_coupons_text_col: str | None = None
        self.current_gift_voucher_col: str | None = None
        self.current_prepaid_pkg_col: str | None = None
        # For price/capacity features
        self.current_flat_rate_col: str | None = None
        self.current_addl_player_fee_col: str | None = None

        # Placeholder for series prefixes that need logit transformation on 'y'
        self.PROB_TARGET_COLUMNS = ["prob"]

        # Initialize lists for known covariate names
        self.known_dynamic_covariates_cols: list[str] = []
        self.known_static_covariates_cols: list[str] = []
        self.ft_channel_is_online_col: str = 'ft_channel_is_online' # Define standard name

        # KPI configuration for aggregation and feature selection

    def _load_and_initial_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial cleaning: timestamp conversion, snaking columns, unique columns."""
        logger.info(f"Starting initial load and clean for CSV: {self.csv_path}")
        
        # --- Lever 1: Timezone conversion at Ingest ---
        if self.raw_ts_col not in df.columns:
            raise ValueError(f"Timestamp column '{self.raw_ts_col}' not found. Columns: {df.columns.tolist()}")
        if self.raw_end_col not in df.columns:
            raise ValueError(f"End timestamp column '{self.raw_end_col}' not found. Columns: {df.columns.tolist()}")

        df[self.raw_ts_col] = pd.to_datetime(df[self.raw_ts_col], errors='coerce')
        df[self.raw_end_col] = pd.to_datetime(df[self.raw_end_col], errors='coerce')
        df.dropna(subset=[self.raw_ts_col, self.raw_end_col], inplace=True)

        try:
            df[self.raw_ts_col] = df[self.raw_ts_col].dt.tz_localize('America/Toronto', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC').dt.tz_localize(None)
            df[self.raw_end_col] = df[self.raw_end_col].dt.tz_localize('America/Toronto', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception as e:
            logger.critical(f"CRITICAL ERROR during timezone localization: {e}. Ensure 'pytz' is installed and timezone is correct.", exc_info=True)
            raise
        df.dropna(subset=[self.raw_ts_col, self.raw_end_col], inplace=True)
        logger.info("Applied timezone conversion: localized to America/Toronto, converted to UTC, made naive.")

        # Store original column names before snaking for robust mapping
        original_cols_map = {c: _snake(c) for c in df.columns}
        df.columns = [_snake(c) for c in df.columns]
        df.columns = make_unique_cols(df.columns.tolist())
        logger.debug(f"Columns after _snake and make_unique_cols: {df.columns.tolist()}")

        # --- Identify current names of key columns AFTER renaming ---
        # This map helps find the new snaked/uniqued name from the original raw name's snaked version
        current_name_map: dict[str, str | None] = {} 
        raw_semantic_to_snaked_map = {
            self.raw_ts_col: _snake(self.raw_ts_col),
            self.raw_end_col: _snake(self.raw_end_col),
            self.raw_participants_col: _snake(self.raw_participants_col),
            "total_gross": _snake("total_gross"),
            "status": _snake("status"),
            "game": _snake("game"), # Often 'Game Name' or similar
            # Use configured raw names for these semantic keys
            self.raw_created_col: _snake(self.raw_created_col),
            "game_location": _snake("game_location"),
            # Promotional raw names (using configured names as keys if they are the semantic meaning)
            self.raw_promo_col: _snake(self.raw_promo_col),
            "number_of_coupons": _snake("number_of_coupons"), # Keep if distinct from coupon_code_col
            self.raw_coupon_col: _snake(self.raw_coupon_col),
            "specific_gift_voucher": _snake("specific_gift_voucher"),
            "prepaid_package": _snake("prepaid_package"),
            # Price/Capacity raw names
            self.raw_flat_rate_col: _snake(self.raw_flat_rate_col),
            self.raw_addl_player_fee_col: _snake(self.raw_addl_player_fee_col),
        }

        for raw_name_key, assumed_snaked_name in raw_semantic_to_snaked_map.items():
            if assumed_snaked_name in df.columns:
                current_name_map[raw_name_key] = assumed_snaked_name
            else: # Check for suffixed versions like "start_1"
                potential_matches = [c for c in df.columns if c.startswith(assumed_snaked_name + "_")]
                if potential_matches:
                    current_name_map[raw_name_key] = potential_matches[0]
                    logger.debug(f"Column '{assumed_snaked_name}' (from raw semantic key '{raw_name_key}') not found directly, using suffixed version '{potential_matches[0]}'.")
                else:
                    # Only raise error for absolutely critical columns for basic processing
                    if raw_name_key in [self.raw_ts_col, self.raw_end_col, "participants", "status", "game"]:
                         raise ValueError(f"Critical base column for '{raw_name_key}' (expected ~'{assumed_snaked_name}') not found after renaming. Columns: {df.columns.tolist()}")
                    logger.info(f"Optional column for '{raw_name_key}' (expected ~'{assumed_snaked_name}') not found. Will proceed without it if possible.")
                    current_name_map[raw_name_key] = None
        
        self.current_ts_col = current_name_map.get(self.raw_ts_col)
        self.current_end_col = current_name_map.get(self.raw_end_col)
        self.current_participants_col = current_name_map.get(self.raw_participants_col)
        self.current_total_gross_col = current_name_map.get("total_gross")
        self.current_status_col = current_name_map.get("status")
        self.current_game_col = current_name_map.get("game")
        # Retrieve based on configured raw names
        self.current_created_col = current_name_map.get(self.raw_created_col)
        self.current_game_location_col = current_name_map.get("game_location")
        
        self.current_promo_col = current_name_map.get(self.raw_promo_col) # Keep promo_col for promo_title
        self.current_num_coupons_col = current_name_map.get("number_of_coupons")
        self.current_coupons_text_col = current_name_map.get(self.raw_coupon_col) # coupons_text_col for coupon codes
        self.current_gift_voucher_col = current_name_map.get("specific_gift_voucher")
        self.current_prepaid_pkg_col = current_name_map.get("prepaid_package")
        
        self.current_flat_rate_col = current_name_map.get(self.raw_flat_rate_col)
        self.current_addl_player_fee_col = current_name_map.get(self.raw_addl_player_fee_col)

        if not self.current_ts_col or not self.current_end_col:
             raise EnvironmentError("Timestamp columns (ts_col, end_col) could not be definitively identified after renaming.")
        if not self.current_status_col:
            raise EnvironmentError("Status column could not be definitively identified.")
        if not self.current_game_col:
            raise EnvironmentError("Game column could not be definitively identified.")
            
        # Apply timezone conversion to identified 'created' column as well, if it exists
        if self.current_created_col and self.current_created_col in df.columns:
            # Ensure it's datetime first (might have been done if it was raw_ts_col/raw_end_col, but good to be sure)
            df[self.current_created_col] = pd.to_datetime(df[self.current_created_col], errors='coerce')
            try:
                df[self.current_created_col] = df[self.current_created_col].dt.tz_localize('America/Toronto', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC').dt.tz_localize(None)
                df.dropna(subset=[self.current_created_col], inplace=True) # Drop rows if TZ conversion results in NaT
                logger.info(f"Applied timezone conversion (Toronto -> UTC -> Naive) to '{self.current_created_col}'.")
            except Exception as e:
                logger.warning(f"Could not apply timezone conversion to '{self.current_created_col}': {e}. It will be used as previously parsed (likely naive).", exc_info=True)

        return df

    def _filter_and_basic_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply status filters, game location filters, and basic calculated columns."""
        # Filter by status
        if self.current_status_col and self.current_status_col in df.columns:
            df = df[df[self.current_status_col].astype(str).str.lower().isin(VALID_STATUSES)]
            df.reset_index(drop=True, inplace=True)
            logger.info(f"Filtered by status column '{self.current_status_col}' using {VALID_STATUSES}.")
        else: # Should have been caught by raise in _load_and_initial_clean
            raise ValueError("Status column not identified, cannot filter.")

        # Filter out online game locations (first pass, can be done again after channel feature)
        if self.current_game_location_col and self.current_game_location_col in df.columns:
            logger.info(f"Original unique values in '{self.current_game_location_col}': {df[self.current_game_location_col].unique().tolist()}")
            initial_row_count = len(df)
            df = df[~df[self.current_game_location_col].astype(str).str.lower().isin(ONLINE_GAME_INDICATORS)]
            logger.info(f"Filtered out {initial_row_count - len(df)} rows based on '{self.current_game_location_col}' being in {ONLINE_GAME_INDICATORS}.")
            if df.empty:
                logger.warning("DataFrame empty after filtering out online game locations.")
        else:
            logger.warning(f"Column '{self.current_game_location_col}' not found for game location filtering.")

        # Basic calculated columns (duration, game_norm, main timestamp)
        if self.current_ts_col and self.current_end_col and self.current_game_col:
            df["duration_minutes"] = (df[self.current_end_col] - df[self.current_ts_col]).dt.total_seconds() / 60
            df["game_norm"] = df[self.current_game_col].astype(str).str.lower().pipe(
                lambda s: s.str.replace(r"[^\w\s]", "", regex=True).str.replace(" ", "_"))
            df["timestamp"] = df[self.current_ts_col].dt.floor("h") # Main timestamp for TS modeling
            logger.info("Added 'duration_minutes', 'game_norm', and floored 'timestamp' columns.")
        else:
            raise ValueError("Essential columns (ts, end, game) not identified for basic transforms.")
        return df

    def _winsorize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize specified numerical features."""
        if self.current_participants_col and self.current_participants_col in df.columns and self.current_game_col in df.columns:
            df[self.current_participants_col] = pd.to_numeric(df[self.current_participants_col], errors='coerce')
            p99_participants = df.groupby(self.current_game_col, observed=True)[self.current_participants_col].quantile(0.99)
            df = df.join(p99_participants.rename(f'{self.current_participants_col}_p99'), on=self.current_game_col)
            df[self.current_participants_col] = np.where(
                df[self.current_participants_col] > df[f'{self.current_participants_col}_p99'],
                df[f'{self.current_participants_col}_p99'],
                df[self.current_participants_col]
            )
            df.drop(columns=[f'{self.current_participants_col}_p99'], inplace=True, errors='ignore')
            logger.info(f"Winsorized '{self.current_participants_col}' at 99th percentile per game.")

        if self.current_total_gross_col and self.current_total_gross_col in df.columns and self.current_game_col in df.columns:
            df[self.current_total_gross_col] = pd.to_numeric(df[self.current_total_gross_col], errors='coerce')
            p99_total_gross = df.groupby(self.current_game_col, observed=True)[self.current_total_gross_col].quantile(0.99)
            df = df.join(p99_total_gross.rename(f'{self.current_total_gross_col}_p99'), on=self.current_game_col)
            df[self.current_total_gross_col] = np.where(
                df[self.current_total_gross_col] > df[f'{self.current_total_gross_col}_p99'],
                df[f'{self.current_total_gross_col}_p99'],
                df[self.current_total_gross_col]
            )
            df.drop(columns=[f'{self.current_total_gross_col}_p99'], inplace=True, errors='ignore')
            logger.info(f"Winsorized '{self.current_total_gross_col}' at 99th percentile per game.")
        return df

    def _derive_channel_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives the 'ft_channel_is_online' feature based on ONLINE_GAME_INDICATORS."""
        if self.current_game_location_col and self.current_game_location_col in df.columns:
            # Ensure ONLINE_GAME_INDICATORS are lowercase for case-insensitive comparison
            # Assuming ONLINE_GAME_INDICATORS is imported from forecast_cli.utils.common
            online_indicators_lower = [str(ind).lower() for ind in ONLINE_GAME_INDICATORS]
            df['ft_channel_is_online'] = df[self.current_game_location_col].astype(str).str.lower()\
                .isin(online_indicators_lower).astype(int)
            # ft_channel_is_online = 1 if it's an online game, 0 if physical/other.
            logger.info(f"Derived 'ft_channel_is_online' from '{self.current_game_location_col}' using ONLINE_GAME_INDICATORS. Distribution: {df['ft_channel_is_online'].value_counts(normalize=True).to_dict()}")
        else:
            df['ft_channel_is_online'] = 0 # Default if game_location is missing (implies physical or unknown)
            logger.warning(f"Column '{self.current_game_location_col}' not found for deriving channel. Defaulting 'ft_channel_is_online' to 0.")
        return df

    def _build_agg_dict(self, df_columns: list[str], target_col: str | None, target_agg_method: str) -> tuple[dict, list[str]]:
        """Helper to build aggregation dictionary for _create_target_series.
        
        Args:
            df_columns: Columns of the DataFrame to be aggregated.
            target_col: The raw column name for the target KPI (e.g., self.current_participants_col). 
                        None if target is derived by size().
            target_agg_method: Aggregation method for the target KPI (e.g., 'size', 'sum', 'mean').

        Returns:
            A tuple containing:
                - agg_dict: The dictionary for groupby().agg().
                - feature_cols_to_keep: List of feature column names after aggregation.
        """
        agg_dict = {}
        feature_cols_to_keep = []

        if target_col: # For KPIs like minutes, revenue, prob (mean part)
            agg_dict[target_col] = target_agg_method
            # Target will be renamed to 'y' later
        elif target_agg_method == 'size': # For bookings
            # .size() is handled differently, no direct column in agg_dict for 'y' yet
            pass

        # Define aggregation for known features
        # Heuristic to determine if the current KPI series is daily or hourly for agg logic
        is_daily_series_aggregation = target_agg_method in ['size', 'sum']

        for col in df_columns:
            if col in ['timestamp', 'game_norm', target_col, self.raw_ts_col, self.raw_end_col, 
                       self.current_ts_col, self.current_end_col, self.current_created_col,
                       self.current_status_col, self.current_game_col, self.current_game_location_col]: # Skip grouping keys and raw unprocessed cols
                continue

            # Default aggregation methods (can be overridden per feature type)
            default_numeric_agg = 'mean'
            default_dummy_agg = 'sum' if is_daily_series_aggregation else 'mean' # 'sum' for daily bookings/rev/min, 'mean' for hourly prob flags

            if col == 'lead_time_days':
                agg_dict[col] = default_numeric_agg # Mean is fine for lead_time_days for all series types
                feature_cols_to_keep.append(col)
            elif (col.startswith('ft_lead_time_bucket_') or
                  col in ['ft_has_promo', 'ft_uses_coupon_code', 'ft_is_gift_redemption', 'ft_is_prepaid_pkg',
                           'ft_flat_rate', 'ft_addl_player_fee', 'ft_capacity_left_placeholder'] or # Added specific price/capacity features here
                  col.startswith('ft_channel_')): # Removed ft_price_ and ft_capacity_ startswith
                
                # For ft_price_ and ft_capacity_ which might be numeric rather than pure flags:
                if col in ['ft_flat_rate', 'ft_addl_player_fee', 'ft_capacity_left_placeholder']:
                     agg_dict[col] = default_numeric_agg # Use 'mean'
                else: # Actual flags like lead time buckets, promo flags, channel flags
                    agg_dict[col] = default_dummy_agg 
                feature_cols_to_keep.append(col)

            elif col.startswith('dt_'): # Calendar features
                if col == 'dt_hour_of_day':
                    if not is_daily_series_aggregation: # Only keep for hourly series
                        agg_dict[col] = 'mean' 
                        feature_cols_to_keep.append(col)
                    # else: Drop for daily series by not adding to agg_dict
                elif col.startswith('dt_is_') or col.startswith('dt_day_name_'): # Boolean flags from calendar features (dt_is_weekend, dt_day_name_Monday, etc.)
                    agg_dict[col] = 'max' # Use 'max' to preserve 0/1 flag for the period (day/hour)
                    feature_cols_to_keep.append(col)
                # Catches dt_week_of_month, dt_day_of_week, dt_month, dt_year etc.
                elif col in ['dt_day_of_week', 'dt_month', 'dt_year', 'dt_quarter', 'dt_day_of_year', 'dt_week_of_year', 'dt_week_of_month']:
                    agg_dict[col] = 'first' # 'first' is good for values constant within the group
                    feature_cols_to_keep.append(col)
                # else: if any other dt_ features exist, they might be missed or need specific handling
            
            elif col.startswith('ext_'): # External features
                if col.startswith('ext_is_'): # Boolean flags like ext_is_major_event
                    agg_dict[col] = 'max' 
                elif col in ["ext_precipitation_mm", "ext_snowfall_cm"]:
                    agg_dict[col] = 'sum' # Sum these specific weather metrics
                else: # Other numeric external features (e.g., temperature, wind speed, humidity, google trends)
                    agg_dict[col] = default_numeric_agg # 'mean'
                feature_cols_to_keep.append(col)
        
        if is_daily_series_aggregation:
            logger.debug(f"_build_agg_dict for DAILY series. features_to_keep: {feature_cols_to_keep}")
        else:
            logger.debug(f"_build_agg_dict for HOURLY series. features_to_keep: {feature_cols_to_keep}")

        return agg_dict, feature_cols_to_keep

    def _create_target_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates multiple target series based on kpi_configs.
        Iterates through self.kpi_configs, determines base KPI type and frequency,
        and aggregates event-level data (now including features) into time series.
        """
        if df.empty:
            logger.warning("Input DataFrame to _create_target_series is empty. Returning empty DataFrame.")
            return pd.DataFrame(columns=['timestamp', 'series_id', 'y'])

        required_cols = ['timestamp', 'game_norm']
        if not self.current_game_col or self.current_game_col not in df.columns:
            raise ValueError(f"Game column '{self.current_game_col}' not found, cannot create target series.")
        if 'timestamp' not in df.columns: # This should be the hourly floored timestamp from _filter_and_basic_transforms
            raise ValueError("'timestamp' column (hourly) not found, cannot create target series.")
        if 'game_norm' not in df.columns:
             raise ValueError("'game_norm' column not found, cannot create target series.")

        all_kpi_dfs = []

        for kpi_name, kpi_config in self.kpi_configs.items():
            logger.info(f"--- Processing KPI: {kpi_name} ---")
            
            # Determine base_kpi_type (e.g., 'bookings', 'participants', 'revenue', 'prob')
            # Assumes kpi_name starts with base_kpi_type_ (e.g., "bookings_daily", "participants_hourly")
            # More robust: add a 'base_kpi_type' field to kpi_config if names are inconsistent
            base_kpi_type = kpi_name.split('_')[0] 
            if base_kpi_type not in ["bookings", "participants", "revenue", "prob"]:
                logger.warning(f"Could not determine base KPI type from kpi_name '{kpi_name}'. Assuming it's '{kpi_name}' itself if no underscore. Skipping if ambiguous.")
                # Fallback or skip logic needed if base_kpi_type is not one of the known
                if base_kpi_type not in self.PROB_TARGET_COLUMNS and base_kpi_type not in ["bookings", "participants", "revenue"]:
                    # A simple check: if the full kpi_name is a known type (e.g. "prob" without _hourly)
                     if kpi_name in self.PROB_TARGET_COLUMNS: base_kpi_type = kpi_name
                     elif kpi_name in ["bookings", "participants", "revenue"]: base_kpi_type = kpi_name
                     else:
                        logger.error(f"Unrecognized base_kpi_type '{base_kpi_type}' derived from '{kpi_name}'. Skipping this KPI.")
                        continue

            target_freq = kpi_config.get('autogluon_freq', 'H') # Default to Hourly if not specified
            target_transform = kpi_config.get('target_transform') # e.g., "log1p"

            logger.info(f"KPI: {kpi_name}, Base Type: {base_kpi_type}, Freq: {target_freq}, Transform: {target_transform}")

            df_kpi_base = df.copy()
            
            # Ensure timestamp is floored to the target_freq for aggregation
            # The input 'timestamp' from _filter_and_basic_transforms is already hourly.
            # If target_freq is 'D', we floor it to Day. If 'H', it's already correct.
            if target_freq == 'D':
                df_kpi_base['timestamp'] = df_kpi_base['timestamp'].dt.floor('D')
            # For 'H', it's already floored to hour.

            group_cols = ['timestamp', 'game_norm']
            
            y_col_name_temp = None # Will hold the name of the column used for 'y' before renaming
            agg_method_for_y = None

            if base_kpi_type == "bookings":
                agg_method_for_y = 'size'
                # y_col_name_temp will be handled by .size() later
            elif base_kpi_type == "participants":
                if not self.current_participants_col or self.current_participants_col not in df_kpi_base.columns:
                    logger.warning(f"Participants column '{self.current_participants_col}' not found. Skipping KPI '{kpi_name}'.")
                    continue
                y_col_name_temp = self.current_participants_col
                agg_method_for_y = 'sum'
            elif base_kpi_type == "revenue":
                if not self.current_total_gross_col or self.current_total_gross_col not in df_kpi_base.columns:
                    logger.warning(f"Total gross column '{self.current_total_gross_col}' not found. Skipping KPI '{kpi_name}'.")
                    continue
                y_col_name_temp = self.current_total_gross_col
                agg_method_for_y = 'sum'
            elif base_kpi_type == "prob": # Probability still uses participants as base
                if not self.current_participants_col or self.current_participants_col not in df_kpi_base.columns:
                    logger.warning(f"Participants column '{self.current_participants_col}' not found for 'prob' type KPI '{kpi_name}'. Skipping.")
                    continue
                y_col_name_temp = self.current_participants_col # Will be used for mean, then normalized
                agg_method_for_y = 'mean' # For prob, we take mean participants, then normalize
            else:
                logger.error(f"Unknown base_kpi_type: {base_kpi_type} for kpi {kpi_name}. Skipping.")
                continue

            # Build aggregation dictionary for features
            # For 'size' based targets, target_col is None. For others, it's y_col_name_temp.
            # target_agg_method passed to _build_agg_dict should be what's applied to that specific column.
            feature_agg_target_col = y_col_name_temp if agg_method_for_y != 'size' else None
            agg_dict_features, features_to_keep = self._build_agg_dict(
                df_kpi_base.columns, 
                target_col=feature_agg_target_col, # Pass the original name of the y-column
                target_agg_method=agg_method_for_y if feature_agg_target_col else 'first' # If y is size, 'first' is just a placeholder for features
            )
            
            # Perform aggregation
            grouped_data = df_kpi_base.groupby(group_cols)
            
            if agg_method_for_y == 'size':
                # If other features are aggregated, agg_dict_features won't be empty.
                if agg_dict_features:
                    aggregated_df = grouped_data.agg(agg_dict_features)
                    size_series = grouped_data.size()
                    aggregated_df = aggregated_df.join(size_series.rename('y')).reset_index()
                else: # Only target is size
                    aggregated_df = grouped_data.size().reset_index(name='y')
            else: # For sum, mean based targets
                # Ensure the y_col_name_temp is part of the agg_dict_features with its specific agg_method_for_y
                # _build_agg_dict might already handle this if y_col_name_temp is not special feature
                # For safety, we make sure it is:
                if y_col_name_temp and y_col_name_temp in agg_dict_features:
                    agg_dict_features[y_col_name_temp] = agg_method_for_y # Ensure correct agg for target
                elif y_col_name_temp: # If it wasn't caught as a feature (e.g. not dt_ or ext_)
                    agg_dict_features[y_col_name_temp] = agg_method_for_y
                
                aggregated_df = grouped_data.agg(agg_dict_features).reset_index()
                if y_col_name_temp and y_col_name_temp in aggregated_df.columns:
                    aggregated_df.rename(columns={y_col_name_temp: 'y'}, inplace=True)
                elif y_col_name_temp: # If it was dropped or not aggregated correctly
                    logger.error(f"Target column {y_col_name_temp} not found in aggregated_df for {kpi_name}. Grouped cols: {aggregated_df.columns.tolist()}")
                    # Fallback: try to re-aggregate just the target if missing
                    temp_y_series = grouped_data[y_col_name_temp].agg(agg_method_for_y).rename('y')
                    aggregated_df = aggregated_df.merge(temp_y_series, on=group_cols, how='left')


            # Handle 'prob' KPI specific normalization for 'y'
            if base_kpi_type == "prob":
                if 'y' in aggregated_df.columns and not aggregated_df['y'].empty:
                    min_val = aggregated_df['y'].min()
                    max_val = aggregated_df['y'].max()
                    if max_val > min_val:
                        aggregated_df['y_prob_normalized'] = (aggregated_df['y'] - min_val) / (max_val - min_val)
                    elif not aggregated_df['y'].isnull().all():
                        aggregated_df['y_prob_normalized'] = 0.5 if max_val > 0 else 0.0
                    else:
                        aggregated_df['y_prob_normalized'] = np.nan
                    
                    # If logit transform is specified for prob, apply it to y_prob_normalized
                    if target_transform == "logit":
                        # Before logit, clip to avoid inf/-inf from 0 or 1
                        epsilon = 1e-6 # Small epsilon
                        aggregated_df['y_prob_clipped'] = aggregated_df['y_prob_normalized'].clip(lower=epsilon, upper=1-epsilon)
                        aggregated_df['y'] = logit_transform(aggregated_df['y_prob_clipped'])
                        logger.info(f"Applied logit transform for {kpi_name}. Original mean y (after normalization): {aggregated_df['y_prob_normalized'].mean()}, Transformed mean y: {aggregated_df['y'].mean()}")
                        aggregated_df.drop(columns=['y_prob_normalized', 'y_prob_clipped'], inplace=True, errors='ignore')
                    else: # If not logit, use the 0-1 normalized value as 'y'
                        aggregated_df['y'] = aggregated_df['y_prob_normalized']
                        aggregated_df.drop(columns=['y_prob_normalized'], inplace=True, errors='ignore')

                else: # 'y' (mean participants) was not created or empty
                    aggregated_df['y'] = np.nan # Ensure 'y' column exists
                logger.info(f"Processed 'prob' KPI '{kpi_name}'. Final 'y' mean: {aggregated_df['y'].mean()}")

            # Apply log1p transform if configured and not 'prob' (prob handled above)
            if target_transform == "log1p" and base_kpi_type != "prob":
                if 'y' in aggregated_df.columns:
                    aggregated_df['y'] = np.log1p(aggregated_df['y'])
                    logger.info(f"Applied log1p transform to 'y' for {kpi_name}")
                else:
                    logger.warning(f"'y' column not found for log1p transform for {kpi_name}")


            # Create series_id: kpi_name + game_norm (e.g., bookings_daily_mygame)
            aggregated_df['series_id'] = kpi_name + '_' + aggregated_df['game_norm']
            
            # Ensure 'y' column exists, fill with 0 if completely missing after all ops
            if 'y' not in aggregated_df.columns:
                aggregated_df['y'] = 0
                logger.warning(f"Column 'y' was missing for {kpi_name} after all processing. Defaulted to 0.")
            else:
                # Fill any remaining NaNs in y with 0 (e.g., if a group had no events for sum/size)
                aggregated_df['y'].fillna(0, inplace=True)


            # Resample to ensure full time index for the kpi's frequency
            # This ensures that even if there were no events for certain periods, those periods exist.
            # This step assumes 'timestamp' and 'series_id' are present.
            # The 'y' values and features for these new rows will be NaN initially.
            
            # For each series_id, create a full date range and reindex
            resampled_dfs_for_kpi = []
            for sid, group in aggregated_df.groupby('series_id'):
                group = group.set_index('timestamp')
                min_ts = group.index.min()
                max_ts = group.index.max()
                
                if pd.isna(min_ts) or pd.isna(max_ts):
                    logger.warning(f"Skipping resampling for series_id {sid} due to NaT timestamps.")
                    resampled_dfs_for_kpi.append(group.reset_index()) # Add as is
                    continue

                # Create the full date range based on the KPI's specific frequency
                full_range = pd.date_range(start=min_ts, end=max_ts, freq=target_freq)
                full_range.name = 'timestamp' # Ensure the index has a name before reset_index
                group_resampled = group.reindex(full_range)
                group_resampled['series_id'] = sid # Re-assign series_id
                
                # Fill 'y' with 0 for new timestamps from reindex
                group_resampled['y'].fillna(0, inplace=True)
                
                # Forward-fill then backward-fill other feature columns
                feature_cols_in_group = [col for col in features_to_keep if col in group_resampled.columns and col != 'y']
                for f_col in feature_cols_in_group:
                    group_resampled[f_col] = group_resampled[f_col].ffill().bfill()
                
                resampled_dfs_for_kpi.append(group_resampled.reset_index())

            if resampled_dfs_for_kpi:
                aggregated_df = pd.concat(resampled_dfs_for_kpi, ignore_index=True)
            else: # If aggregated_df was empty or all groups had NaT timestamps
                logger.warning(f"No data after attempting to resample for KPI {kpi_name}. aggregated_df might have been empty or had timestamp issues.")
                # aggregated_df remains as it was (potentially empty)

            # Define the columns to keep for this KPI's DataFrame
            # Start with essential columns and ensure they are present
            essential_cols = []
            if 'timestamp' in aggregated_df.columns: essential_cols.append('timestamp')
            else: logger.error(f"CRITICAL: 'timestamp' column MISSING from aggregated_df for KPI {kpi_name} before final selection. Columns: {aggregated_df.columns.tolist()}")
            
            if 'series_id' in aggregated_df.columns: essential_cols.append('series_id')
            else: logger.error(f"CRITICAL: 'series_id' column MISSING from aggregated_df for KPI {kpi_name} before final selection. Columns: {aggregated_df.columns.tolist()}")

            if 'y' in aggregated_df.columns: essential_cols.append('y')
            else: logger.error(f"CRITICAL: 'y' column MISSING from aggregated_df for KPI {kpi_name} before final selection. Columns: {aggregated_df.columns.tolist()}")

            # Add other feature columns that should be kept
            other_feature_cols_to_keep = [f for f in features_to_keep if f in aggregated_df.columns and f not in essential_cols]
            
            # Combine and deduplicate
            final_cols = essential_cols + other_feature_cols_to_keep
            final_cols = list(dict.fromkeys(final_cols)) 
            
            # Final check: ensure all selected columns actually exist in aggregated_df (should be redundant but safe)
            final_cols = [col for col in final_cols if col in aggregated_df.columns]
            
            if not aggregated_df.empty:
                all_kpi_dfs.append(aggregated_df[final_cols])
                logger.info(f"Finished KPI: {kpi_name}. Shape: {aggregated_df[final_cols].shape}, Columns: {final_cols}")
            elif not final_cols: # If aggregated_df is empty and final_cols is also empty
                logger.warning(f"KPI {kpi_name} resulted in an empty aggregated_df and no columns to select. Appending an empty DF placeholder if necessary or skipping.")
                # If we need a placeholder for structure, create one, otherwise this KPI contributes nothing to concat
            else: # aggregated_df is empty, but final_cols is not (e.g. ['timestamp', 'series_id', 'y'])
                logger.warning(f"KPI {kpi_name} resulted in an empty aggregated_df, but final_cols were {final_cols}. Appending an empty DF with these columns.")
                all_kpi_dfs.append(pd.DataFrame(columns=final_cols))

        if not all_kpi_dfs:
            logger.warning("No KPI series were generated in _create_target_series. Returning empty DataFrame.")
            return pd.DataFrame(columns=['timestamp', 'series_id', 'y'])

        final_df_all_kpis = pd.concat(all_kpi_dfs, ignore_index=True)
        
        # Clean up game_norm if it's a column and not explicitly kept (it's part of series_id now)
        if 'game_norm' in final_df_all_kpis.columns and 'game_norm' not in final_cols: # Check against last kpi's final_cols (approximate)
             final_df_all_kpis.drop(columns=['game_norm'], inplace=True, errors='ignore')
             
        logger.info(f"Concatenated all processed KPI series. Final shape: {final_df_all_kpis.shape}. Columns: {final_df_all_kpis.columns.tolist()}")
        return final_df_all_kpis

    def process(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Main processing pipeline to generate long format data and static features."""
        try:
            raw_df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            logger.error(f"CSV file not found at {self.csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading CSV {self.csv_path}: {e}", exc_info=True)
            raise
            
        if raw_df.empty:
            logger.warning(f"CSV file {self.csv_path} is empty.")
            # Return empty structures matching expected output
            # Determine expected columns for df_long (based on features that would be added)
            # This is a bit tricky without running through. For now, a minimal set.
            expected_long_cols = ["series_id", "timestamp", "y"] 
            # Plus all covariate names that would have been generated by features module
            # This list should be kept in sync with what features are added
            # For now, an example list (can be made more dynamic later)
            # This is simplified. A more robust way is to get these from the feature modules themselves.
            placeholder_covariate_cols = [
                'dt_hour_of_day', 'dt_day_of_week', 'dt_is_weekend', 'dt_month', 'dt_week_of_year',
                'dt_quarter', 'dt_day_of_year', 'dt_is_month_start', 'dt_is_month_end', 'dt_is_holiday',
                'ft_channel_is_online', 'ft_has_promo', 'ft_uses_coupon_code', 'ft_is_gift_redemption',
                'ft_is_prepaid_pkg', 'lead_time_days', 'ft_lead_time_bucket_0_1d', 
                'ft_lead_time_bucket_2_7d', 'ft_lead_time_bucket_8_30d', 'ft_lead_time_bucket_gt_30d',
                'ft_flat_rate', 'ft_addl_player_fee', 'ft_capacity_left_placeholder',
                'ext_temp_c', 'ext_precip_mm', 'ext_snow_cm', 'ext_google_trends', 'ext_is_major_event'
            ]
            df_long = pd.DataFrame(columns=expected_long_cols + placeholder_covariate_cols)
            static_features_df = create_static_features([], None) # Empty static features
            return df_long, static_features_df

        # Core processing steps from prepare_long_format_data
        df = self._load_and_initial_clean(raw_df.copy()) # Operate on a copy
        df = self._filter_and_basic_transforms(df)
        df = self._winsorize_features(df)

        # --- Apply history cutoff (Lever A from user's OOM guide) ---
        if hasattr(self, 'history_cutoff_str') and self.history_cutoff_str and "timestamp" in df.columns:
            try:
                cutoff_date = pd.to_datetime(self.history_cutoff_str)
                initial_rows = len(df)
                df = df[df["timestamp"] >= cutoff_date]
                logger.info(f"Applied history cutoff: {self.history_cutoff_str}. Rows reduced from {initial_rows} to {len(df)}.")
                if df.empty:
                    logger.warning(f"DataFrame became empty after applying history cutoff {self.history_cutoff_str}.")
            except Exception as e:
                logger.error(f"Error applying history cutoff '{self.history_cutoff_str}': {e}. Proceeding without cutoff.")
        elif hasattr(self, 'history_cutoff_str') and self.history_cutoff_str:
            logger.warning(f"History cutoff '{self.history_cutoff_str}' provided, but 'timestamp' column not found before cutoff application point.")

        # === Feature Engineering on Event-Level Data ===
        # Ensure current_ts_col and current_created_col are valid before using them
        event_timestamp_col_for_lead_time = self.current_ts_col
        if not event_timestamp_col_for_lead_time or event_timestamp_col_for_lead_time not in df.columns:
            logger.warning(f"Event timestamp column '{event_timestamp_col_for_lead_time}' for lead time calculation not found in df. Defaulting to 'timestamp'.")
            event_timestamp_col_for_lead_time = "timestamp" # Fallback, though less ideal for precise lead time

        creation_timestamp_col_for_lead_time = self.current_created_col
        if not creation_timestamp_col_for_lead_time or creation_timestamp_col_for_lead_time not in df.columns:
            logger.warning(f"Creation timestamp column '{creation_timestamp_col_for_lead_time}' for lead time calculation not found in df. Lead time features might be skipped or incorrect.")
            # add_lead_time_features has its own internal check and warning for created_col

        df = add_lead_time_features(
            df,
            timestamp_col=event_timestamp_col_for_lead_time, 
            created_col=creation_timestamp_col_for_lead_time,
            known_covariates_list=self.known_dynamic_covariates_cols
        )
        
        df = add_promotional_features(
            df,
            promo_col=self.current_promo_col,
            num_coupons_col=self.current_num_coupons_col,
            coupons_text_col=self.current_coupons_text_col,
            gift_voucher_col=self.current_gift_voucher_col,
            prepaid_pkg_col=self.current_prepaid_pkg_col,
            known_covariates_list=self.known_dynamic_covariates_cols
        )
        df = add_price_capacity_features(
            df,
            flat_rate_col=self.current_flat_rate_col,
            addl_player_fee_col=self.current_addl_player_fee_col,
            known_covariates_list=self.known_dynamic_covariates_cols
        )
        # Add real weather features first
        df = add_weather_features(df, timestamp_col="timestamp", known_covariates_list=self.known_dynamic_covariates_cols)
        
        # Then add other external placeholders (e.g., Google Trends, Major Events)
        # Placeholder weather features from this function might be redundant if real ones are added,
        # but constant feature dropping should handle them.
        df = add_external_data_placeholders(df, timestamp_col="timestamp", known_covariates_list=self.known_dynamic_covariates_cols)
        
        # Channel feature might use game_location, which is event-level
        df = self._derive_channel_feature(df)
        # Calendar features are based on the 'timestamp' column (which is hourly for events before aggregation)
        df = add_calendar_features(df, timestamp_col="timestamp", known_covariates_list=self.known_dynamic_covariates_cols)
        # === End of Feature Engineering on Event-Level Data ===

        # Create target series by aggregating event-level data (which now includes features)
        df_aggregated_kpis = self._create_target_series(df.copy()) # Pass df with all event-level features

        # The df_aggregated_kpis should now be the final long dataframe, 
        # as _create_target_series handles resampling and full index creation per KPI.
        df_long_final = df_aggregated_kpis

        # Determine all unique covariate columns actually present in the final df_long_final
        # These are needed for AutoGluon's known_covariates_names
        # Exclude 'series_id', 'timestamp', 'y'
        if not df_long_final.empty:
            all_present_columns = df_long_final.columns.tolist()
            reserved_cols_for_ts = ['series_id', 'timestamp', 'y']
            self.final_known_dynamic_covariates_cols = [c for c in all_present_columns if c not in reserved_cols_for_ts]
            logger.info(f"Final known dynamic covariates determined from df_long_final: {self.final_known_dynamic_covariates_cols}")
        else:
            self.final_known_dynamic_covariates_cols = []
            logger.warning("df_long_final is empty, no final dynamic covariates determined.")

        # Final sort for consistency
        if not df_long_final.empty:
            df_long_final = df_long_final.sort_values(by=["series_id", "timestamp"]).reset_index(drop=True)
            logger.info(f"Sorted df_long_final. Shape: {df_long_final.shape}")

        # Drop Constant Engineered Dynamic Covariates
        # Use self.final_known_dynamic_covariates_cols which was derived from df_long_final
        cols_to_drop_dynamically = []
        if not df_long_final.empty:
            for col_name in self.final_known_dynamic_covariates_cols:
                if col_name in df_long_final.columns: # Check if column still exists
                    if df_long_final[col_name].nunique(dropna=False) == 1:
                        cols_to_drop_dynamically.append(col_name)
                        logger.warning(f"Engineered dynamic covariate '{col_name}' is constant. Value: {df_long_final[col_name].iloc[0] if not df_long_final.empty else 'N/A'}. It will be dropped.")
        
        if cols_to_drop_dynamically:
            df_long_final.drop(columns=cols_to_drop_dynamically, inplace=True)
            # Update self.final_known_dynamic_covariates_cols to reflect the dropped columns
            self.final_known_dynamic_covariates_cols = [col for col in self.final_known_dynamic_covariates_cols if col not in cols_to_drop_dynamically]
            logger.info(f"Dropped constant dynamic covariates: {cols_to_drop_dynamically}. Updated final_known_dynamic_covariates_cols.")
        
        logger.info(f"Final df_long_final shape after constant drop: {df_long_final.shape}")

        # Generate static features dataframe
        event_level_df_for_static_features = df # df *after* _derive_channel_feature
        unique_series_ids_for_static = df_long_final["series_id"].unique() if "series_id" in df_long_final.columns and not df_long_final.empty else []
        ids_list_for_static = list(unique_series_ids_for_static) if hasattr(unique_series_ids_for_static, 'tolist') else list(unique_series_ids_for_static)
        
        # Build channel_map for static features
        # static_features expects map value: 0 if online, 1 if physical/IRL
        # ft_channel_is_online: 1 if online, 0 if physical. So, map_value = 1 - ft_channel_is_online
        game_to_channel_value_map = {} # game_norm -> (0 for online, 1 for physical)
        if 'game_norm' in event_level_df_for_static_features.columns and \
           self.ft_channel_is_online_col in event_level_df_for_static_features.columns and \
           not event_level_df_for_static_features[['game_norm', self.ft_channel_is_online_col]].empty:
            try:
                # Create a map from game_norm to the ft_channel_is_online value (0 for physical, 1 for online)
                # The static_feature function might expect the inverse (e.g. is_irl_game), adjust accordingly if needed.
                # For now, this map directly uses ft_channel_is_online value.
                # Static features function create_static_features will handle this map.
                game_norm_to_raw_channel_val = event_level_df_for_static_features.groupby('game_norm')[self.ft_channel_is_online_col]\
                    .first().to_dict() # .first() assumes channel is consistent per game_norm
                logger.info(f"Built game_norm_to_raw_channel_val for static features: {game_norm_to_raw_channel_val}")

                # The create_static_features function might have its own interpretation of channel_map.
                # We pass game_norm_to_raw_channel_val (which maps game_norm to its ft_channel_is_online value).
                # The create_static_features will then iterate through unique_series_ids, parse game_norm, and use this map.
                # This is simpler than creating series_to_channel_map here if create_static_features can handle game_norm map.
                # Rechecking create_static_features: it expects channel_map: series_id -> channel_value.
                # So, we do need to build the series_to_channel_map here.

                series_to_channel_map_for_static = {}
                if game_norm_to_raw_channel_val and ids_list_for_static:
                    for sid in ids_list_for_static:
                        # Extract game_norm part from series_id (e.g., 'bookings_daily_mygame' -> 'mygame')
                        # This logic needs to be robust to series_ids that might not have '_'
                        # The series_id format is kpi_name + "_" + game_norm
                        # Example: bookings_daily_mygame, prob_mygame, revenue_some_other_game
                        game_norm_part = sid.split('_')[-1] # Take the last part as game_norm
                        
                        if game_norm_part in game_norm_to_raw_channel_val:
                            # create_static_features expects: 0 if online, 1 if physical/IRL.
                            # self.ft_channel_is_online_col is 1 if online, 0 if physical.
                            # So, value_for_static = 1 - game_norm_to_raw_channel_val[game_norm_part]
                            series_to_channel_map_for_static[sid] = 1 - game_norm_to_raw_channel_val[game_norm_part]
                        else:
                            logger.debug(f"Game norm '{game_norm_part}' from SID '{sid}' not in game_norm_to_raw_channel_val. Channel for static feature might default.")
                logger.info(f"Built series_to_channel_map_for_static. Example: {dict(list(series_to_channel_map_for_static.items())[:3])}")

            except Exception as e:
                logger.error(f"Error building game_to_channel_value_map or series_to_channel_map_for_static: {e}", exc_info=True)
                series_to_channel_map_for_static = {} # Ensure it's an empty dict on error
        else:
            series_to_channel_map_for_static = {}
            logger.warning(f"Could not build channel map for static features. Relevant columns missing or df empty. game_norm col: {'game_norm' in event_level_df_for_static_features.columns}, ft_channel_is_online col: {self.ft_channel_is_online_col in event_level_df_for_static_features.columns}")


        static_features_df = create_static_features(
            unique_series_ids=ids_list_for_static, # Pass the list of unique series_ids
            channel_map=series_to_channel_map_for_static
        )
        
        return df_long_final, static_features_df 