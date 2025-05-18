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
    VALID_STATUSES
)
from forecast_cli.features import (
    add_calendar_features,
    add_lead_time_features,
    add_promotional_features,
    add_price_capacity_features,
    add_external_data_placeholders,
    create_static_features
)

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing for time series forecasting."""

    def __init__(self, 
                 csv_path: str | Path, 
                 kpi_configs: dict,
                 ts_col: str = "start", 
                 end_col: str = "end",
                 raw_created_col: str = "created",
                 raw_promo_col: str = "promotion",
                 raw_coupon_col: str = "coupons",
                 raw_flat_rate_col: str = "flat_rate",
                 raw_addl_player_fee_col: str = "additional_player_fee"
                 ):
        """Initialize DataPreprocessor.
        # Test comment to force refresh

        Args:
            csv_path: Path to the CSV data.
            kpi_configs: Dictionary of KPI configurations.
            ts_col: Name of the main timestamp column.
            end_col: Name of the end timestamp column.
            raw_created_col: Original name of the 'created at' column.
            raw_promo_col: Original name of the 'promotion title' or similar column.
            raw_coupon_col: Original name of the 'coupon code used' or similar column.
            raw_flat_rate_col: Original name of the 'flat rate booking' indicator column.
            raw_addl_player_fee_col: Original name of the 'additional player fee' column.
        """
        self.csv_path = Path(csv_path)
        self.kpi_configs = kpi_configs
        self.raw_ts_col = ts_col
        self.raw_end_col = end_col
        self.raw_created_col = raw_created_col
        self.raw_promo_col = raw_promo_col
        self.raw_coupon_col = raw_coupon_col
        self.raw_flat_rate_col = raw_flat_rate_col
        self.raw_addl_player_fee_col = raw_addl_player_fee_col
        
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
        self.known_dynamic_covariates_cols = []
        self.known_static_covariates_cols = [] # For static features
        self.ft_channel_is_online_col = 'ft_channel_is_online' # Default name for this key feature


    def _load_and_initial_clean(self) -> pd.DataFrame:
        """Load data from CSV and perform initial cleaning, snaking, and unique col identification."""
        logger.info(f"Starting initial load and clean for CSV: {self.csv_path}")
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            logger.error(f"CSV file not found at {self.csv_path}")
            raise
        except Exception as e: # pylint: disable=broad-except
            logger.error(f"Error reading CSV {self.csv_path}: {e}", exc_info=True)
            raise
            
        if df.empty:
            logger.warning(f"CSV file {self.csv_path} is empty. Cannot proceed with initial clean.")
            # Return an empty DataFrame with expected minimal columns if needed by subsequent steps
            # However, process() method already handles empty raw_df. So we might not need to return anything specific here.
            # For now, returning the empty df.
            return df

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
            "participants": _snake("participants"),
            "total_gross": _snake("total_gross"),
            "status": _snake("status"),
            "game": _snake("game"), # Often 'Game Name' or similar
            self.raw_created_col: _snake(self.raw_created_col),
            "game_location": _snake("game_location"),
            # Promotional raw names (assuming common CSV names before snaking)
            self.raw_promo_col: _snake(self.raw_promo_col), 
            "number_of_coupons": _snake("number_of_coupons"),
            self.raw_coupon_col: _snake(self.raw_coupon_col),
            "specific_gift_voucher": _snake("specific_gift_voucher"),
            "prepaid_package": _snake("prepaid_package"),
            # Price/Capacity raw names
            self.raw_flat_rate_col: _snake(self.raw_flat_rate_col),
            self.raw_addl_player_fee_col: _snake(self.raw_addl_player_fee_col),
        }

        for raw_name_key_semantic, assumed_snaked_name_or_getter in raw_semantic_to_snaked_map.items():
            # Resolve getter functions to actual snaked names
            if callable(assumed_snaked_name_or_getter):
                assumed_snaked_name = assumed_snaked_name_or_getter()
            else:
                assumed_snaked_name = assumed_snaked_name_or_getter
            
            # Handle None case for optional raw columns if _get_snaked_name returns None
            if assumed_snaked_name is None:
                current_name_map[raw_name_key_semantic] = None
                logger.info(f"Optional raw column for semantic key '{raw_name_key_semantic}' was not provided or its original name is None. Skipping.")
                continue

            if assumed_snaked_name in df.columns:
                current_name_map[raw_name_key_semantic] = assumed_snaked_name
            else: # Check for suffixed versions like "start_1"
                potential_matches = [c for c in df.columns if c.startswith(assumed_snaked_name + "_")]
                if potential_matches:
                    current_name_map[raw_name_key_semantic] = potential_matches[0]
                    logger.debug(f"Column '{assumed_snaked_name}' (from raw semantic key '{raw_name_key_semantic}') not found directly, using suffixed version '{potential_matches[0]}'.")
                else:
                    # Only raise error for absolutely critical columns for basic processing
                    if raw_name_key_semantic in [self.raw_ts_col, self.raw_end_col, "participants", "status", "game"]:
                         raise ValueError(f"Critical base column for '{raw_name_key_semantic}' (expected ~'{assumed_snaked_name}') not found after renaming. Columns: {df.columns.tolist()}")
                    logger.info(f"Optional column for '{raw_name_key_semantic}' (expected ~'{assumed_snaked_name}') not found. Will proceed without it if possible.")
                    current_name_map[raw_name_key_semantic] = None
        
        self.current_ts_col = current_name_map.get(self.raw_ts_col)
        self.current_end_col = current_name_map.get(self.raw_end_col)
        self.current_participants_col = current_name_map.get("participants")
        self.current_total_gross_col = current_name_map.get("total_gross")
        self.current_status_col = current_name_map.get("status")
        self.current_game_col = current_name_map.get("game")
        self.current_created_col = current_name_map.get(self.raw_created_col)
        self.current_game_location_col = current_name_map.get("game_location")
        
        self.current_promo_col = current_name_map.get(self.raw_promo_col)
        self.current_num_coupons_col = current_name_map.get("number_of_coupons")
        self.current_coupons_text_col = current_name_map.get(self.raw_coupon_col)
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
        """Derives the 'ft_channel_is_online' feature."""
        if self.current_game_location_col and self.current_game_location_col in df.columns:
            logger.info(f"Unique values in '{self.current_game_location_col}' for channel mapping: {df[self.current_game_location_col].unique().tolist()}")
            df['ft_channel_is_online'] = df[self.current_game_location_col].astype(str).str.lower()\
                .map({'yes': 0, 'no': 1}).fillna(0).astype(int) # 'yes' means in-person (online=0), 'no' means online (online=1)
            logger.info(f"Generated 'ft_channel_is_online' from '{self.current_game_location_col}'. Distribution: {df['ft_channel_is_online'].value_counts(normalize=True).to_dict()}")
        else:
            df['ft_channel_is_online'] = 0 # Default if game_location is missing
            logger.warning(f"Column '{self.current_game_location_col}' not found for deriving channel. Defaulting 'ft_channel_is_online' to 0.")
        return df

    def _resample_and_fill_series(self, df_long_input: pd.DataFrame, final_covariate_cols: list[str]) -> pd.DataFrame:
        """
        Resamples each series to its target frequency (from kpi_configs), fills missing target values ('y')
        using a forward fill, and forward/backward fills covariate columns.
        """
        if df_long_input.empty:
            logger.warning("Input DataFrame for resampling is empty. Returning empty DataFrame.")
            return pd.DataFrame(columns=df_long_input.columns.tolist() + ['y']) # Ensure 'y' is there

        logger.info(f"Step 7: Starting resampling and filling. Input shape: {df_long_input.shape}. Covariates to fill: {final_covariate_cols}")
        
        all_resampled_dfs = []
        
        # Ensure 'timestamp' is datetime
        df_long_input['timestamp'] = pd.to_datetime(df_long_input['timestamp'])

        for series_id_val, group in df_long_input.groupby('series_id'):
            kpi_name_for_series = series_id_val.split('_')[0]
            series_config = self.kpi_configs.get(kpi_name_for_series)

            if not series_config:
                logger.warning(f"No KPI config found for series_id '{series_id_val}' (derived KPI: {kpi_name_for_series}). Skipping resampling for this series.")
                # Optionally, append the group as is, or handle as an error
                # For now, let's append as is, assuming it might be handled downstream or is an anomaly.
                # However, this usually means it won't align with other series if they ARE resampled.
                # A better approach might be to drop it or use a default frequency.
                # For safety, let's append it to not lose data, but log heavily.
                logger.error(f"Series '{series_id_val}' will not be resampled due to missing config. This may cause issues.")
                all_resampled_dfs.append(group.reset_index()) # Reset index before appending
                continue

            frequency = series_config.get('frequency', 'D') # Default to 'D' if not specified
            target_col = 'y' # Assuming 'y' is always the target

            # Set timestamp as index for resampling
            group = group.set_index('timestamp').sort_index()
            
            # Create a full date range for the series
            # Min/max for the specific group to avoid excessive range if data is sparse
            if group.index.min() is pd.NaT or group.index.max() is pd.NaT:
                 logger.warning(f"Skipping resampling for series {series_id_val} due to NaT in index min/max.")
                 all_resampled_dfs.append(group.reset_index()) # Reset index before appending
                 continue

            full_range = pd.date_range(start=group.index.min(), end=group.index.max(), freq=frequency)
            
            # Resample target variable ('y') using sum (or mean/first for different KPIs if needed)
            # The aggregation method for 'y' during resampling depends on the KPI nature.
            # For counts like 'bookings', sum is appropriate. For averages or rates, 'mean' or 'first'.
            # This should ideally be configurable per KPI. For now, using 'sum' as a general default
            # as many of our KPIs are counts/sums over a period.
            y_resampled = group[target_col].resample(frequency).sum() # Example: sum. Could be .mean(), .first() etc.
            
            # For covariates, we typically forward-fill after resampling them.
            # Some covariates might be 'first', 'last', or 'mean' if they change within the new period.
            # For now, let's take 'first' for simplicity, then ffill/bfill.
            covariates_resampled_dict = {}
            for col in final_covariate_cols:
                if col in group.columns and col != target_col: # Ensure col exists and is not the target itself
                    covariates_resampled_dict[col] = group[col].resample(frequency).first() # Or .mean() for numeric that should be averaged
            
            # Combine resampled target and covariates
            resampled_group = pd.DataFrame(index=full_range)
            resampled_group[target_col] = y_resampled
            
            for col_name, resampled_series in covariates_resampled_dict.items():
                resampled_group[col_name] = resampled_series

            # Fill missing 'y' values:
            # 1. If config specifies a fill_value for 'y' (e.g., 0 for counts)
            # 2. Forward fill for remaining NaNs in 'y'
            # 3. Optional: backward fill for leading NaNs in 'y' (model dependent)
            if 'fill_y_na' in series_config:
                resampled_group[target_col] = resampled_group[target_col].fillna(series_config['fill_y_na'])
            
            # Heuristic: if 'y' is log-transformed, 0 is not a good fill.
            # Original values were >= 0. log1p(0) = 0. log1p(positive) > 0.
            # If we filled with 0 *after* log1p, it implies original was -1, which is not good.
            # So, if target_transform was log1p, fill with log1p(0) which is 0.
            if series_config.get("target_transform") == "log1p":
                 resampled_group[target_col] = resampled_group[target_col].fillna(0) # Fill with log1p(0)
            else: # For non-transformed, 0 is often a safe bet for counts/sums
                 resampled_group[target_col] = resampled_group[target_col].fillna(0)


            # Fill covariates: forward fill then backward fill
            for col in final_covariate_cols:
                if col in resampled_group.columns and col != target_col:
                    resampled_group[col] = resampled_group[col].ffill().bfill()
            
            resampled_group['series_id'] = series_id_val # Add back series_id
            all_resampled_dfs.append(resampled_group.reset_index()) # Reset timestamp from index

        if not all_resampled_dfs:
            logger.warning("No data after resampling all series. Returning empty DataFrame.")
            # Construct empty DataFrame with expected columns
            expected_cols = ['timestamp', 'series_id', 'y'] + final_covariate_cols
            return pd.DataFrame(columns=list(set(expected_cols))) # Use set to avoid duplicates if y is in final_covariate_cols

        full_resampled_df = pd.concat(all_resampled_dfs, ignore_index=True)
        
        # Ensure all final_covariate_cols are present, fill with 0 or appropriate default if any are missing post-concat
        for col in final_covariate_cols:
            if col not in full_resampled_df.columns:
                logger.warning(f"Covariate column '{col}' missing after concat of resampled data. Adding as zeros.")
                full_resampled_df[col] = 0 # Or pd.NA, or ffill/bfill based on strategy

        logger.info(f"Step 7 finished. Resampling and filling complete. Final shape: {full_resampled_df.shape}")
        return full_resampled_df

    def _apply_transformations(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Applies transformations like logit for probability series.
        This runs AFTER resampling and filling.
        """
        if df_long.empty:
            return df_long
        logger.info("Step 8: Applying post-resampling transformations (e.g., logit for probability).")

        # Logit transform for probability series (applied again after resampling if necessary)
        # The target_transform="logit" in kpi_configs could also gate this.
        # For now, explicit check for "prob_" prefix.
        prob_series_mask = df_long["series_id"].str.startswith("prob_")
        if prob_series_mask.any():
            # Check if 'y' for these series was already log1p transformed
            # If so, inverse transform before logit, or ensure logit is appropriate.
            # Current _create_target_series_and_base_features applies log1p based on config.
            # Logit usually applies to values in [0,1]. If 'y' is already log1p, this is an issue.
            # For now, assume 'y' for prob_ series is in [0,1] before this step.
            # The _create_target_series_and_base_features handles normalization for prob.
            
            logger.info(f"Applying logit transform to 'y' for {prob_series_mask.sum()} probability series data points.")
            
            # Ensure y is clipped to (epsilon, 1-epsilon) for stability
            epsilon = 1e-6
            df_long.loc[prob_series_mask, 'y'] = np.clip(df_long.loc[prob_series_mask, 'y'], epsilon, 1 - epsilon)
            df_long.loc[prob_series_mask, 'y'] = logit_transform(df_long.loc[prob_series_mask, 'y'])
        else:
            logger.info("No probability series found to apply logit transform post-resampling.")
        
        logger.info("Step 8 finished. Post-resampling transformations applied.")
        return df_long

    def _build_static_features(self, df_processed_long: pd.DataFrame) -> pd.DataFrame:
        """Builds static features DataFrame from the processed long format data."""
        logger.info("Step 10: Building static features.")
        if df_processed_long.empty:
            logger.warning("Cannot build static features from empty long DataFrame.")
            # Return empty static features df with expected columns (series_id, and others)
            return create_static_features([], {}, known_static_cols=self.known_static_covariates_cols)


        unique_sids = df_processed_long['series_id'].unique().tolist()
        
        # Create channel_map for 'sf_is_in_real_life'
        # This uses the 'ft_channel_is_online' which should be stable per series_id after processing.
        channel_map = {}
        if self.ft_channel_is_online_col in df_processed_long.columns:
            try:
                # Use .first() as channel should be static per series_id
                channel_map = df_processed_long.groupby('series_id')[self.ft_channel_is_online_col].first().to_dict()
                logger.info(f"Created channel_map for static features. Example: {dict(list(channel_map.items())[:2])}")
            except Exception as e: # pylint: disable=broad-except
                logger.warning(f"Could not reliably create channel_map for static features from '{self.ft_channel_is_online_col}': {e}", exc_info=True)
        else:
            logger.warning(f"'{self.ft_channel_is_online_col}' not found in df_processed_long. Static feature 'sf_is_in_real_life' might be incorrect or default.")

        # Call the centralized static feature creation function
        # Pass self.known_static_covariates_cols so it knows what to expect/create
        static_features_df = create_static_features(
            unique_sids, 
            channel_map,
            raw_df_for_sf_extraction=self._get_raw_df_for_static_features(), # Pass raw data if needed for other static features
            game_col=self.current_game_col, # Pass current game column name
            known_static_cols=self.known_static_covariates_cols # This list is populated by create_static_features
        )
        
        # self.known_static_covariates_cols will be updated by create_static_features if it adds new ones.
        logger.info(f"Step 10 finished. Static features shape: {static_features_df.shape}. Known static features: {self.known_static_covariates_cols}")
        return static_features_df

    def _get_raw_df_for_static_features(self) -> pd.DataFrame | None:
        """
        Helper to load and provide a minimal version of the raw DataFrame
        if needed by create_static_features for deriving some static attributes
        that are not easily available in the already processed long format data.
        """
        try:
            # Only load essential columns if possible, e.g., game name, and any other raw source for static features
            # For now, reloading and letting create_static_features pick what it needs.
            # This could be optimized to load only a few columns.
            df_raw = pd.read_csv(self.csv_path, usecols=lambda c: c in [self.current_game_col, _snake(self.current_game_col)] + OTHER_RAW_COLS_FOR_SF) # Define these
            df_raw.columns = [_snake(c) for c in df_raw.columns] # snake case
            # Minimal cleaning just for static feature extraction
            # e.g., game_norm if static features are per game_norm
            if self.current_game_col and self.current_game_col in df_raw.columns:
                 df_raw['game_norm'] = df_raw[self.current_game_col].astype(str).str.lower().pipe(
                    lambda s: s.str.replace(r"[^\w\s]", "", regex=True).str.replace(" ", "_"))
            return df_raw
        except Exception as e: # pylint: disable=broad-except
            logger.warning(f"Could not load raw data for static feature extraction: {e}")
            return None

    def _drop_constant_engineered_features(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies and drops engineered dynamic covariate columns that are constant across all series and timestamps.
        Updates self.known_dynamic_covariates_cols.
        """
        if df_long.empty:
            return df_long
            
        logger.info("Step 9: Checking for and dropping constant engineered dynamic covariates.")
        cols_to_drop = []
        
        # We only check columns that are in self.known_dynamic_covariates_cols
        # And are also present in the df_long.
        potential_cols_to_check = [col for col in self.known_dynamic_covariates_cols if col in df_long.columns]

        for col in potential_cols_to_check:
            if col in ['timestamp', 'series_id', 'y', 'game_norm']: # Skip essential/target columns
                continue
            try:
                # A column is constant if it has only one unique value after dropping NaNs
                # If all are NaNs, nunique() is 0. If one value + NaNs, it's 1.
                # If only one unique non-NaN value, it's constant.
                num_unique = df_long[col].nunique(dropna=True)
                if num_unique <= 1:
                    # If num_unique is 0 (all NaN), or 1 (one value, possibly with NaNs), consider it constant.
                    # However, we should only drop if it's truly constant (not all NaN).
                    # If nunique is 0, it means all values are NaN. Such a column might be problematic anyway.
                    # If nunique is 1, it means there's one distinct value (NaNs are ignored by default).
                    
                    # Let's refine: drop if nunique is 1 (constant value)
                    # or if nunique is 0 (all NaNs - effectively a useless feature).
                    # It's generally safer to drop columns that are all NaN or have a single value.
                    
                    # If nunique is 0, it's all NaNs.
                    # If nunique is 1, it's a single value (possibly with NaNs, but nunique ignores NaNs).
                    # We want to drop if there's only one actual value present.
                    
                    # Stricter check: if number of unique non-NaN values is 1, AND not all values are NaN.
                    if num_unique == 1 and not df_long[col].isnull().all():
                        constant_val = df_long[col].dropna().unique()[0]
                        logger.warning(f"Engineered dynamic covariate '{col}' is constant after processing. Value: {constant_val}. It will be dropped.")
                        cols_to_drop.append(col)
                    elif num_unique == 0: # All values are NaN
                        logger.warning(f"Engineered dynamic covariate '{col}' consists of all NaN values. It will be dropped.")
                        cols_to_drop.append(col)
                        
            except Exception as e: # pylint: disable=broad-except
                logger.error(f"Error when checking uniqueness of column '{col}': {e}", exc_info=True)
        
        if cols_to_drop:
            df_long = df_long.drop(columns=cols_to_drop)
            self.known_dynamic_covariates_cols = [c for c in self.known_dynamic_covariates_cols if c not in cols_to_drop]
            logger.info(f"Dropped constant dynamic covariates: {cols_to_drop}")
            logger.info(f"Final df_long_final shape after constant drop: {df_long.shape}. Columns: {df_long.columns.tolist()}")
            logger.info(f"Remaining known dynamic covariates in self.known_dynamic_covariates_cols: {self.known_dynamic_covariates_cols}")
        else:
            logger.info("No constant dynamic covariates found to drop.")
        
        logger.info("Step 9 finished.")
        return df_long

    # Helper to get snaked name for raw columns passed in __init__
    def _get_snaked_name(self, raw_col_name: str | None) -> str | None:
        if raw_col_name and isinstance(raw_col_name, str):
            return _snake(raw_col_name)
        return None

    # Properties for easier access to current column names, assuming _load_and_initial_clean has been run.
    # These could replace direct access to self.current_..._col if desired, offering a small abstraction.
    # Example:
    # @property
    # def timestamp_col(self) -> str:
    #     if not self.current_ts_col:
    #         raise AttributeError("Timestamp column (current_ts_col) not identified. Run _load_and_initial_clean first.")
    #     return self.current_ts_col

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
        df = self._load_and_initial_clean()
        df = self._filter_and_basic_transforms(df)
        df = self._winsorize_features(df)

        # This is where the critical step of forming 'series_id' and 'y' for each KPI occurs.
        # The original `prepare_long_format_data` seems to construct a single `df`
        # which becomes the `full_long_df`. This implies that `df` at this stage
        # needs to be processed to have `series_id` and `y` correctly for all KPIs.
        # This logic is currently a placeholder in `_create_target_series`.
        # For now, we'll assume `_create_target_series` populates them adequately or
        # that the `train.py` `train` function will later filter this `df` by KPI.
        # The most direct port of `prepare_long_format_data` would build one large df
        # with all series for all KPIs.
        
        # If the input CSV is one record per booking, then `_create_target_series` needs to
        # aggregate by `timestamp` and `game_norm` for different metrics (count for bookings, sum(participants) for minutes etc)
        # and then assign `series_id` (e.g. `bookings_gameA`, `minutes_gameA`).
        # THIS IS A MAJOR REFACTORING TASK for `_create_target_series`.
        # For now, this call is here structurally.
        df = self._create_target_series(df) # Placeholder, needs careful implementation

        # Add dynamic features using the new feature modules
        df = self._derive_channel_feature(df) # Derives ft_channel_is_online, must be before static features if it uses it
        df = add_calendar_features(df, timestamp_col="timestamp") # Assumes 'timestamp' is the main model timestamp
        
        # Ensure created_col used by add_lead_time_features is correctly identified
        df = add_lead_time_features(df, timestamp_col="timestamp", created_col=self.current_created_col if self.current_created_col else "_no_created_col_")
        
        df = add_promotional_features(
            df, 
            promo_col=self.current_promo_col,
            num_coupons_col=self.current_num_coupons_col,
            coupons_text_col=self.current_coupons_text_col,
            gift_voucher_col=self.current_gift_voucher_col,
            prepaid_pkg_col=self.current_prepaid_pkg_col
        )
        df = add_price_capacity_features(
            df,
            flat_rate_col=self.current_flat_rate_col,
            addl_player_fee_col=self.current_addl_player_fee_col
        )
        df = add_external_data_placeholders(df)

        # Define all dynamic covariate column names that were created
        # This list needs to be robustly generated based on features added.
        # For now, using the list from the original script as a reference.
        # This should align with columns created by the feature functions.
        all_potential_covariates = [
            'dt_hour_of_day', 'dt_day_of_week', 'dt_is_weekend', 'dt_month', 'dt_week_of_year', 
            'dt_quarter', 'dt_day_of_year', 'dt_is_month_start', 'dt_is_month_end', 
            'dt_is_holiday', 'ft_channel_is_online', 'ft_has_promo',
            'ft_uses_coupon_code', 'ft_is_gift_redemption', 'ft_is_prepaid_pkg',
            'lead_time_days', 
            'ft_lead_time_bucket_0_1d', 'ft_lead_time_bucket_2_7d', 
            'ft_lead_time_bucket_8_30d', 'ft_lead_time_bucket_gt_30d',
            'ft_flat_rate', 'ft_addl_player_fee', 'ft_capacity_left_placeholder',
            'ext_temp_c', 'ext_precip_mm', 'ext_snow_cm', 'ext_google_trends', 'ext_is_major_event'
        ]
        # Filter to only those present in df
        current_dynamic_covariates = [col for col in all_potential_covariates if col in df.columns]
        missing_covs = set(all_potential_covariates) - set(current_dynamic_covariates)
        if missing_covs:
            logger.warning(f"Some expected covariate columns are missing from the DataFrame after feature engineering: {missing_covs}")


        # Resample and fill (this is the long_df)
        # Ensure 'timestamp' is the hourly floored one before this step.
        # Ensure 'series_id' and 'y' are correctly populated.
        df_long = self._resample_and_fill_series(df.copy(), current_dynamic_covariates) # Operate on a copy for resampling

        # Logit transform for probability series
        if not df_long.empty:
            prob_series_mask = df_long["series_id"].str.startswith("prob_")
            if prob_series_mask.any():
                logger.info(f"Applying logit transform to 'y' for {prob_series_mask.sum()} probability series data points.")
                df_long.loc[prob_series_mask, 'y'] = logit_transform(df_long.loc[prob_series_mask, 'y'])
            else:
                logger.info("No probability series found to apply logit transform.")
        
        # Final deduplication and sort
        if not df_long.empty:
            df_long = (df_long
                .drop_duplicates(subset=["timestamp", "series_id"], keep="last")
                .sort_values(["series_id", "timestamp"])
                .reset_index(drop=True))
            logger.info("Final deduplication and sort applied to long format data.")
        
        # Create static features
        static_features_df = pd.DataFrame() # Default to empty
        if not df_long.empty:
            unique_sids = df_long['series_id'].unique().tolist()
            # Pass the channel map for sf_is_in_real_life
            # This assumes ft_channel_is_online is consistent per series_id *before* resampling
            # So we should get it from `df` (before resampling) or ensure it's stable in `df_long`
            # For simplicity, let's try to get it from df_long if it's stable there.
            channel_map = {}
            if 'ft_channel_is_online' in df_long.columns:
                 try: # If series_id does not exist as index, this will fail.
                     channel_map = df_long.groupby('series_id')['ft_channel_is_online'].first().to_dict()
                 except Exception: # Fallback to original df if that was more suitable
                     if 'series_id' in df.columns and 'ft_channel_is_online' in df.columns:
                         channel_map = df.groupby('series_id')['ft_channel_is_online'].first().to_dict()
                     else: # Default
                         logger.warning("Could not reliably create channel_map for static features.")


            static_features_df = create_static_features(unique_sids, channel_map)
        else: # df_long is empty
            static_features_df = create_static_features([], None)


        # Select final columns for df_long (target, timestamp, series_id, and *actual* covariates)
        final_long_columns = ["series_id", "timestamp", "y"] + current_dynamic_covariates
        # Ensure all selected columns actually exist in df_long
        final_long_columns = [col for col in final_long_columns if col in df_long.columns]
        
        if df_long.empty: # Should match the empty df created at the start if raw_df was empty
             logger.info("Preprocessor returning empty long_df and static_features_df.")
             return df_long[final_long_columns if final_long_columns else []], static_features_df # Ensure columns exist

        return df_long[final_long_columns], static_features_df 