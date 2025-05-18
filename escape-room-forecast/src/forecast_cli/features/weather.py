import pandas as pd
import logging
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime # Import datetime

logger = logging.getLogger(__name__)

def add_weather_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    known_covariates_list: list | None = None
) -> pd.DataFrame:
    """Adds historical weather features from Open-Meteo for Kingston, Ontario.

    Args:
        df: Input DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column in df.
        known_covariates_list: Optional list to append newly created feature names to.

    Returns:
        DataFrame with added weather features:
            - ext_temperature_celsius (mean hourly temperature)
            - ext_precipitation_mm (sum of hourly precipitation)
            - ext_snowfall_cm (sum of hourly snowfall)
            - ext_wind_speed_kmh (mean hourly wind speed)
            - ext_relative_humidity_percent (mean hourly relative humidity)
    """
    logger.debug(f"Starting add_weather_features. Input df shape: {df.shape}")
    if df.empty or timestamp_col not in df.columns:
        logger.warning(f"DataFrame is empty or timestamp column '{timestamp_col}' not found. Skipping weather features.")
        # Add placeholder columns if df is not empty but timestamp_col is missing, to avoid breaking downstream
        if not df.empty:
            placeholder_features = [
                "ext_temperature_celsius", "ext_precipitation_mm", "ext_snowfall_cm",
                "ext_wind_speed_kmh", "ext_relative_humidity_percent"
            ]
            for feat in placeholder_features:
                df[feat] = 0 
            if known_covariates_list is not None:
                known_covariates_list.extend(placeholder_features)
        return df

    # Ensure timestamp_col is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    min_date_dt = df[timestamp_col].min()
    max_date_dt = df[timestamp_col].max()

    # Cap max_date_dt at today's date to stay within API limits for historical data
    today_dt = pd.to_datetime(datetime.utcnow().date()) # Use UTC date for comparison
    if max_date_dt > today_dt:
        logger.info(f"Original max_date {max_date_dt.strftime('%Y-%m-%d')} is in the future. Capping at today {today_dt.strftime('%Y-%m-%d')} for weather API request.")
        max_date_dt = today_dt

    if pd.isna(min_date_dt) or pd.isna(max_date_dt):
        logger.warning("Min or Max date is NaT after pd.to_datetime. Skipping weather features.")
        # Add placeholder columns if df is not empty
        if not df.empty:
            placeholder_features = [
                "ext_temperature_celsius", "ext_precipitation_mm", "ext_snowfall_cm",
                "ext_wind_speed_kmh", "ext_relative_humidity_percent"
            ]
            for feat in placeholder_features:
                df[feat] = 0 
            if known_covariates_list is not None:
                known_covariates_list.extend(placeholder_features)
        return df

    min_date_str = min_date_dt.strftime('%Y-%m-%d')
    max_date_str = max_date_dt.strftime('%Y-%m-%d')
        
    logger.info(f"Fetching weather data for Kingston, ON from {min_date_str} to {max_date_str}")

    df_timezone = df[timestamp_col].dt.tz 
    logger.debug(f"Original df timezone for {timestamp_col}: {df_timezone}")

    # Setup the Open-Meteo API client with cache and retry
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1) # Cache indefinitely
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Kingston, Ontario coordinates (approximate)
    latitude = 44.2312
    longitude = -76.4860

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": min_date_str,
        "end_date": max_date_str,
        "hourly": ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "relative_humidity_2m"],
        "timezone": "America/Toronto" # Important for correct alignment
    }
    logger.debug(f"Open-Meteo API request params: {params}")
    
    new_features = [
        "ext_temperature_celsius", "ext_precipitation_mm", "ext_snowfall_cm",
        "ext_wind_speed_kmh", "ext_relative_humidity_percent"
    ]

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0] # Assuming one location
        logger.debug(f"Open-Meteo API response received. Coordinates: {response.Latitude()}°N {response.Longitude()}°E, Elevation: {response.Elevation()}m, TZ: {response.Timezone()} {response.TimezoneAbbreviation()}")
        logger.debug(f"Hourly time range: {pd.to_datetime(response.Hourly().Time(), unit='s', utc=True)} to {pd.to_datetime(response.Hourly().TimeEnd(), unit='s', utc=True)}")


        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
        hourly_snowfall = hourly.Variables(2).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(4).ValuesAsNumpy()

        # Construct the correct date range for the hourly data
        start_time = pd.to_datetime(hourly.Time(), unit="s", utc=True)
        end_time = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
        # The API returns a range inclusive of the start time, up to, but not including the end_time for each variable's period.
        # freq='h' (or '1H') generates timestamps at the start of each hour.
        # The number of values in ValuesAsNumpy() should correspond to the number of hourly intervals.
        num_values = len(hourly_temperature_2m) # Assuming all variables have the same length
        correct_timestamps = pd.date_range(start=start_time, periods=num_values, freq='h', tz='UTC')

        hourly_data = {
            "timestamp": correct_timestamps
        }
        hourly_data["ext_temperature_celsius"] = hourly_temperature_2m
        hourly_data["ext_precipitation_mm"] = hourly_precipitation
        hourly_data["ext_snowfall_cm"] = hourly_snowfall # Typically in cm from Open-Meteo for 'snowfall' param
        hourly_data["ext_wind_speed_kmh"] = hourly_wind_speed_10m
        hourly_data["ext_relative_humidity_percent"] = hourly_relative_humidity_2m
        
        weather_df = pd.DataFrame(data = hourly_data)
        
        # Convert weather_df timestamp (which is UTC from API) to the original df_timezone, then make naive
        # This ensures timestamps align for merging, assuming original df[timestamp_col] was made naive UTC earlier.
        if df_timezone is not None: # If original was tz-aware
             weather_df[timestamp_col] = weather_df[timestamp_col].dt.tz_convert(df_timezone).dt.tz_localize(None)
        else: # If original was already naive (hopefully UTC)
             weather_df[timestamp_col] = weather_df[timestamp_col].dt.tz_localize(None)


        logger.info(f"Weather_df created. Shape: {weather_df.shape}")
        logger.debug(f"Weather_df head before fillna:\\n{weather_df.head()}")
        logger.debug(f"Weather_df describe before fillna:\\n{weather_df.describe(include='all')}")
        logger.debug(f"Weather_df NaNs before fillna:\\n{weather_df.isna().sum()}")

        # Fill NaNs with 0 - consider if this is appropriate or if forward fill/interpolation is better
        weather_df = weather_df.fillna(0)
        logger.debug(f"Weather_df head after fillna:\\n{weather_df.head()}")


        # Ensure the main df timestamp is also floored to hour for merge
        # df_original_timestamps = df[timestamp_col].copy() # Preserve original for potential later use if needed
        df[timestamp_col] = df[timestamp_col].dt.floor('h')
        logger.debug(f"Original df timestamp column '{timestamp_col}' floored to hour for merge.")

        logger.debug(f"Shape of df before merge: {df.shape}")
        logger.debug(f"df['{timestamp_col}'].dtype before merge: {df[timestamp_col].dtype}")
        logger.debug(f"df['{timestamp_col}'].head() before merge:\n{df[[timestamp_col]].head()}")
        logger.debug(f"df['{timestamp_col}'].tail() before merge:\n{df[[timestamp_col]].tail()}")
        
        logger.debug(f"weather_df['{timestamp_col}'].dtype before merge: {weather_df[timestamp_col].dtype}")
        logger.debug(f"weather_df['{timestamp_col}'].head() before merge:\n{weather_df[[timestamp_col]].head()}")
        logger.debug(f"weather_df['{timestamp_col}'].tail() before merge:\n{weather_df[[timestamp_col]].tail()}")

        # Merge weather data
        # Using left merge to keep all original df rows
        df = pd.merge(df, weather_df, on=timestamp_col, how='left')
        logger.info(f"Shape of df after merge with weather_df: {df.shape}")
        if not df.empty:
            logger.debug(f"Merged df head (with weather features if successful):\n{df[new_features + [timestamp_col]].head() if all(f in df.columns for f in new_features) else df.head()}")
        
        # Restore original timestamps if they were not already floored
        # This might not be necessary if downstream processing expects hourly floored.
        # For now, assume hourly floored is acceptable.
        # df[timestamp_col] = df_original_timestamps 

        # Fill any NaNs that might result from merge (e.g., if weather data has slight gaps)
        # using forward fill and then backward fill
        na_counts_before_fill = {}
        for feat in new_features:
            if feat in df.columns:
                na_counts_before_fill[feat] = df[feat].isnull().sum()
                df[feat] = df[feat].ffill().bfill()
                logger.debug(f"Weather feature '{feat}' ffilled and bfilled. NaNs before: {na_counts_before_fill[feat]}, NaNs after: {df[feat].isnull().sum()}")
            else: # Should not happen if merge was successful
                logger.warning(f"Weather feature '{feat}' not found after merge. Defaulting to 0.")
                df[feat] = 0
        
        logger.info(f"Successfully added weather features: {new_features}")

    except Exception as e:
        logger.error(f"Error fetching or processing weather data: {e}. Defaulting weather features to 0.", exc_info=True)
        for feat in new_features:
            df[feat] = 0 # Ensure columns exist even if API call fails
            
    if known_covariates_list is not None:
        # Ensure no duplicates if this function were ever called multiple times with the same list
        for nf in new_features:
            if nf not in known_covariates_list:
                 known_covariates_list.append(nf)
        
    logger.debug(f"Finished add_weather_features. Output df shape: {df.shape}")
    return df 