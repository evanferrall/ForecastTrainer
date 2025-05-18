import pandas as pd
import logging
from workalendar.america import Ontario # Trying direct import of Ontario from america
# from ..utils.logger import setup_logger # Removed problematic import
import numpy as np

# logger = setup_logger(__name__) # Reverted to standard logger
logger = logging.getLogger(__name__) # Standard Python logger

def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp", known_covariates_list: list | None = None) -> pd.DataFrame:
    """Adds calendar-based features to the DataFrame.

    Args:
        df: Input DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column.
        known_covariates_list: Optional list to append newly created feature names to.

    Returns:
        DataFrame with added calendar features.
    """
    if df.empty or timestamp_col not in df.columns:
        logger.warning(f"DataFrame is empty or timestamp column '{timestamp_col}' not found. Skipping calendar features.")
        return df

    # Ensure timestamp_col is datetime
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except Exception as e:
        logger.error(f"Error converting '{timestamp_col}' to datetime: {e}. Skipping calendar features.")
        return df

    new_features = []

    # Basic date components
    df['dt_hour_of_day'] = df[timestamp_col].dt.hour
    new_features.append('dt_hour_of_day')
    df['dt_day_of_week'] = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
    new_features.append('dt_day_of_week')
    df['dt_is_weekend'] = df['dt_day_of_week'].isin([5, 6]).astype(int)
    new_features.append('dt_is_weekend')
    df['dt_month'] = df[timestamp_col].dt.month
    new_features.append('dt_month')
    df['dt_week_of_year'] = df[timestamp_col].dt.isocalendar().week.astype(int)
    new_features.append('dt_week_of_year')
    df['dt_quarter'] = df[timestamp_col].dt.quarter
    new_features.append('dt_quarter')
    df['dt_day_of_year'] = df[timestamp_col].dt.dayofyear
    new_features.append('dt_day_of_year')
    df['dt_is_month_start'] = df[timestamp_col].dt.is_month_start.astype(int)
    new_features.append('dt_is_month_start')
    df['dt_is_month_end'] = df[timestamp_col].dt.is_month_end.astype(int)
    new_features.append('dt_is_month_end')
    logger.info("Added basic calendar features (hour, day, week, month, quarter, etc.).")

    # Day name (one-hot encoded)
    try:
        all_day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name_series = pd.Categorical(df[timestamp_col].dt.day_name(), categories=all_day_names, ordered=False)
        day_names_dummies = pd.get_dummies(day_name_series, prefix='dt_day_name').astype(int)
        df = pd.concat([df, day_names_dummies], axis=1)
        day_names_cols = day_names_dummies.columns.tolist()
        new_features.extend(day_names_cols)
        logger.info(f"Added one-hot encoded day name features: {day_names_cols}")
    except Exception as e:
        logger.error(f"Error adding day_name features: {e}")

    # Week of month
    try:
        df['dt_week_of_month'] = (df[timestamp_col].dt.day - 1) // 7 + 1
        new_features.append('dt_week_of_month')
        logger.info("Added 'dt_week_of_month'.")
    except Exception as e:
        logger.error(f"Error adding week_of_month feature: {e}")

    # Holiday feature
    try:
        cal = Ontario() # Using Ontario 
        logger.info("Initialized workalendar for Canada (Ontario).")
    except Exception as e:
        logger.error(f"Could not initialize workalendar: {e}. Some calendar features might be missing.")
        # If workalendar fails, we can't add holiday-based features, so update list and return
        if known_covariates_list is not None:
            known_covariates_list.extend(new_features)
        return df 
    
    min_year = df[timestamp_col].dt.year.min()
    max_year = df[timestamp_col].dt.year.max()
    holidays_list = []
    if pd.notna(min_year) and pd.notna(max_year) and min_year <= max_year:
        try:
            for year in range(int(min_year), int(max_year) + 1):
                holidays_list.extend([holiday[0] for holiday in cal.holidays(year)])
        except Exception as e:
            logger.error(f"Error generating holiday list for years {min_year}-{max_year}: {e}")
    
    if holidays_list:
        df['dt_is_holiday'] = df[timestamp_col].dt.normalize().isin(holidays_list).astype(int)
        new_features.append('dt_is_holiday')
        logger.info(f"Added 'dt_is_holiday' based on Canadian federal holidays. Found {df['dt_is_holiday'].sum()} holiday occurrences.")
    else:
        df['dt_is_holiday'] = 0 # Default if no holidays could be generated
        new_features.append('dt_is_holiday') # Still add the column, even if all 0s
        logger.warning("Could not generate holiday list or no holidays found in data range. 'dt_is_holiday' set to 0.")

    # Major Holiday Eve feature
    major_holiday_names = [
        "New year",         # Changed from "New Year's Day" to match workalendar output
        "Good Friday", 
        "Canada Day", 
        "Labour Day", 
        "Thanksgiving",     # Changed from "Thanksgiving Day" to match workalendar output
        "Christmas Day"
    ]
    major_holiday_names_lower = [name.lower() for name in major_holiday_names] # Lowercase list for comparison
    major_holiday_eves = set()
    logger.debug(f"Calendar feature: min_year={min_year}, max_year={max_year} for eve calculation. Using major names (lowercase for matching): {major_holiday_names_lower}") # DEBUG
    if pd.notna(min_year) and pd.notna(max_year) and min_year <= max_year:
        try:
            for year_to_fetch_hols in range(int(min_year), int(max_year) + 2):
                logger.debug(f"Fetching holidays for year: {year_to_fetch_hols}") # DEBUG
                year_holidays = cal.holidays(year_to_fetch_hols)
                logger.debug(f"All holidays from workalendar for {year_to_fetch_hols}: {year_holidays}") # PRINT ALL HOLIDAYS
                for hol_date, hol_name in year_holidays:
                    if hol_name.lower() in major_holiday_names_lower: # Case-insensitive match
                        eve_date = pd.Timestamp(hol_date) - pd.Timedelta(days=1) # Ensure pd.Timestamp for eve_date
                        logger.debug(f"Found major holiday: {hol_name} (matched as {hol_name.lower()}) on {hol_date}. Eve: {eve_date}") # DEBUG
                        if int(min_year) <= eve_date.year <= int(max_year):
                            major_holiday_eves.add(eve_date)
                            logger.debug(f"Added eve: {eve_date} to major_holiday_eves set.") # DEBUG
                        else:
                            logger.debug(f"Eve {eve_date} is outside data range [{min_year}, {max_year}]. Not adding.") # DEBUG
        except Exception as e:
            logger.error(f"Error generating major holiday eves list: {e}")
    
    logger.debug(f"Final major_holiday_eves set: {major_holiday_eves}") # DEBUG

    if major_holiday_eves:
        # Ensure comparison is between Timestamps normalized to midnight
        normalized_df_dates = df[timestamp_col].dt.normalize()
        # Convert set of Timestamps to list for isin
        eves_list_for_isin = list(major_holiday_eves)
        df['dt_is_major_holiday_eve'] = normalized_df_dates.isin(eves_list_for_isin).astype(int)
        new_features.append('dt_is_major_holiday_eve')
        logger.info(f"Added 'dt_is_major_holiday_eve'. Found {df['dt_is_major_holiday_eve'].sum()} occurrences.")
    else:
        df['dt_is_major_holiday_eve'] = 0
        new_features.append('dt_is_major_holiday_eve') # Still add the column
        logger.warning("Could not identify any major holiday eves. 'dt_is_major_holiday_eve' set to 0.")

    # Long Weekend Day feature
    long_weekend_dates = set()
    if holidays_list: 
        holiday_timestamps = {pd.Timestamp(h) for h in holidays_list}
        unique_normalized_dates = df[timestamp_col].dt.normalize().unique()
        for d_norm in unique_normalized_dates:
            day_of_week = d_norm.dayofweek
            if day_of_week == 0 and d_norm in holiday_timestamps: # Mon Holiday
                long_weekend_dates.update([d_norm, d_norm - pd.Timedelta(days=1), d_norm - pd.Timedelta(days=2)])
            elif day_of_week == 4 and d_norm in holiday_timestamps: # Fri Holiday
                long_weekend_dates.update([d_norm, d_norm + pd.Timedelta(days=1), d_norm + pd.Timedelta(days=2)])
            elif day_of_week == 5: # Saturday
                if (d_norm + pd.Timedelta(days=2)) in holiday_timestamps: # Mon after is holiday
                    long_weekend_dates.update([d_norm, d_norm + pd.Timedelta(days=1), d_norm + pd.Timedelta(days=2)])
                if (d_norm - pd.Timedelta(days=1)) in holiday_timestamps: # Fri before was holiday
                    long_weekend_dates.add(d_norm)
                    long_weekend_dates.add(d_norm - pd.Timedelta(days=1))
                    if (d_norm - pd.Timedelta(days=1)).dayofweek == 4: long_weekend_dates.add(d_norm + pd.Timedelta(days=1))
            elif day_of_week == 6: # Sunday
                if (d_norm + pd.Timedelta(days=1)) in holiday_timestamps: # Mon after is holiday
                    long_weekend_dates.add(d_norm)
                    long_weekend_dates.add(d_norm + pd.Timedelta(days=1))
                    if (d_norm + pd.Timedelta(days=1)).dayofweek == 0: long_weekend_dates.add(d_norm - pd.Timedelta(days=1))
                if (d_norm - pd.Timedelta(days=2)) in holiday_timestamps: # Fri before was holiday
                    long_weekend_dates.update([d_norm, d_norm - pd.Timedelta(days=1), d_norm - pd.Timedelta(days=2)])

    if long_weekend_dates:
        df['dt_is_long_weekend_day'] = df[timestamp_col].dt.normalize().isin(list(long_weekend_dates)).astype(int)
        new_features.append('dt_is_long_weekend_day')
        logger.info(f"Added 'dt_is_long_weekend_day'. Found {df['dt_is_long_weekend_day'].sum()} occurrences.")
    else:
        df['dt_is_long_weekend_day'] = 0
        new_features.append('dt_is_long_weekend_day') # Still add the column
        logger.warning("Could not identify any long weekend days. 'dt_is_long_weekend_day' set to 0.")

    # Days until next holiday
    df_dates_unique_for_days_until = pd.Series(df[timestamp_col].dt.normalize().unique()).sort_values()
    if holidays_list:
        sorted_holiday_timestamps = sorted([pd.Timestamp(h) for h in holidays_list])
        days_to_next_holiday_map = {}
        for current_date_norm in df_dates_unique_for_days_until:
            next_holidays = [h for h in sorted_holiday_timestamps if h >= current_date_norm]
            days_to_next_holiday_map[current_date_norm] = (next_holidays[0] - current_date_norm).days if next_holidays else 365
        df['dt_days_until_next_holiday'] = df[timestamp_col].dt.normalize().map(days_to_next_holiday_map).fillna(365)
        new_features.append('dt_days_until_next_holiday')
        logger.info(f"Added 'dt_days_until_next_holiday'. Min: {df['dt_days_until_next_holiday'].min()}, Max: {df['dt_days_until_next_holiday'].max()}.")
    else:
        df['dt_days_until_next_holiday'] = 365
        new_features.append('dt_days_until_next_holiday')
        logger.warning("'dt_days_until_next_holiday' set to default 365 as holidays_list was empty.")

    # Days until next long weekend start
    if long_weekend_dates:
        sorted_long_weekend_starts = sorted(list(long_weekend_dates))
        days_to_next_lw_map = {}
        for current_date_norm in df_dates_unique_for_days_until:
            next_lws = [lw for lw in sorted_long_weekend_starts if lw >= current_date_norm]
            days_to_next_lw_map[current_date_norm] = (next_lws[0] - current_date_norm).days if next_lws else 365
        df['dt_days_until_next_long_weekend_day'] = df[timestamp_col].dt.normalize().map(days_to_next_lw_map).fillna(365)
        new_features.append('dt_days_until_next_long_weekend_day')
        logger.info(f"Added 'dt_days_until_next_long_weekend_day'. Min: {df['dt_days_until_next_long_weekend_day'].min()}, Max: {df['dt_days_until_next_long_weekend_day'].max()}.")
    else:
        df['dt_days_until_next_long_weekend_day'] = 365
        new_features.append('dt_days_until_next_long_weekend_day')
        logger.warning("'dt_days_until_next_long_weekend_day' set to default 365 as long_weekend_dates was empty.")

    if known_covariates_list is not None:
        known_covariates_list.extend(new_features)

    return df 