import pandas as pd
import logging
from workalendar.america import Canada # For Canadian holidays

logger = logging.getLogger(__name__)

def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Adds calendar-based features to the DataFrame.

    Args:
        df: Input DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame with added calendar features:
            - dt_hour_of_day
            - dt_day_of_week
            - dt_is_weekend
            - dt_month
            - dt_week_of_year
            - dt_quarter
            - dt_day_of_year
            - dt_is_month_start
            - dt_is_month_end
            - dt_is_holiday (Canadian federal holidays)
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

    # Basic date components
    df['dt_hour_of_day'] = df[timestamp_col].dt.hour
    df['dt_day_of_week'] = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
    df['dt_is_weekend'] = df['dt_day_of_week'].isin([5, 6]).astype(int)
    df['dt_month'] = df[timestamp_col].dt.month
    df['dt_week_of_year'] = df[timestamp_col].dt.isocalendar().week.astype(int)
    df['dt_quarter'] = df[timestamp_col].dt.quarter
    df['dt_day_of_year'] = df[timestamp_col].dt.dayofyear
    df['dt_is_month_start'] = df[timestamp_col].dt.is_month_start.astype(int)
    df['dt_is_month_end'] = df[timestamp_col].dt.is_month_end.astype(int)
    logger.info("Added basic calendar features (hour, day, week, month, quarter, etc.).")

    # Holiday feature
    cal = Canada()  # Default is federal holidays.
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
        logger.info(f"Added 'dt_is_holiday' based on Canadian federal holidays. Found {df['dt_is_holiday'].sum()} holiday occurrences.")
    else:
        df['dt_is_holiday'] = 0 # Default if no holidays could be generated
        logger.warning("Could not generate holiday list or no holidays found in data range. 'dt_is_holiday' set to 0.")

    # TODO (Lever 2 from original script): Add days_to_next_holiday, days_from_prev_holiday if desired

    return df 