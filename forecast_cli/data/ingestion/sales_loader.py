import hashlib
import polars as pl
import os
from pathlib import Path

# Determine Project Root dynamically
# __file__ is the path to the current script: .../DashboardModel/forecast_cli/data/ingestion/sales_loader.py
# Path(__file__).resolve() -> absolute path to script
# .parents[3] should go up three levels: ingestion -> data -> forecast_cli -> DashboardModel (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Define input file paths relative to PROJECT_ROOT
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
BOOKINGS_CSV_PATH = RAW_DATA_DIR / "bookings.csv"

# Define output directory for parquet files relative to PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "forecast_cli" / "data" / "cache"
PROCESSED_BOOKINGS_PARQUET_PATH = OUTPUT_DIR / "bookings_processed.parquet"

def hash_email(email: str) -> str | None:
    """Hashes an email using SHA-256 and returns an 8-byte prefix (16 hex chars)."""
    if email is None or (isinstance(email, str) and not email.strip()):
        return None
    try:
        sha256_hash = hashlib.sha256(str(email).encode('utf-8')).hexdigest()
        return sha256_hash[:16]
    except Exception:
        return None 

def process_booking_data(file_path: Path) -> pl.DataFrame:
    """
    Processes the bookings CSV file:
    - Reads CSV data.
    - Hashes email addresses.
    - Parses timestamps.
    - Converts relevant money columns to CAD cents (Int64).
    - Selects and renames columns for the forecasting model.
    """
    try:
        df = pl.read_csv(file_path, infer_schema_length=10000, null_values=["", "NA", "NULL"], truncate_ragged_lines=True)
    except Exception as e:
        print(f"Error reading booking data from {file_path}: {e}")
        return pl.DataFrame() # Return empty DataFrame on error

    # Hash email
    if 'Email address' in df.columns:
        df = df.with_columns(
            pl.col('Email address').map_elements(hash_email, return_dtype=pl.Utf8).alias('customer_id_hash')
        )
        # Decide whether to drop 'Email address', 'First name', 'Last name', 'Phone' for privacy
        # df = df.drop(['Email address', 'First name', 'Last name', 'Phone'])
    else:
        print("Warning: 'Email address' column not found for hashing.")

    # Timestamp parsing
    # Format: D/M/YYYY H:MM AM/PM. Example: 27/1/2025 2:00 PM
    # These should be localized to a consistent timezone later if needed (e.g., UTC)
    # For now, they are parsed as naive datetimes.
    timestamp_cols_formats = {
        "Start": "%d/%m/%Y %I:%M %p",
        "End": "%d/%m/%Y %I:%M %p",
        "Created": "%d/%m/%Y %I:%M %p",
        "Last changed": "%d/%m/%Y %I:%M %p"
    }
    for col, fmt in timestamp_cols_formats.items():
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).str.strptime(pl.Datetime, format=fmt, strict=False, exact=True).alias(col)
            )
        else:
            print(f"Warning: Timestamp column '{col}' not found.")

    # Money columns to convert to cents (Int64)
    # Assuming these are in dollars and need to be multiplied by 100
    money_cols = [
        'Flat Rate (1-7 Players)', 'Additional players', 'Adjustments',
        'Total net', 'HST', 'Total gross', 'Total paid', 'Total due'
    ]
    for col in money_cols:
        if col in df.columns:
            df = df.with_columns(
                (pl.col(col).cast(pl.Float64, strict=False) * 100).round(0).cast(pl.Int64).alias(col)
            )
        else:
            print(f"Warning: Money column '{col}' not found.")

    # Rename columns for clarity and consistency with model expectations
    # (booking_revenue_cents, players, booking_count will be main targets/features)
    rename_map = {
        "Booking number": "booking_id",
        "Start": "event_start_time",
        "End": "event_end_time",
        "Participants": "players",
        "Total net": "booking_revenue_cents_raw", # Raw revenue in cents from Total net
        "Game": "game_name",
        "Game Location": "branch_id", # Assuming Game Location is the branch identifier
        "Status": "booking_status",
        "Created": "booking_created_at"
        # Add other renames as necessary
    }
    # Filter map for columns that actually exist in df to avoid errors
    actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(actual_rename_map)

    # Add booking_count column (1 for each row, assuming each row is a booking)
    df = df.with_columns(pl.lit(1).cast(pl.Int32).alias("booking_count"))
    
    # Filter out canceled bookings if necessary (e.g., for demand forecasting)
    # Example: df = df.filter(pl.col("booking_status") != "canceled")
    # For now, keeping all statuses.

    # Select columns - choose what's needed for downstream processing
    # This list should be refined based on feature engineering needs.
    final_columns = [
        "booking_id", "event_start_time", "event_end_time", "players",
        "booking_revenue_cents_raw", "customer_id_hash", "game_name", "branch_id",
        "booking_status", "booking_created_at", "booking_count"
    ]
    # Keep only columns that exist to prevent errors
    existing_final_columns = [col for col in final_columns if col in df.columns]
    if not existing_final_columns:
        print("Error: No relevant columns found after processing. Returning empty DataFrame.")
        return pl.DataFrame()
        
    return df.select(existing_final_columns)

def main():
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data Dir: {RAW_DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Starting data ingestion from {BOOKINGS_CSV_PATH}...")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {OUTPUT_DIR}")

    bookings_df = process_booking_data(BOOKINGS_CSV_PATH)
    
    if not bookings_df.is_empty():
        try:
            bookings_df.write_parquet(PROCESSED_BOOKINGS_PARQUET_PATH)
            print(f"Processed booking data saved to {PROCESSED_BOOKINGS_PARQUET_PATH}")
            print("Sample of processed data:")
            print(bookings_df.head())
        except Exception as e:
            print(f"Error writing bookings parquet to {PROCESSED_BOOKINGS_PARQUET_PATH}: {e}")
    else:
        print("No data processed or an error occurred.")
            
    print("Data ingestion finished.")

if __name__ == "__main__":
    # Create dummy raw_data directory and bookings.csv if they don't exist for testing
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created dummy raw_data directory: {RAW_DATA_DIR}")

    if not BOOKINGS_CSV_PATH.exists():
        print(f"Dummy {BOOKINGS_CSV_PATH} not found, creating one for testing.")
        dummy_data_header = "Booking number,Start,End,First name,Last name,Email address,Phone,Participants,Adults,Flat Rate (1-7 Players),Additional players,Players,Participants (details),Game,Seven Dwarfs: Mining Mission,TriWizard Trials,Neverland Heist on the High Seas,Jingles,Café,Undersea Overthrow,Cure for the Common Zombie,Product code,Private event,Status,Promotion,Number of coupons,Coupons,Specific gift voucher,Prepaid credits,Prepaid package,Adjustments,Total adjustments,Total net,HST,HST included,Total gross,Total paid,Total due,Participants (names),Created,Created by,Last changed,Last changed by,Canceled,Canceled by,Reschedule until,Source,IP address,External ref.,Alert,Game Location,Eastern Daylight Time,Horror Experience,In-Real-Life,Location,Online,Beta Test,Time of game,Game Requirements,Online Experience,Eastern Standard Time"
        dummy_data_rows = [
            "1576412191482584,27/1/2025 2:00 PM,27/1/2025 3:00 PM,Casey,Kavanagh,hello@reelivate.com,508-479-0131,5,0,0,0,5,,ONLINE: Neverland: Heist on the High Seas,,,,Neverland Heist on the High Seas,,,,,,41576AFJXYA17676D7483F,FALSE,normal,,,,0,,,Elevent,-108.31,72.24,0.00,0.00,72.24,0.00,72.24,,19/12/2024 9:16 AM,Liz Orenstein,27/1/2025 3:18 PM,Pip Bauer,,,,,,74.15.49.32,,,Branch A,yes,,,Online,,yes,,,",
            "1576501213960737,29/1/2025 7:30 PM,29/1/2025 8:30 PM,Cody,Alexander,cody.alexander@interworks.com,405-385-3849,9,0,0,0,9,,ONLINE: Neverland: Heist on the High Seas,,,,Neverland Heist on the High Seas,,,,,,41576AFJXYA17676D7483F,FALSE,normal,,,,0,,,,324.99,0.00,0.00,324.99,324.99,0.00,,21/1/2025 2:02 PM,CUSTOMER,,,,,,,,68.205.192.232,,,Branch B,yes,,,Online,,yes,,,",
            "1576408143268750,2/10/2024 1:15 PM,2/10/2024 2:15 PM,Testing,Test,liz@improbableescapes.com,343 363 2015,2,0,0,0,2,,ONLINE: Neverland: Heist on the High Seas,,,,,,,,41576AFJXYA17676D7483F,FALSE,canceled,Testing,,,0,,,,0.00,0.00,0.00,0.00,0.00,0.00,,14/8/2024 11:33 AM,Liz Orenstein,14/8/2024 1:35 PM,Liz Orenstein,14/8/2025,,,,142.113.238.200,,,Branch A,no,,,Online,,yes,,,no"
        ]
        with open(BOOKINGS_CSV_PATH, 'w') as f:
            f.write(dummy_data_header + "\n")
            for row in dummy_data_rows:
                f.write(row + "\n")
        print(f"Created dummy bookings file at {BOOKINGS_CSV_PATH}")
    else:
        print(f"Found existing {BOOKINGS_CSV_PATH}. Will use it.")

    main()
