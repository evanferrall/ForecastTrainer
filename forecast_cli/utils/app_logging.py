import logging
from rich.logging import RichHandler
import sys

def setup_rich_logger(level: int = logging.INFO, format_str: str = "%(message)s", date_format_str: str = "[%X]") -> None:
    """
    Sets up a logger that uses RichHandler for beautiful console output.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        format_str: The format string for the log messages.
        date_format_str: The format string for the date/time in log messages.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove any existing handlers to avoid duplicate messages if this is called multiple times
    # or if basicConfig was called before
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]: # Iterate over a copy
            root_logger.removeHandler(handler)
            handler.close() # Close the handler properly

    # Create a RichHandler
    # For more control over specific loggers (e.g. lightning, neuralforecast), 
    # one might get those loggers and add handlers/set levels specifically.
    rich_handler = RichHandler(
        level=level,
        show_time=True,
        show_level=True,
        show_path=False, # Set to True to see file paths in logs
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True, # Can be useful for debugging
        log_time_format=date_format_str 
    )
    
    # Define the format for the handler
    formatter = logging.Formatter(format_str) # RichHandler uses its own formatting largely
    rich_handler.setFormatter(formatter)

    # Add the RichHandler to the root logger
    root_logger.addHandler(rich_handler)
    
    # Set the logging level for the root logger
    root_logger.setLevel(level)

    # Optionally, adjust levels for noisy libraries
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO) # Or logging.WARNING
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING) # PIL can be noisy with debug
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)


if __name__ == '__main__':
    # Example usage:
    setup_rich_logger(level=logging.DEBUG)
    
    log = logging.getLogger("rich_example") # Get a logger instance
    log.debug("This is a debug message with [bold green]markup[/bold green]!")
    log.info("This is an info message.")
    log.warning("This is a warning.")
    log.error("This is an error.")
    log.critical("This is a critical message.")
    
    try:
        x = 1 / 0
    except Exception as e:
        log.exception("An exception occurred!")

    # Test noisy library logging
    # logging.getLogger("PIL.PngImagePlugin").debug("PIL debug message - should be filtered by default setup")
    # logging.getLogger("pytorch_lightning").info("Lightning info - should be visible")
    # logging.getLogger("pytorch_lightning").debug("Lightning debug - should be filtered")

    print(f"Root logger handlers: {logging.getLogger().handlers}")
    print(f"Root logger level: {logging.getLogger().level}")
    print(f"'pytorch_lightning' logger level: {logging.getLogger('pytorch_lightning').level}") 