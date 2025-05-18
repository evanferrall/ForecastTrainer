# Feature engineering package
import logging

# Configure basic logging for the package if not already configured
# This is a simple way to ensure logs are seen during development if no app-level config is set.
# For a more robust setup, application-level logging configuration is preferred.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO) # Or logging.DEBUG for more verbosity

from .calendar import add_calendar_features
from .temporal import add_lead_time_features
from .promotional import add_promotional_features
from .price_capacity import add_price_capacity_features
from .external import add_external_data_placeholders
from .static import create_static_features

__all__ = [
    "add_calendar_features",
    "add_lead_time_features",
    "add_promotional_features",
    "add_price_capacity_features",
    "add_external_data_placeholders",
    "create_static_features",
] 