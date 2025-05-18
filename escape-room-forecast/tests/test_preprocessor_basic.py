import unittest
from pathlib import Path

# Add package path if tests are run from repo root
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from forecast_cli.prep.preprocessor import DataPreprocessor


class TestPreprocessorBasic(unittest.TestCase):
    """Basic integration tests for DataPreprocessor using the example bookings CSV."""

    @classmethod
    def setUpClass(cls):
        kpi_configs = {
            "bookings_daily": {"autogluon_freq": "D", "target_transform": "log1p"},
            "prob": {"autogluon_freq": "H", "target_transform": "logit"},
        }
        csv_path = Path(__file__).resolve().parents[1] / "raw_data" / "bookings.csv"
        cls.preprocessor = DataPreprocessor(
            csv_path=csv_path,
            kpi_configs=kpi_configs,
            ts_col="Start",
            end_col="End",
        )
        cls.long_df, cls.static_df = cls.preprocessor.process()

    def test_long_df_has_basic_columns(self):
        """long_df should contain at least series_id, timestamp, and y."""
        for col in ["series_id", "timestamp", "y"]:
            self.assertIn(col, self.long_df.columns)
        self.assertGreater(len(self.long_df), 0)

    def test_static_features_match_series(self):
        """Static feature series_ids should be subset of long_df series_ids."""
        self.assertIn("series_id", self.static_df.columns)
        self.assertTrue(set(self.static_df["series_id"]).issubset(set(self.long_df["series_id"])))


if __name__ == "__main__":
    unittest.main()
