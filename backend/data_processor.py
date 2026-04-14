"""
Data processing module for water scarcity and sustainability data.

This module handles cleaning, validation, and normalization of data.
"""

import pandas as pd
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean water scarcity and sustainability data."""

    def __init__(self):
        """Initialize data processor."""
        self.logger = logger
        self.water_scarcity_df = None

    def load_water_scarcity_dataset(self, csv_path: str) -> Optional[pd.DataFrame]:
        """
        Load and clean water scarcity dataset.

        Args:
            csv_path: Path to the water scarcity CSV file

        Returns:
            Cleaned DataFrame or None if loading fails
        """
        try:
            df = pd.read_csv(csv_path)
            self.logger.info(f"Loaded water scarcity dataset with {len(df)} rows")

            # Clean dataset
            df = self._clean_dataframe(df)
            self.water_scarcity_df = df
            return df

        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {csv_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            return None

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataframe by handling missing values and normalizing columns.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Handle missing values
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(df[col].mean())

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        self.logger.info("Data cleaning completed")
        return df

    def find_water_stress_level(self, region: str) -> Optional[dict]:
        """
        Find water stress level for a region.

        Args:
            region: Region or country name

        Returns:
            Dictionary with region data or None if not found
        """
        try:
            if self.water_scarcity_df is None:
                self.logger.error("Water scarcity dataset not loaded")
                return None

            # Case-insensitive search
            matches = self.water_scarcity_df[
                self.water_scarcity_df.iloc[:, 0].str.contains(
                    region, case=False, na=False
                )
            ]

            if len(matches) > 0:
                return matches.iloc[0].to_dict()
            else:
                self.logger.warning(f"Region not found: {region}")
                return None

        except Exception as e:
            self.logger.error(f"Error finding water stress level: {str(e)}")
            return None

    def estimate_risk_level(self, water_usage: float, water_stress: str) -> str:
        """
        Estimate environmental risk level.

        Args:
            water_usage: Total water usage in ML/year
            water_stress: Water stress level from dataset

        Returns:
            Risk level: "Low", "Medium", or "High"
        """
        try:
            risk_score = 0

            # Water usage scoring (ML/year)
            if water_usage > 1000000:
                risk_score += 2
            elif water_usage > 500000:
                risk_score += 1

            # Water stress scoring
            stress_map = {
                "low": 0,
                "medium": 1,
                "high": 2,
                "unknown": 1
            }
            risk_score += stress_map.get((water_stress or 'unknown').lower(), 1)

            if risk_score >= 3:
                return "High"
            elif risk_score >= 2:
                return "Medium"
            else:
                return "Low"

        except Exception as e:
            self.logger.error(f"Error estimating risk level: {str(e)}")
            return "Unknown"

    def validate_water_metrics(
        self, water_usage: str, wue: str
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Validate and convert water metrics to numeric values.

        Args:
            water_usage: Water usage as string (e.g., "1.5M liters")
            wue: WUE as string (e.g., "0.85")

        Returns:
            Tuple of (is_valid, water_usage_numeric, wue_numeric)
        """
        try:
            # Parse water usage
            water_numeric = self._parse_water_value(water_usage)
            wue_numeric = self._parse_numeric_value(wue)

            if water_numeric is not None and wue_numeric is not None:
                return True, water_numeric, wue_numeric
            else:
                return False, None, None

        except Exception as e:
            self.logger.error(f"Error validating metrics: {str(e)}")
            return False, None, None

    def _parse_water_value(self, value: str) -> Optional[float]:
        """Parse water value with units (ML, liters, gallons, etc.)."""
        try:
            value = str(value).upper().strip()

            # Remove common units
            value = value.replace("ML", "").replace("LITERS", "").replace("GALLONS", "").strip()

            # Try to extract numeric value
            numeric_str = ""
            for char in value:
                if char.isdigit() or char == ".":
                    numeric_str += char

            return float(numeric_str) if numeric_str else None

        except Exception as e:
            self.logger.error(f"Error parsing water value: {str(e)}")
            return None

    def _parse_numeric_value(self, value: str) -> Optional[float]:
        """Parse numeric value, removing common units."""
        try:
            value = str(value).strip()

            # Remove common units
            for unit in ["L/KWH", "L/WATT", "%", "LITERS/WATT"]:
                value = value.replace(unit, "")

            numeric_str = ""
            for char in value:
                if char.isdigit() or char == ".":
                    numeric_str += char

            return float(numeric_str) if numeric_str else None

        except Exception as e:
            self.logger.error(f"Error parsing numeric value: {str(e)}")
            return None
