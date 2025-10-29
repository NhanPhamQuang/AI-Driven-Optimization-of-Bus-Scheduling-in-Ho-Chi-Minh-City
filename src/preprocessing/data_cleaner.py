"""
Data cleaning and validation module
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .config import PreprocessingConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning and validation"""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.cleaning_stats = {}

    def clean_demand_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean passenger demand data

        Args:
            df: Raw demand DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning demand data: {len(df)} records")
        initial_count = len(df)

        # Make a copy
        df = df.copy()

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])

        # Remove duplicates
        df = df.drop_duplicates(subset=['route', 'stop', 'time'], keep='first')
        duplicates_removed = initial_count - len(df)

        # Validate passenger counts
        df = self._validate_passenger_counts(df)

        # Handle missing values
        df = self._handle_missing_values(df, 'demand')

        # Sort by time
        df = df.sort_values(['route', 'stop', 'time']).reset_index(drop=True)

        # Record stats
        self.cleaning_stats['demand'] = {
            'initial_records': initial_count,
            'final_records': len(df),
            'duplicates_removed': duplicates_removed,
            'records_dropped': initial_count - len(df)
        }

        logger.info(f"Demand data cleaned: {len(df)} records ({duplicates_removed} duplicates removed)")

        return df

    def clean_gps_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean GPS trace data

        Args:
            df: Raw GPS DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning GPS data: {len(df)} records")
        initial_count = len(df)

        df = df.copy()

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])

        # Remove duplicates
        df = df.drop_duplicates(subset=['route', 'bus_id', 'time'], keep='first')

        # Validate coordinates
        df = self._validate_coordinates(df)

        # Validate speed
        df = self._validate_speed(df)

        # Handle missing values
        df = self._handle_missing_values(df, 'gps')

        # Sort by bus and time
        df = df.sort_values(['route', 'bus_id', 'time']).reset_index(drop=True)

        self.cleaning_stats['gps'] = {
            'initial_records': initial_count,
            'final_records': len(df),
            'records_dropped': initial_count - len(df)
        }

        logger.info(f"GPS data cleaned: {len(df)} records")

        return df

    def clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean weather data

        Args:
            df: Raw weather DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning weather data: {len(df)} records")
        initial_count = len(df)

        df = df.copy()

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])

        # Remove duplicates
        df = df.drop_duplicates(subset=['time'], keep='first')

        # Validate temperature
        df = self._validate_temperature(df)

        # Validate humidity
        df = self._validate_humidity(df)

        # Handle missing values
        df = self._handle_missing_values(df, 'weather')

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        self.cleaning_stats['weather'] = {
            'initial_records': initial_count,
            'final_records': len(df),
            'records_dropped': initial_count - len(df)
        }

        logger.info(f"Weather data cleaned: {len(df)} records")

        return df

    def clean_events_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean events data

        Args:
            df: Raw events DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning events data: {len(df)} records")
        initial_count = len(df)

        df = df.copy()

        # Convert times to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        # Remove duplicates
        df = df.drop_duplicates(subset=['name', 'start_time'], keep='first')

        # Validate event durations
        df = df[df['end_time'] > df['start_time']]

        # Validate coordinates if present
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = self._validate_coordinates(df)

        # Handle missing values
        df = self._handle_missing_values(df, 'events')

        # Sort by start time
        df = df.sort_values('start_time').reset_index(drop=True)

        self.cleaning_stats['events'] = {
            'initial_records': initial_count,
            'final_records': len(df),
            'records_dropped': initial_count - len(df)
        }

        logger.info(f"Events data cleaned: {len(df)} records")

        return df

    def _validate_passenger_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix passenger counts"""
        # Check for negative values
        for col in ['boarding_count', 'alighting_count', 'passengers']:
            if col in df.columns:
                invalid = df[col] < self.config.MIN_PASSENGERS
                if invalid.any():
                    logger.warning(f"Found {invalid.sum()} negative {col}, setting to 0")
                    df.loc[invalid, col] = 0

                # Check for unrealistic high values
                invalid = df[col] > self.config.MAX_PASSENGERS
                if invalid.any():
                    logger.warning(f"Found {invalid.sum()} invalid {col} (>{self.config.MAX_PASSENGERS})")
                    df = df[~invalid]

        return df

    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate GPS coordinates"""
        # HCMC approximate bounds: lat 10.3-11.2, lng 106.3-107.0
        if 'latitude' in df.columns:
            invalid_lat = (df['latitude'] < 10.0) | (df['latitude'] > 11.5)
            if invalid_lat.any():
                logger.warning(f"Found {invalid_lat.sum()} invalid latitudes")
                df = df[~invalid_lat]

        if 'longitude' in df.columns:
            invalid_lng = (df['longitude'] < 106.0) | (df['longitude'] > 107.5)
            if invalid_lng.any():
                logger.warning(f"Found {invalid_lng.sum()} invalid longitudes")
                df = df[~invalid_lng]

        return df

    def _validate_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix speed values"""
        if 'speed' not in df.columns:
            return df

        # Check for negative speeds
        invalid = df['speed'] < self.config.MIN_SPEED_KMH
        if invalid.any():
            logger.warning(f"Found {invalid.sum()} negative speeds, setting to 0")
            df.loc[invalid, 'speed'] = 0

        # Check for unrealistic high speeds
        invalid = df['speed'] > self.config.MAX_SPEED_KMH
        if invalid.any():
            logger.warning(f"Found {invalid.sum()} excessive speeds (>{self.config.MAX_SPEED_KMH} km/h)")
            df.loc[invalid, 'speed'] = self.config.MAX_SPEED_KMH

        return df

    def _validate_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate temperature values"""
        if 'temperature' not in df.columns:
            return df

        invalid = (df['temperature'] < self.config.MIN_TEMPERATURE_C) | \
                  (df['temperature'] > self.config.MAX_TEMPERATURE_C)

        if invalid.any():
            logger.warning(f"Found {invalid.sum()} invalid temperatures")
            df = df[~invalid]

        return df

    def _validate_humidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate humidity values"""
        if 'humidity' not in df.columns:
            return df

        invalid = (df['humidity'] < self.config.MIN_HUMIDITY) | \
                  (df['humidity'] > self.config.MAX_HUMIDITY)

        if invalid.any():
            logger.warning(f"Found {invalid.sum()} invalid humidity values")
            df.loc[invalid, 'humidity'] = df.loc[~invalid, 'humidity'].median()

        return df

    def _handle_missing_values(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Handle missing values based on data type"""
        initial_nulls = df.isnull().sum().sum()

        if initial_nulls == 0:
            return df

        logger.info(f"Handling {initial_nulls} missing values in {data_type} data")

        # Strategy depends on data type
        if data_type == 'demand':
            # Forward fill for temporal data, then drop remaining
            df = df.sort_values(['route', 'stop', 'time'])
            df = df.groupby(['route', 'stop']).fillna(method='ffill')
            df = df.dropna()

        elif data_type == 'gps':
            # Interpolate for GPS traces
            df = df.sort_values(['route', 'bus_id', 'time'])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df.groupby(['route', 'bus_id'])[numeric_cols].apply(
                lambda x: x.interpolate(method='linear', limit=3)
            )
            df = df.dropna()

        elif data_type == 'weather':
            # Interpolate weather data
            df = df.sort_values('time')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            df = df.fillna(method='ffill').fillna(method='bfill')

        elif data_type == 'events':
            # Drop rows with critical missing values
            df = df.dropna(subset=['start_time', 'end_time'])

        final_nulls = df.isnull().sum().sum()
        logger.info(f"Missing values reduced from {initial_nulls} to {final_nulls}")

        return df

    def get_cleaning_report(self) -> Dict:
        """Get cleaning statistics"""
        return self.cleaning_stats

    def print_cleaning_report(self):
        """Print cleaning report"""
        print("\n" + "=" * 80)
        print("DATA CLEANING REPORT")
        print("=" * 80)

        for data_type, stats in self.cleaning_stats.items():
            print(f"\n{data_type.upper()}:")
            for key, value in stats.items():
                print(f"  {key}: {value:,}")

        print("\n" + "=" * 80)
