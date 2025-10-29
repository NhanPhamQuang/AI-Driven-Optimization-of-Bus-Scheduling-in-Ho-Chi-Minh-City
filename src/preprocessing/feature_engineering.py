"""
Feature engineering for bus scheduling data
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

from .config import PreprocessingConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates engineered features from raw data"""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def create_temporal_features(self, df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
        """
        Create temporal features from timestamp

        Args:
            df: DataFrame with time column
            time_col: Name of time column

        Returns:
            DataFrame with additional temporal features
        """
        logger.info("Creating temporal features")

        df = df.copy()

        # Ensure datetime
        df[time_col] = pd.to_datetime(df[time_col])

        # Basic time features
        df['hour'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek  # 0=Monday
        df['day_of_month'] = df[time_col].dt.day
        df['month'] = df[time_col].dt.month
        df['week_of_year'] = df[time_col].dt.isocalendar().week
        df['quarter'] = df[time_col].dt.quarter

        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Peak hours
        df['is_peak_hour'] = df['hour'].isin([7, 8, 16, 17]).astype(int)
        df['is_morning_peak'] = df['hour'].isin([7, 8]).astype(int)
        df['is_evening_peak'] = df['hour'].isin([16, 17]).astype(int)

        # Time of day categories
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[-1, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )

        # Cyclical encoding for hour (sin/cos transformation)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Minutes since midnight
        df['time_since_midnight'] = df['hour'] * 60 + df[time_col].dt.minute

        # Holiday indicator
        df['date_str'] = df[time_col].dt.strftime('%Y-%m-%d')
        df['is_holiday'] = df['date_str'].isin(self.config.HOLIDAYS).astype(int)
        df = df.drop('date_str', axis=1)

        # Rush hour intensity (0-1 scale)
        df['rush_hour_intensity'] = df.apply(
            lambda x: self._calculate_rush_intensity(x['hour'], x['day_of_week']),
            axis=1
        )

        logger.info(f"Created {len([c for c in df.columns if c.startswith(('hour', 'day', 'is_', 'time', 'dow', 'rush'))])} temporal features")

        return df

    def create_spatial_features(
        self,
        df: pd.DataFrame,
        stops_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create spatial features from GPS coordinates

        Args:
            df: DataFrame with latitude/longitude
            stops_df: DataFrame with stop information

        Returns:
            DataFrame with spatial features
        """
        logger.info("Creating spatial features")

        df = df.copy()

        # Distance from city center
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['distance_from_center'] = df.apply(
                lambda row: self._haversine_distance(
                    row['latitude'], row['longitude'],
                    self.config.CITY_CENTER_LAT, self.config.CITY_CENTER_LNG
                ),
                axis=1
            )

            # Quadrant (relative to city center)
            df['is_north'] = (df['latitude'] > self.config.CITY_CENTER_LAT).astype(int)
            df['is_east'] = (df['longitude'] > self.config.CITY_CENTER_LNG).astype(int)

        # If we have stop information
        if stops_df is not None and 'stop' in df.columns:
            # Add stop coordinates
            stop_coords = stops_df[['stop', 'latitude', 'longitude', 'order']].rename(
                columns={'latitude': 'stop_lat', 'longitude': 'stop_lng', 'order': 'stop_order'}
            )
            df = df.merge(stop_coords, on='stop', how='left', suffixes=('', '_stop'))

            # Total stops on route
            route_stops = stops_df.groupby('route')['stop'].count().reset_index()
            route_stops.columns = ['route', 'total_stops']
            df = df.merge(route_stops, on='route', how='left')

            # Progress along route (0-1)
            df['route_progress'] = df['stop_order'] / df['total_stops']

            # Distance to next stop (if we can calculate it)
            df = self._add_distance_to_next_stop(df, stops_df)

        logger.info("Spatial features created")

        return df

    def create_contextual_features(
        self,
        df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create contextual features (weather, events)

        Args:
            df: Main DataFrame with time column
            weather_df: Weather data
            events_df: Events data

        Returns:
            DataFrame with contextual features
        """
        logger.info("Creating contextual features")

        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])

        # Merge weather data
        if weather_df is not None:
            weather_df = weather_df.copy()
            weather_df['time'] = pd.to_datetime(weather_df['time'])

            # Round times to nearest 15 minutes for merging
            df['time_rounded'] = df['time'].dt.floor('15min')
            weather_df['time_rounded'] = weather_df['time'].dt.floor('15min')

            # Merge
            weather_cols = ['time_rounded', 'temperature', 'humidity', 'rain_mm', 'wind_speed', 'conditions']
            weather_cols = [c for c in weather_cols if c in weather_df.columns]

            df = df.merge(
                weather_df[weather_cols],
                on='time_rounded',
                how='left'
            )

            # Weather-derived features
            if 'rain_mm' in df.columns:
                df['is_raining'] = (df['rain_mm'] > 0).astype(int)
                df['rain_intensity'] = pd.cut(
                    df['rain_mm'],
                    bins=[-0.1, 0, 1, 5, 100],
                    labels=['none', 'light', 'moderate', 'heavy']
                )

            if 'temperature' in df.columns:
                df['temp_category'] = pd.cut(
                    df['temperature'],
                    bins=[0, 25, 30, 35, 50],
                    labels=['cool', 'comfortable', 'hot', 'very_hot']
                )

            # Heat index (simplified)
            if 'temperature' in df.columns and 'humidity' in df.columns:
                df['heat_index'] = df['temperature'] * (1 + df['humidity'] / 200)

            df = df.drop('time_rounded', axis=1, errors='ignore')

        # Merge events data
        if events_df is not None:
            df = self._add_event_features(df, events_df)

        # Commute patterns
        df['is_commute_time'] = df.apply(
            lambda x: 1 if ((x['hour'] in [7, 8, 9]) or (x['hour'] in [16, 17, 18]))
            and x.get('is_weekend', 0) == 0 else 0,
            axis=1
        )

        logger.info("Contextual features created")

        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        lags: List[int] = [1, 2, 3, 6, 12],
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series

        Args:
            df: DataFrame sorted by time
            target_col: Column to create lags for
            lags: List of lag periods
            group_cols: Columns to group by (e.g., ['route', 'stop'])

        Returns:
            DataFrame with lag features
        """
        logger.info(f"Creating lag features for {target_col}")

        df = df.copy()

        if group_cols:
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
        else:
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

        logger.info(f"Created {len(lags)} lag features")

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        windows: List[int] = [3, 6, 12],
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features

        Args:
            df: DataFrame sorted by time
            target_col: Column to create rolling features for
            windows: List of window sizes
            group_cols: Columns to group by

        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling features for {target_col}")

        df = df.copy()

        for window in windows:
            if group_cols:
                # Rolling mean
                df[f'{target_col}_rolling_mean_{window}'] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                # Rolling std
                df[f'{target_col}_rolling_std_{window}'] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            else:
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(
                    window=window, min_periods=1
                ).mean()
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(
                    window=window, min_periods=1
                ).std()

        logger.info(f"Created {len(windows) * 2} rolling features")

        return df

    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two GPS coordinates (in km)

        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c  # Radius of earth in kilometers

        return km

    def _calculate_rush_intensity(self, hour: int, day_of_week: int) -> float:
        """
        Calculate rush hour intensity (0-1)

        Args:
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday)

        Returns:
            Intensity value between 0 and 1
        """
        # Weekend has lower intensity
        if day_of_week >= 5:
            base = 0.3
        else:
            base = 1.0

        # Morning peak (7-9 AM)
        if 7 <= hour <= 9:
            intensity = base * (1.0 if hour == 8 else 0.8)
        # Evening peak (4-6 PM)
        elif 16 <= hour <= 18:
            intensity = base * (1.0 if hour == 17 else 0.8)
        # Shoulder hours
        elif hour in [6, 9, 15, 18]:
            intensity = base * 0.5
        # Off-peak
        else:
            intensity = base * 0.2

        return intensity

    def _add_distance_to_next_stop(
        self,
        df: pd.DataFrame,
        stops_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add distance to next stop feature"""
        # Group stops by route and order
        stops_ordered = stops_df.sort_values(['route', 'order'])

        # Calculate distance between consecutive stops
        distances = []
        for route in stops_ordered['route'].unique():
            route_stops = stops_ordered[stops_ordered['route'] == route].reset_index(drop=True)

            for i in range(len(route_stops) - 1):
                stop1 = route_stops.loc[i]
                stop2 = route_stops.loc[i + 1]

                dist = self._haversine_distance(
                    stop1['latitude'], stop1['longitude'],
                    stop2['latitude'], stop2['longitude']
                )

                distances.append({
                    'route': route,
                    'stop': stop1['stop'],
                    'distance_to_next_stop': dist
                })

        if distances:
            dist_df = pd.DataFrame(distances)
            df = df.merge(dist_df, on=['route', 'stop'], how='left')

        return df

    def _add_event_features(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add event-related features"""
        df = df.copy()
        events_df = events_df.copy()

        # Convert times
        events_df['start_time'] = pd.to_datetime(events_df['start_time'])
        events_df['end_time'] = pd.to_datetime(events_df['end_time'])

        # Initialize features
        df['nearby_events'] = 0
        df['event_impact_factor'] = 1.0

        # For each row, check if there are nearby events
        for idx, row in df.iterrows():
            time = row['time']

            # Find events happening at this time
            active_events = events_df[
                (events_df['start_time'] <= time) &
                (events_df['end_time'] >= time)
            ]

            if len(active_events) > 0:
                df.at[idx, 'nearby_events'] = len(active_events)

                # Calculate combined impact factor
                if 'impact_factor' in active_events.columns:
                    df.at[idx, 'event_impact_factor'] = active_events['impact_factor'].max()

        return df
