"""
Data preparation for LSTM forecasting
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import MinMaxScaler
import pickle

from .config import ForecastingConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceGenerator:
    """Generates sequences for LSTM training"""

    def __init__(self, config: Optional[ForecastingConfig] = None):
        self.config = config or ForecastingConfig()
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.target_column = self.config.TARGET_COLUMN

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for sequence generation

        Args:
            df: Input DataFrame
            feature_columns: List of feature column names
            fit_scaler: Whether to fit the scaler

        Returns:
            Prepared DataFrame
        """
        logger.info("Preparing data for sequence generation")

        df = df.copy()

        # Sort by time
        if 'time' in df.columns:
            df = df.sort_values('time').reset_index(drop=True)

        # Select feature columns
        if feature_columns is None:
            feature_columns = self.config.get_feature_columns()

        # Filter to available columns
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_columns = available_features

        # Add target column
        if self.target_column not in self.feature_columns:
            self.feature_columns.append(self.target_column)

        logger.info(f"Using {len(self.feature_columns)} features: {self.feature_columns}")

        # Select and fill missing values
        df = df[self.feature_columns].fillna(method='ffill').fillna(method='bfill')

        # Scale features
        if fit_scaler:
            df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
            logger.info("Fitted scaler on data")
        else:
            df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
            logger.info("Transformed data using existing scaler")

        return df

    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        step_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training

        Args:
            data: Input data array
            sequence_length: Length of input sequence
            forecast_horizon: Number of steps to forecast
            step_size: Step size for sliding window

        Returns:
            Tuple of (X, y) sequences
        """
        sequence_length = sequence_length or self.config.SEQUENCE_LENGTH
        forecast_horizon = forecast_horizon or self.config.FORECAST_HORIZON
        step_size = step_size or self.config.STEP_SIZE

        logger.info(f"Creating sequences: seq_len={sequence_length}, horizon={forecast_horizon}, step={step_size}")

        X, y = [], []

        for i in range(0, len(data) - sequence_length - forecast_horizon + 1, step_size):
            # Input sequence
            X.append(data[i:i + sequence_length])

            # Target (forecast_horizon steps ahead)
            target_idx = i + sequence_length + forecast_horizon - 1
            y.append(data[target_idx, -1])  # Target is last column (passengers)

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} sequences with shape X={X.shape}, y={y.shape}")

        return X, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_split: Optional[float] = None,
        val_split: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets

        Args:
            X: Input sequences
            y: Target values
            train_split: Fraction for training
            val_split: Fraction for validation

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train_split = train_split or self.config.TRAIN_SPLIT
        val_split = val_split or self.config.VAL_SPLIT

        n_samples = len(X)
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)

        X_train = X[:n_train]
        y_train = y[:n_train]

        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]

        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]

        logger.info(f"Data split:")
        logger.info(f"  Train: {len(X_train)} samples ({train_split*100:.1f}%)")
        logger.info(f"  Val:   {len(X_val)} samples ({val_split*100:.1f}%)")
        logger.info(f"  Test:  {len(X_test)} samples ({(1-train_split-val_split)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        route_id: Optional[str] = None,
        stop_id: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preparation pipeline for training

        Args:
            df: Input DataFrame
            feature_columns: List of feature columns
            route_id: Filter by route ID
            stop_id: Filter by stop ID

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("=" * 80)
        logger.info("PREPARING DATA FOR TRAINING")
        logger.info("=" * 80)

        # Filter by route/stop if specified
        if route_id:
            df = df[df['route'] == route_id]
            logger.info(f"Filtered to route: {route_id}")

        if stop_id:
            df = df[df['stop'] == stop_id]
            logger.info(f"Filtered to stop: {stop_id}")

        logger.info(f"Data shape after filtering: {df.shape}")

        # Prepare data
        prepared_df = self.prepare_data(df, feature_columns, fit_scaler=True)

        # Convert to numpy array
        data = prepared_df.values

        # Create sequences
        X, y = self.create_sequences(data)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        logger.info("=" * 80)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("=" * 80)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform target values to original scale

        Args:
            y: Normalized target values

        Returns:
            Original scale target values
        """
        # Target is last column in scaler
        target_idx = self.feature_columns.index(self.target_column)

        # Create dummy array for inverse transform
        dummy = np.zeros((len(y), len(self.feature_columns)))
        dummy[:, target_idx] = y

        # Inverse transform
        inverse = self.scaler.inverse_transform(dummy)

        return inverse[:, target_idx]

    def save_scaler(self, path: Optional[str] = None):
        """Save fitted scaler"""
        path = path or self.config.get_scaler_path()

        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)

        logger.info(f"Saved scaler to {path}")

    def load_scaler(self, path: Optional[str] = None):
        """Load fitted scaler"""
        path = path or self.config.get_scaler_path()

        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']

        logger.info(f"Loaded scaler from {path}")

    def get_data_info(self) -> Dict:
        """Get information about prepared data"""
        return {
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns),
            'target_column': self.target_column,
            'sequence_length': self.config.SEQUENCE_LENGTH,
            'forecast_horizon': self.config.FORECAST_HORIZON,
            'scaler_fitted': self.scaler is not None
        }
