"""
Data normalization and standardization
"""

import logging
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder

from .config import PreprocessingConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataNormalizer:
    """Handles data normalization and standardization"""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.scalers = {}
        self.encoders = {}
        self.feature_ranges = {}

    def normalize_features(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        method: str = 'minmax',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize numeric features

        Args:
            df: DataFrame with features
            numeric_cols: Columns to normalize (if None, auto-detect)
            method: 'minmax', 'standard', or 'robust'
            fit: If True, fit scaler. If False, use existing scaler

        Returns:
            DataFrame with normalized features
        """
        logger.info(f"Normalizing features using {method} method")

        df = df.copy()

        # Auto-detect numeric columns if not specified
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude time-related columns that shouldn't be normalized
            exclude_cols = ['time', 'year', 'month', 'day', 'hour', 'minute']
            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        if not numeric_cols:
            logger.warning("No numeric columns found to normalize")
            return df

        # Select scaler
        scaler_name = f'features_{method}'

        if fit:
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Fit and transform
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

            # Store scaler
            self.scalers[scaler_name] = scaler

            # Store feature ranges for reference
            self.feature_ranges[scaler_name] = {
                'columns': numeric_cols,
                'original_min': df[numeric_cols].min().to_dict(),
                'original_max': df[numeric_cols].max().to_dict()
            }

            logger.info(f"Fitted and normalized {len(numeric_cols)} features")
        else:
            # Use existing scaler
            if scaler_name not in self.scalers:
                raise ValueError(f"Scaler {scaler_name} not found. Set fit=True first.")

            scaler = self.scalers[scaler_name]
            df[numeric_cols] = scaler.transform(df[numeric_cols])

            logger.info(f"Normalized {len(numeric_cols)} features using existing scaler")

        return df

    def inverse_normalize(
        self,
        df: pd.DataFrame,
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        Inverse normalization (denormalize)

        Args:
            df: DataFrame with normalized features
            method: Normalization method used

        Returns:
            DataFrame with original scale
        """
        logger.info(f"Inverse normalizing features")

        df = df.copy()
        scaler_name = f'features_{method}'

        if scaler_name not in self.scalers:
            raise ValueError(f"Scaler {scaler_name} not found")

        scaler = self.scalers[scaler_name]
        numeric_cols = self.feature_ranges[scaler_name]['columns']

        df[numeric_cols] = scaler.inverse_transform(df[numeric_cols])

        logger.info(f"Inverse normalized {len(numeric_cols)} features")

        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features

        Args:
            df: DataFrame with categorical features
            categorical_cols: Columns to encode (if None, auto-detect)
            fit: If True, fit encoder. If False, use existing encoder

        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features")

        df = df.copy()

        # Auto-detect categorical columns if not specified
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            logger.warning("No categorical columns found to encode")
            return df

        for col in categorical_cols:
            encoder_name = f'encoder_{col}'

            if fit:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[encoder_name] = encoder
                logger.info(f"Encoded {col}: {len(encoder.classes_)} unique values")
            else:
                if encoder_name not in self.encoders:
                    logger.warning(f"Encoder for {col} not found, fitting new encoder")
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    self.encoders[encoder_name] = encoder
                else:
                    encoder = self.encoders[encoder_name]
                    # Handle unseen categories
                    df[col] = df[col].apply(
                        lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                    )
                    df[col] = encoder.transform(df[col].astype(str))

        return df

    def one_hot_encode(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        drop_first: bool = False
    ) -> pd.DataFrame:
        """
        One-hot encode categorical features

        Args:
            df: DataFrame with categorical features
            categorical_cols: Columns to one-hot encode
            drop_first: Drop first category to avoid multicollinearity

        Returns:
            DataFrame with one-hot encoded features
        """
        logger.info(f"One-hot encoding {len(categorical_cols)} categorical features")

        df = df.copy()

        for col in categorical_cols:
            if col in df.columns:
                # Create dummies
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)

                # Add to dataframe
                df = pd.concat([df, dummies], axis=1)

                # Drop original column
                df = df.drop(col, axis=1)

                logger.info(f"One-hot encoded {col}: {len(dummies.columns)} new features")

        return df

    def normalize_target(
        self,
        df: pd.DataFrame,
        target_col: str,
        method: str = 'minmax',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize target variable

        Args:
            df: DataFrame with target
            target_col: Target column name
            method: Normalization method
            fit: If True, fit scaler

        Returns:
            DataFrame with normalized target
        """
        logger.info(f"Normalizing target: {target_col}")

        df = df.copy()
        scaler_name = f'target_{target_col}_{method}'

        if fit:
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Reshape for sklearn
            df[target_col] = scaler.fit_transform(df[[target_col]])

            self.scalers[scaler_name] = scaler
            logger.info(f"Normalized target {target_col}")
        else:
            if scaler_name not in self.scalers:
                raise ValueError(f"Scaler {scaler_name} not found")

            scaler = self.scalers[scaler_name]
            df[target_col] = scaler.transform(df[[target_col]])

        return df

    def inverse_normalize_target(
        self,
        values: Union[np.ndarray, pd.Series, pd.DataFrame],
        target_col: str,
        method: str = 'minmax'
    ) -> np.ndarray:
        """
        Inverse normalize target predictions

        Args:
            values: Normalized values
            target_col: Target column name
            method: Normalization method used

        Returns:
            Original scale values
        """
        scaler_name = f'target_{target_col}_{method}'

        if scaler_name not in self.scalers:
            raise ValueError(f"Scaler {scaler_name} not found")

        scaler = self.scalers[scaler_name]

        # Handle different input types
        if isinstance(values, pd.Series):
            values = values.values.reshape(-1, 1)
        elif isinstance(values, pd.DataFrame):
            values = values.values
        elif isinstance(values, np.ndarray):
            if values.ndim == 1:
                values = values.reshape(-1, 1)

        # Inverse transform
        original_values = scaler.inverse_transform(values)

        logger.info(f"Inverse normalized {len(values)} predictions")

        return original_values.flatten()

    def save_scalers(self, directory: str = None):
        """
        Save fitted scalers to disk

        Args:
            directory: Directory to save scalers (default: models dir from config)
        """
        if directory is None:
            directory = self.config.MODELS_DIR

        os.makedirs(directory, exist_ok=True)

        # Save scalers
        scalers_path = os.path.join(directory, 'scalers.pkl')
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Saved {len(self.scalers)} scalers to {scalers_path}")

        # Save encoders
        encoders_path = os.path.join(directory, 'encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.encoders, f)
        logger.info(f"Saved {len(self.encoders)} encoders to {encoders_path}")

        # Save feature ranges
        ranges_path = os.path.join(directory, 'feature_ranges.pkl')
        with open(ranges_path, 'wb') as f:
            pickle.dump(self.feature_ranges, f)
        logger.info(f"Saved feature ranges to {ranges_path}")

    def load_scalers(self, directory: str = None):
        """
        Load fitted scalers from disk

        Args:
            directory: Directory to load scalers from
        """
        if directory is None:
            directory = self.config.MODELS_DIR

        # Load scalers
        scalers_path = os.path.join(directory, 'scalers.pkl')
        if os.path.exists(scalers_path):
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
            logger.info(f"Loaded {len(self.scalers)} scalers from {scalers_path}")

        # Load encoders
        encoders_path = os.path.join(directory, 'encoders.pkl')
        if os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
            logger.info(f"Loaded {len(self.encoders)} encoders from {encoders_path}")

        # Load feature ranges
        ranges_path = os.path.join(directory, 'feature_ranges.pkl')
        if os.path.exists(ranges_path):
            with open(ranges_path, 'rb') as f:
                self.feature_ranges = pickle.load(f)
            logger.info(f"Loaded feature ranges from {ranges_path}")

    def get_normalization_info(self) -> Dict:
        """
        Get information about fitted scalers

        Returns:
            Dictionary with scaler information
        """
        info = {
            'num_scalers': len(self.scalers),
            'num_encoders': len(self.encoders),
            'scalers': list(self.scalers.keys()),
            'encoders': list(self.encoders.keys()),
            'feature_ranges': self.feature_ranges
        }
        return info

    def print_normalization_info(self):
        """Print normalization information"""
        info = self.get_normalization_info()

        print("\n" + "=" * 80)
        print("NORMALIZATION INFO")
        print("=" * 80)
        print(f"Number of scalers: {info['num_scalers']}")
        print(f"Number of encoders: {info['num_encoders']}")

        if info['scalers']:
            print("\nScalers:")
            for scaler in info['scalers']:
                print(f"  - {scaler}")

        if info['encoders']:
            print("\nEncoders:")
            for encoder in info['encoders']:
                print(f"  - {encoder}")

        print("=" * 80)
