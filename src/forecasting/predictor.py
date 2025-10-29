"""
Prediction and inference for demand forecasting
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from scipy import stats

from .config import ForecastingConfig
from .lstm_model import LSTMForecaster
from .data_preparation import SequenceGenerator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandPredictor:
    """Makes demand predictions using trained LSTM model"""

    def __init__(
        self,
        forecaster: Optional[LSTMForecaster] = None,
        sequence_generator: Optional[SequenceGenerator] = None,
        config: Optional[ForecastingConfig] = None
    ):
        self.config = config or ForecastingConfig()
        self.forecaster = forecaster or LSTMForecaster(config)
        self.sequence_generator = sequence_generator or SequenceGenerator(config)

    def predict(
        self,
        X: np.ndarray,
        return_original_scale: bool = True
    ) -> np.ndarray:
        """
        Make predictions on input sequences

        Args:
            X: Input sequences (batch_size, sequence_length, n_features)
            return_original_scale: Whether to return predictions in original scale

        Returns:
            Predictions array
        """
        if self.forecaster.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Make predictions
        predictions = self.forecaster.model.predict(X, verbose=0).flatten()

        # Convert to original scale if requested
        if return_original_scale:
            predictions = self.sequence_generator.inverse_transform_target(predictions)

        return predictions

    def predict_single(
        self,
        sequence: np.ndarray,
        return_original_scale: bool = True
    ) -> float:
        """
        Make prediction for a single sequence

        Args:
            sequence: Input sequence (sequence_length, n_features)
            return_original_scale: Whether to return prediction in original scale

        Returns:
            Single prediction value
        """
        # Add batch dimension
        X = np.expand_dims(sequence, axis=0)

        # Make prediction
        prediction = self.predict(X, return_original_scale=return_original_scale)

        return prediction[0]

    def predict_with_confidence(
        self,
        X: np.ndarray,
        n_iterations: int = 100,
        confidence_level: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals using Monte Carlo dropout

        Args:
            X: Input sequences
            n_iterations: Number of MC iterations
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        confidence_level = confidence_level or self.config.CONFIDENCE_INTERVAL

        if self.forecaster.model is None:
            raise ValueError("Model not loaded")

        logger.info(f"Computing confidence intervals with {n_iterations} iterations")

        # Enable dropout at test time for MC dropout
        predictions_list = []

        for _ in range(n_iterations):
            preds = self.forecaster.model(X, training=True)  # Enable dropout
            preds = preds.numpy().flatten()
            predictions_list.append(preds)

        predictions_array = np.array(predictions_list)

        # Calculate statistics
        predictions = np.mean(predictions_array, axis=0)
        std_dev = np.std(predictions_array, axis=0)

        # Confidence intervals
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * std_dev

        lower_bounds = predictions - margin
        upper_bounds = predictions + margin

        # Convert to original scale
        predictions = self.sequence_generator.inverse_transform_target(predictions)
        lower_bounds = self.sequence_generator.inverse_transform_target(lower_bounds)
        upper_bounds = self.sequence_generator.inverse_transform_target(upper_bounds)

        return predictions, lower_bounds, upper_bounds

    def predict_future(
        self,
        initial_sequence: np.ndarray,
        n_steps: int,
        return_original_scale: bool = True
    ) -> np.ndarray:
        """
        Make multi-step ahead predictions

        Args:
            initial_sequence: Initial sequence (sequence_length, n_features)
            n_steps: Number of steps to predict ahead
            return_original_scale: Whether to return in original scale

        Returns:
            Array of predictions
        """
        if self.forecaster.model is None:
            raise ValueError("Model not loaded")

        logger.info(f"Predicting {n_steps} steps ahead")

        predictions = []
        current_sequence = initial_sequence.copy()

        for step in range(n_steps):
            # Predict next step
            pred = self.predict_single(current_sequence, return_original_scale=False)
            predictions.append(pred)

            # Update sequence for next prediction
            # Shift sequence and add prediction
            new_row = current_sequence[-1].copy()
            new_row[-1] = pred  # Update target column

            current_sequence = np.vstack([current_sequence[1:], new_row])

        predictions = np.array(predictions)

        # Convert to original scale if requested
        if return_original_scale:
            predictions = self.sequence_generator.inverse_transform_target(predictions)

        return predictions

    def predict_real_time(
        self,
        recent_data: pd.DataFrame,
        feature_columns: list
    ) -> float:
        """
        Make real-time prediction from recent data

        Args:
            recent_data: Recent data (must contain at least sequence_length rows)
            feature_columns: List of feature columns

        Returns:
            Prediction value
        """
        if len(recent_data) < self.config.SEQUENCE_LENGTH:
            raise ValueError(
                f"Need at least {self.config.SEQUENCE_LENGTH} rows, got {len(recent_data)}"
            )

        # Take last sequence_length rows
        recent_data = recent_data.tail(self.config.SEQUENCE_LENGTH)

        # Prepare data (without fitting scaler)
        prepared_df = self.sequence_generator.prepare_data(
            recent_data,
            feature_columns=feature_columns,
            fit_scaler=False
        )

        # Convert to sequence
        sequence = prepared_df.values

        # Make prediction
        prediction = self.predict_single(sequence, return_original_scale=True)

        logger.info(f"Real-time prediction: {prediction:.2f} passengers")

        return prediction

    def load_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Load trained model and scaler

        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
        """
        logger.info("Loading model and scaler for prediction")

        # Load model
        self.forecaster.load_model(model_path)

        # Load scaler
        self.sequence_generator.load_scaler(scaler_path)

        logger.info("Model and scaler loaded successfully")

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        route_id: Optional[str] = None,
        stop_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Make predictions for a DataFrame

        Args:
            df: Input DataFrame
            feature_columns: List of feature columns
            route_id: Filter by route
            stop_id: Filter by stop

        Returns:
            DataFrame with predictions
        """
        logger.info("Making predictions for DataFrame")

        # Filter if needed
        if route_id:
            df = df[df['route'] == route_id]
        if stop_id:
            df = df[df['stop'] == stop_id]

        # Prepare data
        prepared_df = self.sequence_generator.prepare_data(
            df,
            feature_columns=feature_columns,
            fit_scaler=False
        )

        # Create sequences
        data = prepared_df.values
        X, y_actual = self.sequence_generator.create_sequences(data)

        # Make predictions
        y_pred = self.predict(X, return_original_scale=True)
        y_actual_orig = self.sequence_generator.inverse_transform_target(y_actual)

        # Create result DataFrame
        result_df = pd.DataFrame({
            'actual': y_actual_orig,
            'predicted': y_pred,
            'error': y_actual_orig - y_pred,
            'abs_error': np.abs(y_actual_orig - y_pred),
            'pct_error': np.abs((y_actual_orig - y_pred) / (y_actual_orig + 1e-10)) * 100
        })

        logger.info(f"Made {len(result_df)} predictions")

        return result_df
