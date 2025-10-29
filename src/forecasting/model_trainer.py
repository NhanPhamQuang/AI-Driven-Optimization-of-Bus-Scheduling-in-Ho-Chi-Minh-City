"""
Model training pipeline for LSTM forecasting
"""

import logging
import numpy as np
import time
from typing import Optional, Tuple, Dict
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    Callback
)

from .config import ForecastingConfig
from .lstm_model import LSTMForecaster


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingProgressCallback(Callback):
    """Custom callback for training progress"""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss', 0):.4f}, "
            f"mae={logs.get('mae', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}, "
            f"val_mae={logs.get('val_mae', 0):.4f}"
        )


class ModelTrainer:
    """Trains LSTM forecasting model"""

    def __init__(
        self,
        forecaster: Optional[LSTMForecaster] = None,
        config: Optional[ForecastingConfig] = None
    ):
        self.config = config or ForecastingConfig()
        self.forecaster = forecaster or LSTMForecaster(config)
        self.history = None
        self.training_time = None

    def get_callbacks(
        self,
        model_path: Optional[str] = None
    ) -> list:
        """
        Create training callbacks

        Args:
            model_path: Path to save best model

        Returns:
            List of callbacks
        """
        model_path = model_path or self.config.get_model_path()

        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config.MONITOR_METRIC,
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        logger.info(f"Added EarlyStopping (patience={self.config.EARLY_STOPPING_PATIENCE})")

        # Model checkpoint
        model_checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor=self.config.MONITOR_METRIC,
            save_best_only=self.config.SAVE_BEST_ONLY,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        logger.info(f"Added ModelCheckpoint (path={model_path})")

        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=self.config.MONITOR_METRIC,
            factor=self.config.REDUCE_LR_FACTOR,
            patience=self.config.REDUCE_LR_PATIENCE,
            min_lr=self.config.MIN_LEARNING_RATE,
            verbose=1
        )
        callbacks.append(reduce_lr)
        logger.info(f"Added ReduceLROnPlateau (patience={self.config.REDUCE_LR_PATIENCE})")

        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=self.config.TENSORBOARD_LOG_DIR,
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        logger.info(f"Added TensorBoard (log_dir={self.config.TENSORBOARD_LOG_DIR})")

        # Training progress
        progress = TrainingProgressCallback()
        callbacks.append(progress)

        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> keras.callbacks.History:
        """
        Train the model

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        logger.info("=" * 80)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 80)

        epochs = epochs or self.config.EPOCHS
        batch_size = batch_size or self.config.BATCH_SIZE

        # Build model if not already built
        if self.forecaster.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.forecaster.build_model(input_shape)

        # Print model summary
        logger.info("\nModel Architecture:")
        self.forecaster.summary()

        # Training info
        logger.info(f"\nTraining Configuration:")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Validation samples: {len(X_val)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Learning rate: {self.config.LEARNING_RATE}")
        logger.info(f"  Optimizer: {self.config.OPTIMIZER}")
        logger.info(f"  Loss: {self.config.LOSS}")

        # Get callbacks
        callbacks = self.get_callbacks()

        # Train model
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING IN PROGRESS...")
        logger.info("=" * 80 + "\n")

        start_time = time.time()

        self.history = self.forecaster.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0  # We use custom callback for logging
        )

        self.training_time = time.time() - start_time

        # Training complete
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Training time: {self.training_time:.2f}s ({self.training_time/60:.2f} minutes)")
        logger.info(f"Epochs completed: {len(self.history.history['loss'])}")

        # Best metrics
        best_epoch = np.argmin(self.history.history['val_loss'])
        logger.info(f"\nBest Epoch: {best_epoch + 1}")
        logger.info(f"  Train Loss: {self.history.history['loss'][best_epoch]:.4f}")
        logger.info(f"  Train MAE: {self.history.history['mae'][best_epoch]:.4f}")
        logger.info(f"  Val Loss: {self.history.history['val_loss'][best_epoch]:.4f}")
        logger.info(f"  Val MAE: {self.history.history['val_mae'][best_epoch]:.4f}")

        return self.history

    def get_training_history(self) -> Optional[Dict]:
        """Get training history"""
        if self.history is None:
            return None

        return {
            'loss': self.history.history['loss'],
            'mae': self.history.history['mae'],
            'rmse': self.history.history['rmse'],
            'val_loss': self.history.history['val_loss'],
            'val_mae': self.history.history['val_mae'],
            'val_rmse': self.history.history['val_rmse'],
            'epochs': len(self.history.history['loss']),
            'training_time': self.training_time
        }

    def get_best_metrics(self) -> Dict:
        """Get best metrics from training"""
        if self.history is None:
            return {}

        best_epoch = np.argmin(self.history.history['val_loss'])

        return {
            'best_epoch': best_epoch + 1,
            'train_loss': self.history.history['loss'][best_epoch],
            'train_mae': self.history.history['mae'][best_epoch],
            'train_rmse': self.history.history['rmse'][best_epoch],
            'val_loss': self.history.history['val_loss'][best_epoch],
            'val_mae': self.history.history['val_mae'][best_epoch],
            'val_rmse': self.history.history['val_rmse'][best_epoch]
        }
