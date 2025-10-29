"""
Configuration for demand forecasting module
"""

import os


class ForecastingConfig:
    """Configuration for LSTM demand forecasting"""

    # Data directories
    PROCESSED_DATA_DIR = "data/processed"
    FEATURES_DATA_DIR = "data/features"
    MODELS_DIR = "models"
    FORECASTING_MODELS_DIR = "models/forecasting"

    # Sequence parameters
    SEQUENCE_LENGTH = 12  # Number of time steps to look back (12 * 15min = 3 hours)
    FORECAST_HORIZON = 2  # Number of steps ahead to predict (2 * 15min = 30 min)
    STEP_SIZE = 1  # Step size for sliding window

    # Train/validation/test split
    TRAIN_SPLIT = 0.7  # 70% for training
    VAL_SPLIT = 0.15   # 15% for validation
    TEST_SPLIT = 0.15  # 15% for testing

    # LSTM architecture
    LSTM_UNITS = [128, 64, 32]  # Units for each LSTM layer
    DROPOUT_RATE = 0.2
    RECURRENT_DROPOUT_RATE = 0.1
    DENSE_UNITS = [32, 16]  # Dense layers before output
    ACTIVATION = 'relu'
    OUTPUT_ACTIVATION = 'linear'
    USE_BIDIRECTIONAL = False  # Use bidirectional LSTM
    USE_ATTENTION = True  # Use attention mechanism

    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    OPTIMIZER = 'adam'  # 'adam', 'rmsprop', 'sgd'
    LOSS = 'mse'  # 'mse', 'mae', 'huber'

    # Callbacks
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-7

    # Model checkpoint
    SAVE_BEST_ONLY = True
    MONITOR_METRIC = 'val_loss'

    # Target accuracy thresholds
    TARGET_MAE = None  # Will be computed from data
    TARGET_RMSE = None  # Will be computed from data
    TARGET_MAPE = 15.0  # 15% MAPE (85% accuracy)

    # Feature groups
    TEMPORAL_FEATURES = [
        'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
        'is_morning_peak', 'is_evening_peak', 'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos', 'rush_hour_intensity', 'is_holiday'
    ]

    SPATIAL_FEATURES = [
        'stop_order', 'route_progress', 'distance_to_next_stop'
    ]

    CONTEXTUAL_FEATURES = [
        'temperature', 'humidity', 'is_raining', 'heat_index',
        'nearby_events', 'event_impact_factor', 'is_commute_time'
    ]

    # Target variable
    TARGET_COLUMN = 'passengers'

    # Additional feature engineering for sequences
    USE_LAG_FEATURES = True
    LAG_STEPS = [1, 2, 3, 6]  # Additional lag features
    USE_ROLLING_FEATURES = True
    ROLLING_WINDOWS = [3, 6]  # Rolling window sizes

    # Prediction parameters
    CONFIDENCE_INTERVAL = 0.95  # 95% confidence interval
    ENSEMBLE_MODELS = 3  # Number of models for ensemble (if > 1)

    # Visualization
    PLOT_TRAINING_HISTORY = True
    PLOT_PREDICTIONS = True
    SAVE_PLOTS = True
    PLOTS_DIR = "results/forecasting/plots"

    # Logging
    LOG_LEVEL = "INFO"
    TENSORBOARD_LOG_DIR = "logs/tensorboard"

    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.FEATURES_DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.FORECASTING_MODELS_DIR, exist_ok=True)
        os.makedirs(cls.PLOTS_DIR, exist_ok=True)
        os.makedirs(cls.TENSORBOARD_LOG_DIR, exist_ok=True)

    @classmethod
    def get_feature_columns(cls):
        """Get all feature columns"""
        return (
            cls.TEMPORAL_FEATURES +
            cls.SPATIAL_FEATURES +
            cls.CONTEXTUAL_FEATURES
        )

    @classmethod
    def get_model_path(cls, model_name: str = "lstm_demand_forecaster"):
        """Get model save path"""
        return os.path.join(cls.FORECASTING_MODELS_DIR, f"{model_name}.keras")

    @classmethod
    def get_scaler_path(cls, scaler_name: str = "forecast_scaler"):
        """Get scaler save path"""
        return os.path.join(cls.FORECASTING_MODELS_DIR, f"{scaler_name}.pkl")
