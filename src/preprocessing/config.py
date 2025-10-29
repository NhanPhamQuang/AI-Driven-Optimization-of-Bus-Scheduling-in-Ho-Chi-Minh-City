"""
Configuration for data preprocessing
"""

import os


class PreprocessingConfig:
    """Configuration for preprocessing operations"""

    # Data directories
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    FEATURES_DATA_DIR = "data/features"
    MODELS_DIR = "models"

    # Database configuration
    USE_TIMESCALE = os.environ.get('USE_TIMESCALE', 'false').lower() == 'true'
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = int(os.environ.get('DB_PORT', 5432))
    DB_NAME = os.environ.get('DB_NAME', 'bus_scheduling')
    DB_USER = os.environ.get('DB_USER', 'postgres')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')

    # Data validation thresholds
    MAX_SPEED_KMH = 80  # Maximum realistic bus speed
    MIN_SPEED_KMH = 0
    MAX_PASSENGERS = 200  # Maximum bus capacity
    MIN_PASSENGERS = 0
    MAX_TEMPERATURE_C = 45
    MIN_TEMPERATURE_C = 15
    MAX_HUMIDITY = 100
    MIN_HUMIDITY = 0

    # Feature engineering parameters
    TEMPORAL_FEATURES = [
        'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
        'is_morning_peak', 'is_evening_peak', 'time_since_midnight',
        'day_of_month', 'week_of_year', 'is_holiday'
    ]

    SPATIAL_FEATURES = [
        'latitude', 'longitude', 'distance_from_center',
        'distance_to_next_stop', 'stop_order', 'total_stops'
    ]

    CONTEXTUAL_FEATURES = [
        'temperature', 'humidity', 'rain_mm', 'wind_speed',
        'nearby_events', 'event_impact_factor'
    ]

    # Normalization
    NORMALIZATION_METHOD = 'minmax'  # 'minmax', 'standard', 'robust'
    SAVE_SCALERS = True

    # Ho Chi Minh City center coordinates
    CITY_CENTER_LAT = 10.7769
    CITY_CENTER_LNG = 106.7009

    # Vietnamese holidays (simplified - 2025)
    HOLIDAYS = [
        '2025-01-01',  # New Year
        '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31',  # Tet
        '2025-02-01', '2025-02-02', '2025-02-03', '2025-02-04',
        '2025-04-30',  # Reunification Day
        '2025-05-01',  # Labor Day
        '2025-09-02',  # National Day
    ]

    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.FEATURES_DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)

    @classmethod
    def get_db_connection_string(cls):
        """Get database connection string"""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
