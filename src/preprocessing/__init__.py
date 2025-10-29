"""
Data Preprocessing & Storage Layer
"""

from .data_cleaner import DataCleaner
from .feature_engineering import FeatureEngineer
from .normalizer import DataNormalizer

# Optional database support (requires sqlalchemy and psycopg2)
try:
    from .database_manager import DatabaseManager
    __all__ = [
        'DataCleaner',
        'FeatureEngineer',
        'DataNormalizer',
        'DatabaseManager'
    ]
except ImportError:
    DatabaseManager = None
    __all__ = [
        'DataCleaner',
        'FeatureEngineer',
        'DataNormalizer'
    ]
