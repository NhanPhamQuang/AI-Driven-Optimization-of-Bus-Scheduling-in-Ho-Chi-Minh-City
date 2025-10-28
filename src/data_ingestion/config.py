"""
Configuration for data ingestion
"""

import os
from typing import Dict, Any


class DataIngestionConfig:
    """Configuration for all data collectors"""

    # Base URLs
    XEBUYT_BASE_URL = "https://xebuyt.net"
    XEBUYT_API_BASE_URL = "https://api.xebuyt.net/businfo"
    XEBUYT_ROUTE_LIST_URL = "https://xebuyt.net/tuyen-xe-buyt"

    # API Configuration
    REQUEST_TIMEOUT = 10  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    # Rate limiting
    REQUEST_DELAY = 0.5  # seconds between requests

    # Weather API
    WEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
    WEATHER_API_URL = "https://api.openweathermap.org/data/2.5"
    WEATHER_CITY = "Ho Chi Minh City"
    WEATHER_COUNTRY_CODE = "VN"

    # Event scraping sources
    EVENT_SOURCES = {
        'vnexpress': 'https://vnexpress.net',
        'tuoitre': 'https://tuoitre.vn',
        'thanhnien': 'https://thanhnien.vn'
    }

    # Data directories
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    # User agent for web scraping
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        """Get default headers for HTTP requests"""
        return {
            'User-Agent': cls.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    @classmethod
    def ensure_data_dirs(cls):
        """Create data directories if they don't exist"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
