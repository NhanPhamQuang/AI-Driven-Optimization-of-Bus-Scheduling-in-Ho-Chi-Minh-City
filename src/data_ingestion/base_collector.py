"""
Base collector class for data ingestion
"""

import time
import logging
from typing import Optional, Dict, Any
import requests
from abc import ABC, abstractmethod

from .config import DataIngestionConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BaseCollector(ABC):
    """Base class for all data collectors"""

    def __init__(self, config: Optional[DataIngestionConfig] = None):
        """
        Initialize base collector

        Args:
            config: Configuration object. If None, uses default config
        """
        self.config = config or DataIngestionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()
        self.session.headers.update(self.config.get_headers())

    def _make_request(
        self,
        url: str,
        method: str = 'GET',
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic

        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            json_data: JSON data for POST requests
            retry_count: Current retry attempt

        Returns:
            Response object or None if failed
        """
        try:
            self.logger.debug(f"Making {method} request to {url}")

            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                timeout=self.config.REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                return response
            else:
                self.logger.warning(f"Request failed with status {response.status_code}: {url}")

                # Retry for certain status codes
                if retry_count < self.config.MAX_RETRIES and response.status_code in [429, 500, 502, 503, 504]:
                    time.sleep(self.config.RETRY_DELAY * (retry_count + 1))
                    return self._make_request(url, method, params, json_data, retry_count + 1)

                return None

        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout: {url}")
            if retry_count < self.config.MAX_RETRIES:
                time.sleep(self.config.RETRY_DELAY * (retry_count + 1))
                return self._make_request(url, method, params, json_data, retry_count + 1)
            return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {e}")
            if retry_count < self.config.MAX_RETRIES:
                time.sleep(self.config.RETRY_DELAY * (retry_count + 1))
                return self._make_request(url, method, params, json_data, retry_count + 1)
            return None

    def _rate_limit(self):
        """Apply rate limiting between requests"""
        time.sleep(self.config.REQUEST_DELAY)

    @abstractmethod
    def collect(self, **kwargs) -> Any:
        """
        Collect data from source

        This method should be implemented by subclasses
        """
        pass

    def __del__(self):
        """Clean up session on deletion"""
        if hasattr(self, 'session'):
            self.session.close()
