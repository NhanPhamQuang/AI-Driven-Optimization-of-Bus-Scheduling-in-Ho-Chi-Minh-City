"""
Base collector class for data ingestion
"""

import time
import logging
from typing import Optional, Dict, Any
import requests
from abc import ABC, abstractmethod
from contextlib import contextmanager

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

    @contextmanager
    def _timer(self, operation_name: str):
        """
        Context manager for timing operations

        Args:
            operation_name: Name of the operation being timed

        Yields:
            Dictionary to store timing info
        """
        start_time = time.time()
        timing_info = {'start': start_time}

        self.logger.info(f"⏳ Starting: {operation_name}")

        try:
            yield timing_info
        finally:
            elapsed = time.time() - start_time
            timing_info['elapsed'] = elapsed
            self.logger.info(f"✅ Completed: {operation_name} (took {elapsed:.2f}s)")

    def _print_progress(self, current: int, total: int, prefix: str = "Progress"):
        """
        Print progress bar to console

        Args:
            current: Current item number
            total: Total number of items
            prefix: Prefix text for progress bar
        """
        bar_length = 40
        filled_length = int(bar_length * current / total)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        percent = 100 * (current / total)
        try:
            print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
        except UnicodeEncodeError:
            # Fallback for systems with encoding issues
            print(f'\r{prefix}: {percent:.1f}% ({current}/{total})', end='', flush=True)
        if current == total:
            print()  # New line when complete

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
