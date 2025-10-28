"""
Weather data collector from OpenWeatherMap API
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

from .base_collector import BaseCollector
from .config import DataIngestionConfig


class WeatherCollector(BaseCollector):
    """Collector for weather data from OpenWeatherMap API"""

    def __init__(self, config: Optional[DataIngestionConfig] = None):
        super().__init__(config)

        if not self.config.WEATHER_API_KEY:
            self.logger.warning(
                "No WEATHER_API_KEY found in environment. "
                "Set OPENWEATHER_API_KEY environment variable to use real weather data."
            )

    def collect_current_weather(self) -> Dict[str, Any]:
        """
        Collect current weather data

        Returns:
            Dictionary with current weather information
        """
        if not self.config.WEATHER_API_KEY:
            self.logger.warning("No API key, returning empty data")
            return {}

        url = f"{self.config.WEATHER_API_URL}/weather"
        params = {
            'q': f"{self.config.WEATHER_CITY},{self.config.WEATHER_COUNTRY_CODE}",
            'appid': self.config.WEATHER_API_KEY,
            'units': 'metric'  # Celsius
        }

        response = self._make_request(url, params=params)
        if not response:
            return {}

        try:
            data = response.json()

            # Extract relevant fields
            weather_data = {
                'time': datetime.now().isoformat(),
                'temperature': data.get('main', {}).get('temp'),
                'humidity': data.get('main', {}).get('humidity'),
                'pressure': data.get('main', {}).get('pressure'),
                'wind_speed': data.get('wind', {}).get('speed'),
                'conditions': data.get('weather', [{}])[0].get('main', ''),
                'description': data.get('weather', [{}])[0].get('description', ''),
                'rain_1h': data.get('rain', {}).get('1h', 0),
                'rain_3h': data.get('rain', {}).get('3h', 0)
            }

            self.logger.info(f"Collected current weather: {weather_data['temperature']}°C, {weather_data['conditions']}")
            return weather_data

        except (KeyError, ValueError) as e:
            self.logger.error(f"Error parsing weather data: {e}")
            return {}

    def collect_forecast(self, days: int = 5) -> List[Dict[str, Any]]:
        """
        Collect weather forecast

        Args:
            days: Number of days to forecast (max 5 for free tier)

        Returns:
            List of forecast data points
        """
        if not self.config.WEATHER_API_KEY:
            self.logger.warning("No API key, returning empty data")
            return []

        url = f"{self.config.WEATHER_API_URL}/forecast"
        params = {
            'q': f"{self.config.WEATHER_CITY},{self.config.WEATHER_COUNTRY_CODE}",
            'appid': self.config.WEATHER_API_KEY,
            'units': 'metric',
            'cnt': min(days * 8, 40)  # 3-hour intervals, max 40 (5 days)
        }

        response = self._make_request(url, params=params)
        if not response:
            return []

        try:
            data = response.json()
            forecasts = []

            for item in data.get('list', []):
                forecast = {
                    'time': item.get('dt_txt'),
                    'temperature': item.get('main', {}).get('temp'),
                    'humidity': item.get('main', {}).get('humidity'),
                    'pressure': item.get('main', {}).get('pressure'),
                    'wind_speed': item.get('wind', {}).get('speed'),
                    'conditions': item.get('weather', [{}])[0].get('main', ''),
                    'description': item.get('weather', [{}])[0].get('description', ''),
                    'rain_3h': item.get('rain', {}).get('3h', 0),
                    'clouds': item.get('clouds', {}).get('all', 0),
                    'pop': item.get('pop', 0)  # Probability of precipitation
                }
                forecasts.append(forecast)

            self.logger.info(f"Collected {len(forecasts)} forecast data points")
            return forecasts

        except (KeyError, ValueError) as e:
            self.logger.error(f"Error parsing forecast data: {e}")
            return []

    def generate_synthetic_weather(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Generate synthetic weather data for testing

        Args:
            start_time: Start datetime
            end_time: End datetime
            interval_minutes: Interval between data points in minutes

        Returns:
            DataFrame with synthetic weather data
        """
        import numpy as np

        self.logger.info(f"Generating synthetic weather from {start_time} to {end_time}")

        time_slots = []
        current = start_time
        while current <= end_time:
            time_slots.append(current)
            current += timedelta(minutes=interval_minutes)

        data = []
        for t in time_slots:
            hour = t.hour

            # Temperature varies by time of day (26-34°C)
            base_temp = 30
            temp_variation = 4 * np.sin((hour - 6) * np.pi / 12)  # Peak at 2 PM
            temp = base_temp + temp_variation + np.random.normal(0, 1)

            # Rain probability higher in afternoon
            rain_prob = 0.3 if 13 <= hour <= 18 else 0.1
            rain_mm = 0
            if np.random.random() < rain_prob:
                rain_mm = np.random.choice([0.5, 1, 2, 5], p=[0.5, 0.3, 0.15, 0.05])

            # Humidity inversely correlated with temperature
            humidity = 70 + (34 - temp) * 2 + np.random.normal(0, 5)
            humidity = max(40, min(100, humidity))

            data.append({
                'time': t.strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': round(temp, 1),
                'humidity': round(humidity, 1),
                'rain_mm': rain_mm,
                'wind_speed': round(np.random.uniform(2, 8), 1),
                'conditions': 'Rain' if rain_mm > 0 else 'Clear'
            })

        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} synthetic weather records")
        return df

    def collect(self, **kwargs) -> Dict[str, Any]:
        """
        Main collection method

        Args:
            mode: 'current', 'forecast', or 'synthetic' (default: 'current')
            start_time: Start time for synthetic data
            end_time: End time for synthetic data
            interval_minutes: Interval for synthetic data (default: 15)

        Returns:
            Dictionary with weather data
        """
        mode = kwargs.get('mode', 'current')

        if mode == 'current':
            return {'current': self.collect_current_weather()}

        elif mode == 'forecast':
            days = kwargs.get('days', 5)
            return {'forecast': self.collect_forecast(days)}

        elif mode == 'synthetic':
            start_time = kwargs.get('start_time', datetime.now())
            end_time = kwargs.get('end_time', datetime.now() + timedelta(days=1))
            interval_minutes = kwargs.get('interval_minutes', 15)

            df = self.generate_synthetic_weather(start_time, end_time, interval_minutes)
            return {'synthetic': df}

        else:
            self.logger.error(f"Unknown mode: {mode}")
            return {}
