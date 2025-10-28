# Data Ingestion Layer

This module provides a comprehensive data ingestion layer for the AI-Driven Bus Scheduling Optimization project.

## Features

- **Bus Route Collection**: Scrapes and collects bus route data from xebuyt.net
- **Weather Data**: Integrates with OpenWeatherMap API (with synthetic fallback)
- **Event Data**: Collects events from news websites (with synthetic generation)
- **Synthetic Data Generation**: Creates realistic test data for development
- **Robust Error Handling**: Retry logic, rate limiting, and comprehensive logging

## Architecture

```
data_ingestion/
├── __init__.py                    # Package initialization
├── config.py                      # Configuration management
├── base_collector.py              # Base class for all collectors
├── bus_route_collector.py         # Bus route data collection
├── weather_collector.py           # Weather data collection
├── event_collector.py             # Event data collection
├── synthetic_data_generator.py    # Synthetic data generation
└── main.py                        # Main pipeline orchestration
```

## Quick Start

### Basic Usage

```python
from src.data_ingestion import BusRouteCollector

# Collect bus route data
collector = BusRouteCollector()
routes = collector.collect_route_list()
print(f"Found {len(routes)} routes")
```

### Full Pipeline

```python
from src.data_ingestion.main import DataIngestionPipeline

# Create pipeline
pipeline = DataIngestionPipeline()

# Collect synthetic data (for testing)
data = pipeline.collect_synthetic_data(save=True)

# Collect real data (requires internet)
# data = pipeline.collect_all_real_data(save=True)
```

### Command Line

```bash
# Generate synthetic data
python -m src.data_ingestion.main --mode synthetic

# Collect real data
python -m src.data_ingestion.main --mode real

# Collect both
python -m src.data_ingestion.main --mode both
```

## Components

### 1. BusRouteCollector

Collects bus route information from xebuyt.net website and API.

**Features:**
- Scrapes route list from website
- Extracts detailed route information
- Fetches route variants, stops, and paths from API
- Handles pagination and rate limiting

**Example:**
```python
from src.data_ingestion import BusRouteCollector

collector = BusRouteCollector()

# Get all routes
routes_df = collector.collect_route_list()

# Get complete data for a specific route
route_data = collector.collect_complete_route_data(route_id=1)
```

### 2. WeatherCollector

Collects weather data from OpenWeatherMap API with synthetic fallback.

**Features:**
- Current weather data
- Weather forecasts (up to 5 days)
- Synthetic weather generation for testing

**Example:**
```python
from src.data_ingestion import WeatherCollector
from datetime import datetime, timedelta

collector = WeatherCollector()

# Get current weather (requires API key)
current = collector.collect_current_weather()

# Generate synthetic weather
start = datetime.now()
end = start + timedelta(days=1)
weather_df = collector.generate_synthetic_weather(start, end)
```

**Setup:**
Set environment variable for real weather data:
```bash
export OPENWEATHER_API_KEY="your_api_key_here"
```

### 3. EventCollector

Collects event data from Vietnamese news websites.

**Features:**
- Web scraping from VnExpress, Tuoi Tre, Thanh Nien
- Synthetic event generation
- Event filtering by time range

**Example:**
```python
from src.data_ingestion import EventCollector
from datetime import datetime, timedelta

collector = EventCollector()

# Generate synthetic events
start = datetime.now()
end = start + timedelta(days=30)
events_df = collector.generate_synthetic_events(start, end, num_events=20)
```

### 4. SyntheticDataGenerator

Generates realistic synthetic data for testing and development.

**Features:**
- Passenger demand with peak/off-peak patterns
- GPS traces for buses
- Route and stop generation
- Realistic temporal patterns

**Example:**
```python
from src.data_ingestion import SyntheticDataGenerator
from datetime import datetime

generator = SyntheticDataGenerator()

# Generate complete dataset
data = generator.generate_complete_dataset(
    num_routes=5,
    stops_per_route=5,
    buses_per_route=3
)

# Access different components
routes_df = data['routes']
stops_df = data['stops']
demand_df = data['demand']
gps_df = data['gps']
```

## Configuration

All configuration is managed in `config.py`. Key settings:

```python
from src.data_ingestion.config import DataIngestionConfig

config = DataIngestionConfig()

# API endpoints
config.XEBUYT_API_BASE_URL
config.WEATHER_API_URL

# Rate limiting
config.REQUEST_TIMEOUT = 10  # seconds
config.MAX_RETRIES = 3
config.REQUEST_DELAY = 0.5

# Data directories
config.RAW_DATA_DIR
config.PROCESSED_DATA_DIR
```

## Data Output

Data is saved to the following directories:

```
data/
├── raw/                          # Raw collected data
│   ├── bus_routes.csv
│   ├── bus_route_details.csv
│   ├── bus_api_data.json
│   ├── weather_current.json
│   ├── events.csv
│   ├── synthetic_*.csv
│   └── ...
└── processed/                    # Processed data (for future use)
```

## Examples

See `examples/data_ingestion_example.py` for comprehensive usage examples:

```bash
python examples/data_ingestion_example.py
```

The example file includes:
1. Collecting bus routes
2. Using the xebuyt API
3. Generating synthetic data
4. Collecting weather data
5. Generating events
6. Running the full pipeline

## Error Handling

The module includes robust error handling:

- **Retry Logic**: Automatic retries for failed requests
- **Rate Limiting**: Prevents overwhelming external APIs
- **Logging**: Comprehensive logging at all levels
- **Graceful Degradation**: Falls back to synthetic data when real data unavailable

## Testing

```python
# Test individual collectors
from src.data_ingestion import BusRouteCollector

collector = BusRouteCollector()
routes = collector.collect_route_list()
assert len(routes) > 0

# Test with synthetic data (no internet required)
from src.data_ingestion import SyntheticDataGenerator

generator = SyntheticDataGenerator()
data = generator.generate_complete_dataset()
assert 'routes' in data
assert 'demand' in data
```

## API Reference

### Base Classes

**BaseCollector**
- `_make_request(url, method, params, json_data, retry_count)`: Make HTTP request with retry
- `_rate_limit()`: Apply rate limiting
- `collect(**kwargs)`: Abstract method to be implemented by subclasses

### Bus Route Collector Methods

- `collect_route_list()`: Get all bus routes
- `collect_route_details(route_url)`: Get details for a route
- `collect_route_variants_api(route_id, direction)`: Get route variants
- `collect_stops_by_variant(route_id, direction, variant_id)`: Get stops
- `collect_path_by_variant(route_id, direction, variant_id)`: Get geographic path
- `collect_complete_route_data(route_id)`: Get all data for a route

### Weather Collector Methods

- `collect_current_weather()`: Get current weather
- `collect_forecast(days)`: Get weather forecast
- `generate_synthetic_weather(start_time, end_time, interval_minutes)`: Generate synthetic data

### Event Collector Methods

- `scrape_vnexpress_events(max_pages)`: Scrape VnExpress
- `generate_synthetic_events(start_date, end_date, num_events)`: Generate synthetic events
- `get_events_for_timerange(start_time, end_time, events_df)`: Filter events

### Synthetic Data Generator Methods

- `generate_passenger_demand(routes, stops_per_route, start_time, end_time)`: Generate demand
- `generate_gps_traces(routes, stops_per_route, buses_per_route, start_time, end_time)`: Generate GPS
- `generate_complete_dataset(num_routes, stops_per_route, buses_per_route)`: Generate all data

## Best Practices

1. **Use Synthetic Data for Development**: Start with synthetic data to develop and test your models
2. **Set API Keys**: Configure API keys in environment variables, not in code
3. **Respect Rate Limits**: Use the built-in rate limiting to avoid overwhelming external services
4. **Save Intermediate Results**: Enable saving to avoid re-collecting data
5. **Monitor Logs**: Check logs for warnings and errors during collection

## Future Enhancements

- [ ] Implement real-time data streaming
- [ ] Add support for more weather APIs
- [ ] Complete event scraping implementation
- [ ] Add data validation and quality checks
- [ ] Implement incremental updates
- [ ] Add support for historical data collection
- [ ] Create data versioning system

## Troubleshooting

**Problem**: "No API key" warning for weather data
**Solution**: Set `OPENWEATHER_API_KEY` environment variable or use synthetic mode

**Problem**: HTTP timeout errors
**Solution**: Increase `REQUEST_TIMEOUT` in config or check internet connection

**Problem**: Rate limiting errors
**Solution**: Increase `REQUEST_DELAY` in config

**Problem**: No data returned
**Solution**: Check logs for specific errors, verify URLs are accessible

## License

See main project LICENSE file.
