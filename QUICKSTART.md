# Quick Start Guide

Get started with the AI-Driven Bus Scheduling Optimization project in minutes.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/NhanPhamQuang/AI-Driven-Optimization-of-Bus-Scheduling-in-Ho-Chi-Minh-City.git
cd AI-Driven-Optimization-of-Bus-Scheduling-in-Ho-Chi-Minh-City
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **(Optional) Set up API keys**
```bash
# For real weather data
export OPENWEATHER_API_KEY="your_api_key_here"
```

## Quick Examples

### 1. Generate Synthetic Data (Recommended for Testing)

```bash
# Run the data ingestion pipeline with synthetic data
python -m src.data_ingestion.main --mode synthetic
```

This will generate:
- Synthetic bus routes and stops
- Passenger demand data
- GPS traces for buses
- Weather data
- Event data

Output files will be in `data/raw/`:
- `synthetic_routes.csv`
- `synthetic_stops.csv`
- `synthetic_demand.csv`
- `synthetic_gps.csv`
- `synthetic_weather.csv`
- `synthetic_events.csv`

### 2. Collect Real Bus Route Data

```bash
# Collect real data from xebuyt.net (requires internet)
python -m src.data_ingestion.main --mode real
```

This will collect:
- Bus route list
- Detailed route information
- Route variants, stops, and paths from API
- Current weather (if API key is set)

### 3. Use in Python Scripts

```python
# Import the data ingestion components
from src.data_ingestion import (
    BusRouteCollector,
    SyntheticDataGenerator
)

# Generate synthetic data
generator = SyntheticDataGenerator()
data = generator.generate_complete_dataset(
    num_routes=5,
    stops_per_route=5,
    buses_per_route=3
)

# Access the data
routes_df = data['routes']
demand_df = data['demand']
print(f"Generated {len(demand_df)} demand records")
```

### 4. Run Examples

```bash
# Run all examples (mostly synthetic data, no internet required)
python examples/data_ingestion_example.py
```

## Project Structure

```
AI-Driven-Optimization-of-Bus-Scheduling-in-Ho-Chi-Minh-City/
├── src/                          # Source code
│   └── data_ingestion/           # Data collection modules
│       ├── bus_route_collector.py
│       ├── weather_collector.py
│       ├── event_collector.py
│       ├── synthetic_data_generator.py
│       └── main.py
├── data/                         # Data storage
│   ├── raw/                      # Raw collected data
│   └── processed/                # Processed data
├── examples/                     # Example scripts
│   └── data_ingestion_example.py
├── requirements.txt              # Python dependencies
└── implement_plan.md             # Detailed implementation plan
```

## Next Steps

1. **Explore the Examples**: Run `python examples/data_ingestion_example.py` to see various usage patterns

2. **Read the Documentation**: Check `src/data_ingestion/README.md` for detailed API documentation

3. **Generate Your Own Data**: Customize the synthetic data generation parameters

4. **Build Models**: Use the generated data to develop LSTM forecasting and ACO optimization models

5. **Review the Implementation Plan**: See `implement_plan.md` for the full project roadmap

## Common Tasks

### Generate Data for a Specific Time Period

```python
from src.data_ingestion import SyntheticDataGenerator
from datetime import datetime, timedelta

generator = SyntheticDataGenerator()

start_time = datetime(2025, 1, 1, 6, 0, 0)
end_time = datetime(2025, 1, 1, 18, 0, 0)

data = generator.generate_complete_dataset(
    num_routes=5,
    stops_per_route=5,
    buses_per_route=3,
    start_time=start_time,
    end_time=end_time
)
```

### Collect Specific Route Data

```python
from src.data_ingestion import BusRouteCollector

collector = BusRouteCollector()

# Get complete data for route 1
route_data = collector.collect_complete_route_data(route_id=1)

# Access variants
variants = route_data['variants']

# Access stops
stops = route_data['stops']

# Access geographic paths
paths = route_data['paths']
```

### Generate Weather Data

```python
from src.data_ingestion import WeatherCollector
from datetime import datetime, timedelta

collector = WeatherCollector()

start = datetime.now()
end = start + timedelta(days=1)

weather_df = collector.generate_synthetic_weather(
    start_time=start,
    end_time=end,
    interval_minutes=15
)
```

## Troubleshooting

**Issue**: ModuleNotFoundError
**Solution**: Make sure you've installed all dependencies with `pip install -r requirements.txt`

**Issue**: No data generated
**Solution**: Check that the `data/raw/` directory exists and you have write permissions

**Issue**: API timeout errors
**Solution**: You may be rate-limited. Wait a few minutes and try again, or use synthetic data

**Issue**: Import errors
**Solution**: Make sure you're running from the project root directory

## Getting Help

- Check `src/data_ingestion/README.md` for detailed documentation
- Review `implement_plan.md` for project architecture
- Run examples with verbose logging: Set `logging.basicConfig(level=logging.DEBUG)`

## Contributing

See the main README for contribution guidelines.

## License

See LICENSE file for details.
