# Data Preprocessing & Storage

Complete data preprocessing pipeline with cleaning, feature engineering, normalization, and database storage.

## Features

- **Data Cleaning & Validation**: Remove duplicates, handle missing values, validate data ranges
- **Feature Engineering**: Create temporal, spatial, and contextual features
- **Normalization**: MinMax, Standard, and Robust scaling with scaler persistence
- **Database Storage**: PostgreSQL/TimescaleDB support for time-series data
- **Pipeline Orchestration**: End-to-end preprocessing workflow

## Quick Start

### Basic Usage

```python
from src.preprocessing import PreprocessingPipeline
import pandas as pd

# Load raw data
demand_df = pd.read_csv('data/raw/synthetic_demand.csv')
stops_df = pd.read_csv('data/raw/synthetic_stops.csv')

# Create pipeline
pipeline = PreprocessingPipeline()

# Process demand data
processed_df = pipeline.process_demand_data(
    demand_df,
    stops_df=stops_df,
    normalize=True,
    save_to_db=False
)

print(f"Processed {len(processed_df)} records with {len(processed_df.columns)} features")
```

### Complete Pipeline

```python
from src.preprocessing import PreprocessingPipeline
import pandas as pd

# Load all data
routes_df = pd.read_csv('data/raw/synthetic_routes.csv')
stops_df = pd.read_csv('data/raw/synthetic_stops.csv')
demand_df = pd.read_csv('data/raw/synthetic_demand.csv')
gps_df = pd.read_csv('data/raw/synthetic_gps.csv')
weather_df = pd.read_csv('data/raw/synthetic_weather.csv')

# Create and run pipeline
pipeline = PreprocessingPipeline()

processed_data = pipeline.process_complete_dataset(
    routes_df=routes_df,
    stops_df=stops_df,
    demand_df=demand_df,
    gps_df=gps_df,
    weather_df=weather_df,
    normalize=True,
    save_to_files=True,
    save_to_db=False
)

# Access processed data
demand = processed_data['demand']
gps = processed_data['gps']
```

## Components

### 1. DataCleaner

Handles data validation and cleaning.

```python
from src.preprocessing import DataCleaner

cleaner = DataCleaner()

# Clean demand data
cleaned_demand = cleaner.clean_demand_data(demand_df)

# Clean GPS data
cleaned_gps = cleaner.clean_gps_data(gps_df)

# Clean weather data
cleaned_weather = cleaner.clean_weather_data(weather_df)

# Get cleaning statistics
cleaner.print_cleaning_report()
```

**Features:**
- Remove duplicates
- Validate value ranges (passenger counts, coordinates, speed, temperature, etc.)
- Handle missing values (forward fill, interpolation, dropping)
- Track cleaning statistics

### 2. FeatureEngineer

Creates engineered features from raw data.

```python
from src.preprocessing import FeatureEngineer

engineer = FeatureEngineer()

# Temporal features
df = engineer.create_temporal_features(df, time_col='time')
# Creates: hour, day_of_week, is_weekend, is_peak_hour,
#          hour_sin, hour_cos, rush_hour_intensity, etc.

# Spatial features
df = engineer.create_spatial_features(df, stops_df=stops_df)
# Creates: distance_from_center, route_progress,
#          distance_to_next_stop, etc.

# Contextual features
df = engineer.create_contextual_features(df, weather_df=weather_df, events_df=events_df)
# Creates: weather-derived features, event impact factors,
#          commute time indicators, etc.

# Lag features
df = engineer.create_lag_features(
    df,
    target_col='passengers',
    lags=[1, 2, 3, 6, 12],
    group_cols=['route', 'stop']
)

# Rolling window features
df = engineer.create_rolling_features(
    df,
    target_col='passengers',
    windows=[3, 6, 12],
    group_cols=['route', 'stop']
)
```

**Temporal Features:**
- Basic: hour, day_of_week, day_of_month, month, week_of_year, quarter
- Binary: is_weekend, is_peak_hour, is_morning_peak, is_evening_peak, is_holiday
- Cyclical: hour_sin, hour_cos, dow_sin, dow_cos
- Derived: time_since_midnight, rush_hour_intensity

**Spatial Features:**
- distance_from_center (Haversine distance from HCMC center)
- Quadrant indicators (is_north, is_east)
- Route progress (normalized position along route)
- distance_to_next_stop

**Contextual Features:**
- Weather: temperature, humidity, rain_mm, wind_speed, conditions
- Weather-derived: is_raining, rain_intensity, temp_category, heat_index
- Events: nearby_events, event_impact_factor
- Patterns: is_commute_time

### 3. DataNormalizer

Normalizes and encodes features.

```python
from src.preprocessing import DataNormalizer

normalizer = DataNormalizer()

# Normalize features
normalized_df = normalizer.normalize_features(
    df,
    numeric_cols=['passengers', 'temperature', 'distance'],
    method='minmax',  # or 'standard', 'robust'
    fit=True
)

# Encode categorical features
encoded_df = normalizer.encode_categorical(
    df,
    categorical_cols=['time_of_day', 'conditions'],
    fit=True
)

# One-hot encoding
one_hot_df = normalizer.one_hot_encode(
    df,
    categorical_cols=['time_of_day'],
    drop_first=True
)

# Normalize target variable
df = normalizer.normalize_target(df, target_col='passengers', method='minmax')

# Inverse normalization (for predictions)
predictions = normalizer.inverse_normalize_target(
    normalized_predictions,
    target_col='passengers',
    method='minmax'
)

# Save/load scalers
normalizer.save_scalers('models/')
normalizer.load_scalers('models/')
```

**Normalization Methods:**
- **MinMax**: Scales to [0, 1] range
- **Standard**: Z-score normalization (mean=0, std=1)
- **Robust**: Uses median and IQR (robust to outliers)

### 4. DatabaseManager

Manages PostgreSQL/TimescaleDB storage.

```python
from src.preprocessing import DatabaseManager

db = DatabaseManager()

# Connect to database
db.connect()

# Create tables (including TimescaleDB hypertables)
db.create_tables()

# Save DataFrame to database
db.save_dataframe(demand_df, 'passenger_demand', if_exists='append')

# Load data from database
recent_demand = db.load_dataframe(
    'passenger_demand',
    filters={'route_id': 'Tuyen_A'},
    time_range=(start_time, end_time),
    limit=1000
)

# Get recent data
last_24h = db.get_recent_data('passenger_demand', hours=24, route_id='Tuyen_A')

# Execute custom query
results = db.execute_query("SELECT * FROM passenger_demand WHERE hour >= 7 AND hour <= 9")

# Close connection
db.close()

# Or use context manager
with DatabaseManager() as db:
    db.save_dataframe(df, 'passenger_demand')
```

**Database Tables:**
- `routes`: Route information
- `stops`: Stop locations and metadata
- `passenger_demand`: Time-series demand data (hypertable)
- `gps_traces`: Time-series GPS data (hypertable)
- `weather_data`: Time-series weather data (hypertable)
- `events`: City events and impact factors
- `processed_features`: Preprocessed features (hypertable)
- `predictions`: Model predictions (hypertable)

## Configuration

Configure preprocessing via `PreprocessingConfig`:

```python
from src.preprocessing.config import PreprocessingConfig

config = PreprocessingConfig()

# Data directories
config.RAW_DATA_DIR = "data/raw"
config.PROCESSED_DATA_DIR = "data/processed"
config.FEATURES_DATA_DIR = "data/features"

# Database configuration (via environment variables)
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=bus_scheduling
# DB_USER=postgres
# DB_PASSWORD=your_password
# USE_TIMESCALE=true

# Validation thresholds
config.MAX_SPEED_KMH = 80
config.MAX_PASSENGERS = 200
config.MAX_TEMPERATURE_C = 45

# Ho Chi Minh City center coordinates
config.CITY_CENTER_LAT = 10.7769
config.CITY_CENTER_LNG = 106.7009

# Holidays
config.HOLIDAYS = ['2025-01-01', '2025-01-28', ...]
```

## Examples

Run the comprehensive examples:

```bash
# Run all examples
python examples/preprocessing_example.py

# Run specific example
python examples/preprocessing_example.py --example 1
```

### Available Examples

1. **Data Cleaning**: Basic cleaning and validation
2. **Feature Engineering**: Create temporal, spatial features
3. **Normalization**: Normalize and encode features
4. **Complete Pipeline**: End-to-end preprocessing
5. **Feature Importance**: Analyze feature correlations
6. **Lag and Rolling Features**: Time-series features

## Pipeline Workflow

The complete preprocessing pipeline follows these steps:

```
1. Data Cleaning
   ├─ Remove duplicates
   ├─ Validate ranges
   └─ Handle missing values

2. Feature Engineering
   ├─ Temporal features (hour, day_of_week, cyclical encoding)
   ├─ Spatial features (distances, route progress)
   ├─ Contextual features (weather, events)
   ├─ Lag features (time-series history)
   └─ Rolling features (moving averages)

3. Normalization
   ├─ Encode categorical variables
   ├─ Normalize numeric features
   └─ Save scalers for inference

4. Storage
   ├─ Save to CSV files (data/processed/)
   └─ Save to database (PostgreSQL/TimescaleDB)
```

## Output

### Processed Files

After running the pipeline, find processed data in:

```
data/processed/
├── processed_routes.csv
├── processed_stops.csv
├── processed_demand.csv     # Main dataset with all features
├── processed_gps.csv
├── processed_weather.csv
└── processed_events.csv
```

### Saved Models

Scalers and encoders are saved for inference:

```
models/
├── scalers.pkl              # Feature and target scalers
├── encoders.pkl             # Categorical encoders
└── feature_ranges.pkl       # Original value ranges
```

## Database Setup

### PostgreSQL + TimescaleDB

```bash
# Install PostgreSQL
# Then add TimescaleDB extension

# Connect to database
psql -U postgres

# Create database
CREATE DATABASE bus_scheduling;

# Connect to database
\c bus_scheduling

# Add TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

# Verify
SELECT * FROM pg_extension WHERE extname = 'timescaledb';
```

### Environment Variables

Create `.env` file:

```
USE_TIMESCALE=true
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bus_scheduling
DB_USER=postgres
DB_PASSWORD=your_password
```

## Features Created

### Temporal Features (18+)
- hour, day_of_week, day_of_month, month, week_of_year, quarter
- is_weekend, is_peak_hour, is_morning_peak, is_evening_peak, is_holiday
- hour_sin, hour_cos, dow_sin, dow_cos
- time_since_midnight, rush_hour_intensity
- time_of_day (categories: night, morning, afternoon, evening)

### Spatial Features (6+)
- distance_from_center
- is_north, is_east (quadrant indicators)
- stop_order, total_stops, route_progress
- distance_to_next_stop

### Contextual Features (10+)
- temperature, humidity, rain_mm, wind_speed, conditions
- is_raining, rain_intensity, temp_category, heat_index
- nearby_events, event_impact_factor, is_commute_time

### Time-Series Features (configurable)
- Lag features: passengers_lag_1, passengers_lag_2, ...
- Rolling mean: passengers_rolling_mean_3, passengers_rolling_mean_6, ...
- Rolling std: passengers_rolling_std_3, passengers_rolling_std_6, ...

## Best Practices

### 1. Data Cleaning First
Always clean data before feature engineering:
```python
cleaner = DataCleaner()
df = cleaner.clean_demand_data(raw_df)
```

### 2. Use Pipeline for Consistency
Use the complete pipeline to ensure consistent preprocessing:
```python
pipeline = PreprocessingPipeline()
processed_data = pipeline.process_complete_dataset(...)
```

### 3. Save Scalers for Inference
Always save scalers when training:
```python
normalizer.save_scalers('models/')
```

Load them for inference:
```python
normalizer.load_scalers('models/')
normalized_new_data = normalizer.normalize_features(new_data, fit=False)
```

### 4. Group Time-Series Features
When creating lag/rolling features, always group by identifiers:
```python
df = engineer.create_lag_features(
    df,
    target_col='passengers',
    group_cols=['route', 'stop']  # Important!
)
```

### 5. Handle Missing Values Appropriately
- Temporal data: Forward fill
- GPS traces: Linear interpolation (limited)
- Weather: Interpolation + forward/backward fill
- Events: Drop if critical fields missing

## Troubleshooting

### Issue: Database connection failed
**Solution**: Check database is running and environment variables are set correctly.

```python
# Test connection
from src.preprocessing import DatabaseManager
db = DatabaseManager()
db.connect()
```

### Issue: Missing values after normalization
**Solution**: Check for NaN/infinite values before normalization.

```python
# Check for issues
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])

# Handle before normalization
df = df.dropna()  # or use appropriate filling strategy
```

### Issue: Feature dimensions mismatch
**Solution**: Ensure consistent feature engineering and use saved scalers.

```python
# Training
normalizer.normalize_features(train_df, fit=True)
normalizer.save_scalers()

# Inference
normalizer.load_scalers()
normalizer.normalize_features(test_df, fit=False)  # Use existing scaler
```

## Performance

Typical processing times (on average hardware):

| Dataset Size | Records | Processing Time |
|--------------|---------|-----------------|
| Small        | 1K      | ~1s             |
| Medium       | 10K     | ~5s             |
| Large        | 100K    | ~30s            |
| Very Large   | 1M      | ~5min           |

## Next Steps

After preprocessing:

1. **Exploratory Data Analysis**: Analyze feature distributions and correlations
2. **Model Training**: Use processed data for LSTM demand forecasting
3. **Optimization**: Use features in ACO scheduling optimization
4. **Real-time Pipeline**: Set up automated preprocessing for new data

## See Also

- **Data Ingestion**: `src/data_ingestion/README.md`
- **Examples**: `examples/preprocessing_example.py`
- **Configuration**: `src/preprocessing/config.py`
