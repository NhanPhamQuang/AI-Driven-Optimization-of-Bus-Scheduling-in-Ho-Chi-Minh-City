# Data Ingestion Examples

This directory contains comprehensive examples for using the data ingestion layer.

## Quick Start

### Run All Offline Examples (Recommended)
```bash
python examples/data_ingestion_example.py
```

This will run examples 3-6 which don't require internet:
- **Example 3**: Generate Synthetic Data
- **Example 4**: Collect Weather Data
- **Example 5**: Collect Event Data
- **Example 6**: Full Pipeline with Data Saving ✅

### Run Specific Example
```bash
# Run only example 6 (generates and saves all data)
python examples/data_ingestion_example.py --example 6
```

### Include Internet-Dependent Examples
```bash
python examples/data_ingestion_example.py --include-web
```

This includes:
- **Example 1**: Collect Bus Routes (scrapes xebuyt.net)
- **Example 2**: Collect API Data (calls xebuyt.net API)

## Available Examples

### Example 1: Collect Bus Routes (Requires Internet)
Scrapes the xebuyt.net website to get a list of all bus routes.

```bash
python examples/data_ingestion_example.py --example 1
```

**What it does:**
- Fetches route list from xebuyt.net
- Extracts route numbers, names, and URLs
- Displays first 10 routes

**Output:**
```
Found 140 routes:
Route_Number  Route_Name                     URL
01            Bến Thành- BX Chợ Lớn         https://...
02            Bến Thành- BX Miền Tây        https://...
...
```

### Example 2: Collect API Data (Requires Internet)
Uses the xebuyt.net API to get detailed route information.

```bash
python examples/data_ingestion_example.py --example 2
```

**What it does:**
- Collects complete data for route 1
- Gets route variants (outbound/inbound)
- Fetches stops with coordinates
- Gets geographic path data

**Output:**
```
Route variants:
  outbound: 1 variants
    - Bến Thành - Bến Xe Chợ Lớn: 8600m, 35 mins

Stops:
  outbound_1: 25 stops

Paths:
  outbound_1: 150 coordinates
```

### Example 3: Generate Synthetic Data
Creates realistic synthetic passenger demand and GPS data.

```bash
python examples/data_ingestion_example.py --example 3
```

**What it does:**
- Generates 3 routes with 5 stops each
- Creates passenger demand data (4 hours, 15-min intervals)
- Generates GPS traces for 2 buses per route
- Shows progress bars and timing

**Output:**
```
Generated datasets:
  routes: 3 records
  stops: 15 records
  demand: 255 records
  gps: 306 records
```

### Example 4: Collect Weather Data
Generates synthetic weather data for testing.

```bash
python examples/data_ingestion_example.py --example 4
```

**What it does:**
- Generates 24 hours of weather data
- 60-minute intervals
- Includes temperature, humidity, rain, wind
- Realistic daily patterns (cooler morning, hot afternoon)

**Output:**
```
Generated 25 weather records:
                 time  temperature  humidity  rain_mm  wind_speed
2025-10-29 00:00:00         26.3      82.4      0.0         3.2
2025-10-29 01:00:00         25.8      84.1      0.0         2.9
...
```

### Example 5: Collect Event Data
Generates synthetic city events that affect transportation.

```bash
python examples/data_ingestion_example.py --example 5
```

**What it does:**
- Generates 15 events over 14 days
- Includes concerts, festivals, sports, conferences
- Assigns realistic locations in HCMC
- Calculates impact factors on transport

**Output:**
```
Generated 15 events:
                name event_type              location  expected_attendance
  Concert Event 1     concert    Nhà văn hóa TN             45,000
  Festival Event 2    festival   Công viên Tao Đàn         120,000
...

Event types distribution:
concert      5
festival     4
sports       3
...
```

### Example 6: Full Pipeline with Data Saving ⭐

**The most important example** - Runs the complete data generation pipeline and SAVES all data to files.

```bash
python examples/data_ingestion_example.py --example 6
```

**What it does:**
- Generates 5 routes with 5 stops each
- Creates 12 hours of passenger demand (6 AM - 6 PM)
- Generates GPS traces for 3 buses per route
- Creates weather and event data
- **SAVES everything to data/raw/ directory**

**Output:**
```
================================================================================
[SAVE] SAVING DATA TO FILES
================================================================================
  OK Saved 5 records to synthetic_routes.csv
  OK Saved 25 records to synthetic_stops.csv
  OK Saved 1,225 records to synthetic_demand.csv
  OK Saved 5,415 records to synthetic_gps.csv
  OK Saved 49 weather records
  OK Saved 10 events

[TIME] Data saved in 0.06s
[FOLDER] Location: data\raw
```

**Generated Files:**
```
data/raw/
├── synthetic_routes.csv      (5 records)
├── synthetic_stops.csv       (25 records)
├── synthetic_demand.csv      (1,225 records)
├── synthetic_gps.csv         (5,415 records)
├── synthetic_weather.csv     (49 records)
└── synthetic_events.csv      (10 records)
```

## Understanding the Data

### Routes (synthetic_routes.csv)
```csv
route,num_stops,num_buses
Tuyen_A,5,3
Tuyen_B,5,3
...
```

### Stops (synthetic_stops.csv)
```csv
route,stop,order,latitude,longitude
Tuyen_A,Tuyen_A_Stop_1,1,10.750234,106.650123
Tuyen_A,Tuyen_A_Stop_2,2,10.765123,106.675234
...
```

### Passenger Demand (synthetic_demand.csv)
```csv
route,stop,time,boarding_count,alighting_count,passengers,hour,is_peak_hour
Tuyen_A,Tuyen_A_Stop_1,2025-10-29 07:00:00,55,48,103,7,1
Tuyen_A,Tuyen_A_Stop_1,2025-10-29 07:15:00,62,51,113,7,1
...
```

Key fields:
- **boarding_count**: Passengers getting on
- **alighting_count**: Passengers getting off
- **passengers**: Total passenger movement
- **is_peak_hour**: 1 if 7-8 AM or 4-5 PM (peak hours)

### GPS Traces (synthetic_gps.csv)
```csv
route,bus_id,time,latitude,longitude,speed,heading
Tuyen_A,Tuyen_A_Bus_1,2025-10-29 06:00:00,10.750234,106.650123,25.3,180.5
Tuyen_A,Tuyen_A_Bus_1,2025-10-29 06:02:00,10.750456,106.650345,32.1,175.2
...
```

2-minute intervals showing real-time bus positions.

### Weather (synthetic_weather.csv)
```csv
time,temperature,humidity,rain_mm,wind_speed,conditions
2025-10-29 06:00:00,27.2,78.5,0.0,4.2,Clear
2025-10-29 06:15:00,27.8,76.3,0.0,4.5,Clear
2025-10-29 14:30:00,33.5,62.1,2.0,6.8,Rain
...
```

### Events (synthetic_events.csv)
```csv
event_id,name,event_type,location,start_time,expected_attendance,impact_factor
1,Concert Event 1,concert,Nhà văn hóa TN,2025-10-30 19:00:00,45000,1.9
2,Festival Event 2,festival,Công viên Tao Đàn,2025-11-02 10:00:00,95000,2.9
...
```

## Command Line Options

### Basic Usage
```bash
# Run default examples (3-6, no internet needed)
python examples/data_ingestion_example.py

# Run specific example
python examples/data_ingestion_example.py --example 6

# Include internet-dependent examples
python examples/data_ingestion_example.py --include-web
```

### Help
```bash
python examples/data_ingestion_example.py --help
```

## Next Steps

After generating data:

1. **Explore the data**:
   ```bash
   head data/raw/synthetic_demand.csv
   wc -l data/raw/*.csv
   ```

2. **Use in Jupyter notebooks**:
   ```python
   import pandas as pd

   demand = pd.read_csv('data/raw/synthetic_demand.csv')
   gps = pd.read_csv('data/raw/synthetic_gps.csv')
   ```

3. **Build LSTM models** using the demand data

4. **Implement ACO optimization** using routes and stops

## Troubleshooting

### No data files generated?
Make sure you're running **Example 6**:
```bash
python examples/data_ingestion_example.py --example 6
```

### Unicode errors on Windows?
The examples use `safe_print()` which handles encoding automatically. If you still see errors, try:
```bash
python examples/data_ingestion_example.py 2>&1 | more
```

### Want more/less data?
Edit the example functions in `data_ingestion_example.py`:
```python
# In example_6_full_pipeline()
bus_data = self.synthetic_generator.collect(
    num_routes=10,        # Increase routes
    stops_per_route=10,   # More stops
    buses_per_route=5,    # More buses
)
```

## Performance

Typical execution times (on average hardware):

| Example | Time | Data Generated |
|---------|------|----------------|
| 1 | 2-5s | None (display only) |
| 2 | 3-10s | None (display only) |
| 3 | 0.5s | ~600 records |
| 4 | 0.2s | 25 records |
| 5 | 0.2s | 15 records |
| 6 | 0.3s | **6,735 records** |

Example 6 is optimized for quick generation while producing enough data for meaningful analysis.

## See Also

- **Main CLI**: `python -m src.data_ingestion.main --mode synthetic`
- **API Documentation**: `src/data_ingestion/README.md`
- **Quick Start Guide**: `QUICKSTART.md`
