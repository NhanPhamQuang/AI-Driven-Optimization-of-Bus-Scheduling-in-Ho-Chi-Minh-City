"""
Example usage of data ingestion layer
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion import (
    BusRouteCollector,
    WeatherCollector,
    EventCollector,
    SyntheticDataGenerator
)
from src.data_ingestion.main import DataIngestionPipeline


def example_1_collect_bus_routes():
    """Example 1: Collect bus route information"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Collect Bus Route Information")
    print("=" * 80)

    collector = BusRouteCollector()

    # Get list of all routes
    print("\nFetching route list...")
    routes_df = collector.collect_route_list()
    print(f"\nFound {len(routes_df)} routes:")
    print(routes_df.head(10))

    # Get details for first route
    if len(routes_df) > 0:
        first_route_url = routes_df.iloc[0]['URL']
        print(f"\nFetching details for first route: {first_route_url}")
        details = collector.collect_route_details(first_route_url)
        print("\nRoute details:")
        for key, value in details.items():
            print(f"  {key}: {value}")


def example_2_collect_api_data():
    """Example 2: Collect data from xebuyt API"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Collect Data from xebuyt API")
    print("=" * 80)

    collector = BusRouteCollector()

    # Collect data for route 1
    route_id = 1
    print(f"\nCollecting complete data for route {route_id}...")

    route_data = collector.collect_complete_route_data(route_id)

    print("\nRoute variants:")
    for direction, variants in route_data['variants'].items():
        print(f"  {direction}: {len(variants)} variants")
        for v in variants:
            print(f"    - {v.get('RouteVarName')}: {v.get('Distance')}m, {v.get('RunningTime')} mins")

    print("\nStops:")
    for key, stops in route_data['stops'].items():
        print(f"  {key}: {len(stops)} stops")

    print("\nPaths:")
    for key, path in route_data['paths'].items():
        print(f"  {key}: {len(path.get('lat', []))} coordinates")


def example_3_generate_synthetic_data():
    """Example 3: Generate synthetic data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Generate Synthetic Data")
    print("=" * 80)

    generator = SyntheticDataGenerator()

    # Generate complete dataset
    print("\nGenerating synthetic dataset...")
    data = generator.generate_complete_dataset(
        num_routes=3,
        stops_per_route=5,
        buses_per_route=2,
        start_time=datetime.now().replace(hour=6, minute=0, second=0, microsecond=0),
        end_time=datetime.now().replace(hour=10, minute=0, second=0, microsecond=0),
        demand_interval=15,
        gps_interval=5
    )

    print("\nGenerated datasets:")
    for name, df in data.items():
        print(f"  {name}: {len(df)} records")
        print(f"    Columns: {list(df.columns)}")
        print(f"    Sample:\n{df.head(3)}\n")


def example_4_collect_weather():
    """Example 4: Collect weather data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Collect Weather Data")
    print("=" * 80)

    collector = WeatherCollector()

    # Generate synthetic weather
    print("\nGenerating synthetic weather data...")
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    weather_df = collector.generate_synthetic_weather(start, end, interval_minutes=60)

    print(f"\nGenerated {len(weather_df)} weather records:")
    print(weather_df.head(10))

    print("\nStatistics:")
    print(weather_df.describe())


def example_5_collect_events():
    """Example 5: Collect event data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Collect Event Data")
    print("=" * 80)

    collector = EventCollector()

    # Generate synthetic events
    print("\nGenerating synthetic events...")
    start_date = datetime.now()
    end_date = start_date + timedelta(days=14)

    events_df = collector.generate_synthetic_events(start_date, end_date, num_events=15)

    print(f"\nGenerated {len(events_df)} events:")
    print(events_df[['name', 'event_type', 'location', 'start_time', 'expected_attendance']])

    print("\nEvent types distribution:")
    print(events_df['event_type'].value_counts())


def example_6_full_pipeline():
    """Example 6: Run full data ingestion pipeline"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Full Data Ingestion Pipeline")
    print("=" * 80)

    pipeline = DataIngestionPipeline()

    # Collect synthetic data (faster for demo)
    print("\nRunning synthetic data collection pipeline...")
    result = pipeline.collect_synthetic_data(save=False)

    print("\n\nPipeline results:")
    print("=" * 80)

    # Bus data
    if 'bus_data' in result:
        print("\nBus Data:")
        for key, df in result['bus_data'].items():
            print(f"  {key}: {len(df)} records")

    # Weather data
    if 'weather_data' in result and 'synthetic' in result['weather_data']:
        print(f"\nWeather Data: {len(result['weather_data']['synthetic'])} records")

    # Event data
    if 'event_data' in result and 'synthetic' in result['event_data']:
        print(f"\nEvent Data: {len(result['event_data']['synthetic'])} events")


def main():
    """Run all examples"""
    examples = [
        # example_1_collect_bus_routes,  # Uncomment to run (requires internet)
        # example_2_collect_api_data,     # Uncomment to run (requires internet)
        example_3_generate_synthetic_data,
        example_4_collect_weather,
        example_5_collect_events,
        example_6_full_pipeline,
    ]

    print("\n" + "=" * 80)
    print("DATA INGESTION LAYER EXAMPLES")
    print("=" * 80)
    print("\nNote: Examples 1-2 are commented out by default as they require internet.")
    print("Uncomment them in the code to run.")

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\n[ERROR in example {i}]: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
