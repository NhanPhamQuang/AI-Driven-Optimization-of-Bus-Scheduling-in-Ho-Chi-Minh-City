"""
Main data ingestion script
"""

import os
import json
from datetime import datetime, timedelta
import pandas as pd

from .config import DataIngestionConfig
from .bus_route_collector import BusRouteCollector
from .weather_collector import WeatherCollector
from .event_collector import EventCollector
from .synthetic_data_generator import SyntheticDataGenerator


class DataIngestionPipeline:
    """Main pipeline for collecting all data"""

    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()
        self.config.ensure_data_dirs()

        self.bus_collector = BusRouteCollector(self.config)
        self.weather_collector = WeatherCollector(self.config)
        self.event_collector = EventCollector(self.config)
        self.synthetic_generator = SyntheticDataGenerator(self.config)

    def collect_all_real_data(self, save: bool = True) -> dict:
        """
        Collect all real data from external sources

        Args:
            save: Whether to save data to files

        Returns:
            Dictionary with all collected data
        """
        print("=" * 80)
        print("Starting real data collection...")
        print("=" * 80)

        result = {}

        # 1. Collect bus route data
        print("\n[1/3] Collecting bus route data...")
        bus_data = self.bus_collector.collect(
            include_details=True,
            include_api_data=True,
            max_route_id=150  # Adjust based on needs
        )
        result['bus_data'] = bus_data

        if save:
            # Save route list
            if 'route_list' in bus_data:
                bus_data['route_list'].to_csv(
                    os.path.join(self.config.RAW_DATA_DIR, 'bus_routes.csv'),
                    index=False
                )
                print(f"  Saved {len(bus_data['route_list'])} routes to bus_routes.csv")

            # Save route details
            if 'route_details' in bus_data:
                bus_data['route_details'].to_csv(
                    os.path.join(self.config.RAW_DATA_DIR, 'bus_route_details.csv'),
                    index=False
                )
                print(f"  Saved route details to bus_route_details.csv")

            # Save API data as JSON
            if 'api_data' in bus_data:
                with open(os.path.join(self.config.RAW_DATA_DIR, 'bus_api_data.json'), 'w', encoding='utf-8') as f:
                    json.dump(bus_data['api_data'], f, ensure_ascii=False, indent=2)
                print(f"  Saved API data for {len(bus_data['api_data'])} routes")

        # 2. Collect weather data
        print("\n[2/3] Collecting weather data...")
        weather_data = self.weather_collector.collect(mode='current')
        result['weather_data'] = weather_data

        if save and 'current' in weather_data:
            with open(os.path.join(self.config.RAW_DATA_DIR, 'weather_current.json'), 'w') as f:
                json.dump(weather_data['current'], f, indent=2)
            print(f"  Saved current weather data")

        # 3. Collect event data (synthetic for now)
        print("\n[3/3] Collecting event data...")
        event_data = self.event_collector.collect(
            mode='synthetic',
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            num_events=20
        )
        result['event_data'] = event_data

        if save and 'synthetic' in event_data:
            event_data['synthetic'].to_csv(
                os.path.join(self.config.RAW_DATA_DIR, 'events.csv'),
                index=False
            )
            print(f"  Saved {len(event_data['synthetic'])} events to events.csv")

        print("\n" + "=" * 80)
        print("Real data collection complete!")
        print("=" * 80)

        return result

    def collect_synthetic_data(self, save: bool = True) -> dict:
        """
        Collect synthetic data for testing

        Args:
            save: Whether to save data to files

        Returns:
            Dictionary with all synthetic data
        """
        print("=" * 80)
        print("Starting synthetic data generation...")
        print("=" * 80)

        # Set time range (e.g., one day from 6 AM to 6 PM)
        start_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
        end_time = start_time.replace(hour=18, minute=0, second=0)

        # Generate complete dataset
        print("\n[1/3] Generating bus demand and GPS data...")
        bus_data = self.synthetic_generator.collect(
            num_routes=5,
            stops_per_route=5,
            buses_per_route=3,
            start_time=start_time,
            end_time=end_time
        )

        # Generate weather data
        print("\n[2/3] Generating weather data...")
        weather_data = self.weather_collector.collect(
            mode='synthetic',
            start_time=start_time,
            end_time=end_time,
            interval_minutes=15
        )

        # Generate event data
        print("\n[3/3] Generating event data...")
        event_data = self.event_collector.collect(
            mode='synthetic',
            start_date=start_time,
            end_date=start_time + timedelta(days=7),
            num_events=10
        )

        result = {
            'bus_data': bus_data,
            'weather_data': weather_data,
            'event_data': event_data
        }

        if save:
            # Save bus data
            for key, df in bus_data.items():
                filename = f"synthetic_{key}.csv"
                df.to_csv(
                    os.path.join(self.config.RAW_DATA_DIR, filename),
                    index=False
                )
                print(f"  Saved {len(df)} records to {filename}")

            # Save weather data
            if 'synthetic' in weather_data:
                weather_data['synthetic'].to_csv(
                    os.path.join(self.config.RAW_DATA_DIR, 'synthetic_weather.csv'),
                    index=False
                )
                print(f"  Saved weather data")

            # Save event data
            if 'synthetic' in event_data:
                event_data['synthetic'].to_csv(
                    os.path.join(self.config.RAW_DATA_DIR, 'synthetic_events.csv'),
                    index=False
                )
                print(f"  Saved event data")

        print("\n" + "=" * 80)
        print("Synthetic data generation complete!")
        print("=" * 80)

        return result


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Data ingestion for bus scheduling optimization')
    parser.add_argument(
        '--mode',
        choices=['real', 'synthetic', 'both'],
        default='synthetic',
        help='Data collection mode'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save data to files'
    )

    args = parser.parse_args()

    pipeline = DataIngestionPipeline()

    if args.mode == 'real' or args.mode == 'both':
        pipeline.collect_all_real_data(save=not args.no_save)

    if args.mode == 'synthetic' or args.mode == 'both':
        pipeline.collect_synthetic_data(save=not args.no_save)


if __name__ == '__main__':
    main()
