"""
Synthetic data generator for passenger demand and GPS traces
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from .base_collector import BaseCollector
from .config import DataIngestionConfig
from .utils import safe_print, EMOJI


class SyntheticDataGenerator(BaseCollector):
    """Generate synthetic passenger demand and GPS data"""

    def __init__(self, config: Optional[DataIngestionConfig] = None):
        super().__init__(config)
        np.random.seed(42)  # For reproducibility

    def generate_passenger_demand(
        self,
        routes: List[str],
        stops_per_route: Dict[str, List[str]],
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Generate synthetic passenger demand data

        Args:
            routes: List of route IDs
            stops_per_route: Dictionary mapping route IDs to list of stop names
            start_time: Start datetime
            end_time: End datetime
            interval_minutes: Time interval in minutes

        Returns:
            DataFrame with passenger demand data
        """
        with self._timer("Generating passenger demand data"):
            # Create time slots
            time_slots = []
            current = start_time
            while current <= end_time:
                time_slots.append(current)
                current += timedelta(minutes=interval_minutes)

            safe_print(f"{EMOJI['chart']} Generating demand for {len(routes)} routes, {len(time_slots)} time slots...")

            data = []
            total_iterations = len(routes)

            for route_idx, route in enumerate(routes, 1):
                stops = stops_per_route.get(route, [])
                if not stops:
                    continue

                for stop in stops:
                    for t in time_slots:
                        hour = t.hour
                        day_of_week = t.weekday()  # 0 = Monday

                        # Base demand varies by hour
                        if hour in [7, 8, 16, 17]:  # Peak hours
                            mean = 120
                            std = 30
                        elif hour in [6, 9, 15, 18]:  # Shoulder hours
                            mean = 80
                            std = 20
                        else:  # Off-peak
                            mean = 40
                            std = 15

                        # Weekend reduction
                        if day_of_week >= 5:  # Saturday or Sunday
                            mean *= 0.7

                        # Generate demand
                        passengers = max(0, int(np.random.normal(mean, std)))

                        # Boarding vs alighting (rough approximation)
                        boarding = int(passengers * np.random.uniform(0.4, 0.6))
                        alighting = passengers - boarding

                        data.append({
                            'route': route,
                            'stop': stop,
                            'time': t.strftime('%Y-%m-%d %H:%M:%S'),
                            'boarding_count': boarding,
                            'alighting_count': alighting,
                            'passengers': passengers,
                            'hour': hour,
                            'day_of_week': day_of_week,
                            'is_weekend': int(day_of_week >= 5),
                            'is_peak_hour': int(hour in [7, 8, 16, 17])
                        })

                self._print_progress(route_idx, total_iterations, "Generating demand")

            df = pd.DataFrame(data)
            return df

    def generate_gps_traces(
        self,
        routes: List[str],
        stops_per_route: Dict[str, List[Tuple[str, float, float]]],  # (stop_name, lat, lng)
        buses_per_route: int,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 2
    ) -> pd.DataFrame:
        """
        Generate synthetic GPS traces for buses

        Args:
            routes: List of route IDs
            stops_per_route: Dictionary mapping route IDs to list of (stop_name, lat, lng)
            buses_per_route: Number of buses per route
            start_time: Start datetime
            end_time: End datetime
            interval_minutes: GPS update interval in minutes

        Returns:
            DataFrame with GPS traces
        """
        with self._timer("Generating GPS traces"):
            # Create time slots
            time_slots = []
            current = start_time
            while current <= end_time:
                time_slots.append(current)
                current += timedelta(minutes=interval_minutes)

            safe_print(f"{EMOJI['location']} Generating GPS for {len(routes)} routes, {buses_per_route} buses each...")

            data = []
            total_iterations = len(routes)

            for route_idx, route in enumerate(routes, 1):
                stops = stops_per_route.get(route, [])
                if not stops:
                    continue

                for bus_id in range(1, buses_per_route + 1):
                    bus_identifier = f"{route}_Bus_{bus_id}"

                    for t in time_slots:
                        # Randomly select a stop (bus is somewhere on route)
                        stop_name, base_lat, base_lng = stops[np.random.randint(0, len(stops))]

                        # Add small random variation to simulate movement
                        lat = base_lat + np.random.uniform(-0.003, 0.003)
                        lng = base_lng + np.random.uniform(-0.003, 0.003)

                        # Speed varies (0-60 km/h typical for city bus)
                        speed = np.random.uniform(0, 60)

                        # Heading (0-360 degrees)
                        heading = np.random.uniform(0, 360)

                        data.append({
                            'route': route,
                            'bus_id': bus_identifier,
                            'time': t.strftime('%Y-%m-%d %H:%M:%S'),
                            'latitude': round(lat, 6),
                            'longitude': round(lng, 6),
                            'speed': round(speed, 1),
                            'heading': round(heading, 1)
                        })

                self._print_progress(route_idx, total_iterations, "Generating GPS")

            df = pd.DataFrame(data)
            return df

    def generate_route_stops(
        self,
        num_routes: int = 5,
        stops_per_route: int = 5,
        base_lat: float = 10.75,
        base_lng: float = 106.65
    ) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[Tuple[str, float, float]]]]:
        """
        Generate synthetic route and stop data

        Args:
            num_routes: Number of routes to generate
            stops_per_route: Number of stops per route
            base_lat: Base latitude (Ho Chi Minh City)
            base_lng: Base longitude

        Returns:
            Tuple of (route_list, stops_dict, stops_with_coords_dict)
        """
        self.logger.info(f"Generating {num_routes} routes with {stops_per_route} stops each")

        route_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        routes = [f"Tuyen_{route_letters[i]}" for i in range(num_routes)]

        stops_dict = {}
        stops_with_coords = {}

        for route in routes:
            stops = []
            stops_coords = []

            for i in range(stops_per_route):
                stop_name = f"{route}_Stop_{i+1}"

                # Generate coordinates along a rough line
                lat = base_lat + 0.015 * i + np.random.uniform(-0.005, 0.005)
                lng = base_lng + 0.025 * i + np.random.uniform(-0.005, 0.005)

                stops.append(stop_name)
                stops_coords.append((stop_name, round(lat, 6), round(lng, 6)))

            stops_dict[route] = stops
            stops_with_coords[route] = stops_coords

        return routes, stops_dict, stops_with_coords

    def generate_complete_dataset(
        self,
        num_routes: int = 5,
        stops_per_route: int = 5,
        buses_per_route: int = 3,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        demand_interval: int = 15,
        gps_interval: int = 2
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a complete synthetic dataset

        Args:
            num_routes: Number of routes
            stops_per_route: Number of stops per route
            buses_per_route: Number of buses per route
            start_time: Start datetime (default: today 6 AM)
            end_time: End datetime (default: today 6 PM)
            demand_interval: Passenger demand interval in minutes
            gps_interval: GPS trace interval in minutes

        Returns:
            Dictionary with DataFrames: 'routes', 'stops', 'demand', 'gps'
        """
        if start_time is None:
            start_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
        if end_time is None:
            end_time = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)

        self.logger.info("Generating complete synthetic dataset")

        # Generate routes and stops
        routes, stops_dict, stops_with_coords = self.generate_route_stops(
            num_routes, stops_per_route
        )

        # Create route DataFrame
        route_data = []
        for route in routes:
            route_data.append({
                'route': route,
                'num_stops': len(stops_dict[route]),
                'num_buses': buses_per_route
            })
        routes_df = pd.DataFrame(route_data)

        # Create stops DataFrame
        stops_data = []
        for route, stops in stops_with_coords.items():
            for i, (stop_name, lat, lng) in enumerate(stops):
                stops_data.append({
                    'route': route,
                    'stop': stop_name,
                    'order': i + 1,
                    'latitude': lat,
                    'longitude': lng
                })
        stops_df = pd.DataFrame(stops_data)

        # Generate passenger demand
        demand_df = self.generate_passenger_demand(
            routes, stops_dict, start_time, end_time, demand_interval
        )

        # Generate GPS traces
        gps_df = self.generate_gps_traces(
            routes, stops_with_coords, buses_per_route,
            start_time, end_time, gps_interval
        )

        self.logger.info("Complete dataset generated")

        return {
            'routes': routes_df,
            'stops': stops_df,
            'demand': demand_df,
            'gps': gps_df
        }

    def collect(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Main collection method

        Args:
            num_routes: Number of routes (default: 5)
            stops_per_route: Stops per route (default: 5)
            buses_per_route: Buses per route (default: 3)
            start_time: Start datetime
            end_time: End datetime

        Returns:
            Dictionary with generated DataFrames
        """
        return self.generate_complete_dataset(**kwargs)
