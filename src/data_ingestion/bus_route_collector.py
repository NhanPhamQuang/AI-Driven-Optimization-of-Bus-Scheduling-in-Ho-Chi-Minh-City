"""
Bus route data collector from xebuyt.net
"""

import json
from typing import List, Dict, Optional, Any
import pandas as pd
from bs4 import BeautifulSoup

from .base_collector import BaseCollector
from .config import DataIngestionConfig


class BusRouteCollector(BaseCollector):
    """Collector for bus route data from xebuyt.net website and API"""

    def __init__(self, config: Optional[DataIngestionConfig] = None):
        super().__init__(config)
        self.routes_cache = None

    def collect_route_list(self) -> pd.DataFrame:
        """
        Collect list of all bus routes from xebuyt.net

        Returns:
            DataFrame with columns: Route_Number, Route_Name, URL
        """
        self.logger.info("Collecting bus route list from xebuyt.net")

        response = self._make_request(self.config.XEBUYT_ROUTE_LIST_URL)
        if not response:
            self.logger.error("Failed to fetch route list")
            return pd.DataFrame()

        soup = BeautifulSoup(response.content, 'html.parser')
        div_result = soup.find('div', id='divResult')

        if not div_result:
            self.logger.error("Could not find route list container")
            return pd.DataFrame()

        route_links = div_result.find_all('a', class_='cms-button')
        routes = []

        for link in route_links:
            href = link.get('href')
            if not href:
                continue

            full_url = f"{self.config.XEBUYT_BASE_URL}{href}" if href.startswith('/') else href

            # Get route number from span.icon
            icon_span = link.find('span', class_='icon')
            route_number = icon_span.get_text(strip=True) if icon_span else ''

            # Get route name from div.routetrip
            routetrip_div = link.find('div', class_='routetrip')
            route_name = routetrip_div.get_text(strip=True) if routetrip_div else ''

            routes.append({
                'Route_Number': route_number,
                'Route_Name': route_name,
                'URL': full_url
            })

            self._rate_limit()

        df = pd.DataFrame(routes)
        self.routes_cache = df
        self.logger.info(f"Collected {len(df)} routes")

        return df

    def collect_route_details(self, route_url: str) -> Dict[str, Any]:
        """
        Collect detailed information for a specific route

        Args:
            route_url: URL of the route detail page

        Returns:
            Dictionary with route details
        """
        self.logger.debug(f"Collecting route details from {route_url}")

        response = self._make_request(route_url)
        if not response:
            return {}

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find route info table
        rou_info_div = soup.find('div', id='rouInfo')
        if not rou_info_div:
            self.logger.warning(f"No route info found for {route_url}")
            return {}

        table = rou_info_div.find('table', class_='tbl100')
        if not table:
            return {}

        details = {
            'Operator': '',
            'Operating_Type': '',
            'Distance': '',
            'Vehicle_Type': '',
            'Operating_Hours': '',
            'Ticket_Price': '',
            'Number_of_Trips': '',
            'Trip_Time': '',
            'Frequency': ''
        }

        # Parse table rows
        tbody = table.find('tbody')
        if tbody:
            rows = tbody.find_all('tr')
            for tr in rows:
                tds = tr.find_all('td')
                if len(tds) < 2:
                    continue

                label = tds[0].get_text(strip=True)
                content = tds[1].get_text(separator='\n').strip()

                if 'Đơn vị đảm nhận' in label:
                    details['Operator'] = content

                # Parse ul lists
                if len(tds) >= 3:
                    left_ul = tds[1].find('ul')
                    right_ul = tds[2].find('ul') if len(tds) > 2 else None

                    if left_ul:
                        for li in left_ul.find_all('li'):
                            item = li.get_text(strip=True)
                            if 'Loại hình hoạt động' in item:
                                details['Operating_Type'] = item.replace('Loại hình hoạt động:', '').strip()
                            elif 'Cự ly' in item:
                                details['Distance'] = item.replace('Cự ly:', '').strip()
                            elif 'Loại xe' in item:
                                details['Vehicle_Type'] = item.replace('Loại xe:', '').strip()
                            elif 'Thời gian hoạt động' in item:
                                details['Operating_Hours'] = item.replace('Thời gian hoạt động:', '').strip()

                    if right_ul:
                        for li in right_ul.find_all('li'):
                            item = li.get_text(separator='\n').strip()
                            if 'Giá vé' in item:
                                details['Ticket_Price'] = item.replace('Giá vé:', '').strip()
                            elif 'Số chuyến' in item:
                                details['Number_of_Trips'] = item.replace('Số chuyến:', '').strip()
                            elif 'Thời gian chuyến' in item:
                                details['Trip_Time'] = item.replace('Thời gian chuyến:', '').strip()
                            elif 'Giãn cách chuyến' in item:
                                details['Frequency'] = item.replace('Giãn cách chuyến:', '').strip()

        self._rate_limit()
        return details

    def collect_all_route_details(self) -> pd.DataFrame:
        """
        Collect detailed information for all routes

        Returns:
            DataFrame with detailed route information
        """
        if self.routes_cache is None:
            self.collect_route_list()

        self.logger.info("Collecting detailed information for all routes")

        all_details = []
        for _, route in self.routes_cache.iterrows():
            details = self.collect_route_details(route['URL'])
            all_details.append({
                'Route_Number': route['Route_Number'],
                'Route_Name': route['Route_Name'],
                'URL': route['URL'],
                **details
            })

        return pd.DataFrame(all_details)

    def collect_route_variants_api(self, route_id: int, direction: int = 1) -> List[Dict[str, Any]]:
        """
        Get route variants from API

        Args:
            route_id: Route ID (e.g., 1 for route 01)
            direction: Direction (1 for outbound, 2 for inbound)

        Returns:
            List of route variants
        """
        url = f"{self.config.XEBUYT_API_BASE_URL}/getvarsbyroute/{route_id}_{direction}"

        response = self._make_request(url)
        if not response:
            return []

        try:
            data = response.json()
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON response from {url}")
            return []

    def collect_stops_by_variant(self, route_id: int, direction: int, variant_id: int) -> List[Dict[str, Any]]:
        """
        Get stops for a specific route variant

        Args:
            route_id: Route ID
            direction: Direction (1 or 2)
            variant_id: Variant ID

        Returns:
            List of stops with details
        """
        url = f"{self.config.XEBUYT_API_BASE_URL}/getstopsbyvar/{route_id}_{direction}/{variant_id}"

        response = self._make_request(url)
        if not response:
            return []

        try:
            data = response.json()
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON response from {url}")
            return []

    def collect_path_by_variant(self, route_id: int, direction: int, variant_id: int) -> Dict[str, List[float]]:
        """
        Get geographic path for a specific route variant

        Args:
            route_id: Route ID
            direction: Direction (1 or 2)
            variant_id: Variant ID

        Returns:
            Dictionary with 'lat' and 'lng' arrays
        """
        url = f"{self.config.XEBUYT_API_BASE_URL}/getpathsbyvar/{route_id}_{direction}/{variant_id}"

        response = self._make_request(url)
        if not response:
            return {'lat': [], 'lng': []}

        try:
            data = response.json()
            if isinstance(data, dict):
                return data
            return {'lat': [], 'lng': []}
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON response from {url}")
            return {'lat': [], 'lng': []}

    def collect_complete_route_data(self, route_id: int) -> Dict[str, Any]:
        """
        Collect complete data for a route (variants, stops, paths)

        Args:
            route_id: Route ID

        Returns:
            Dictionary with complete route data
        """
        self.logger.info(f"Collecting complete data for route {route_id}")

        route_data = {
            'route_id': route_id,
            'variants': {},
            'stops': {},
            'paths': {}
        }

        # Collect both directions
        for direction in [1, 2]:
            direction_name = 'outbound' if direction == 1 else 'inbound'

            # Get variants
            variants = self.collect_route_variants_api(route_id, direction)
            route_data['variants'][direction_name] = variants

            # For each variant, get stops and paths
            for variant in variants:
                variant_id = variant.get('RouteVarId')
                if not variant_id:
                    continue

                key = f"{direction_name}_{variant_id}"

                # Get stops
                stops = self.collect_stops_by_variant(route_id, direction, variant_id)
                route_data['stops'][key] = stops

                # Get path
                path = self.collect_path_by_variant(route_id, direction, variant_id)
                route_data['paths'][key] = path

                self._rate_limit()

        return route_data

    def collect_all_routes_api(self, max_route_id: int = 200) -> Dict[int, Dict[str, Any]]:
        """
        Collect data for all routes using API

        Args:
            max_route_id: Maximum route ID to check

        Returns:
            Dictionary mapping route IDs to their data
        """
        self.logger.info(f"Collecting data for routes 1-{max_route_id}")

        all_routes = {}

        for route_id in range(1, max_route_id + 1):
            # First check if route exists
            variants = self.collect_route_variants_api(route_id, 1)
            if not variants:
                self.logger.debug(f"No data for route {route_id}")
                continue

            # Collect complete data
            route_data = self.collect_complete_route_data(route_id)
            all_routes[route_id] = route_data

            self.logger.info(f"Collected data for route {route_id}")

        self.logger.info(f"Collected data for {len(all_routes)} routes")
        return all_routes

    def collect(self, **kwargs) -> Dict[str, Any]:
        """
        Main collection method

        Args:
            include_details: Whether to collect detailed route info (default: True)
            include_api_data: Whether to collect API data (default: True)
            max_route_id: Maximum route ID for API collection (default: 200)

        Returns:
            Dictionary with all collected data
        """
        include_details = kwargs.get('include_details', True)
        include_api_data = kwargs.get('include_api_data', True)
        max_route_id = kwargs.get('max_route_id', 200)

        result = {}

        # Collect route list
        route_list = self.collect_route_list()
        result['route_list'] = route_list

        # Collect detailed info
        if include_details:
            route_details = self.collect_all_route_details()
            result['route_details'] = route_details

        # Collect API data
        if include_api_data:
            api_data = self.collect_all_routes_api(max_route_id)
            result['api_data'] = api_data

        return result
