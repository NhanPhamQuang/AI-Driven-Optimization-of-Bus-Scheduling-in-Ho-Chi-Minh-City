"""
Database storage layer with TimescaleDB support
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, DateTime, MetaData, Index
from sqlalchemy.exc import SQLAlchemyError
import json

from .config import PreprocessingConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handles database operations with TimescaleDB support"""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.engine = None
        self.metadata = MetaData()
        self.use_timescale = self.config.USE_TIMESCALE

    def connect(self):
        """Establish database connection"""
        try:
            connection_string = self.config.get_db_connection_string()
            self.engine = create_engine(connection_string)

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            logger.info(f"Connected to database: {self.config.DB_NAME}")

            # Check if TimescaleDB is available
            if self.use_timescale:
                self._check_timescale_extension()

        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _check_timescale_extension(self):
        """Check if TimescaleDB extension is available"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'timescaledb'"))
                if result.fetchone():
                    logger.info("TimescaleDB extension detected")
                else:
                    logger.warning("TimescaleDB extension not found, using regular PostgreSQL")
                    self.use_timescale = False
        except SQLAlchemyError as e:
            logger.warning(f"Could not check TimescaleDB extension: {e}")
            self.use_timescale = False

    def create_tables(self):
        """Create necessary tables"""
        logger.info("Creating database tables")

        try:
            with self.engine.begin() as conn:
                # Routes table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS routes (
                        route_id VARCHAR(50) PRIMARY KEY,
                        route_name VARCHAR(200),
                        num_stops INTEGER,
                        num_buses INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                # Stops table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS stops (
                        stop_id VARCHAR(100) PRIMARY KEY,
                        route_id VARCHAR(50) REFERENCES routes(route_id),
                        stop_name VARCHAR(200),
                        stop_order INTEGER,
                        latitude DOUBLE PRECISION,
                        longitude DOUBLE PRECISION,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                # Passenger demand table (time-series)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS passenger_demand (
                        id SERIAL,
                        time TIMESTAMP NOT NULL,
                        route_id VARCHAR(50) REFERENCES routes(route_id),
                        stop_id VARCHAR(100) REFERENCES stops(stop_id),
                        boarding_count INTEGER,
                        alighting_count INTEGER,
                        passengers INTEGER,
                        hour INTEGER,
                        is_peak_hour INTEGER,
                        is_weekend INTEGER,
                        is_holiday INTEGER,
                        PRIMARY KEY (time, route_id, stop_id)
                    )
                """))

                # GPS traces table (time-series)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS gps_traces (
                        id SERIAL,
                        time TIMESTAMP NOT NULL,
                        route_id VARCHAR(50) REFERENCES routes(route_id),
                        bus_id VARCHAR(100),
                        latitude DOUBLE PRECISION,
                        longitude DOUBLE PRECISION,
                        speed DOUBLE PRECISION,
                        heading DOUBLE PRECISION,
                        PRIMARY KEY (time, bus_id)
                    )
                """))

                # Weather data table (time-series)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS weather_data (
                        time TIMESTAMP PRIMARY KEY,
                        temperature DOUBLE PRECISION,
                        humidity DOUBLE PRECISION,
                        rain_mm DOUBLE PRECISION,
                        wind_speed DOUBLE PRECISION,
                        conditions VARCHAR(100)
                    )
                """))

                # Events table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS events (
                        event_id VARCHAR(100) PRIMARY KEY,
                        event_name VARCHAR(200),
                        event_type VARCHAR(100),
                        location VARCHAR(200),
                        latitude DOUBLE PRECISION,
                        longitude DOUBLE PRECISION,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        expected_attendance INTEGER,
                        impact_factor DOUBLE PRECISION
                    )
                """))

                # Processed features table (time-series)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS processed_features (
                        id SERIAL,
                        time TIMESTAMP NOT NULL,
                        route_id VARCHAR(50),
                        stop_id VARCHAR(100),
                        features JSONB,
                        PRIMARY KEY (time, route_id, stop_id)
                    )
                """))

                # Model predictions table (time-series)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL,
                        time TIMESTAMP NOT NULL,
                        route_id VARCHAR(50),
                        stop_id VARCHAR(100),
                        predicted_demand INTEGER,
                        confidence_lower DOUBLE PRECISION,
                        confidence_upper DOUBLE PRECISION,
                        model_version VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (time, route_id, stop_id)
                    )
                """))

                logger.info("Tables created successfully")

                # Create TimescaleDB hypertables if available
                if self.use_timescale:
                    self._create_hypertables(conn)

                # Create indexes
                self._create_indexes(conn)

        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def _create_hypertables(self, conn):
        """Convert tables to TimescaleDB hypertables"""
        logger.info("Creating TimescaleDB hypertables")

        time_series_tables = [
            'passenger_demand',
            'gps_traces',
            'weather_data',
            'processed_features',
            'predictions'
        ]

        for table in time_series_tables:
            try:
                # Check if already a hypertable
                result = conn.execute(text(f"""
                    SELECT * FROM timescaledb_information.hypertables
                    WHERE hypertable_name = '{table}'
                """))

                if not result.fetchone():
                    conn.execute(text(f"""
                        SELECT create_hypertable('{table}', 'time', if_not_exists => TRUE)
                    """))
                    logger.info(f"Created hypertable: {table}")
            except SQLAlchemyError as e:
                logger.warning(f"Could not create hypertable for {table}: {e}")

    def _create_indexes(self, conn):
        """Create database indexes"""
        logger.info("Creating indexes")

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_demand_route ON passenger_demand(route_id)",
            "CREATE INDEX IF NOT EXISTS idx_demand_stop ON passenger_demand(stop_id)",
            "CREATE INDEX IF NOT EXISTS idx_demand_time ON passenger_demand(time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_gps_route ON gps_traces(route_id)",
            "CREATE INDEX IF NOT EXISTS idx_gps_bus ON gps_traces(bus_id)",
            "CREATE INDEX IF NOT EXISTS idx_gps_time ON gps_traces(time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_weather_time ON weather_data(time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_time ON events(start_time, end_time)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(time DESC)"
        ]

        for index_sql in indexes:
            try:
                conn.execute(text(index_sql))
            except SQLAlchemyError as e:
                logger.warning(f"Could not create index: {e}")

    def save_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = 'append'
    ) -> int:
        """
        Save DataFrame to database

        Args:
            df: DataFrame to save
            table_name: Target table name
            if_exists: 'append', 'replace', or 'fail'

        Returns:
            Number of rows saved
        """
        if self.engine is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        logger.info(f"Saving {len(df)} records to {table_name}")

        try:
            rows_saved = df.to_sql(
                table_name,
                self.engine,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )

            logger.info(f"Saved {len(df)} records to {table_name}")
            return len(df)

        except SQLAlchemyError as e:
            logger.error(f"Failed to save to {table_name}: {e}")
            raise

    def load_dataframe(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load data from database into DataFrame

        Args:
            table_name: Source table name
            filters: Dictionary of column:value filters
            time_range: Tuple of (start_time, end_time)
            columns: List of columns to select
            limit: Maximum number of rows

        Returns:
            DataFrame with loaded data
        """
        if self.engine is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Build query
        query_parts = []

        # SELECT clause
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"

        query_parts.append(f"SELECT {select_clause} FROM {table_name}")

        # WHERE clause
        where_conditions = []

        if filters:
            for col, val in filters.items():
                if isinstance(val, str):
                    where_conditions.append(f"{col} = '{val}'")
                else:
                    where_conditions.append(f"{col} = {val}")

        if time_range:
            start_time, end_time = time_range
            where_conditions.append(f"time >= '{start_time}'")
            where_conditions.append(f"time <= '{end_time}'")

        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))

        # ORDER BY
        query_parts.append("ORDER BY time DESC")

        # LIMIT
        if limit:
            query_parts.append(f"LIMIT {limit}")

        query = " ".join(query_parts)

        logger.info(f"Loading data from {table_name}")

        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} records from {table_name}")
            return df

        except SQLAlchemyError as e:
            logger.error(f"Failed to load from {table_name}: {e}")
            raise

    def save_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> int:
        """
        Save processed features to database

        Args:
            df: DataFrame with time, route_id, stop_id, and features
            feature_columns: List of feature column names

        Returns:
            Number of rows saved
        """
        logger.info(f"Saving {len(df)} feature records")

        # Convert features to JSONB format
        records = []
        for _, row in df.iterrows():
            features_dict = {}
            for col in feature_columns:
                if col in row:
                    val = row[col]
                    # Handle NaN values
                    if pd.isna(val):
                        features_dict[col] = None
                    else:
                        features_dict[col] = float(val) if isinstance(val, (np.integer, np.floating)) else val

            record = {
                'time': row['time'],
                'route_id': row.get('route', None),
                'stop_id': row.get('stop', None),
                'features': json.dumps(features_dict)
            }
            records.append(record)

        features_df = pd.DataFrame(records)
        return self.save_dataframe(features_df, 'processed_features')

    def save_predictions(
        self,
        predictions_df: pd.DataFrame
    ) -> int:
        """
        Save model predictions to database

        Args:
            predictions_df: DataFrame with predictions

        Returns:
            Number of rows saved
        """
        logger.info(f"Saving {len(predictions_df)} predictions")
        return self.save_dataframe(predictions_df, 'predictions')

    def get_recent_data(
        self,
        table_name: str,
        hours: int = 24,
        route_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get recent data from time-series table

        Args:
            table_name: Table name
            hours: Number of hours to look back
            route_id: Optional route filter

        Returns:
            DataFrame with recent data
        """
        query = f"""
            SELECT * FROM {table_name}
            WHERE time >= NOW() - INTERVAL '{hours} hours'
        """

        if route_id:
            query += f" AND route_id = '{route_id}'"

        query += " ORDER BY time DESC"

        logger.info(f"Loading recent {hours}h data from {table_name}")

        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} recent records")
            return df

        except SQLAlchemyError as e:
            logger.error(f"Failed to load recent data: {e}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute custom SQL query

        Args:
            query: SQL query string

        Returns:
            DataFrame with query results
        """
        logger.info("Executing custom query")

        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Query returned {len(df)} records")
            return df

        except SQLAlchemyError as e:
            logger.error(f"Query failed: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
