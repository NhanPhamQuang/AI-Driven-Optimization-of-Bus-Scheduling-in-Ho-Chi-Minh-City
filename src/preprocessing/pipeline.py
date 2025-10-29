"""
Preprocessing pipeline orchestrator
"""

import logging
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

from .config import PreprocessingConfig
from .data_cleaner import DataCleaner
from .feature_engineering import FeatureEngineer
from .normalizer import DataNormalizer
from .database_manager import DatabaseManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Orchestrates the complete preprocessing workflow"""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.config.ensure_dirs()

        # Initialize components
        self.cleaner = DataCleaner(config)
        self.feature_engineer = FeatureEngineer(config)
        self.normalizer = DataNormalizer(config)
        self.db_manager = DatabaseManager(config)

        self.pipeline_stats = {}

    def process_demand_data(
        self,
        demand_df: pd.DataFrame,
        stops_df: Optional[pd.DataFrame] = None,
        weather_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        normalize: bool = True,
        save_to_db: bool = False
    ) -> pd.DataFrame:
        """
        Process passenger demand data through complete pipeline

        Args:
            demand_df: Raw demand DataFrame
            stops_df: Stops information
            weather_df: Weather data
            events_df: Events data
            normalize: Whether to normalize features
            save_to_db: Whether to save to database

        Returns:
            Processed DataFrame with features
        """
        logger.info("=" * 80)
        logger.info("PROCESSING DEMAND DATA")
        logger.info("=" * 80)

        start_time = time.time()

        # Step 1: Clean data
        logger.info("[1/5] Cleaning data...")
        step_start = time.time()
        df = self.cleaner.clean_demand_data(demand_df)
        logger.info(f"[TIME] Cleaning completed in {time.time() - step_start:.2f}s")

        # Step 2: Create temporal features
        logger.info("[2/5] Creating temporal features...")
        step_start = time.time()
        df = self.feature_engineer.create_temporal_features(df, time_col='time')
        logger.info(f"[TIME] Temporal features completed in {time.time() - step_start:.2f}s")

        # Step 3: Create spatial features
        if stops_df is not None:
            logger.info("[3/5] Creating spatial features...")
            step_start = time.time()
            df = self.feature_engineer.create_spatial_features(df, stops_df)
            logger.info(f"[TIME] Spatial features completed in {time.time() - step_start:.2f}s")
        else:
            logger.info("[3/5] Skipping spatial features (no stops data)")

        # Step 4: Create contextual features
        if weather_df is not None or events_df is not None:
            logger.info("[4/5] Creating contextual features...")
            step_start = time.time()
            df = self.feature_engineer.create_contextual_features(df, weather_df, events_df)
            logger.info(f"[TIME] Contextual features completed in {time.time() - step_start:.2f}s")
        else:
            logger.info("[4/5] Skipping contextual features (no weather/events data)")

        # Step 5: Normalize
        if normalize:
            logger.info("[5/5] Normalizing features...")
            step_start = time.time()

            # Identify feature columns (exclude identifiers and targets)
            exclude_cols = ['time', 'route', 'stop', 'boarding_count', 'alighting_count', 'passengers']
            feature_cols = [c for c in df.columns if c not in exclude_cols]

            # Separate numeric and categorical
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

            # Encode categorical
            if categorical_cols:
                df = self.normalizer.encode_categorical(df, categorical_cols, fit=True)

            # Normalize numeric
            if numeric_cols:
                df = self.normalizer.normalize_features(
                    df,
                    numeric_cols,
                    method=self.config.NORMALIZATION_METHOD,
                    fit=True
                )

            # Save scalers
            if self.config.SAVE_SCALERS:
                self.normalizer.save_scalers()

            logger.info(f"[TIME] Normalization completed in {time.time() - step_start:.2f}s")
        else:
            logger.info("[5/5] Skipping normalization")

        # Save to database
        if save_to_db:
            logger.info("Saving to database...")
            try:
                self.db_manager.connect()
                self.db_manager.save_dataframe(df, 'passenger_demand', if_exists='append')
                self.db_manager.close()
            except Exception as e:
                logger.error(f"Failed to save to database: {e}")

        total_time = time.time() - start_time

        # Record stats
        self.pipeline_stats['demand'] = {
            'records_processed': len(df),
            'features_created': len(df.columns),
            'time_taken': total_time
        }

        logger.info("=" * 80)
        logger.info(f"COMPLETED: Processed {len(df)} records in {total_time:.2f}s")
        logger.info(f"Features: {len(df.columns)} columns")
        logger.info("=" * 80)

        return df

    def process_gps_data(
        self,
        gps_df: pd.DataFrame,
        normalize: bool = True,
        save_to_db: bool = False
    ) -> pd.DataFrame:
        """
        Process GPS trace data

        Args:
            gps_df: Raw GPS DataFrame
            normalize: Whether to normalize features
            save_to_db: Whether to save to database

        Returns:
            Processed DataFrame
        """
        logger.info("=" * 80)
        logger.info("PROCESSING GPS DATA")
        logger.info("=" * 80)

        start_time = time.time()

        # Clean data
        logger.info("[1/3] Cleaning data...")
        df = self.cleaner.clean_gps_data(gps_df)

        # Create temporal features
        logger.info("[2/3] Creating temporal features...")
        df = self.feature_engineer.create_temporal_features(df, time_col='time')

        # Normalize
        if normalize:
            logger.info("[3/3] Normalizing features...")
            numeric_cols = ['latitude', 'longitude', 'speed', 'heading']
            numeric_cols = [c for c in numeric_cols if c in df.columns]

            if numeric_cols:
                df = self.normalizer.normalize_features(
                    df,
                    numeric_cols,
                    method=self.config.NORMALIZATION_METHOD,
                    fit=True
                )
        else:
            logger.info("[3/3] Skipping normalization")

        # Save to database
        if save_to_db:
            try:
                self.db_manager.connect()
                self.db_manager.save_dataframe(df, 'gps_traces', if_exists='append')
                self.db_manager.close()
            except Exception as e:
                logger.error(f"Failed to save to database: {e}")

        total_time = time.time() - start_time

        logger.info("=" * 80)
        logger.info(f"COMPLETED: Processed {len(df)} GPS records in {total_time:.2f}s")
        logger.info("=" * 80)

        return df

    def process_complete_dataset(
        self,
        routes_df: pd.DataFrame,
        stops_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        gps_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        normalize: bool = True,
        save_to_files: bool = True,
        save_to_db: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Process complete dataset

        Args:
            routes_df: Routes data
            stops_df: Stops data
            demand_df: Demand data
            gps_df: GPS data
            weather_df: Weather data (optional)
            events_df: Events data (optional)
            normalize: Whether to normalize
            save_to_files: Whether to save to CSV files
            save_to_db: Whether to save to database

        Returns:
            Dictionary of processed DataFrames
        """
        logger.info("=" * 80)
        logger.info("PROCESSING COMPLETE DATASET")
        logger.info("=" * 80)

        overall_start = time.time()

        # Process auxiliary data first
        processed_data = {}

        # Clean routes and stops
        logger.info("Processing routes and stops...")
        processed_data['routes'] = routes_df.copy()
        processed_data['stops'] = stops_df.copy()

        # Clean weather if provided
        if weather_df is not None:
            logger.info("Processing weather data...")
            processed_data['weather'] = self.cleaner.clean_weather_data(weather_df)

        # Clean events if provided
        if events_df is not None:
            logger.info("Processing events data...")
            processed_data['events'] = self.cleaner.clean_events_data(events_df)

        # Process demand data (main pipeline)
        processed_data['demand'] = self.process_demand_data(
            demand_df,
            stops_df=stops_df,
            weather_df=processed_data.get('weather'),
            events_df=processed_data.get('events'),
            normalize=normalize,
            save_to_db=False  # Will batch save later
        )

        # Process GPS data
        processed_data['gps'] = self.process_gps_data(
            gps_df,
            normalize=normalize,
            save_to_db=False
        )

        # Save to files
        if save_to_files:
            logger.info("=" * 80)
            logger.info("SAVING PROCESSED DATA")
            logger.info("=" * 80)

            output_dir = self.config.PROCESSED_DATA_DIR
            os.makedirs(output_dir, exist_ok=True)

            for name, df in processed_data.items():
                output_path = os.path.join(output_dir, f'processed_{name}.csv')
                df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(df)} records to {output_path}")

        # Save to database
        if save_to_db:
            logger.info("=" * 80)
            logger.info("SAVING TO DATABASE")
            logger.info("=" * 80)

            try:
                self.db_manager.connect()
                self.db_manager.create_tables()

                # Save each dataset
                table_mapping = {
                    'routes': 'routes',
                    'stops': 'stops',
                    'demand': 'passenger_demand',
                    'gps': 'gps_traces',
                    'weather': 'weather_data',
                    'events': 'events'
                }

                for name, df in processed_data.items():
                    if name in table_mapping:
                        table_name = table_mapping[name]
                        self.db_manager.save_dataframe(df, table_name, if_exists='append')

                self.db_manager.close()

            except Exception as e:
                logger.error(f"Failed to save to database: {e}")

        total_time = time.time() - overall_start

        # Print summary
        logger.info("=" * 80)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 80)
        for name, df in processed_data.items():
            logger.info(f"{name}: {len(df)} records, {len(df.columns)} features")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("=" * 80)

        return processed_data

    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        return self.pipeline_stats

    def print_cleaning_report(self):
        """Print data cleaning report"""
        self.cleaner.print_cleaning_report()

    def print_normalization_info(self):
        """Print normalization information"""
        self.normalizer.print_normalization_info()
