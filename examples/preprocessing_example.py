"""
Data Preprocessing Examples

This script demonstrates how to use the preprocessing pipeline to clean,
transform, and prepare data for machine learning models.
"""

import os
import sys
import pandas as pd
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataCleaner, FeatureEngineer, DataNormalizer
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.config import PreprocessingConfig

# Check if database support is available
try:
    from src.preprocessing import DatabaseManager
    HAS_DATABASE = True
except (ImportError, AttributeError):
    DatabaseManager = None
    HAS_DATABASE = False
    print("Note: Database support not available. Install with: pip install sqlalchemy psycopg2-binary")


class PreprocessingExamples:
    """Examples for data preprocessing"""

    def __init__(self):
        self.config = PreprocessingConfig()
        self.config.ensure_dirs()

    def example_1_data_cleaning(self):
        """Example 1: Basic data cleaning"""
        print("\n" + "=" * 80)
        print("EXAMPLE 1: DATA CLEANING")
        print("=" * 80)

        # Load synthetic data
        data_dir = "data/raw"
        demand_df = pd.read_csv(os.path.join(data_dir, "synthetic_demand.csv"))

        print(f"\nOriginal data: {len(demand_df)} records")
        print(f"Columns: {list(demand_df.columns)}")

        # Clean data
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_demand_data(demand_df)

        print(f"\nCleaned data: {len(cleaned_df)} records")

        # Print cleaning report
        cleaner.print_cleaning_report()

        return cleaned_df

    def example_2_feature_engineering(self):
        """Example 2: Feature engineering"""
        print("\n" + "=" * 80)
        print("EXAMPLE 2: FEATURE ENGINEERING")
        print("=" * 80)

        # Load data
        data_dir = "data/raw"
        demand_df = pd.read_csv(os.path.join(data_dir, "synthetic_demand.csv"))
        stops_df = pd.read_csv(os.path.join(data_dir, "synthetic_stops.csv"))

        print(f"\nOriginal features: {len(demand_df.columns)}")
        print(f"Features: {list(demand_df.columns)}")

        # Create features
        feature_engineer = FeatureEngineer()

        # Temporal features
        print("\n[1/2] Creating temporal features...")
        df = feature_engineer.create_temporal_features(demand_df, time_col='time')

        # Spatial features
        print("[2/2] Creating spatial features...")
        df = feature_engineer.create_spatial_features(df, stops_df)

        print(f"\nAfter feature engineering: {len(df.columns)} features")
        print(f"\nNew features added:")

        original_cols = set(demand_df.columns)
        new_cols = set(df.columns) - original_cols

        for col in sorted(new_cols):
            print(f"  - {col}")

        # Show sample
        print("\nSample of engineered features:")
        feature_cols = ['hour', 'is_peak_hour', 'is_weekend', 'hour_sin', 'hour_cos',
                        'distance_from_center', 'route_progress']
        feature_cols = [c for c in feature_cols if c in df.columns]

        print(df[feature_cols].head().to_string())

        return df

    def example_3_normalization(self):
        """Example 3: Data normalization"""
        print("\n" + "=" * 80)
        print("EXAMPLE 3: DATA NORMALIZATION")
        print("=" * 80)

        # Load data
        data_dir = "data/raw"
        demand_df = pd.read_csv(os.path.join(data_dir, "synthetic_demand.csv"))

        # Create some features first
        feature_engineer = FeatureEngineer()
        df = feature_engineer.create_temporal_features(demand_df, time_col='time')

        # Prepare columns
        numeric_cols = ['boarding_count', 'alighting_count', 'passengers', 'hour',
                        'day_of_week', 'hour_sin', 'hour_cos']
        numeric_cols = [c for c in numeric_cols if c in df.columns]

        print(f"\nNormalizing {len(numeric_cols)} numeric features")
        print("Features:", numeric_cols)

        # Show before normalization
        print("\nBefore normalization:")
        print(df[numeric_cols].describe())

        # Normalize
        normalizer = DataNormalizer()
        normalized_df = normalizer.normalize_features(
            df,
            numeric_cols,
            method='minmax',
            fit=True
        )

        # Show after normalization
        print("\nAfter normalization (MinMax):")
        print(normalized_df[numeric_cols].describe())

        # Test inverse normalization
        print("\nTesting inverse normalization...")
        denormalized_df = normalizer.inverse_normalize(normalized_df, method='minmax')

        print("After inverse normalization:")
        print(denormalized_df[numeric_cols].describe())

        # Save scalers
        normalizer.save_scalers()
        print("\nScalers saved to models/")

        return normalized_df

    def example_4_complete_pipeline(self):
        """Example 4: Complete preprocessing pipeline"""
        print("\n" + "=" * 80)
        print("EXAMPLE 4: COMPLETE PREPROCESSING PIPELINE")
        print("=" * 80)

        # Load all data
        data_dir = "data/raw"

        print("Loading data...")
        routes_df = pd.read_csv(os.path.join(data_dir, "synthetic_routes.csv"))
        stops_df = pd.read_csv(os.path.join(data_dir, "synthetic_stops.csv"))
        demand_df = pd.read_csv(os.path.join(data_dir, "synthetic_demand.csv"))
        gps_df = pd.read_csv(os.path.join(data_dir, "synthetic_gps.csv"))

        # Optional: load weather and events
        weather_path = os.path.join(data_dir, "synthetic_weather.csv")
        events_path = os.path.join(data_dir, "synthetic_events.csv")

        weather_df = pd.read_csv(weather_path) if os.path.exists(weather_path) else None
        events_df = pd.read_csv(events_path) if os.path.exists(events_path) else None

        print(f"\nData loaded:")
        print(f"  Routes: {len(routes_df)} records")
        print(f"  Stops: {len(stops_df)} records")
        print(f"  Demand: {len(demand_df)} records")
        print(f"  GPS: {len(gps_df)} records")
        if weather_df is not None:
            print(f"  Weather: {len(weather_df)} records")
        if events_df is not None:
            print(f"  Events: {len(events_df)} records")

        # Create pipeline
        pipeline = PreprocessingPipeline()

        # Process complete dataset
        processed_data = pipeline.process_complete_dataset(
            routes_df=routes_df,
            stops_df=stops_df,
            demand_df=demand_df,
            gps_df=gps_df,
            weather_df=weather_df,
            events_df=events_df,
            normalize=True,
            save_to_files=True,
            save_to_db=False  # Set to True if you have database configured
        )

        print("\n" + "=" * 80)
        print("PROCESSED DATA SUMMARY")
        print("=" * 80)

        for name, df in processed_data.items():
            print(f"\n{name.upper()}:")
            print(f"  Records: {len(df)}")
            print(f"  Features: {len(df.columns)}")
            print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # Show sample of processed demand data
        print("\n" + "=" * 80)
        print("SAMPLE OF PROCESSED DEMAND DATA")
        print("=" * 80)
        print(processed_data['demand'].head(10))

        # Print reports
        pipeline.print_cleaning_report()

        return processed_data

    def example_5_feature_importance(self):
        """Example 5: Analyze feature importance"""
        print("\n" + "=" * 80)
        print("EXAMPLE 5: FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)

        # Load processed data
        processed_dir = "data/processed"
        processed_path = os.path.join(processed_dir, "processed_demand.csv")

        if not os.path.exists(processed_path):
            print("Processed data not found. Run Example 4 first.")
            return None

        df = pd.read_csv(processed_path)

        print(f"\nAnalyzing {len(df.columns)} features")

        # Calculate correlations with target
        target = 'passengers'
        if target not in df.columns:
            print(f"Target column '{target}' not found")
            return None

        # Select numeric columns
        numeric_df = df.select_dtypes(include=['number'])

        # Calculate correlations
        correlations = numeric_df.corr()[target].sort_values(ascending=False)

        print(f"\nTop 15 features correlated with {target}:")
        print("=" * 60)
        print(f"{'Feature':<40} {'Correlation':>15}")
        print("=" * 60)

        for feature, corr in correlations.head(15).items():
            if feature != target:
                print(f"{feature:<40} {corr:>15.4f}")

        print("\n" + "=" * 60)

        return correlations

    def example_6_lag_and_rolling_features(self):
        """Example 6: Create lag and rolling features"""
        print("\n" + "=" * 80)
        print("EXAMPLE 6: LAG AND ROLLING FEATURES")
        print("=" * 80)

        # Load data
        data_dir = "data/raw"
        demand_df = pd.read_csv(os.path.join(data_dir, "synthetic_demand.csv"))

        # Sort by time
        demand_df = demand_df.sort_values(['route', 'stop', 'time']).reset_index(drop=True)

        print(f"\nOriginal data: {len(demand_df.columns)} features")

        # Create feature engineer
        feature_engineer = FeatureEngineer()

        # Create lag features
        print("\nCreating lag features...")
        df = feature_engineer.create_lag_features(
            demand_df,
            target_col='passengers',
            lags=[1, 2, 3],
            group_cols=['route', 'stop']
        )

        # Create rolling features
        print("Creating rolling features...")
        df = feature_engineer.create_rolling_features(
            df,
            target_col='passengers',
            windows=[3, 6],
            group_cols=['route', 'stop']
        )

        print(f"\nAfter lag/rolling features: {len(df.columns)} features")

        # Show sample
        print("\nSample with lag and rolling features:")
        feature_cols = ['time', 'route', 'stop', 'passengers',
                        'passengers_lag_1', 'passengers_lag_2', 'passengers_lag_3',
                        'passengers_rolling_mean_3', 'passengers_rolling_std_3']
        feature_cols = [c for c in feature_cols if c in df.columns]

        print(df[feature_cols].head(10).to_string())

        return df


def main():
    """Run examples"""
    parser = argparse.ArgumentParser(description='Data Preprocessing Examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Run specific example (1-6)'
    )

    args = parser.parse_args()

    examples = PreprocessingExamples()

    if args.example:
        # Run specific example
        example_map = {
            1: examples.example_1_data_cleaning,
            2: examples.example_2_feature_engineering,
            3: examples.example_3_normalization,
            4: examples.example_4_complete_pipeline,
            5: examples.example_5_feature_importance,
            6: examples.example_6_lag_and_rolling_features
        }
        example_map[args.example]()
    else:
        # Run examples that don't require database
        print("\n" + "=" * 80)
        print("RUNNING PREPROCESSING EXAMPLES")
        print("=" * 80)

        try:
            examples.example_1_data_cleaning()
            input("\nPress Enter to continue to Example 2...")

            examples.example_2_feature_engineering()
            input("\nPress Enter to continue to Example 3...")

            examples.example_3_normalization()
            input("\nPress Enter to continue to Example 4...")

            examples.example_4_complete_pipeline()
            input("\nPress Enter to continue to Example 5...")

            examples.example_5_feature_importance()
            input("\nPress Enter to continue to Example 6...")

            examples.example_6_lag_and_rolling_features()

            print("\n" + "=" * 80)
            print("ALL EXAMPLES COMPLETED!")
            print("=" * 80)
            print("\nProcessed data saved to: data/processed/")
            print("Scalers saved to: models/")

        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user")
        except Exception as e:
            print(f"\n\nError running examples: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
