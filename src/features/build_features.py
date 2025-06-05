import pandas as pd
from pathlib import Path
from typing import Dict, Any
from src.utils.helpers import load_yaml
import logging


class FeatureBuilder:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature builder with configuration.
        
        Args:
            config_path: Path to config file with feature engineering rules
        """
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate feature engineering configuration.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dictionary with feature engineering rules
        """
        try:
            config = load_yaml(config_path)
            feature_config = config.get('feature_engineering', {})
            
            # Validate required fields exist
            required = ['lag_features', 'moving_averages', 'seasonal_features']
            if not all(k in feature_config for k in required):
                raise ValueError("Missing required feature config fields")
                
            return feature_config
        except Exception as e:
            logging.error(f"Error loading feature config: {e}")
            raise

    def _setup_logging(self):
        """Configure logging for feature engineering operations."""
        logging.basicConfig(
            filename='logs/feature_engineering.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features according to configuration rules.
        
        Args:
            df: Input dataframe to transform
            
        Returns:
            Dataframe with engineered features
        """
        try:
            # Create lag features
            for lag_col in self.config['lag_features']:
                base_col = lag_col.split('_lag_')[0]
                # Map to actual column names in our dataset
                column_mapping = {
                    'precipitation': 'Precipitation_mm',
                    'temp': 'Temp_avg',
                    'humidity': 'Relative_Humidity'
                }
                actual_base_col = column_mapping.get(base_col, base_col)
                lag_period = int(lag_col.split('_lag_')[1])
                df[lag_col] = df[actual_base_col].shift(lag_period)
                logging.info(f"Created lag feature: {lag_col}")

            # Create moving averages
            for base_col, window in self.config['moving_averages'].items():
                # Map to actual column names in our dataset
                column_mapping = {
                    'precipitation': 'Precipitation_mm',
                    'temp': 'Temp_avg',
                    'humidity': 'Relative_Humidity'
                }
                actual_base_col = column_mapping.get(base_col, base_col)
                ma_col = f"{base_col}_ma_{window}"
                df[ma_col] = df[actual_base_col].rolling(window=window).mean()
                msg = f"Created moving average: {ma_col} (window={window})"
                logging.info(msg)

            # Create seasonal features
            if 'monsoon_months' in self.config['seasonal_features']:
                df['is_monsoon'] = df['Date'].dt.month.isin(
                    self.config['seasonal_features']['monsoon_months']
                )
                logging.info("Created monsoon season feature")

            if 'dry_months' in self.config['seasonal_features']:
                df['is_dry'] = df['Date'].dt.month.isin(
                    self.config['seasonal_features']['dry_months']
                )
                logging.info("Created dry season feature")

            logging.info("Feature engineering completed successfully")
            return df
        except Exception as e:
            logging.error(f"Feature engineering failed: {e}")
            raise

    def build_features_from_file(self, input_path: str, output_path: str) -> None:
        """Build features from file and save transformed data.
        
        Args:
            input_path: Path to input data file
            output_path: Path to save transformed data
        """
        try:
            df = pd.read_csv(input_path, parse_dates=['Date'])
            transformed_df = self.build_features(df)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            transformed_df.to_csv(output_path, index=False)
            msg = f"Saved transformed data to {output_path}"
            logging.info(msg)
        except Exception as e:
            logging.error("Feature engineering failed")
            logging.error(str(e)[:70])
            raise
