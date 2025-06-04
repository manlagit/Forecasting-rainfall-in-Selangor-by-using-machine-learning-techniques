"""
Data loader module for rainfall forecasting project.
Handles data loading, validation, and initial preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and validation for rainfall forecasting."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataLoader with configuration."""
        self.config = self._load_config(config_path)
        self.data_config = self.config['data']
        self.preprocessing_config = self.config['preprocessing']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and merge both CSV files.
        
        Returns:
            pd.DataFrame: Merged dataset
        """
        logger.info("Loading data files...")
        
        # Construct file paths
        raw_path = Path(self.data_config['raw_path'])
        file1_path = raw_path / self.data_config['file1']
        file2_path = raw_path / self.data_config['file2']
        
        # Load datasets
        try:
            df1 = pd.read_csv(file1_path)
            df2 = pd.read_csv(file2_path)
            logger.info(f"Loaded {len(df1)} records from {self.data_config['file1']}")
            logger.info(f"Loaded {len(df2)} records from {self.data_config['file2']}")
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            raise
        
        # Merge datasets and remove duplicates
        df_merged = pd.concat([df1, df2], ignore_index=True)
        initial_count = len(df_merged)
        
        # Remove duplicates
        df_merged = df_merged.drop_duplicates()
        duplicate_count = initial_count - len(df_merged)
        
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate records")
        
        # Sort by date
        df_merged['Date'] = pd.to_datetime(df_merged['Date'])
        df_merged = df_merged.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Final dataset contains {len(df_merged)} records")
        
        return df_merged
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data schema and ranges.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Validated DataFrame
        """
        logger.info("Validating data...")
        
        # Check required columns
        required_columns = ['Date', 'Temp_avg', 'Relative_Humidity', 
                          'Wind_kmh', 'Precipitation_mm', 'Week_Number', 'Year']
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data types
        df['Date'] = pd.to_datetime(df['Date'])
        numeric_cols = ['Temp_avg', 'Relative_Humidity', 'Wind_kmh', 'Precipitation_mm']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Log validation summary
        logger.info("Data validation completed successfully")
        self._log_data_summary(df)
        
        return df
    
    def _log_data_summary(self, df: pd.DataFrame):
        """Log summary statistics of the dataset."""
        logger.info("\n=== Data Summary ===")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Number of records: {len(df)}")
        logger.info(f"Number of features: {len(df.columns)}")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning("Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                logger.warning(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        else:
            logger.info("No missing values detected")
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Load and validate data in one step.
        
        Returns:
            pd.DataFrame: Loaded and validated dataset
        """
        df = self.load_data()
        df = self.validate_data(df)
        
        # Save interim data
        interim_path = Path(self.data_config['interim_path'])
        interim_path.mkdir(parents=True, exist_ok=True)
        
        output_file = interim_path / 'merged_validated_data.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved validated data to {output_file}")
        
        return df
