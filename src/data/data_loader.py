"""
Enhanced Data Acquisition & Integration Module
Handles loading, validation, and integration of rainfall datasets.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict
import yaml


class DataLoader:
    """
    Enhanced data loader with robust error handling and validation.
    Cost-efficient approach using existing datasets without external API calls.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config_path: Path to config file
        """
        self.logger = logging.getLogger(__name__)
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.data_dir = Path(self.config['data']['raw_path'])
            self.logger.info(
                f"Initialized DataLoader with config from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise
        
        # Expected data schema
        self.expected_columns = [
            'Date', 'Temp_avg', 'Relative_Humidity',
            'Wind_kmh', 'Precipitation_mm', 'Week_Number', 'Year'
        ]
        
        # Optional columns
        self.optional_columns = ['Week_Number', 'Year']
        
        # Data validation ranges
        self.validation_ranges = {
            'Temp_avg': (20, 35),           # Temperature in °C
            'Relative_Humidity': (0, 100),  # Humidity in %
            'Wind_kmh': (0, 15),            # Wind speed in km/h
            'Precipitation_mm': (0, 400)    # Precipitation in mm
        }
    
    def load_csv_with_validation(self, file_path: Path) -> pd.DataFrame:
        """
        Load CSV file with comprehensive validation.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Validated DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If validation fails
        """
        try:
            # Check file exists
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            # Load CSV
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} records from {file_path.name}")
            
            # Derive Year from Date if missing
            if 'Year' not in df.columns and 'Date' in df.columns:
                df['Year'] = pd.to_datetime(df['Date']).dt.year
                self.logger.info(f"Derived Year column from Date for {file_path.name}")
            
            # Validate schema
            self._validate_schema(df, file_path.name)
            
            # Validate data ranges
            self._validate_ranges(df, file_path.name)
            
            # Convert Date column
            df['Date'] = pd.to_datetime(df['Date'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def _validate_schema(self, df: pd.DataFrame, filename: str) -> None:
        """
        Validate DataFrame schema against expected columns.
        
        Args:
            df: DataFrame to validate
            filename: Name of file for error reporting
        """
        required_cols = set(self.expected_columns) - set(self.optional_columns)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in {filename}: {missing_cols}")

        extra_cols = set(df.columns) - set(self.expected_columns)
        if extra_cols:
            self.logger.warning(f"Extra columns in {filename}: {extra_cols}")

        self.logger.info(f"Schema validation passed for {filename}")
    
    def _validate_ranges(self, df: pd.DataFrame, filename: str) -> None:
        """
        Validate data ranges for key variables.
        
        Args:
            df: DataFrame to validate
            filename: Name of file for error reporting
        """
        validation_results = {}
        
        for column, (min_val, max_val) in self.validation_ranges.items():
            if column in df.columns:
                out_of_range = df[
                    (df[column] < min_val) | (df[column] > max_val)
                ][column]
                
                if len(out_of_range) > 0:
                    validation_results[column] = {
                        'count': len(out_of_range),
                        'percentage': (len(out_of_range) / len(df)) * 100,
                        'values': out_of_range.tolist()
                    }
                    
                    self.logger.warning(
                        f"{filename}: {len(out_of_range)} values out of range "
                        f"for {column} ({validation_results[column]['percentage']:.2f}%)"
                    )
        
        self.logger.info(f"Range validation completed for {filename}")
        return validation_results
    
    def detect_duplicates(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """
        Detect duplicates between two DataFrames.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            Dictionary with duplicate detection results
        """
        # Find exact duplicates
        merged = pd.concat([df1, df2], ignore_index=True)
        duplicates = merged[merged.duplicated(keep=False)]
        
        # Find date overlaps
        date_overlap = set(df1['Date']).intersection(set(df2['Date']))
        
        results = {
            'exact_duplicates': len(duplicates),
            'date_overlaps': len(date_overlap),
            'overlap_dates': sorted(list(date_overlap))
        }
        
        self.logger.info(
            f"Duplicate detection: {results['exact_duplicates']} exact, "
            f"{results['date_overlaps']} date overlaps")
        
        return results
    
    def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge two datasets with duplicate handling.
        
        Args:
            df1: First DataFrame (primary)
            df2: Second DataFrame (validation)
            
        Returns:
            Merged DataFrame
        """
        # Detect duplicates first
        dup_results = self.detect_duplicates(df1, df2)
        
        if dup_results['exact_duplicates'] > 0:
            self.logger.info("Found exact duplicates - using for validation")
            # If exact duplicates exist, prioritize df1
            merged = pd.concat([df1, df2], ignore_index=True)
            merged = merged.drop_duplicates(keep='first')
        else:
            # If no exact duplicates, merge all data
            merged = pd.concat([df1, df2], ignore_index=True)
        
        # Sort by date
        merged = merged.sort_values('Date').reset_index(drop=True)
        
        self.logger.info(f"Merged dataset: {len(merged)} total records")
        return merged
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Main method to load and validate all data with error handling.
        
        Returns:
            Validated and merged DataFrame
        """
        try:
            # Define file paths from config
            file1 = self.data_dir / self.config['data']['file1']
            file2 = self.data_dir / self.config['data']['file2']
            
            # Load both files
            self.logger.info("Loading primary dataset...")
            df1 = self.load_csv_with_validation(file1)
            
            self.logger.info("Loading validation dataset...")
            df2 = self.load_csv_with_validation(file2)
            
            # Merge datasets
            self.logger.info("Merging datasets...")
            df_merged = self.merge_datasets(df1, df2)
            
            # Final validation
            self._perform_final_validation(df_merged)
            
            self.logger.info(
                "Data loading and validation completed successfully")
            return df_merged
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _perform_final_validation(self, df: pd.DataFrame) -> None:
        """
        Perform final validation checks on merged dataset.
        
        Args:
            df: Merged DataFrame to validate
        """
        # Check for missing values
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            self.logger.warning(f"Missing values found:\n{missing_summary}")
        
        # Check date range
        date_range = (df['Date'].min(), df['Date'].max())
        self.logger.info(f"Date range: {date_range[0]} to {date_range[1]}")
        
        # Check data completeness
        expected_weeks = (date_range[1] - date_range[0]).days // 7
        actual_weeks = len(df)
        completeness = (actual_weeks / expected_weeks) * 100
        
        self.logger.info(
            f"Data completeness: {completeness:.1f}% "
            f"({actual_weeks}/{expected_weeks} weeks)")
        
        # Summary statistics
        numeric_cols = [
            'Temp_avg', 'Relative_Humidity', 'Wind_kmh', 'Precipitation_mm']
        summary = df[numeric_cols].describe()
        self.logger.info(f"Summary statistics:\n{summary}")
    
    def save_sample_data(self, df: pd.DataFrame, filename: str = "sample_data.csv") -> None:
        """
        Save sample data for system testing.
        
        Args:
            df: DataFrame to save
            filename: Output file name
        """
        sample_path = self.data_dir / filename
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(sample_path, index=False)
        self.logger.info(f"Saved sample data to {sample_path}")
        
    def save_processed_data(self, df: pd.DataFrame, 
                           filepath: str = "data/interim/merged_data.csv") -> None:
        """
        Save processed data to interim directory.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved merged data to {output_path}")
