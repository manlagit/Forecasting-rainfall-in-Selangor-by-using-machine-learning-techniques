import yaml
import pandas as pd
import logging
from typing import Dict, Any


class DataValidator:
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize with configuration path"""
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading config: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data against config ranges and handle missing values"""
        # Get validation rules from config
        valid_ranges = self.config['data_validation']['valid_ranges']
        imputation_method = self.config['data_validation']['imputation_method']
        
        # 1. Range validation
        for col, (min_val, max_val) in valid_ranges.items():
            if col in df.columns:
                invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                if invalid_mask.any():
                    self.logger.warning(
                        f"{invalid_mask.sum()} rows in '{col}' outside "
                        f"valid range [{min_val}, {max_val}]"
                    )
        
        # 2. Missing value imputation
        if imputation_method == 'mean':
            df = df.fillna(df.mean())
        elif imputation_method == 'median':
            df = df.fillna(df.median())
        elif imputation_method == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown imputation method: {imputation_method}")
            
        self.logger.info(
            f"Applied {imputation_method} imputation for missing values"
        )
        return df
