import pandas as pd
from src.utils.helpers import load_yaml
from pathlib import Path
import logging


class DataValidator:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data validator with configuration.
        
        Args:
            config_path: Path to config file with validation rules
        """
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: str) -> dict:
        """Load and validate configuration.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dictionary with validation rules
        """
        try:
            config = load_yaml(config_path)
            validation_config = config.get('preprocessing', {})
            
            # Validate required fields exist
            required = ['valid_ranges', 'imputation_method']
            if not all(k in validation_config for k in required):
                raise ValueError("Missing required validation config fields")
                
            return validation_config
        except Exception as e:
            logging.error(f"Error loading validation config: {e}")
            raise

    def _setup_logging(self):
        """Configure logging for validation operations."""
        logging.basicConfig(
            filename='logs/validation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean dataframe according to config rules.
        
        Args:
            df: Input dataframe to validate
            
        Returns:
            Cleaned and validated dataframe
        """
        try:
            # Apply value range constraints
            for col, (min_val, max_val) in self.config['valid_ranges'].items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=min_val, upper=max_val)
                    outliers = (
                        (df[col] < min_val) | 
                        (df[col] > max_val)
                    ).sum()
                    if outliers > 0:
                        msg = f"{outliers} outliers clipped in {col}"
                        logging.warning(msg)

            # Handle missing values
            impute_method = self.config['imputation_method']
            if impute_method == "mean":
                df.fillna(df.mean(), inplace=True)
            elif impute_method == "median":
                df.fillna(df.median(), inplace=True)
            elif impute_method == "drop":
                df.dropna(inplace=True)
            else:
                msg = f"Unknown imputation method: {impute_method}"
                raise ValueError(msg)

            logging.info("Data validation completed successfully")
            return df
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            raise

    def validate_file(self, input_path: str, output_path: str) -> None:
        """Validate data from file and save cleaned output.
        
        Args:
            input_path: Path to input data file
            output_path: Path to save cleaned data
        """
        try:
            df = pd.read_csv(input_path)
            validated_df = self.validate(df)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            validated_df.to_csv(output_path, index=False)
            logging.info(f"Saved validated data to {output_path}")
        except Exception as e:
            logging.error(f"File validation failed: {e}")
            raise
