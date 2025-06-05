import pytest
import pandas as pd
import numpy as np
from src.data.data_validator import DataValidator
from unittest.mock import patch


class TestDataValidator:
    @pytest.fixture
    def sample_data(self):
        """Create sample test data"""
        return pd.DataFrame({
            'temp': [25, 30, 35, 40, 45],
            'humidity': [50, 55, 60, 65, 70],
            'precipitation': [0, 5, 10, 15, 20]
        })

    @pytest.fixture
    def validator(self):
        """Create validator instance with test config"""
        test_config = {
            'preprocessing': {
                'valid_ranges': {
                    'temp': (20, 50),
                    'humidity': (0, 100),
                    'precipitation': (0, 50)
                },
                'imputation_method': 'mean'
            }
        }
        with patch(
            'src.data.data_validator.load_yaml',
            return_value=test_config
        ):
            return DataValidator()

    def test_validate_clips_values(self, validator, sample_data):
        """Test that values are clipped to valid ranges"""
        # Add some out-of-range values
        test_data = sample_data.copy()
        test_data.loc[0, 'temp'] = 10  # Below min
        test_data.loc[1, 'humidity'] = 110  # Above max
        
        validated = validator.validate(test_data)
        
        assert validated['temp'].min() == 20
        assert validated['humidity'].max() == 100

    def test_validate_handles_missing_values(self, validator, sample_data):
        """Test missing value imputation"""
        test_data = sample_data.copy()
        test_data.loc[0, 'precipitation'] = np.nan
        
        validated = validator.validate(test_data)
        
        assert not validated.isnull().any().any()
        # mean of [5,10,15,20]
        assert validated.loc[0, 'precipitation'] == pytest.approx(10)

    def test_validate_file(self, validator, sample_data, tmp_path):
        """Test file validation workflow"""
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        
        sample_data.to_csv(input_path, index=False)
        validator.validate_file(
            str(input_path), 
            str(output_path)
        )
        
        assert output_path.exists()
        loaded = pd.read_csv(output_path)
        assert len(loaded) == len(sample_data)

    def test_invalid_config_raises_error(self):
        """Test that invalid config raises error"""
        bad_config = {
            'preprocessing': {
                'imputation_method': 'invalid'
            }
        }
        with patch(
            'src.data.data_validator.load_yaml', 
            return_value=bad_config
        ):
            with pytest.raises(ValueError):
                DataValidator()
