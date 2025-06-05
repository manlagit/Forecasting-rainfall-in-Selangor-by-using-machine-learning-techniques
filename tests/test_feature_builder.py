import pytest
import pandas as pd
from src.features.build_features import FeatureBuilder
from unittest.mock import patch


class TestFeatureBuilder:
    @pytest.fixture
    def sample_data(self):
        """Create sample test data with dates"""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='W')
        return pd.DataFrame({
            'Date': dates,
            'temp': [25, 26, 27, 28, 29, 30, 29, 28, 27, 26],
            'humidity': [50, 55, 60, 65, 70, 75, 70, 65, 60, 55],
            'precipitation': [0, 5, 10, 15, 20, 25, 20, 15, 10, 5]
        })

    @pytest.fixture
    def builder(self):
        """Create feature builder with test config"""
        test_config = {
            'feature_engineering': {
                'lag_features': ['temp_lag_1', 'humidity_lag_2'],
                'moving_averages': {
                    'temp_ma_3': 3,
                    'precipitation_ma_2': 2
                },
                'seasonal_features': {
                    'monsoon_months': [4, 10, 11, 12],
                    'dry_months': [6, 7, 8]
                }
            }
        }
        with patch(
            'src.features.build_features.load_yaml',
            return_value=test_config
        ):
            return FeatureBuilder()

    def test_lag_features(self, builder, sample_data):
        """Test lag features"""
        transformed = builder.build_features(sample_data.copy())
        
        cols = transformed.columns
        assert 'temp_lag_1' in cols
        assert 'humidity_lag_2' in cols
        temp_lag_nans = (
            transformed['temp_lag_1']
            .isna().sum()
        )
        humidity_lag_nans = (
            transformed['humidity_lag_2']
            .isna().sum()
        )
        assert temp_lag_nans == 1
        assert humidity_lag_nans == 2

    def test_moving_averages(self, builder, sample_data):
        """Test moving averages"""
        transformed = builder.build_features(sample_data.copy())
        
        cols = transformed.columns
        assert 'temp_ma_3' in cols
        assert 'precipitation_ma_2' in cols
        temp_ma_nans = (
            transformed['temp_ma_3']
            .isna().sum()
        )
        precip_ma_nans = (
            transformed['precipitation_ma_2']
            .isna().sum()
        )
        assert temp_ma_nans == 2
        assert precip_ma_nans == 1

    def test_seasonal_features(self, builder, sample_data):
        """Test seasonal indicators"""
        transformed = builder.build_features(sample_data.copy())
        
        assert 'is_monsoon' in transformed.columns
        assert 'is_dry' in transformed.columns
        assert transformed['is_monsoon'].dtype == bool
        assert transformed['is_dry'].dtype == bool

    def test_build_features_from_file(self, builder, sample_data, tmp_path):
        """Test file-based feature building"""
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        
        sample_data.to_csv(input_path, index=False)
        builder.build_features_from_file(
            str(input_path),
            str(output_path)
        )
        
        assert output_path.exists()
        loaded = pd.read_csv(output_path, parse_dates=['Date'])
        assert len(loaded) == len(sample_data)
        assert 'temp_lag_1' in loaded.columns
