import pytest
import pandas as pd
import numpy as np
from src.models.model_trainer import ModelTrainer
from unittest.mock import patch as mock_patch, MagicMock


class TestModelTrainer:
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='W')
        return pd.DataFrame({
            'Date': dates,
            'target': np.random.rand(100) * 100,
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })

    @pytest.fixture
    def trainer(self):
        """Create model trainer with test config"""
        c = {
            'model_training': {
                'target_col': 'target',
                'features': ['feature1', 'feature2'],
                'test_size': 0.2,
                'random_state': 42
            }
        }
        with mock_patch(
            'src.models.model_trainer.load_yaml',
            return_value=c
        ):
            return ModelTrainer()

    def test_train_test_split(self, trainer, sample_data):
        """Test train-test split"""
        X_train, X_test, y_train, y_test = trainer._train_test_split(sample_data)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns

    @pytest.fixture
    def mock_rf(self):
        """Fixture to mock RandomForestRegressor"""
        with mock_patch('RandomForestRegressor') as mock:
            yield mock

    def test_train_model(self, trainer, sample_data, mock_rf):
        """Test model training"""
        mock_model = MagicMock()
        mock_rf.return_value = mock_model
        
        model = trainer.train_model(sample_data)
            
        mock_fit = mock_model.fit
        mock_fit.assert_called_once()
        assert model == mock_model

    def test_evaluate_model(self, trainer, sample_data):
        """Test model evaluation"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(20)
        
        metrics = trainer.evaluate_model(
            mock_model,
            sample_data.iloc[:20],
            sample_data.iloc[:20]['target']
        )
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics

    def test_train_and_evaluate(self, trainer, sample_data, mock_rf):
        """Test full training pipeline"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(20)
        mock_rf.return_value = mock_model
        
        results = trainer.train_and_evaluate(sample_data)
            
        assert 'model' in results
        assert 'metrics' in results
        assert 'train_metrics' in results['metrics']
        assert 'test_metrics' in results['metrics']
