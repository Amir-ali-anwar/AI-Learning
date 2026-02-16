"""
Unit tests for the Used Cars Price Prediction pipeline.

Covers:
- Input validation
- Feature engineering
- Preprocessing pipeline
- Prediction pipeline
- Data validator
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root and src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'url': ['http://a', 'http://b', 'http://c', 'http://d', 'http://e'],
        'region': ['los angeles', 'new york', 'chicago', 'houston', 'phoenix'],
        'region_url': ['http://a', 'http://b', 'http://c', 'http://d', 'http://e'],
        'price': [15000, 25000, 8000, 35000, 12000],
        'year': [2018, 2020, 2015, 2021, 2017],
        'manufacturer': ['ford', 'toyota', 'honda', 'bmw', 'chevrolet'],
        'model': ['f-150', 'camry', 'civic', 'x5', 'malibu'],
        'condition': ['good', 'excellent', 'fair', 'like new', 'good'],
        'cylinders': ['6 cylinders', '4 cylinders', '4 cylinders', '6 cylinders', '4 cylinders'],
        'fuel': ['gas', 'gas', 'gas', 'gas', 'gas'],
        'odometer': [45000, 12000, 120000, 5000, 80000],
        'title_status': ['clean', 'clean', 'clean', 'clean', 'clean'],
        'transmission': ['automatic', 'automatic', 'manual', 'automatic', 'automatic'],
        'VIN': ['VIN1', 'VIN2', 'VIN3', 'VIN4', 'VIN5'],
        'drive': ['4wd', 'fwd', 'fwd', 'rwd', '4wd'],
        'type': ['truck', 'sedan', 'sedan', 'SUV', 'sedan'],
        'paint_color': ['white', 'black', 'red', 'blue', 'silver'],
        'image_url': ['http://a', 'http://b', 'http://c', 'http://d', 'http://e'],
        'description': ['desc1', 'desc2', 'desc3', 'desc4', 'desc5'],
        'county': [None, None, None, None, None],
        'state': ['ca', 'ny', 'il', 'tx', 'az'],
        'lat': [34.05, 40.71, 41.88, 29.76, 33.45],
        'long': [-118.24, -74.01, -87.63, -95.37, -112.07],
        'posting_date': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01'],
    })


@pytest.fixture
def sample_processed_data():
    """Create sample processed data (post-preprocessing) for testing."""
    return pd.DataFrame({
        'year': [2018, 2020, 2015, 2021, 2017],
        'odometer': [45000, 12000, 120000, 5000, 80000],
        'condition': [2, 3, 1, 4, 2],
        'lat': [0.1, 0.5, 0.3, -0.2, 0.0],
        'long': [-0.5, 0.2, -0.3, 0.1, -0.8],
        'region': [1, 2, 3, 4, 5],
        'state': [1, 2, 3, 4, 5],
    })


@pytest.fixture
def sample_prediction_input():
    """Create sample prediction input."""
    return {
        'year': 2020,
        'odometer': 4500,
        'manufacturer': 'ford',
        'model': 'f-150',
        'fuel': 'gas'
    }


# ============================================================================
# Test Input Validation (predict.py)
# ============================================================================

class TestInputValidation:
    """Test input validation for predictions."""
    
    def test_valid_input_passes(self, sample_prediction_input):
        from predict import validate_input
        errors = validate_input(sample_prediction_input)
        assert len(errors) == 0, f"Valid input should pass, got errors: {errors}"
    
    def test_missing_required_field(self):
        from predict import validate_input
        incomplete = {'year': 2020, 'odometer': 5000}  # missing manufacturer, model, fuel
        errors = validate_input(incomplete)
        assert len(errors) >= 3, "Should report missing required fields"
        assert any("manufacturer" in e for e in errors)
    
    def test_invalid_year_range(self):
        from predict import validate_input
        data = {
            'year': 1800,  # too old
            'odometer': 5000,
            'manufacturer': 'ford',
            'model': 'f-150',
            'fuel': 'gas'
        }
        errors = validate_input(data)
        assert any("year" in e.lower() for e in errors), "Should flag invalid year"
    
    def test_negative_odometer(self):
        from predict import validate_input
        data = {
            'year': 2020,
            'odometer': -500,
            'manufacturer': 'ford',
            'model': 'f-150',
            'fuel': 'gas'
        }
        errors = validate_input(data)
        assert any("odometer" in e.lower() for e in errors), "Should flag negative odometer"
    
    def test_invalid_fuel_type(self):
        from predict import validate_input
        data = {
            'year': 2020,
            'odometer': 5000,
            'manufacturer': 'ford',
            'model': 'f-150',
            'fuel': 'nuclear'  # invalid
        }
        errors = validate_input(data)
        assert any("fuel" in e.lower() for e in errors), "Should flag invalid fuel"
    
    def test_invalid_type_for_year(self):
        from predict import validate_input
        data = {
            'year': 'not_a_number',
            'odometer': 5000,
            'manufacturer': 'ford',
            'model': 'f-150',
            'fuel': 'gas'
        }
        errors = validate_input(data)
        assert any("year" in e.lower() for e in errors), "Should flag non-numeric year"


# ============================================================================
# Test Feature Engineering
# ============================================================================

class TestFeatureEngineering:
    """Test feature engineering module."""
    
    def test_create_age_feature(self, sample_processed_data):
        from feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        fe._fit_year = 2026
        fe.is_fitted_ = False
        result = fe.create_age_feature(sample_processed_data)
        assert 'vehicle_age' in result.columns, "Should create vehicle_age"
        assert (result['vehicle_age'] >= 0).all(), "Vehicle age should be non-negative"
        assert result.loc[result['year'] == 2021, 'vehicle_age'].values[0] == 5
    
    def test_create_mileage_features(self, sample_processed_data):
        from feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        fe._fit_year = 2026
        fe.is_fitted_ = False
        
        # First create age (mileage features depend on it)
        df = fe.create_age_feature(sample_processed_data)
        result = fe.create_mileage_features(df)
        
        assert 'avg_miles_per_year' in result.columns
        assert 'mileage_category' in result.columns
        assert 'odometer_log' in result.columns
    
    def test_no_price_features_created(self, sample_processed_data):
        """Verify that no price-derived features are created (data leakage prevention)."""
        from feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_processed_data)
        
        price_features = [c for c in result.columns if 'price' in c.lower()]
        assert len(price_features) == 0, (
            f"No price-derived features should exist to prevent data leakage. "
            f"Found: {price_features}"
        )
    
    def test_transform_does_not_refit(self, sample_processed_data):
        """Verify transform() doesn't reset state."""
        from feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        
        # Fit
        fe.fit_transform(sample_processed_data)
        n_features_after_fit = len(fe.feature_names_created_)
        
        # Transform
        fe.transform(sample_processed_data)
        n_features_after_transform = len(fe.feature_names_created_)
        
        assert n_features_after_fit == n_features_after_transform, (
            f"transform() should not add to feature_names_created_. "
            f"Before: {n_features_after_fit}, After: {n_features_after_transform}"
        )
    
    def test_dynamic_year(self):
        """Verify current year is dynamically set."""
        from feature_engineering import FeatureEngineer
        from datetime import datetime
        
        fe = FeatureEngineer()
        fe.fit_transform(pd.DataFrame({'year': [2020], 'odometer': [5000]}))
        
        assert fe._fit_year == datetime.now().year, (
            f"Fit year should be {datetime.now().year}, got {fe._fit_year}"
        )
    
    def test_boolean_features_are_binary(self, sample_processed_data):
        from feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_processed_data)
        
        binary_cols = ['is_low_mileage', 'is_high_mileage', 'is_recent_model', 'is_vintage']
        for col in binary_cols:
            if col in result.columns:
                unique_vals = result[col].dropna().unique()
                assert set(unique_vals).issubset({0, 1}), (
                    f"{col} should be binary (0/1), got {unique_vals}"
                )


# ============================================================================
# Test Data Validator
# ============================================================================

class TestDataValidator:
    """Test data validation utilities."""
    
    def test_validate_empty_dataframe(self):
        from data_validator import DataValidator
        validator = DataValidator({'min_rows': 100})
        is_valid, issues = validator.validate_dataframe(pd.DataFrame())
        assert not is_valid, "Empty DataFrame should fail validation"
        assert any("empty" in i.lower() for i in issues)
    
    def test_validate_target_column_missing(self, sample_raw_data):
        from data_validator import DataValidator
        validator = DataValidator({})
        is_valid, issues = validator.validate_target_column(sample_raw_data, 'nonexistent')
        assert not is_valid
        assert any("not found" in i.lower() for i in issues)
    
    def test_validate_target_column_valid(self, sample_raw_data):
        from data_validator import DataValidator
        validator = DataValidator({'min_unique_target': 3})
        is_valid, issues = validator.validate_target_column(sample_raw_data, 'price')
        assert is_valid, f"Valid target should pass. Issues: {issues}"
    
    def test_check_data_leakage_no_overlap(self):
        from data_validator import check_data_leakage
        train = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        test = pd.DataFrame({'a': [4, 5, 6]}, index=[3, 4, 5])
        result = check_data_leakage(train, test)
        assert not result['has_leakage'], "Non-overlapping data should have no leakage"
    
    def test_check_data_leakage_with_overlap(self):
        from data_validator import check_data_leakage
        train = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        test = pd.DataFrame({'a': [4, 5, 6]}, index=[1, 2, 3])  # index 1,2 overlap
        result = check_data_leakage(train, test)
        assert result['has_leakage'], "Overlapping indices should be detected"
        assert len(result['common_indices']) == 2


# ============================================================================
# Test Preprocessing Pipeline
# ============================================================================

class TestPreprocessingPipeline:
    """Test preprocessing pipeline components."""
    
    def test_outlier_handler_iqr(self):
        from preprocessing_pipeline import OutlierHandler
        handler = OutlierHandler(method='iqr', handle_method='cap')
        
        df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 100, -50]})
        handler.fit(df, ['values'])
        result = handler.transform(df)
        
        assert result['values'].max() <= handler.bounds_['values']['upper']
        assert result['values'].min() >= handler.bounds_['values']['lower']
    
    def test_outlier_handler_remove(self):
        from preprocessing_pipeline import OutlierHandler
        handler = OutlierHandler(method='iqr', handle_method='remove')
        
        df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 100, -50]})
        handler.fit(df, ['values'])
        result = handler.transform(df)
        
        assert len(result) < len(df), "Outlier rows should be removed"
    
    def test_drop_columns(self, sample_raw_data):
        from preprocessing_pipeline import DataPreprocessor
        config = {'COLUMNS_TO_DROP': ['id', 'url', 'VIN']}
        preprocessor = DataPreprocessor(config)
        result = preprocessor.drop_columns(sample_raw_data)
        
        assert 'id' not in result.columns
        assert 'url' not in result.columns
        assert 'VIN' not in result.columns
        assert 'price' in result.columns  # should keep price


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
