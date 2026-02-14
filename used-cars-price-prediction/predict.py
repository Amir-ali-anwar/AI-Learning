"""
Used Car Price Prediction - Inference Script

Predict the price of a used car by passing raw feature values.
The pipeline handles all preprocessing, feature engineering, and feature selection.

Usage:
    python predict.py

    Or import and call predict_price() from other scripts:
        from predict import predict_price
        result = predict_price({'year': 2020, 'manufacturer': 'ford', ...})
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, Any, Optional

# Add src to path so pickle can deserialize custom classes
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom classes required for pickle deserialization
try:
    from preprocessing_pipeline import DataPreprocessor, OutlierHandler
    from feature_engineering import FeatureEngineer
    from feature_selection import FeatureSelector
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Default values for required columns if not provided by the user.
# These are necessary because the pipeline expects certain columns to exist
# for scaling/encoding, even if we are only interested in a few primary features.
DEFAULTS = {
    'condition': 'good',
    'cylinders': '6 cylinders',
    'title_status': 'clean',
    'transmission': 'automatic',
    'drive': '4wd',
    'type': 'sedan',
    'paint_color': 'white',
    'state': 'ca',
    'region': 'los angeles',
    'lat': 34.05,
    'long': -118.24
}

# Input validation rules
VALID_RANGES = {
    'year': (1990, 2027),
    'odometer': (0, 500000),
    'lat': (-90, 90),
    'long': (-180, 180),
}

VALID_CATEGORIES = {
    'manufacturer': [
        'acura', 'alfa-romeo', 'aston-martin', 'audi', 'bmw', 'buick',
        'cadillac', 'chevrolet', 'chrysler', 'datsun', 'dodge', 'ferrari',
        'fiat', 'ford', 'gmc', 'harley-davidson', 'honda', 'hyundai',
        'infiniti', 'jaguar', 'jeep', 'kia', 'land rover', 'lexus',
        'lincoln', 'mazda', 'mercedes-benz', 'mercury', 'mini',
        'mitsubishi', 'nissan', 'pontiac', 'porsche', 'ram', 'rover',
        'saturn', 'subaru', 'tesla', 'toyota', 'volkswagen', 'volvo'
    ],
    'fuel': ['gas', 'diesel', 'electric', 'hybrid', 'other'],
    'transmission': ['automatic', 'manual', 'other'],
    'drive': ['fwd', 'rwd', '4wd'],
    'type': [
        'sedan', 'SUV', 'truck', 'pickup', 'coupe', 'hatchback',
        'wagon', 'van', 'convertible', 'mini-van', 'offroad', 'bus', 'other'
    ],
    'condition': ['salvage', 'fair', 'good', 'excellent', 'like new', 'new'],
    'title_status': ['clean', 'rebuilt', 'salvage', 'missing', 'lien', 'parts only'],
    'paint_color': [
        'black', 'blue', 'brown', 'custom', 'green', 'grey', 'orange',
        'purple', 'red', 'silver', 'white', 'yellow', 'unknown'
    ],
    'cylinders': [
        '3 cylinders', '4 cylinders', '5 cylinders', '6 cylinders',
        '8 cylinders', '10 cylinders', '12 cylinders', 'other'
    ],
}

REQUIRED_FIELDS = ['year', 'odometer', 'manufacturer', 'model', 'fuel']

# Cache for loaded pipeline components (avoid reloading on every call)
_pipeline_cache = {}


class PredictionInputError(ValueError):
    """Raised when prediction input fails validation."""
    pass


def validate_input(user_input: dict) -> list:
    """
    Validate user input for prediction.
    
    Args:
        user_input: Dictionary with car features.
        
    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in user_input:
            errors.append(f"Missing required field: '{field}'")
    
    # Validate ranges for numerical fields
    for field, (min_val, max_val) in VALID_RANGES.items():
        if field in user_input:
            val = user_input[field]
            if not isinstance(val, (int, float)):
                errors.append(f"Field '{field}' must be numeric, got {type(val).__name__}")
            elif val < min_val or val > max_val:
                errors.append(f"Field '{field}' value {val} is out of range [{min_val}, {max_val}]")
    
    # Validate categorical fields
    for field, valid_values in VALID_CATEGORIES.items():
        if field in user_input:
            val = str(user_input[field]).lower()
            if val not in [v.lower() for v in valid_values]:
                errors.append(
                    f"Field '{field}' value '{user_input[field]}' is not valid. "
                    f"Choose from: {valid_values[:5]}{'...' if len(valid_values) > 5 else ''}"
                )
    
    # Type checks
    if 'year' in user_input and isinstance(user_input['year'], float):
        if user_input['year'] != int(user_input['year']):
            errors.append(f"Field 'year' should be an integer, got {user_input['year']}")
    
    if 'odometer' in user_input and isinstance(user_input['odometer'], (int, float)):
        if user_input['odometer'] < 0:
            errors.append(f"Field 'odometer' cannot be negative")
    
    if 'model' in user_input and not isinstance(user_input['model'], str):
        errors.append(f"Field 'model' must be a string")
    
    return errors


def _get_best_model_name() -> str:
    """Determine the best model name from metadata or fallback."""
    base_path = os.path.join(os.path.dirname(__file__), 'models')
    metadata_path = os.path.join(base_path, 'best_model_metadata.json')
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            model_name = metadata.get('model_name', 'xgboost')
            logger.info(f"Best model from metadata: {model_name}")
            return model_name
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not read model metadata: {e}")
    
    # Fallback: check which model files exist, prefer xgboost > lightgbm > random_forest
    for name in ['xgboost', 'lightgbm', 'random_forest']:
        if os.path.exists(os.path.join(base_path, f'{name}.pkl')):
            logger.info(f"Falling back to model: {name}")
            return name
    
    raise FileNotFoundError("No model files found in models directory")


def load_pipeline():
    """Load all pipeline components. Uses cache to avoid repeated disk I/O."""
    if _pipeline_cache:
        return (
            _pipeline_cache['preprocessor'],
            _pipeline_cache['feature_engineer'],
            _pipeline_cache['feature_selector'],
            _pipeline_cache['model'],
            _pipeline_cache['model_name']
        )
    
    base_path = os.path.join(os.path.dirname(__file__), 'models')
    
    logger.info("Loading pipeline components...")
    
    try:
        with open(os.path.join(base_path, 'preprocessor.pkl'), 'rb') as f:
            preprocessor = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessor: {e}")
        
    try:
        with open(os.path.join(base_path, 'feature_engineer.pkl'), 'rb') as f:
            feature_engineer = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load feature engineer: {e}")
        
    try:
        with open(os.path.join(base_path, 'feature_selector.pkl'), 'rb') as f:
            feature_selector = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load feature selector: {e}")
    
    # Load the best model dynamically
    model_name = _get_best_model_name()
    model_path = os.path.join(base_path, f'{model_name}.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}' from {model_path}: {e}")
    
    logger.info(f"Loaded model: {model_name}")
    
    _pipeline_cache['preprocessor'] = preprocessor
    _pipeline_cache['feature_engineer'] = feature_engineer
    _pipeline_cache['feature_selector'] = feature_selector
    _pipeline_cache['model'] = model
    _pipeline_cache['model_name'] = model_name
    
    return preprocessor, feature_engineer, feature_selector, model, model_name


def clear_pipeline_cache():
    """Clear the pipeline cache to force reload on next prediction."""
    _pipeline_cache.clear()
    logger.info("Pipeline cache cleared")


def predict_price(user_input: dict, validate: bool = True) -> dict:
    """
    Predict price from a raw dictionary of car features.
    
    Merges user_input with DEFAULTS to ensure all required columns are present.
    
    Args:
        user_input: Dictionary with car features. At minimum, provide:
            - year (int): Vehicle model year
            - odometer (int): Mileage reading
            - manufacturer (str): e.g. 'ford', 'toyota'
            - model (str): e.g. 'f-150', 'camry'
            - fuel (str): e.g. 'gas', 'diesel'
        validate: Whether to validate input (default True)
            
    Returns:
        Dictionary with:
            - predicted_price (float): The predicted price
            - model_used (str): Which model made the prediction
            - input_warnings (list): Any validation warnings
    """
    warnings_list = []
    
    # Validate input
    if validate:
        errors = validate_input(user_input)
        if errors:
            raise PredictionInputError(
                f"Input validation failed with {len(errors)} error(s):\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
    
    # Merge user input with defaults
    data = DEFAULTS.copy()
    data.update(user_input)
    
    preprocessor, feature_engineer, feature_selector, model, model_name = load_pipeline()
    
    # Create a DataFrame with a single row
    df = pd.DataFrame([data])
    
    # Disable drop_first for single-row inference to prevent the only 
    # category present from being dropped during one-hot encoding
    if 'ENCODING_CONFIG' in preprocessor.config:
        preprocessor.config['ENCODING_CONFIG']['drop_first'] = False

    # 1. Preprocessing (Scaling, Categorical Encoding)
    df_processed = preprocessor.transform(df)
        
    # Align columns with training data:
    # Add missing one-hot columns (fill with 0), remove extras, enforce column order
    if preprocessor.feature_names_:
        missing_cols = [c for c in preprocessor.feature_names_ if c not in df_processed.columns]
        for c in missing_cols:
            df_processed[c] = 0
        df_processed = df_processed[preprocessor.feature_names_]
    
    # 2. Feature Engineering
    df_eng = feature_engineer.transform(df_processed)
    
    # 3. Feature Selection
    df_final = feature_selector.transform(df_eng)
    
    # 4. Prediction
    prediction = model.predict(df_final)
    predicted_price = float(prediction[0])
    
    # Sanity check: price should be positive
    if predicted_price < 0:
        warnings_list.append(f"Model predicted negative price ({predicted_price:.2f}), clamped to 0")
        predicted_price = 0.0
    
    return {
        'predicted_price': predicted_price,
        'model_used': model_name,
        'input_warnings': warnings_list
    }


if __name__ == "__main__":
    # Example: predict price for a 2020 Ford F-150
    sample_input = {
        'year': 2020,
        'odometer': 4500,
        'manufacturer': 'ford',
        'model': 'f-150',
        'fuel': 'gas'
    }
    
    print("\nInput Data:")
    for k, v in sample_input.items():
        print(f"  {k}: {v}")
    
    try:
        result = predict_price(sample_input)
        print(f"\nPredicted Price: ${result['predicted_price']:,.2f}")
        print(f"Model Used: {result['model_used']}")
        if result['input_warnings']:
            print(f"Warnings: {result['input_warnings']}")
    except PredictionInputError as e:
        print(f"\nValidation Error: {e}")
    except Exception as e:
        print(f"\nError: {e}")
