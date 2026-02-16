"""
Model Training module for training multiple machine learning models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage multiple regression models."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_default_models(self) -> Dict[str, Any]:
        """
        Get dictionary of default models with baseline parameters.
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {
            # Linear Models
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            
            # Tree-based Models
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            
            # Boosting Models
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            ),
            'adaboost': AdaBoostRegressor(
                n_estimators=100,
                random_state=42
            ),
            
            # Other Models
            'knn': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
            'svr': SVR(kernel='rbf')
        }
        
        # Optional boosting libraries (only add if installed)
        if XGBRegressor is not None:
            models['xgboost'] = XGBRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        if LGBMRegressor is not None:
            models['lightgbm'] = LGBMRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        if CatBoostRegressor is not None:
            models['catboost'] = CatBoostRegressor(
                iterations=100,
                random_state=42,
                verbose=False
            )
        
        return models
    
    def get_optimized_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models with optimized parameters.
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {
            'random_forest_optimized': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting_optimized': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        
        # Optional boosting libraries (only add if installed)
        if XGBRegressor is not None:
            models['xgboost_optimized'] = XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        if LGBMRegressor is not None:
            models['lightgbm_optimized'] = LGBMRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        if CatBoostRegressor is not None:
            models['catboost_optimized'] = CatBoostRegressor(
                iterations=200,
                depth=7,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        
        return models
    
    def train_single_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                          model_name: str = None) -> Any:
        """
        Train a single model.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            
        Returns:
            Trained model
        """
        if model_name:
            logger.info(f"Training {model_name}...")
        
        try:
            model.fit(X_train, y_train)
            if model_name:
                logger.info(f"{model_name} training completed")
            return model
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            return None
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        model_subset: List[str] = None,
                        use_optimized: bool = False) -> Dict[str, Any]:
        """
        Train multiple models.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_subset: List of model names to train (None = all)
            use_optimized: Whether to use optimized parameters
            
        Returns:
            Dictionary of trained models
        """
        logger.info("="*80)
        logger.info("TRAINING MODELS")
        logger.info("="*80)
        logger.info(f"Training set size: {X_train.shape}")
        
        # Get models
        if use_optimized:
            self.models = self.get_optimized_models()
            logger.info("Using optimized model parameters")
        else:
            self.models = self.get_default_models()
            logger.info("Using default model parameters")
        
        # Filter to subset if specified
        if model_subset:
            self.models = {k: v for k, v in self.models.items() if k in model_subset}
            logger.info(f"Training subset of {len(self.models)} models: {list(self.models.keys())}")
        else:
            logger.info(f"Training all {len(self.models)} models")
        
        # Train each model
        for name, model in self.models.items():
            trained_model = self.train_single_model(model, X_train, y_train, name)
            if trained_model is not None:
                self.trained_models[name] = trained_model
        
        logger.info(f"\nSuccessfully trained {len(self.trained_models)} models")
        logger.info("="*80)
        
        return self.trained_models
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model
            X: Features
            
        Returns:
            Predictions array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        model = self.trained_models[model_name]
        return model.predict(X)
    
    def predict_all(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using all trained models.
        
        Args:
            X: Features
            
        Returns:
            Dictionary of model name to predictions
        """
        predictions = {}
        
        for name, model in self.trained_models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logger.error(f"Error predicting with {name}: {str(e)}")
        
        return predictions
    
    def save_model(self, model_name: str, save_path: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        model = self.trained_models[model_name]
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model '{model_name}' saved to: {save_path}")
    
    def save_all_models(self, save_dir: str):
        """
        Save all trained models to directory.
        
        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.trained_models.items():
            model_path = save_path / f"{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved: {name}.pkl")
        
        logger.info(f"\nAll models saved to: {save_dir}")
    
    def load_model(self, model_name: str, load_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name to assign to the model
            load_path: Path to load the model from
        """
        with open(load_path, 'rb') as f:
            model = pickle.load(f)
        
        self.trained_models[model_name] = model
        logger.info(f"Model '{model_name}' loaded from: {load_path}")
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a trained model by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Trained model instance
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        return self.trained_models[model_name]
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all trained models."""
        return self.trained_models
    
    def set_best_model(self, model_name: str):
        """
        Set the best performing model.
        
        Args:
            model_name: Name of the best model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        self.best_model_name = model_name
        self.best_model = self.trained_models[model_name]
        logger.info(f"Best model set to: {model_name}")
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Returns:
            Tuple of (model name, model instance)
        """
        if self.best_model is None:
            raise ValueError("Best model not set. Call set_best_model first.")
        
        return self.best_model_name, self.best_model
    
    def get_model_params(self, model_name: str) -> Dict:
        """
        Get parameters of a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of model parameters
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        model = self.trained_models[model_name]
        return model.get_params()
    
    def get_feature_importance(self, model_name: str, 
                              feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        model = self.trained_models[model_name]
        
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model '{model_name}' does not have feature importances")
            return None
        
        importances = pd.DataFrame({
            'feature': feature_names if feature_names else range(len(model.feature_importances_)),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importances


class EnsembleModel:
    """Create ensemble of multiple models."""
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        """
        Initialize EnsembleModel.
        
        Args:
            models: Dictionary of model name to trained model
            weights: Dictionary of model name to weight (None = equal weights)
        """
        self.models = models
        
        if weights is None:
            # Equal weights
            self.weights = {name: 1.0 / len(models) for name in models.keys()}
        else:
            # Normalize weights
            total_weight = sum(weights.values())
            self.weights = {name: w / total_weight for name, w in weights.items()}
        
        logger.info(f"Ensemble created with {len(models)} models")
        logger.info(f"Weights: {self.weights}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make weighted ensemble predictions.
        
        Args:
            X: Features
            
        Returns:
            Ensemble predictions
        """
        predictions = np.zeros(len(X))
        
        for name, model in self.models.items():
            weight = self.weights[name]
            predictions += weight * model.predict(X)
        
        return predictions
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit all models in ensemble.
        
        Args:
            X: Training features
            y: Training target
        """
        for name, model in self.models.items():
            logger.info(f"Training {name} for ensemble...")
            model.fit(X, y)
        
        logger.info("Ensemble training completed")
