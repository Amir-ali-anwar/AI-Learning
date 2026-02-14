"""
Hyperparameter Tuning module for optimizing model parameters.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Tune hyperparameters for machine learning models."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize HyperparameterTuner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.best_params_ = {}
        self.best_models_ = {}
        self.tuning_results_ = {}
        
    def get_param_grid(self, model_name: str) -> Dict:
        """
        Get parameter grid for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of parameter grid
        """
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, 9],
                'num_leaves': [31, 50, 70],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            'catboost': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5]
            }
        }
        
        return param_grids.get(model_name, {})
    
    def get_param_distributions(self, model_name: str) -> Dict:
        """
        Get parameter distributions for random search.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of parameter distributions
        """
        from scipy.stats import randint, uniform
        
        param_distributions = {
            'random_forest': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(5, 50),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': randint(50, 500),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 15),
                'subsample': uniform(0.6, 0.4),
                'min_samples_split': randint(2, 20)
            },
            'xgboost': {
                'n_estimators': randint(50, 500),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 15),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'gamma': uniform(0, 0.5)
            },
            'lightgbm': {
                'n_estimators': randint(50, 500),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 15),
                'num_leaves': randint(20, 100),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            },
            'catboost': {
                'iterations': randint(50, 500),
                'learning_rate': uniform(0.01, 0.3),
                'depth': randint(3, 12),
                'l2_leaf_reg': uniform(1, 10)
            }
        }
        
        return param_distributions.get(model_name, {})
    
    def grid_search(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                   param_grid: Dict, cv: int = 5, scoring: str = 'r2',
                   n_jobs: int = -1) -> Tuple[Any, Dict]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info(f"Starting grid search with {len(param_grid)} parameters...")
        logger.info(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Grid search completed!")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def random_search(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                     param_distributions: Dict, n_iter: int = 50, cv: int = 5,
                     scoring: str = 'r2', n_jobs: int = -1) -> Tuple[Any, Dict]:
        """
        Perform random search for hyperparameter tuning.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training target
            param_distributions: Parameter distributions
            n_iter: Number of iterations
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info(f"Starting random search with {n_iter} iterations...")
        logger.info(f"Parameter distributions: {param_distributions}")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"Random search completed!")
        logger.info(f"Best score: {random_search.best_score_:.4f}")
        logger.info(f"Best parameters: {random_search.best_params_}")
        
        return random_search.best_estimator_, random_search.best_params_
    
    def tune_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                          method: str = 'grid', **kwargs) -> Tuple[Any, Dict]:
        """
        Tune Random Forest hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Tuning method ('grid' or 'random')
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info("="*80)
        logger.info("TUNING RANDOM FOREST")
        logger.info("="*80)
        
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        if method == 'grid':
            param_grid = kwargs.get('param_grid', self.get_param_grid('random_forest'))
            best_model, best_params = self.grid_search(
                model, X_train, y_train, param_grid,
                cv=kwargs.get('cv', 5),
                scoring=kwargs.get('scoring', 'r2')
            )
        else:  # random
            param_dist = kwargs.get('param_distributions', self.get_param_distributions('random_forest'))
            best_model, best_params = self.random_search(
                model, X_train, y_train, param_dist,
                n_iter=kwargs.get('n_iter', 50),
                cv=kwargs.get('cv', 5),
                scoring=kwargs.get('scoring', 'r2')
            )
        
        self.best_params_['random_forest'] = best_params
        self.best_models_['random_forest'] = best_model
        
        return best_model, best_params
    
    def tune_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                    method: str = 'grid', **kwargs) -> Tuple[Any, Dict]:
        """
        Tune XGBoost hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Tuning method ('grid' or 'random')
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info("="*80)
        logger.info("TUNING XGBOOST")
        logger.info("="*80)
        
        model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        
        if method == 'grid':
            param_grid = kwargs.get('param_grid', self.get_param_grid('xgboost'))
            best_model, best_params = self.grid_search(
                model, X_train, y_train, param_grid,
                cv=kwargs.get('cv', 5),
                scoring=kwargs.get('scoring', 'r2')
            )
        else:  # random
            param_dist = kwargs.get('param_distributions', self.get_param_distributions('xgboost'))
            best_model, best_params = self.random_search(
                model, X_train, y_train, param_dist,
                n_iter=kwargs.get('n_iter', 50),
                cv=kwargs.get('cv', 5),
                scoring=kwargs.get('scoring', 'r2')
            )
        
        self.best_params_['xgboost'] = best_params
        self.best_models_['xgboost'] = best_model
        
        return best_model, best_params
    
    def tune_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                     method: str = 'grid', **kwargs) -> Tuple[Any, Dict]:
        """
        Tune LightGBM hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Tuning method ('grid' or 'random')
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info("="*80)
        logger.info("TUNING LIGHTGBM")
        logger.info("="*80)
        
        model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        
        if method == 'grid':
            param_grid = kwargs.get('param_grid', self.get_param_grid('lightgbm'))
            best_model, best_params = self.grid_search(
                model, X_train, y_train, param_grid,
                cv=kwargs.get('cv', 5),
                scoring=kwargs.get('scoring', 'r2')
            )
        else:  # random
            param_dist = kwargs.get('param_distributions', self.get_param_distributions('lightgbm'))
            best_model, best_params = self.random_search(
                model, X_train, y_train, param_dist,
                n_iter=kwargs.get('n_iter', 50),
                cv=kwargs.get('cv', 5),
                scoring=kwargs.get('scoring', 'r2')
            )
        
        self.best_params_['lightgbm'] = best_params
        self.best_models_['lightgbm'] = best_model
        
        return best_model, best_params
    
    def tune_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     method: str = 'grid', **kwargs) -> Tuple[Any, Dict]:
        """
        Tune CatBoost hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Tuning method ('grid' or 'random')
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info("="*80)
        logger.info("TUNING CATBOOST")
        logger.info("="*80)
        
        model = CatBoostRegressor(random_state=42, verbose=False)
        
        if method == 'grid':
            param_grid = kwargs.get('param_grid', self.get_param_grid('catboost'))
            best_model, best_params = self.grid_search(
                model, X_train, y_train, param_grid,
                cv=kwargs.get('cv', 5),
                scoring=kwargs.get('scoring', 'r2')
            )
        else:  # random
            param_dist = kwargs.get('param_distributions', self.get_param_distributions('catboost'))
            best_model, best_params = self.random_search(
                model, X_train, y_train, param_dist,
                n_iter=kwargs.get('n_iter', 50),
                cv=kwargs.get('cv', 5),
                scoring=kwargs.get('scoring', 'r2')
            )
        
        self.best_params_['catboost'] = best_params
        self.best_models_['catboost'] = best_model
        
        return best_model, best_params
    
    def tune_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                              method: str = 'grid', **kwargs) -> Tuple[Any, Dict]:
        """
        Tune Gradient Boosting hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Tuning method ('grid' or 'random')
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info("="*80)
        logger.info("TUNING GRADIENT BOOSTING")
        logger.info("="*80)
        
        model = GradientBoostingRegressor(random_state=42)
        
        if method == 'grid':
            param_grid = kwargs.get('param_grid', self.get_param_grid('gradient_boosting'))
            best_model, best_params = self.grid_search(
                model, X_train, y_train, param_grid,
                cv=kwargs.get('cv', 5),
                scoring=kwargs.get('scoring', 'r2')
            )
        else:  # random
            param_dist = kwargs.get('param_distributions', self.get_param_distributions('gradient_boosting'))
            best_model, best_params = self.random_search(
                model, X_train, y_train, param_dist,
                n_iter=kwargs.get('n_iter', 50),
                cv=kwargs.get('cv', 5),
                scoring=kwargs.get('scoring', 'r2')
            )
        
        self.best_params_['gradient_boosting'] = best_params
        self.best_models_['gradient_boosting'] = best_model
        
        return best_model, best_params
    
    def tune_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                       models: List[str] = None, method: str = 'random',
                       **kwargs) -> Dict[str, Tuple[Any, Dict]]:
        """
        Tune multiple models.
        
        Args:
            X_train: Training features
            y_train: Training target
            models: List of model names to tune
            method: Tuning method ('grid' or 'random')
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of model name to (best model, best parameters)
        """
        if models is None:
            models = ['random_forest', 'xgboost', 'lightgbm']
        
        logger.info("="*80)
        logger.info(f"TUNING {len(models)} MODELS")
        logger.info("="*80)
        logger.info(f"Models: {models}")
        logger.info(f"Method: {method}")
        
        results = {}
        
        for model_name in models:
            try:
                if model_name == 'random_forest':
                    best_model, best_params = self.tune_random_forest(X_train, y_train, method, **kwargs)
                elif model_name == 'xgboost':
                    best_model, best_params = self.tune_xgboost(X_train, y_train, method, **kwargs)
                elif model_name == 'lightgbm':
                    best_model, best_params = self.tune_lightgbm(X_train, y_train, method, **kwargs)
                elif model_name == 'catboost':
                    best_model, best_params = self.tune_catboost(X_train, y_train, method, **kwargs)
                elif model_name == 'gradient_boosting':
                    best_model, best_params = self.tune_gradient_boosting(X_train, y_train, method, **kwargs)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                results[model_name] = (best_model, best_params)
                
            except Exception as e:
                logger.error(f"Error tuning {model_name}: {str(e)}")
        
        logger.info(f"\nSuccessfully tuned {len(results)} models")
        
        return results
    
    def save_best_params(self, save_path: str):
        """
        Save best parameters to file.
        
        Args:
            save_path: Path to save parameters
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.best_params_, f)
        
        logger.info(f"Best parameters saved to: {save_path}")
    
    def load_best_params(self, load_path: str):
        """
        Load best parameters from file.
        
        Args:
            load_path: Path to load parameters from
        """
        with open(load_path, 'rb') as f:
            self.best_params_ = pickle.load(f)
        
        logger.info(f"Best parameters loaded from: {load_path}")
    
    def get_best_params(self, model_name: str = None) -> Dict:
        """
        Get best parameters.
        
        Args:
            model_name: Name of the model (None = all)
            
        Returns:
            Dictionary of best parameters
        """
        if model_name:
            return self.best_params_.get(model_name, {})
        return self.best_params_
    
    def get_best_model(self, model_name: str) -> Any:
        """
        Get best model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Best model instance
        """
        return self.best_models_.get(model_name)
    
    def get_all_best_models(self) -> Dict[str, Any]:
        """Get all best models."""
        return self.best_models_
