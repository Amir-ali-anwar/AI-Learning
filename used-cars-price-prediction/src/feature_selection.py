"""
Feature Selection module for selecting the most important features.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select the most important features for modeling."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FeatureSelector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.selected_features_ = None
        self.feature_scores_ = None
        self.selector_ = None
        
    def remove_low_variance_features(self, X: pd.DataFrame, 
                                     threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with low variance.
        
        Args:
            X: Feature DataFrame
            threshold: Variance threshold
            
        Returns:
            Tuple of (filtered DataFrame, removed features)
        """
        logger.info(f"Removing low variance features (threshold: {threshold})")
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        # Get feature names
        selected_features = X.columns[selector.get_support()].tolist()
        removed_features = X.columns[~selector.get_support()].tolist()
        
        X_filtered = X[selected_features]
        
        logger.info(f"Removed {len(removed_features)} low variance features")
        if removed_features:
            logger.info(f"Removed features: {removed_features[:10]}...")
        
        return X_filtered, removed_features
    
    def remove_correlated_features(self, X: pd.DataFrame, 
                                   threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold
            
        Returns:
            Tuple of (filtered DataFrame, removed features)
        """
        logger.info(f"Removing highly correlated features (threshold: {threshold})")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        X_filtered = X.drop(columns=to_drop)
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        if to_drop:
            logger.info(f"Removed features: {to_drop[:10]}...")
        
        return X_filtered, to_drop
    
    def select_k_best_features(self, X: pd.DataFrame, y: pd.Series, 
                               k: int = 50, score_func=f_regression) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select K best features using statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            score_func: Scoring function
            
        Returns:
            Tuple of (selected features DataFrame, scores DataFrame)
        """
        logger.info(f"Selecting {k} best features using {score_func.__name__}")
        
        # Adjust k if larger than number of features
        k = min(k, X.shape[1])
        
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        # Get feature scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        X_selected = X[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Top 10 features: {scores.head(10)['feature'].tolist()}")
        
        return X_selected, scores
    
    def select_with_mutual_info(self, X: pd.DataFrame, y: pd.Series, 
                                k: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select features using mutual information.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            
        Returns:
            Tuple of (selected features DataFrame, scores DataFrame)
        """
        logger.info(f"Selecting {k} features using mutual information")
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create scores DataFrame
        scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Select top k features
        k = min(k, len(scores))
        selected_features = scores.head(k)['feature'].tolist()
        X_selected = X[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Top 10 features: {scores.head(10)['feature'].tolist()}")
        
        return X_selected, scores
    
    def select_with_rfe(self, X: pd.DataFrame, y: pd.Series, 
                       n_features: int = 50, step: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            step: Number of features to remove at each iteration
            
        Returns:
            Tuple of (selected features DataFrame, ranking DataFrame)
        """
        logger.info(f"Selecting {n_features} features using RFE")
        
        # Adjust n_features if larger than number of features
        n_features = min(n_features, X.shape[1])
        
        # Use RandomForest as estimator
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
        selector.fit(X, y)
        
        # Get feature rankings
        rankings = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_,
            'selected': selector.support_
        }).sort_values('ranking')
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        X_selected = X[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Top 10 features: {rankings[rankings['selected']].head(10)['feature'].tolist()}")
        
        return X_selected, rankings
    
    def select_with_lasso(self, X: pd.DataFrame, y: pd.Series, 
                         n_features: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select features using Lasso regularization.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Approximate number of features to select
            
        Returns:
            Tuple of (selected features DataFrame, coefficients DataFrame)
        """
        logger.info(f"Selecting features using Lasso regularization")
        
        # Fit Lasso with cross-validation
        lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
        lasso.fit(X, y)
        
        # Get feature coefficients
        coefs = pd.DataFrame({
            'feature': X.columns,
            'coefficient': np.abs(lasso.coef_)
        }).sort_values('coefficient', ascending=False)
        
        # Select features with non-zero coefficients
        selected_features = coefs[coefs['coefficient'] > 0]['feature'].tolist()
        
        # If too many features, select top n_features
        if len(selected_features) > n_features:
            selected_features = selected_features[:n_features]
        
        X_selected = X[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features with non-zero coefficients")
        logger.info(f"Top 10 features: {coefs.head(10)['feature'].tolist()}")
        
        return X_selected, coefs
    
    def select_with_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                                  n_features: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select features using Random Forest feature importance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            
        Returns:
            Tuple of (selected features DataFrame, importance DataFrame)
        """
        logger.info(f"Selecting {n_features} features using Random Forest importance")
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top n_features
        n_features = min(n_features, len(importances))
        selected_features = importances.head(n_features)['feature'].tolist()
        X_selected = X[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Top 10 features: {importances.head(10)['feature'].tolist()}")
        
        return X_selected, importances
    
    def select_features_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                                n_features: int = 50,
                                methods: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select features using ensemble of multiple methods.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            methods: List of methods to use
            
        Returns:
            Tuple of (selected features DataFrame, combined scores DataFrame)
        """
        logger.info(f"Selecting {n_features} features using ensemble approach")
        
        if methods is None:
            methods = ['mutual_info', 'random_forest', 'lasso']
        
        all_scores = pd.DataFrame({'feature': X.columns})
        
        # Mutual Information
        if 'mutual_info' in methods:
            _, mi_scores = self.select_with_mutual_info(X, y, k=X.shape[1])
            # Normalize scores to 0-1
            mi_scores['mi_score_norm'] = (mi_scores['mi_score'] - mi_scores['mi_score'].min()) / \
                                         (mi_scores['mi_score'].max() - mi_scores['mi_score'].min())
            all_scores = all_scores.merge(mi_scores[['feature', 'mi_score_norm']], on='feature')
        
        # Random Forest
        if 'random_forest' in methods:
            _, rf_importance = self.select_with_random_forest(X, y, n_features=X.shape[1])
            # Normalize scores to 0-1
            rf_importance['importance_norm'] = (rf_importance['importance'] - rf_importance['importance'].min()) / \
                                               (rf_importance['importance'].max() - rf_importance['importance'].min())
            all_scores = all_scores.merge(rf_importance[['feature', 'importance_norm']], on='feature')
        
        # Lasso
        if 'lasso' in methods:
            _, lasso_coefs = self.select_with_lasso(X, y, n_features=X.shape[1])
            # Normalize scores to 0-1
            lasso_coefs['coefficient_norm'] = (lasso_coefs['coefficient'] - lasso_coefs['coefficient'].min()) / \
                                              (lasso_coefs['coefficient'].max() - lasso_coefs['coefficient'].min())
            all_scores = all_scores.merge(lasso_coefs[['feature', 'coefficient_norm']], on='feature')
        
        # Calculate average score
        score_columns = [col for col in all_scores.columns if col.endswith('_norm')]
        all_scores['avg_score'] = all_scores[score_columns].mean(axis=1)
        all_scores = all_scores.sort_values('avg_score', ascending=False)
        
        # Select top n_features
        n_features = min(n_features, len(all_scores))
        selected_features = all_scores.head(n_features)['feature'].tolist()
        X_selected = X[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features using ensemble")
        logger.info(f"Top 10 features: {all_scores.head(10)['feature'].tolist()}")
        
        self.feature_scores_ = all_scores
        self.selected_features_ = selected_features
        
        return X_selected, all_scores
    
    def plot_feature_importance(self, scores: pd.DataFrame, top_n: int = 20, 
                               score_col: str = 'avg_score', save_path: str = None):
        """
        Plot feature importance scores.
        
        Args:
            scores: DataFrame with feature scores
            top_n: Number of top features to plot
            score_col: Name of score column
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 8))
        
        top_features = scores.head(top_n)
        
        plt.barh(range(len(top_features)), top_features[score_col], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, 
                     method: str = 'ensemble', n_features: int = 50) -> pd.DataFrame:
        """
        Select features using specified method.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method
            n_features: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Starting feature selection using {method} method")
        logger.info(f"Initial features: {X.shape[1]}")
        
        # Remove low variance features first
        X_filtered, _ = self.remove_low_variance_features(
            X, threshold=self.config.get('variance_threshold', 0.01)
        )
        
        # Remove highly correlated features
        X_filtered, _ = self.remove_correlated_features(
            X_filtered, threshold=self.config.get('correlation_threshold', 0.95)
        )
        
        # Select features using specified method
        if method == 'k_best':
            X_selected, scores = self.select_k_best_features(X_filtered, y, k=n_features)
        elif method == 'mutual_info':
            X_selected, scores = self.select_with_mutual_info(X_filtered, y, k=n_features)
        elif method == 'rfe':
            X_selected, scores = self.select_with_rfe(X_filtered, y, n_features=n_features)
        elif method == 'lasso':
            X_selected, scores = self.select_with_lasso(X_filtered, y, n_features=n_features)
        elif method == 'random_forest':
            X_selected, scores = self.select_with_random_forest(X_filtered, y, n_features=n_features)
        elif method == 'ensemble':
            X_selected, scores = self.select_features_ensemble(X_filtered, y, n_features=n_features)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.selected_features_ = X_selected.columns.tolist()
        self.feature_scores_ = scores
        
        logger.info(f"Feature selection completed!")
        logger.info(f"Final features: {X_selected.shape[1]}")
        
        return X_selected
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using selected features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with selected features
        """
        if self.selected_features_ is None:
            raise ValueError("FeatureSelector not fitted. Call fit_transform first.")
        
        # Select only the features that were selected during fit
        available_features = [f for f in self.selected_features_ if f in X.columns]
        
        if len(available_features) < len(self.selected_features_):
            missing = set(self.selected_features_) - set(available_features)
            logger.warning(f"Missing {len(missing)} features in new data: {list(missing)[:5]}...")
        
        return X[available_features]
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        return self.selected_features_
    
    def get_feature_scores(self) -> pd.DataFrame:
        """Get feature scores DataFrame."""
        return self.feature_scores_
