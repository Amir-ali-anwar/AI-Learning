"""
Feature Engineering module for creating new features from existing ones.

PRODUCTION NOTE: This module deliberately excludes any features derived from
the target variable (price) to prevent data leakage. Features like
price_per_mile, price_per_year, and price aggregations by group would give
the model access to the target during training and be unavailable at
inference time.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create new features from existing data.
    
    Uses a fit/transform pattern:
    - fit_transform(): learns parameters from training data and creates features
    - transform(): applies the same feature engineering using stored parameters
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_names_created_ = []
        self.is_fitted_ = False
        # Store the current year at fit time for consistency
        self._fit_year = None
        
    def create_age_feature(self, df: pd.DataFrame, year_col: str = 'year', 
                          copy: bool = True) -> pd.DataFrame:
        """
        Create vehicle age feature.
        
        Args:
            df: DataFrame
            year_col: Name of year column
            
        Returns:
            DataFrame with age feature
        """
        df_new = df.copy() if copy else df
        
        if year_col in df_new.columns:
            df_new['vehicle_age'] = self._fit_year - df_new[year_col]
            # Ensure non-negative age (future model years get age 0)
            df_new['vehicle_age'] = df_new['vehicle_age'].clip(lower=0)
            if not self.is_fitted_:
                self.feature_names_created_.append('vehicle_age')
            logger.info(f"Created feature: vehicle_age (mean: {df_new['vehicle_age'].mean():.2f} years)")
        
        return df_new
    
    def create_mileage_features(self, df: pd.DataFrame, odometer_col: str = 'odometer', 
                                copy: bool = True) -> pd.DataFrame:
        """
        Create mileage-related features.
        
        Args:
            df: DataFrame
            odometer_col: Name of odometer column
            
        Returns:
            DataFrame with mileage features
        """
        df_new = df.copy() if copy else df
        
        if odometer_col in df_new.columns:
            # Average miles per year
            if 'vehicle_age' in df_new.columns:
                df_new['avg_miles_per_year'] = df_new[odometer_col] / (df_new['vehicle_age'] + 1)
                if not self.is_fitted_:
                    self.feature_names_created_.append('avg_miles_per_year')
                logger.info(f"Created feature: avg_miles_per_year")
            
            # Mileage category
            df_new['mileage_category'] = pd.cut(
                df_new[odometer_col],
                bins=[0, 30000, 60000, 100000, 150000, float('inf')],
                labels=False
            )
            if not self.is_fitted_:
                self.feature_names_created_.append('mileage_category')
            logger.info(f"Created feature: mileage_category")
            
            # Log transformation for skewed distribution
            df_new['odometer_log'] = np.log1p(df_new[odometer_col])
            if not self.is_fitted_:
                self.feature_names_created_.append('odometer_log')
            logger.info(f"Created feature: odometer_log")
        
        return df_new
    
    def create_age_categories(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Create age category features.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with age categories
        """
        df_new = df.copy() if copy else df
        
        if 'vehicle_age' in df_new.columns:
            # Age category
            df_new['age_category'] = pd.cut(
                df_new['vehicle_age'],
                bins=[0, 3, 7, 12, 20, float('inf')],
                labels=False
            )
            if not self.is_fitted_:
                self.feature_names_created_.append('age_category')
            logger.info(f"Created feature: age_category")
            
            # Is vintage (>25 years)
            df_new['is_vintage'] = (df_new['vehicle_age'] > 25).astype(int)
            if not self.is_fitted_:
                self.feature_names_created_.append('is_vintage')
            logger.info(f"Created feature: is_vintage")
        
        return df_new
    
    def create_interaction_features(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df_new = df.copy() if copy else df
        
        # Age * Mileage interaction
        if 'vehicle_age' in df_new.columns and 'odometer' in df_new.columns:
            df_new['age_mileage_interaction'] = df_new['vehicle_age'] * df_new['odometer']
            if not self.is_fitted_:
                self.feature_names_created_.append('age_mileage_interaction')
            logger.info(f"Created feature: age_mileage_interaction")
        
        return df_new
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                   columns: List[str] = None,
                                   degree: int = 2, copy: bool = True) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Args:
            df: DataFrame
            columns: List of column names for polynomial features
            degree: Degree of polynomial
            
        Returns:
            DataFrame with polynomial features
        """
        df_new = df.copy() if copy else df
        
        if columns is None:
            columns = ['vehicle_age', 'odometer']
        
        for col in columns:
            if col in df_new.columns:
                for d in range(2, degree + 1):
                    feature_name = f'{col}_pow{d}'
                    df_new[feature_name] = df_new[col] ** d
                    if not self.is_fitted_:
                        self.feature_names_created_.append(feature_name)
                    logger.info(f"Created feature: {feature_name}")
        
        return df_new
    
    def create_boolean_features(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Create boolean indicator features.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with boolean features
        """
        df_new = df.copy() if copy else df
        
        # Low mileage indicator
        if 'odometer' in df_new.columns:
            df_new['is_low_mileage'] = (df_new['odometer'] < 50000).astype(int)
            if not self.is_fitted_:
                self.feature_names_created_.append('is_low_mileage')
            logger.info(f"Created feature: is_low_mileage")
        
        # High mileage indicator
        if 'odometer' in df_new.columns:
            df_new['is_high_mileage'] = (df_new['odometer'] > 150000).astype(int)
            if not self.is_fitted_:
                self.feature_names_created_.append('is_high_mileage')
            logger.info(f"Created feature: is_high_mileage")
        
        # Recent model indicator
        if 'vehicle_age' in df_new.columns:
            df_new['is_recent_model'] = (df_new['vehicle_age'] <= 3).astype(int)
            if not self.is_fitted_:
                self.feature_names_created_.append('is_recent_model')
            logger.info(f"Created feature: is_recent_model")
        
        return df_new
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to create all features (shared by fit_transform and transform).
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        df_engineered = df.copy()
        
        # Create features in order (some depend on others)
        df_engineered = self.create_age_feature(df_engineered, copy=False)
        df_engineered = self.create_mileage_features(df_engineered, copy=False)
        df_engineered = self.create_age_categories(df_engineered, copy=False)
        df_engineered = self.create_interaction_features(df_engineered, copy=False)
        df_engineered = self.create_boolean_features(df_engineered, copy=False)
        
        # Optional: polynomial features (can create many features)
        if self.config.get('create_polynomial', False):
            df_engineered = self.create_polynomial_features(df_engineered, copy=False)
        
        return df_engineered
    
    def fit_transform(self, df: pd.DataFrame, create_all: bool = True) -> pd.DataFrame:
        """
        Fit the feature engineer on training data and create features.
        
        This records which features are created and stores parameters
        (like current year) for consistent transform on new data.
        
        Args:
            df: Training DataFrame
            create_all: Whether to create all features or use config
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering (fit_transform)...")
        logger.info(f"Initial shape: {df.shape}")
        
        # Reset state on re-fit
        self.feature_names_created_ = []
        self.is_fitted_ = False
        
        # Store the current year at fit time for consistency
        self._fit_year = datetime.now().year
        
        df_engineered = self._create_features(df)
        
        # Mark as fitted
        self.is_fitted_ = True
        
        logger.info(f"Feature engineering completed!")
        logger.info(f"Final shape: {df_engineered.shape}")
        logger.info(f"Created {len(self.feature_names_created_)} new features: {self.feature_names_created_}")
        
        return df_engineered
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the same feature engineering learned during fit.
        
        Unlike fit_transform, this does NOT reset feature_names_created_ and
        uses the same year stored during fit for consistency.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted_:
            logger.warning("FeatureEngineer not fitted yet. Calling fit_transform instead.")
            return self.fit_transform(df)
        
        logger.info("Starting feature engineering (transform)...")
        logger.info(f"Initial shape: {df.shape}")
        
        df_engineered = self._create_features(df)
        
        logger.info(f"Feature engineering completed!")
        logger.info(f"Final shape: {df_engineered.shape}")
        
        return df_engineered
    
    def get_feature_names(self) -> List[str]:
        """Get list of created feature names."""
        return self.feature_names_created_
