"""
Production-ready preprocessing pipeline for used cars price prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import warnings
# Only suppress known noisy warnings; do NOT suppress all warnings globally
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*X does not have valid feature names.*')

logger = logging.getLogger(__name__)


class OutlierHandler:
    """Handle outliers in numerical data."""
    
    def __init__(self, method: str = 'iqr', iqr_multiplier: float = 1.5, 
                 zscore_threshold: float = 3, handle_method: str = 'cap'):
        """
        Initialize OutlierHandler.
        
        Args:
            method: Detection method ('iqr' or 'zscore')
            iqr_multiplier: Multiplier for IQR method
            zscore_threshold: Threshold for z-score method
            handle_method: How to handle outliers ('remove', 'cap', 'keep')
        """
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.handle_method = handle_method
        self.bounds_ = {}
        
    def detect_outliers_iqr(self, series: pd.Series) -> Tuple[float, float]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.iqr_multiplier * IQR
        upper_bound = Q3 + self.iqr_multiplier * IQR
        return lower_bound, upper_bound
    
    def detect_outliers_zscore(self, series: pd.Series) -> Tuple[float, float]:
        """Detect outliers using z-score method."""
        mean = series.mean()
        std = series.std()
        lower_bound = mean - self.zscore_threshold * std
        upper_bound = mean + self.zscore_threshold * std
        return lower_bound, upper_bound
    
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'OutlierHandler':
        """
        Fit outlier bounds on data.
        
        Args:
            df: DataFrame to fit
            columns: List of numerical columns
            
        Returns:
            self
        """
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
            
            if self.method == 'iqr':
                lower, upper = self.detect_outliers_iqr(df[col].dropna())
            elif self.method == 'zscore':
                lower, upper = self.detect_outliers_zscore(df[col].dropna())
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.bounds_[col] = {'lower': lower, 'upper': upper}
            
            outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
            logger.info(f"Column '{col}': {outlier_count} outliers detected (bounds: [{lower:.2f}, {upper:.2f}])")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by handling outliers.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        df_transformed = df.copy()
        
        for col, bounds in self.bounds_.items():
            if col not in df_transformed.columns:
                continue
            
            lower, upper = bounds['lower'], bounds['upper']
            
            if self.handle_method == 'remove':
                # Remove rows with outliers
                mask = (df_transformed[col] >= lower) & (df_transformed[col] <= upper)
                removed_count = (~mask).sum()
                df_transformed = df_transformed[mask]
                logger.info(f"Column '{col}': Removed {removed_count} outlier rows")
                
            elif self.handle_method == 'cap':
                # Cap outliers to bounds
                capped_lower = (df_transformed[col] < lower).sum()
                capped_upper = (df_transformed[col] > upper).sum()
                df_transformed[col] = df_transformed[col].clip(lower, upper)
                logger.info(f"Column '{col}': Capped {capped_lower} lower and {capped_upper} upper outliers")
                
            elif self.handle_method == 'keep':
                # Keep outliers as is
                logger.info(f"Column '{col}': Keeping outliers")
            else:
                raise ValueError(f"Unknown handle_method: {self.handle_method}")
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, columns).transform(df)


class DataPreprocessor:
    """Main preprocessing pipeline."""
    
    def __init__(self, config: Dict):
        """
        Initialize DataPreprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.outlier_handler = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names_ = None
        
    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns."""
        columns_to_drop = self.config.get('COLUMNS_TO_DROP', [])
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        
        if existing_cols:
            df_dropped = df.drop(columns=existing_cols)
            logger.info(f"Dropped {len(existing_cols)} columns: {existing_cols}")
            return df_dropped
        
        return df
    
    def filter_by_domain_knowledge(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on domain knowledge."""
        df_filtered = df.copy()
        initial_rows = len(df_filtered)
        
        # Filter price
        price_filter = self.config.get('PRICE_FILTER', {})
        if 'price' in df_filtered.columns and price_filter:
            df_filtered = df_filtered[
                (df_filtered['price'] >= price_filter.get('min', 0)) &
                (df_filtered['price'] <= price_filter.get('max', float('inf')))
            ]
            logger.info(f"Price filter: Removed {initial_rows - len(df_filtered)} rows")
        
        # Filter year
        year_filter = self.config.get('YEAR_FILTER', {})
        if 'year' in df_filtered.columns and year_filter:
            initial_rows = len(df_filtered)
            df_filtered = df_filtered[
                (df_filtered['year'] >= year_filter.get('min', 0)) &
                (df_filtered['year'] <= year_filter.get('max', float('inf')))
            ]
            logger.info(f"Year filter: Removed {initial_rows - len(df_filtered)} rows")
        
        # Filter odometer
        odometer_filter = self.config.get('ODOMETER_FILTER', {})
        if 'odometer' in df_filtered.columns and odometer_filter:
            initial_rows = len(df_filtered)
            df_filtered = df_filtered[
                (df_filtered['odometer'] >= odometer_filter.get('min', 0)) &
                (df_filtered['odometer'] <= odometer_filter.get('max', float('inf')))
            ]
            logger.info(f"Odometer filter: Removed {initial_rows - len(df_filtered)} rows")
        
        return df_filtered
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_rows = len(df)
        df_no_duplicates = df.drop_duplicates()
        removed = initial_rows - len(df_no_duplicates)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df_no_duplicates
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using configured strategies."""
        df_imputed = df.copy()
        
        imputation_strategy = self.config.get('IMPUTATION_STRATEGY', {})
        constant_values = self.config.get('CONSTANT_VALUES', {})
        
        for col, strategy in imputation_strategy.items():
            if col not in df_imputed.columns:
                continue
            
            missing_count = df_imputed[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'mode':
                mode_value = df_imputed[col].mode()[0] if len(df_imputed[col].mode()) > 0 else None
                if mode_value is not None:
                    df_imputed[col].fillna(mode_value, inplace=True)
                    logger.info(f"Column '{col}': Imputed {missing_count} missing values with mode '{mode_value}'")
                    
            elif strategy == 'constant':
                constant_value = constant_values.get(col, 'unknown')
                df_imputed[col].fillna(constant_value, inplace=True)
                logger.info(f"Column '{col}': Imputed {missing_count} missing values with constant '{constant_value}'")
                
            elif strategy == 'median':
                median_value = df_imputed[col].median()
                df_imputed[col].fillna(median_value, inplace=True)
                logger.info(f"Column '{col}': Imputed {missing_count} missing values with median {median_value:.2f}")
                
            elif strategy == 'mean':
                mean_value = df_imputed[col].mean()
                df_imputed[col].fillna(mean_value, inplace=True)
                logger.info(f"Column '{col}': Imputed {missing_count} missing values with mean {mean_value:.2f}")
        
        # Drop rows with missing target
        target_col = self.config.get('TARGET_COLUMN')
        if target_col and target_col in df_imputed.columns:
            initial_rows = len(df_imputed)
            df_imputed = df_imputed.dropna(subset=[target_col])
            removed = initial_rows - len(df_imputed)
            if removed > 0:
                logger.info(f"Removed {removed} rows with missing target values")
        
        return df_imputed
    
    def handle_outliers(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle outliers in numerical columns."""
        outlier_config = self.config.get('OUTLIER_CONFIG', {})
        numerical_cols = self.config.get('NUMERICAL_COLUMNS', [])
        
        # Filter to existing columns
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if not numerical_cols:
            logger.warning("No numerical columns found for outlier handling")
            return df
        
        if fit:
            self.outlier_handler = OutlierHandler(
                method=outlier_config.get('method', 'iqr'),
                iqr_multiplier=outlier_config.get('iqr_multiplier', 1.5),
                zscore_threshold=outlier_config.get('zscore_threshold', 3),
                handle_method=outlier_config.get('handle_method', 'cap')
            )
            return self.outlier_handler.fit_transform(df, numerical_cols)
        else:
            if self.outlier_handler is None:
                raise ValueError("OutlierHandler not fitted. Call with fit=True first.")
            return self.outlier_handler.transform(df)
    
    def encode_categorical_variables(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df_encoded = df.copy()
        encoding_config = self.config.get('ENCODING_CONFIG', {})
        
        # Ordinal encoding
        ordinal_columns = encoding_config.get('ordinal_columns', {})
        for col, categories in ordinal_columns.items():
            if col not in df_encoded.columns:
                continue
            
            # Create mapping
            category_mapping = {cat: idx for idx, cat in enumerate(categories)}
            df_encoded[col] = df_encoded[col].map(category_mapping)
            if fit:
                logger.info(f"Ordinal encoded column '{col}' with {len(categories)} categories")
        
        # One-hot encoding
        onehot_columns = encoding_config.get('onehot_columns', [])
        onehot_columns = [col for col in onehot_columns if col in df_encoded.columns]
        
        if onehot_columns:
            df_encoded = pd.get_dummies(
                df_encoded,
                columns=onehot_columns,
                drop_first=encoding_config.get('drop_first', True),
                prefix=onehot_columns
            )
            if fit:
                logger.info(f"One-hot encoded {len(onehot_columns)} columns: {onehot_columns}")
        
        # Label encoding for remaining object columns (if any)
        object_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Label encoded column '{col}'")
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )
        
        # Ensure column consistency between fit and transform
        if fit:
            self.encoded_feature_names_ = df_encoded.columns.tolist()
        else:
            if hasattr(self, 'encoded_feature_names_'):
                # Reindex to ensure all columns from training are present
                # This handles missing categories in new data (fills with 0)
                # and removes extra columns from new categories (drops them)
                df_encoded = df_encoded.reindex(columns=self.encoded_feature_names_, fill_value=0)
            else:
                logger.warning("encoded_feature_names_ not found. Skipping column alignment.")
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        scaling_config = self.config.get('SCALING_CONFIG', {})
        columns_to_scale = scaling_config.get('columns', [])
        columns_to_scale = [col for col in columns_to_scale if col in df.columns]
        
        if not columns_to_scale:
            logger.info("No columns to scale")
            return df
        
        df_scaled = df.copy()
        
        if fit:
            method = scaling_config.get('method', 'standard')
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            # Ensure columns are numeric
            for col in columns_to_scale:
                df_scaled[col] = pd.to_numeric(df_scaled[col], errors='coerce')
            
            df_scaled[columns_to_scale] = self.scaler.fit_transform(df_scaled[columns_to_scale])
            logger.info(f"Scaled {len(columns_to_scale)} columns using {method} scaling")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df_scaled[columns_to_scale] = self.scaler.transform(df_scaled[columns_to_scale])
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the entire preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")
        logger.info(f"Initial shape: {df.shape}")
        
        # Step 1: Drop unnecessary columns
        df = self.drop_columns(df)
        logger.info(f"After dropping columns: {df.shape}")
        
        # Step 2: Filter by domain knowledge
        df = self.filter_by_domain_knowledge(df)
        logger.info(f"After domain filtering: {df.shape}")
        
        # Step 3: Remove duplicates
        df = self.remove_duplicates(df)
        logger.info(f"After removing duplicates: {df.shape}")
        
        # Step 4: Handle missing values
        df = self.handle_missing_values(df)
        logger.info(f"After handling missing values: {df.shape}")
        
        # Step 5: Handle outliers
        df = self.handle_outliers(df, fit=True)
        logger.info(f"After handling outliers: {df.shape}")
        
        # Step 6: Encode categorical variables
        df = self.encode_categorical_variables(df, fit=True)
        logger.info(f"After encoding: {df.shape}")
        
        # Step 7: Scale features
        df = self.scale_features(df, fit=True)
        logger.info(f"After scaling: {df.shape}")
        
        # Store feature names
        self.feature_names_ = df.columns.tolist()
        
        logger.info("Preprocessing pipeline completed!")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Transforming new data...")
        
        df = self.drop_columns(df)
        df = self.filter_by_domain_knowledge(df)
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df, fit=False)
        df = self.encode_categorical_variables(df, fit=False)
        df = self.scale_features(df, fit=False)
        
        logger.info("Transformation completed!")
        return df


def split_data(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        df: Preprocessed DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    target_col = config.get('TARGET_COLUMN')
    split_config = config.get('TRAIN_TEST_SPLIT', {})
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_config.get('test_size', 0.2),
        random_state=split_config.get('random_state', 42),
        stratify=split_config.get('stratify')
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test
