"""
Data validation utilities for the preprocessing pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and integrity."""
    
    def __init__(self, config: Dict):
        """
        Initialize DataValidator.
        
        Args:
            config: Configuration dictionary with validation thresholds
        """
        self.config = config
        
    def validate_dataframe(self, df: pd.DataFrame, stage: str = "raw") -> Tuple[bool, List[str]]:
        """
        Validate dataframe against quality checks.
        
        Args:
            df: DataFrame to validate
            stage: Stage of processing ('raw', 'cleaned', 'processed')
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if dataframe is empty
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check minimum rows
        if len(df) < self.config.get('min_rows', 1000):
            issues.append(f"DataFrame has only {len(df)} rows, minimum required: {self.config.get('min_rows', 1000)}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")
        
        # Check missing values
        max_missing_pct = self.config.get('max_missing_percentage', 50)
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > max_missing_pct:
                issues.append(f"Column '{col}' has {missing_pct:.2f}% missing values (max allowed: {max_missing_pct}%)")
        
        # Check data types
        if stage == "processed":
            # After processing, there should be no object columns (all should be encoded)
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            if len(object_cols) > 0:
                issues.append(f"Found {len(object_cols)} object columns in processed data: {object_cols}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_target_column(self, df: pd.DataFrame, target_col: str) -> Tuple[bool, List[str]]:
        """
        Validate target column.
        
        Args:
            df: DataFrame containing target column
            target_col: Name of target column
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if target_col not in df.columns:
            issues.append(f"Target column '{target_col}' not found in DataFrame")
            return False, issues
        
        # Check for missing values in target
        missing_target = df[target_col].isnull().sum()
        if missing_target > 0:
            issues.append(f"Target column has {missing_target} missing values")
        
        # Check for minimum unique values
        min_unique = self.config.get('min_unique_target', 10)
        unique_values = df[target_col].nunique()
        if unique_values < min_unique:
            issues.append(f"Target column has only {unique_values} unique values (minimum: {min_unique})")
        
        # Check for negative values (if target is price)
        if target_col.lower() == 'price':
            negative_values = (df[target_col] < 0).sum()
            if negative_values > 0:
                issues.append(f"Target column has {negative_values} negative values")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_numerical_columns(self, df: pd.DataFrame, numerical_cols: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate numerical columns.
        
        Args:
            df: DataFrame to validate
            numerical_cols: List of numerical column names
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        for col in numerical_cols:
            if col not in df.columns:
                issues.append(f"Numerical column '{col}' not found in DataFrame")
                continue
            
            # Check if column is actually numerical
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' is not numerical (dtype: {df[col].dtype})")
            
            # Check for infinite values
            if np.isinf(df[col]).any():
                inf_count = np.isinf(df[col]).sum()
                issues.append(f"Column '{col}' has {inf_count} infinite values")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_categorical_columns(self, df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate categorical columns.
        
        Args:
            df: DataFrame to validate
            categorical_cols: List of categorical column names
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        for col in categorical_cols:
            if col not in df.columns:
                issues.append(f"Categorical column '{col}' not found in DataFrame")
                continue
            
            # Check for single value columns
            unique_values = df[col].nunique()
            if unique_values == 1:
                issues.append(f"Column '{col}' has only 1 unique value")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with quality metrics for each column
        """
        report_data = []
        
        for col in df.columns:
            col_data = {
                'Column': col,
                'Dtype': str(df[col].dtype),
                'Missing_Count': df[col].isnull().sum(),
                'Missing_Percentage': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
                'Unique_Values': df[col].nunique(),
                'Duplicate_Count': df[col].duplicated().sum()
            }
            
            # Add numerical statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_data['Mean'] = f"{df[col].mean():.2f}" if not df[col].isnull().all() else 'N/A'
                col_data['Std'] = f"{df[col].std():.2f}" if not df[col].isnull().all() else 'N/A'
                col_data['Min'] = f"{df[col].min():.2f}" if not df[col].isnull().all() else 'N/A'
                col_data['Max'] = f"{df[col].max():.2f}" if not df[col].isnull().all() else 'N/A'
                col_data['Zeros'] = (df[col] == 0).sum()
                col_data['Negatives'] = (df[col] < 0).sum() if not df[col].isnull().all() else 0
            else:
                col_data['Top_Value'] = df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'
                col_data['Top_Value_Count'] = df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
            
            report_data.append(col_data)
        
        return pd.DataFrame(report_data)


def check_data_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, any]:
    """
    Check for data leakage between train and test sets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        Dictionary with leakage information
    """
    leakage_info = {
        'has_leakage': False,
        'common_indices': [],
        'percentage': 0.0
    }
    
    # Check for common indices
    if hasattr(train_df, 'index') and hasattr(test_df, 'index'):
        common_idx = train_df.index.intersection(test_df.index)
        if len(common_idx) > 0:
            leakage_info['has_leakage'] = True
            leakage_info['common_indices'] = common_idx.tolist()
            leakage_info['percentage'] = (len(common_idx) / len(train_df)) * 100
            logger.warning(f"Data leakage detected! {len(common_idx)} common indices found ({leakage_info['percentage']:.2f}%)")
    
    return leakage_info


def validate_train_test_split(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                               y_train: pd.Series, y_test: pd.Series) -> Tuple[bool, List[str]]:
    """
    Validate train-test split.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check shapes match
    if len(X_train) != len(y_train):
        issues.append(f"X_train ({len(X_train)}) and y_train ({len(y_train)}) have different lengths")
    
    if len(X_test) != len(y_test):
        issues.append(f"X_test ({len(X_test)}) and y_test ({len(y_test)}) have different lengths")
    
    # Check columns match
    if not X_train.columns.equals(X_test.columns):
        issues.append("X_train and X_test have different columns")
    
    # Check for data leakage
    leakage = check_data_leakage(X_train, X_test)
    if leakage['has_leakage']:
        issues.append(f"Data leakage detected: {len(leakage['common_indices'])} common samples")
    
    is_valid = len(issues) == 0
    return is_valid, issues
