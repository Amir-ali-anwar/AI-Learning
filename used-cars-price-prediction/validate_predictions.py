"""
Model Prediction Validation Script

This script validates model predictions by:
1. Testing predictions on known test data
2. Calculating error metrics and identifying problematic predictions
3. Validating individual predictions with confidence intervals
4. Providing detailed error analysis
5. Running k-fold cross-validation to confirm metric stability
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
import logging
from typing import Dict
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom classes
try:
    from preprocessing_pipeline import DataPreprocessor, OutlierHandler
    from feature_engineering import FeatureEngineer
    from feature_selection import FeatureSelector
    from model_evaluation import ModelEvaluator
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionValidator:
    """Validate model predictions for accuracy and errors."""
    
    def __init__(self, model_path: str = 'models'):
        """
        Initialize the validator.
        
        Args:
            model_path: Path to the models directory
        """
        self.model_path = model_path
        self.preprocessor = None
        self.feature_engineer = None
        self.feature_selector = None
        self.model = None
        self.model_name = None
        self.evaluator = ModelEvaluator()
        
        self._load_pipeline()
    
    def _get_best_model_name(self) -> str:
        """Determine the best model name from metadata."""
        metadata_path = os.path.join(self.model_path, 'best_model_metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('model_name', 'xgboost')
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Fallback
        for name in ['xgboost', 'lightgbm', 'random_forest']:
            if os.path.exists(os.path.join(self.model_path, f'{name}.pkl')):
                return name
        
        raise FileNotFoundError("No model files found")
    
    def _load_pipeline(self):
        """Load all pipeline components."""
        logger.info("Loading pipeline components...")
        
        try:
            with open(os.path.join(self.model_path, 'preprocessor.pkl'), 'rb') as f:
                self.preprocessor = pickle.load(f)
            with open(os.path.join(self.model_path, 'feature_engineer.pkl'), 'rb') as f:
                self.feature_engineer = pickle.load(f)
            with open(os.path.join(self.model_path, 'feature_selector.pkl'), 'rb') as f:
                self.feature_selector = pickle.load(f)
            
            self.model_name = self._get_best_model_name()
            model_file = os.path.join(self.model_path, f'{self.model_name}.pkl')
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"✓ All pipeline components loaded (model: {self.model_name})")
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise
    
    def validate_on_test_data(self, test_features_path: str = 'data/processed/X_test.csv', 
                             test_labels_path: str = 'data/processed/y_test.csv') -> Dict:
        """
        Validate predictions on test dataset.
        
        Args:
            test_features_path: Path to test features CSV
            test_labels_path: Path to test labels CSV
            
        Returns:
            Dictionary with validation results
        """
        logger.info("=" * 80)
        logger.info("VALIDATING MODEL ON TEST DATA")
        logger.info("=" * 80)
        
        # Check if files exist
        if not os.path.exists(test_features_path) or not os.path.exists(test_labels_path):
            logger.error(f"Test files not found: {test_features_path} or {test_labels_path}")
            print("Test files not found. Please ensure the pipeline has been run.")
            return {}

        # Load test data
        logger.info(f"Loading test features from: {test_features_path}")
        try:
            X_test = pd.read_csv(test_features_path, index_col=0)
            if X_test.shape[1] == 0:
                logger.warning("Loaded X_test has 0 columns with index_col=0, reloading without it...")
                X_test = pd.read_csv(test_features_path)
        except Exception:
            X_test = pd.read_csv(test_features_path)
        
        logger.info(f"Loading test labels from: {test_labels_path}")
        try:
            y_test_df = pd.read_csv(test_labels_path, index_col=0)
            if y_test_df.shape[1] == 0:
                logger.info("Loaded y_test has 0 columns with index_col=0, reloading without it...")
                y_test_df = pd.read_csv(test_labels_path)
        except Exception:
            y_test_df = pd.read_csv(test_labels_path)
        
        if y_test_df.shape[1] == 0:
            raise ValueError(f"y_test has {y_test_df.shape} shape - effectively empty?")
             
        y_true = y_test_df.iloc[:, 0].values
        
        logger.info(f"Test set size: {len(X_test)} samples")
        
        # Schema alignment: handle missing features for model compatibility
        if 'region' not in X_test.columns:
            logger.warning("Column 'region' not found in test data. Adding with default value 0.")
            X_test['region'] = 0
        
        try:
            # 1. Feature Engineering
            logger.info("Applying feature engineering...")
            logger.info(f"X_test shape: {X_test.shape}")
            
            X_eng = self.feature_engineer.transform(X_test)
            logger.info(f"After feature engineering: {X_eng.shape}")
            
            # 2. Feature Selection
            logger.info("Applying feature selection...")
            X_sel = self.feature_selector.transform(X_eng)
            logger.info(f"After feature selection: {X_sel.shape}")
            
            # 3. Prediction
            logger.info("Making predictions...")
            y_pred = self.model.predict(X_sel)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            if hasattr(self.model, 'feature_names_in_'):
                logger.error(f"Model expects features: {self.model.feature_names_in_[:10]}...")
            raise
        
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(y_true, y_pred)
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION METRICS")
        logger.info("=" * 80)
        logger.info(f"Model:                           {self.model_name}")
        logger.info(f"RMSE (Root Mean Squared Error): ${metrics['rmse']:,.2f}")
        logger.info(f"MAE (Mean Absolute Error):      ${metrics['mae']:,.2f}")
        logger.info(f"R² Score:                        {metrics['r2']:.4f}")
        logger.info(f"MAPE (Mean Absolute % Error):    {metrics['mape']:.2f}%")
        logger.info(f"Mean Error (Bias):               ${metrics['mean_error']:,.2f}")
        logger.info(f"Std Error:                       ${metrics['std_error']:,.2f}")
        logger.info(f"Max Error:                       ${metrics['max_error']:,.2f}")
        
        # Analyze errors
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_errors = (abs_errors / y_true) * 100
            pct_errors = np.nan_to_num(pct_errors, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info("\n" + "=" * 80)
        logger.info("ERROR ANALYSIS")
        logger.info("=" * 80)
        within_10 = np.sum(pct_errors <= 10) / len(pct_errors) * 100
        within_20 = np.sum(pct_errors <= 20) / len(pct_errors) * 100
        within_30 = np.sum(pct_errors <= 30) / len(pct_errors) * 100
        logger.info(f"Predictions within ±10% of actual: {within_10:.2f}%")
        logger.info(f"Predictions within ±20% of actual: {within_20:.2f}%")
        logger.info(f"Predictions within ±30% of actual: {within_30:.2f}%")
        
        # Find worst predictions
        worst_indices = np.argsort(abs_errors)[-10:][::-1]
        
        logger.info("\n" + "=" * 80)
        logger.info("TOP 10 WORST PREDICTIONS")
        logger.info("=" * 80)
        for i, idx in enumerate(worst_indices, 1):
            logger.info(f"{i}. Actual: ${y_true[idx]:,.2f} | Predicted: ${y_pred[idx]:,.2f} | "
                       f"Error: ${errors[idx]:,.2f} ({pct_errors[idx]:.1f}%)")
        
        # Find best predictions
        best_indices = np.argsort(abs_errors)[:10]
        
        logger.info("\n" + "=" * 80)
        logger.info("TOP 10 BEST PREDICTIONS")
        logger.info("=" * 80)
        for i, idx in enumerate(best_indices, 1):
            logger.info(f"{i}. Actual: ${y_true[idx]:,.2f} | Predicted: ${y_pred[idx]:,.2f} | "
                       f"Error: ${errors[idx]:,.2f} ({pct_errors[idx]:.1f}%)")
        
        return {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'errors': errors,
            'abs_errors': abs_errors,
            'pct_errors': pct_errors,
            'within_10pct': within_10,
            'within_20pct': within_20,
            'within_30pct': within_30,
        }
    
    def validate_single_prediction(self, input_data: Dict, expected_price: float = None) -> Dict:
        """
        Validate a single prediction with confidence interval.
        
        Args:
            input_data: Dictionary with car features
            expected_price: Optional expected price for comparison
            
        Returns:
            Dictionary with prediction and validation info
        """
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING SINGLE PREDICTION")
        logger.info("=" * 80)
        
        # Display input
        logger.info("\nInput Features:")
        for key, value in input_data.items():
            logger.info(f"  {key}: {value}")
        
        from predict import predict_price
        
        # Make prediction
        try:
            result = predict_price(input_data)
            predicted_price = result['predicted_price']
            logger.info(f"\nModel used: {result['model_used']}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            predicted_price = 0.0

        # Get prediction interval if model supports it (Random Forest / ensemble trees)
        if hasattr(self.model, 'estimators_'):
            from predict import DEFAULTS
            
            full_input = DEFAULTS.copy()
            full_input.update(input_data)
            
            input_df = pd.DataFrame([full_input])
            
            try:
                # Pipeline transformation chain
                X_proc = self.preprocessor.transform(input_df)
                X_eng = self.feature_engineer.transform(X_proc)
                X_sel = self.feature_selector.transform(X_eng)
                
                # Get predictions from all trees for confidence interval
                tree_predictions = np.array([tree.predict(X_sel)[0] for tree in self.model.estimators_])
                
                # Calculate 95% confidence interval
                lower_bound = np.percentile(tree_predictions, 2.5)
                upper_bound = np.percentile(tree_predictions, 97.5)
                std_dev = np.std(tree_predictions)
                
                logger.info("\n" + "=" * 80)
                logger.info("PREDICTION RESULTS")
                logger.info("=" * 80)
                logger.info(f"Predicted Price:     ${predicted_price:,.2f}")
                logger.info(f"95% Confidence Int:  ${lower_bound:,.2f} - ${upper_bound:,.2f}")
                logger.info(f"Prediction Std Dev:  ${std_dev:,.2f}")
                logger.info(f"Uncertainty Range:   ±${(upper_bound - lower_bound) / 2:,.2f}")
                
                result_dict = {
                    'predicted_price': predicted_price,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'std_dev': std_dev,
                    'expected_price': expected_price,
                    'error': None,
                    'pct_error': None
                }
                
                if expected_price:
                    error = predicted_price - expected_price
                    pct_error = (abs(error) / expected_price) * 100
                    within_interval = lower_bound <= expected_price <= upper_bound
                    
                    logger.info("\n" + "=" * 80)
                    logger.info("VALIDATION AGAINST EXPECTED PRICE")
                    logger.info("=" * 80)
                    logger.info(f"Expected Price:      ${expected_price:,.2f}")
                    logger.info(f"Prediction Error:    ${error:,.2f}")
                    logger.info(f"Percentage Error:    {pct_error:.2f}%")
                    logger.info(f"Within 95% CI:       {'✓ YES' if within_interval else '✗ NO'}")
                    
                    if pct_error <= 10:
                        quality = "EXCELLENT"
                    elif pct_error <= 20:
                        quality = "GOOD"
                    elif pct_error <= 30:
                        quality = "FAIR"
                    else:
                        quality = "POOR"
                    
                    logger.info(f"Prediction Quality:  {quality}")
                    result_dict['error'] = error
                    result_dict['pct_error'] = pct_error
                
                return result_dict
            except Exception as e:
                logger.error(f"Confidence interval calculation failed: {e}")
                return {'predicted_price': predicted_price, 'error': str(e)}
        else:
            logger.info(f"\nPredicted Price: ${predicted_price:,.2f}")
            
            result_dict = {
                'predicted_price': predicted_price,
                'expected_price': expected_price,
                'error': None,
                'pct_error': None
            }
            
            if expected_price:
                error = predicted_price - expected_price
                pct_error = (abs(error) / expected_price) * 100
                logger.info(f"Expected Price:  ${expected_price:,.2f}")
                logger.info(f"Error:           ${error:,.2f} ({pct_error:.2f}%)")
                result_dict['error'] = error
                result_dict['pct_error'] = pct_error
            
            return result_dict
    
    def run_cross_validation(self, n_folds: int = 5) -> Dict:
        """
        Run k-fold cross-validation to confirm metric stability.
        
        Args:
            n_folds: Number of folds
            
        Returns:
            Dictionary with CV results
        """
        from sklearn.model_selection import cross_val_score, KFold
        
        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING {n_folds}-FOLD CROSS-VALIDATION")
        logger.info("=" * 80)
        
        # Load training data
        X_train_path = 'data/processed/X_train.csv'
        y_train_path = 'data/processed/y_train.csv'
        
        if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
            logger.error("Training data not found for cross-validation")
            return {}
        
        try:
            X_train = pd.read_csv(X_train_path, index_col=0)
            if X_train.shape[1] == 0:
                X_train = pd.read_csv(X_train_path)
        except Exception:
            X_train = pd.read_csv(X_train_path)
        
        try:
            y_train_df = pd.read_csv(y_train_path, index_col=0)
            if y_train_df.shape[1] == 0:
                y_train_df = pd.read_csv(y_train_path)
        except Exception:
            y_train_df = pd.read_csv(y_train_path)
        
        y_train = y_train_df.iloc[:, 0].values
        
        # Schema alignment
        if 'region' not in X_train.columns:
            X_train['region'] = 0
        
        # Feature engineering and selection
        X_eng = self.feature_engineer.transform(X_train)
        X_sel = self.feature_selector.transform(X_eng)
        
        logger.info(f"Training data shape: {X_sel.shape}")
        
        # Cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # R² scores
        r2_scores = cross_val_score(self.model, X_sel, y_train, cv=kf, scoring='r2', n_jobs=-1)
        
        # Negative MAE (sklearn convention)
        mae_scores = -cross_val_score(self.model, X_sel, y_train, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
        
        # Negative RMSE
        rmse_scores = np.sqrt(-cross_val_score(self.model, X_sel, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1))
        
        logger.info("\n" + "=" * 80)
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Folds: {n_folds}")
        logger.info("")
        logger.info(f"R² Score:  {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
        logger.info(f"  Per fold: {[f'{s:.4f}' for s in r2_scores]}")
        logger.info(f"MAE:       ${mae_scores.mean():,.2f} ± ${mae_scores.std():,.2f}")
        logger.info(f"  Per fold: {[f'${s:,.2f}' for s in mae_scores]}")
        logger.info(f"RMSE:      ${rmse_scores.mean():,.2f} ± ${rmse_scores.std():,.2f}")
        logger.info(f"  Per fold: {[f'${s:,.2f}' for s in rmse_scores]}")
        
        # Assess stability
        r2_cv = r2_scores.std() / r2_scores.mean() if r2_scores.mean() != 0 else float('inf')
        if r2_cv < 0.05:
            stability = "EXCELLENT - very stable across folds"
        elif r2_cv < 0.10:
            stability = "GOOD - reasonably stable"
        elif r2_cv < 0.20:
            stability = "FAIR - some variability"
        else:
            stability = "POOR - high variability, may indicate overfitting"
        
        logger.info(f"\nStability (R² CV): {r2_cv:.4f} → {stability}")
        
        return {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'r2_scores': r2_scores.tolist(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'stability': stability,
            'n_folds': n_folds,
        }
    
    def plot_validation_results(self, validation_results: Dict, save_path: str = None):
        """
        Create visualization of validation results.
        
        Args:
            validation_results: Results from validate_on_test_data
            save_path: Optional path to save plots
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        y_true = validation_results['y_true']
        y_pred = validation_results['y_pred']
        errors = validation_results['errors']
        pct_errors = validation_results['pct_errors']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Predictions vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Price ($)', fontsize=12)
        axes[0, 0].set_ylabel('Predicted Price ($)', fontsize=12)
        axes[0, 0].set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        axes[0, 1].scatter(y_pred, errors, alpha=0.5, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Price ($)', fontsize=12)
        axes[0, 1].set_ylabel('Residuals ($)', fontsize=12)
        axes[0, 1].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error Distribution
        axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals ($)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Percentage Error Distribution
        axes[1, 1].hist(pct_errors[pct_errors < 100], bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=20, color='orange', linestyle='--', lw=2, label='20% threshold')
        axes[1, 1].axvline(x=10, color='green', linestyle='--', lw=2, label='10% threshold')
        axes[1, 1].set_xlabel('Percentage Error (%)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Percentage Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation plots saved to: {save_path}")
        
        plt.close(fig)


def main():
    """Main validation workflow."""
    
    # Initialize validator
    validator = PredictionValidator()
    
    # 1. Validate on test data
    print("\n" + "=" * 80)
    print("STEP 1: Validate on Test Dataset")
    print("=" * 80)
    
    if os.path.exists('data/processed/X_test.csv') and os.path.exists('data/processed/y_test.csv'):
        results = validator.validate_on_test_data()
        
        if results:
            validator.plot_validation_results(results, save_path='models/validation_plots.png')
    else:
        logger.warning("Test data files X_test.csv/y_test.csv not found in data/processed/")
    
    # 2. K-Fold Cross-Validation
    print("\n" + "=" * 80)
    print("STEP 2: K-Fold Cross-Validation")
    print("=" * 80)
    
    cv_results = validator.run_cross_validation(n_folds=5)
    
    # 3. Validate single prediction
    print("\n" + "=" * 80)
    print("STEP 3: Validate Single Prediction")
    print("=" * 80)
    
    sample_input = {
        'year': 2020,
        'odometer': 4500,
        'manufacturer': 'ford',
        'model': 'f-150',
        'fuel': 'gas'
    }
    
    validator.validate_single_prediction(sample_input, expected_price=None)
    
    # 4. Production readiness summary
    print("\n" + "=" * 80)
    print("PRODUCTION READINESS SUMMARY")
    print("=" * 80)
    
    if results:
        r2 = results['metrics']['r2']
        mape = results['metrics']['mape']
        w10 = results.get('within_10pct', 0)
        w20 = results.get('within_20pct', 0)
        
        checks = {
            'R² > 0.85': r2 > 0.85,
            'MAPE < 20%': mape < 20,
            'Within ±10% > 50%': w10 > 50,
            'Within ±20% > 75%': w20 > 75,
        }
        
        if cv_results:
            checks['CV R² Stable (std < 0.05)'] = cv_results.get('r2_std', 1) < 0.05
        
        all_pass = all(checks.values())
        
        for check, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {check}")
        
        print(f"\n  Overall: {'✅ PRODUCTION READY' if all_pass else '⚠️ NEEDS IMPROVEMENT'}")
    
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
