"""
Model Improvement Script

This script improves the model by:
1. Checking for data leakage between train and test sets
2. Training multiple models (XGBoost, LightGBM, Gradient Boosting)
3. Comparing all models side-by-side
4. Saving the best model for production use

Run this AFTER main.py has generated the processed data.
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_evaluation import ModelEvaluator
from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('improvement.log')
    ]
)
logger = logging.getLogger(__name__)

# Paths
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'


def load_processed_data():
    """Load the processed train/test data."""
    logger.info("=" * 80)
    logger.info("LOADING PROCESSED DATA")
    logger.info("=" * 80)
    
    # Load features
    try:
        X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index_col=0)
        if X_train.shape[1] == 0:
            X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
    except Exception:
        X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
    
    try:
        X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index_col=0)
        if X_test.shape[1] == 0:
            X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
    except Exception:
        X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
    
    # Load labels
    try:
        y_train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index_col=0)
        if y_train_df.shape[1] == 0:
            y_train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'))
    except Exception:
        y_train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'))
    
    try:
        y_test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index_col=0)
        if y_test_df.shape[1] == 0:
            y_test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'))
    except Exception:
        y_test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'))
    
    y_train = y_train_df.iloc[:, 0].values
    y_test = y_test_df.iloc[:, 0].values
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape:  {X_test.shape}")
    logger.info(f"y_train range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    logger.info(f"y_test range:  ${y_test.min():,.0f} - ${y_test.max():,.0f}")
    
    return X_train, X_test, y_train, y_test


def check_data_leakage(X_train, X_test, y_train, y_test):
    """
    Check for data leakage between train and test sets.
    This is critical for ensuring honest model evaluation.
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: CHECKING FOR DATA LEAKAGE")
    logger.info("=" * 80)
    
    issues_found = []
    
    # Check 1: Duplicate rows between train and test
    logger.info("\n--- Check 1: Duplicate rows between train/test ---")
    train_hashes = pd.util.hash_pandas_object(X_train).values
    test_hashes = pd.util.hash_pandas_object(X_test).values
    overlap = set(train_hashes) & set(test_hashes)
    
    if len(overlap) > 0:
        overlap_pct = len(overlap) / len(X_test) * 100
        logger.warning(f"⚠ LEAKAGE FOUND: {len(overlap)} duplicate rows ({overlap_pct:.2f}% of test set)")
        issues_found.append(f"Duplicate rows: {len(overlap)}")
    else:
        logger.info("✓ No duplicate rows between train and test sets")
    
    # Check 2: Target distribution similarity
    logger.info("\n--- Check 2: Target distribution comparison ---")
    train_mean = np.mean(y_train)
    test_mean = np.mean(y_test)
    train_std = np.std(y_train)
    test_std = np.std(y_test)
    
    mean_diff_pct = abs(train_mean - test_mean) / train_mean * 100
    std_diff_pct = abs(train_std - test_std) / train_std * 100
    
    logger.info(f"  Train: mean=${train_mean:,.2f}, std=${train_std:,.2f}")
    logger.info(f"  Test:  mean=${test_mean:,.2f}, std=${test_std:,.2f}")
    logger.info(f"  Mean difference: {mean_diff_pct:.2f}%")
    logger.info(f"  Std difference:  {std_diff_pct:.2f}%")
    
    if mean_diff_pct > 10:
        logger.warning(f"⚠ Large mean difference ({mean_diff_pct:.2f}%) - possible distribution shift")
        issues_found.append(f"Mean difference: {mean_diff_pct:.2f}%")
    else:
        logger.info("✓ Train/test target distributions are similar")
    
    # Check 3: Perfect predictions from existing RF model
    logger.info("\n--- Check 3: Looking for suspiciously perfect predictions ---")
    rf_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    if os.path.exists(rf_path):
        logger.info("Loading existing Random Forest to check...")
        
        # Load pipeline components
        with open(os.path.join(MODELS_DIR, 'feature_engineer.pkl'), 'rb') as f:
            fe = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'feature_selector.pkl'), 'rb') as f:
            fs = pickle.load(f)
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)
        
        # Add region if missing
        X_test_copy = X_test.copy()
        if 'region' not in X_test_copy.columns:
            X_test_copy['region'] = 0
        
        X_eng = fe.transform(X_test_copy)
        X_sel = fs.transform(X_eng)
        y_pred = rf_model.predict(X_sel)
        
        zero_error = np.sum(np.abs(y_pred - y_test) < 0.01)
        if zero_error > 0:
            logger.warning(f"⚠ {zero_error} predictions have essentially ZERO error!")
            logger.warning("  This suggests possible data leakage or memorization")
            issues_found.append(f"Zero-error predictions: {zero_error}")
        else:
            logger.info("✓ No suspiciously perfect predictions found")
    
    # Summary
    logger.info("\n--- Data Leakage Summary ---")
    if issues_found:
        logger.warning(f"⚠ Found {len(issues_found)} potential issue(s):")
        for issue in issues_found:
            logger.warning(f"  - {issue}")
        logger.warning("  Note: Duplicate rows are the most serious issue.")
        logger.warning("  Zero-error predictions could also be from very common car listings.")
    else:
        logger.info("✓ No data leakage issues detected!")
    
    return issues_found


def apply_feature_pipeline(X_train, X_test):
    """Apply feature engineering and selection using saved pipeline components."""
    logger.info("\nApplying feature engineering and selection...")
    
    with open(os.path.join(MODELS_DIR, 'feature_engineer.pkl'), 'rb') as f:
        feature_engineer = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'feature_selector.pkl'), 'rb') as f:
        feature_selector = pickle.load(f)
    
    # Add region column if missing
    for df in [X_train, X_test]:
        if 'region' not in df.columns:
            df['region'] = 0
    
    # Feature engineering
    X_train_eng = feature_engineer.transform(X_train)
    X_test_eng = feature_engineer.transform(X_test)
    
    # Feature selection
    X_train_sel = feature_selector.transform(X_train_eng)
    X_test_sel = feature_selector.transform(X_test_eng)
    
    logger.info(f"After pipeline: train={X_train_sel.shape}, test={X_test_sel.shape}")
    
    return X_train_sel, X_test_sel


def train_multiple_models(X_train, y_train, X_test, y_test):
    """
    Train XGBoost, LightGBM, and Gradient Boosting models.
    Compare with existing Random Forest.
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: TRAINING MULTIPLE MODELS")
    logger.info("=" * 80)
    
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    
    evaluator = ModelEvaluator()
    models = {}
    training_times = {}
    model_sizes = {}
    
    # ---- Model 1: XGBoost (optimized) ----
    logger.info("\n--- Training XGBoost ---")
    xgb_model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    start = time.time()
    xgb_model.fit(X_train, y_train)
    training_times['xgboost'] = time.time() - start
    models['xgboost'] = xgb_model
    logger.info(f"  Training time: {training_times['xgboost']:.1f}s")
    
    # Save XGBoost
    xgb_path = os.path.join(MODELS_DIR, 'xgboost.pkl')
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    model_sizes['xgboost'] = os.path.getsize(xgb_path) / (1024 * 1024)
    logger.info(f"  Model size: {model_sizes['xgboost']:.1f} MB")
    
    # ---- Model 2: LightGBM (optimized) ----
    logger.info("\n--- Training LightGBM ---")
    lgb_model = LGBMRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=63,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    start = time.time()
    lgb_model.fit(X_train, y_train)
    training_times['lightgbm'] = time.time() - start
    models['lightgbm'] = lgb_model
    logger.info(f"  Training time: {training_times['lightgbm']:.1f}s")
    
    # Save LightGBM
    lgb_path = os.path.join(MODELS_DIR, 'lightgbm.pkl')
    with open(lgb_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    model_sizes['lightgbm'] = os.path.getsize(lgb_path) / (1024 * 1024)
    logger.info(f"  Model size: {model_sizes['lightgbm']:.1f} MB")
    
    # ---- Model 3: HistGradient Boosting (handles NaN natively) ----
    logger.info("\n--- Training HistGradient Boosting ---")
    from sklearn.ensemble import HistGradientBoostingRegressor
    gb_model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=10,
        max_leaf_nodes=63,
        random_state=42
    )
    
    start = time.time()
    gb_model.fit(X_train, y_train)
    training_times['hist_gradient_boosting'] = time.time() - start
    models['hist_gradient_boosting'] = gb_model
    logger.info(f"  Training time: {training_times['hist_gradient_boosting']:.1f}s")
    
    # Save HistGradient Boosting
    gb_path = os.path.join(MODELS_DIR, 'hist_gradient_boosting.pkl')
    with open(gb_path, 'wb') as f:
        pickle.dump(gb_model, f)
    model_sizes['hist_gradient_boosting'] = os.path.getsize(gb_path) / (1024 * 1024)
    logger.info(f"  Model size: {model_sizes['hist_gradient_boosting']:.1f} MB")
    
    # ---- Load existing Random Forest for comparison ----
    rf_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    if os.path.exists(rf_path):
        logger.info("\n--- Loading existing Random Forest ---")
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)
        models['random_forest'] = rf_model
        model_sizes['random_forest'] = os.path.getsize(rf_path) / (1024 * 1024)
        training_times['random_forest'] = 0  # Already trained
        logger.info(f"  Model size: {model_sizes['random_forest']:.1f} MB")
    
    return models, training_times, model_sizes


def compare_models(models, X_test, y_test, training_times, model_sizes):
    """Compare all trained models and select the best one."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: MODEL COMPARISON")
    logger.info("=" * 80)
    
    evaluator = ModelEvaluator()
    results = []
    
    for name, model in models.items():
        logger.info(f"\n--- Evaluating: {name} ---")
        y_pred = model.predict(X_test)
        
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        
        # Additional custom metrics
        abs_errors = np.abs(y_pred - y_test)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_errors = (abs_errors / y_test) * 100
            pct_errors = np.nan_to_num(pct_errors, nan=0.0, posinf=0.0, neginf=0.0)
        
        median_ape = np.median(pct_errors)
        within_10 = np.sum(pct_errors <= 10) / len(pct_errors) * 100
        within_20 = np.sum(pct_errors <= 20) / len(pct_errors) * 100
        within_30 = np.sum(pct_errors <= 30) / len(pct_errors) * 100
        
        result = {
            'model': name,
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'mape': metrics['mape'],
            'median_ape': median_ape,
            'within_10pct': within_10,
            'within_20pct': within_20,
            'within_30pct': within_30,
            'mean_error_bias': metrics['mean_error'],
            'max_error': metrics['max_error'],
            'training_time_s': training_times.get(name, 0),
            'model_size_mb': model_sizes.get(name, 0)
        }
        results.append(result)
        
        logger.info(f"  R²:          {metrics['r2']:.4f}")
        logger.info(f"  RMSE:        ${metrics['rmse']:,.2f}")
        logger.info(f"  MAE:         ${metrics['mae']:,.2f}")
        logger.info(f"  MAPE:        {metrics['mape']:.2f}%")
        logger.info(f"  Median APE:  {median_ape:.2f}%")
        logger.info(f"  Within ±10%: {within_10:.1f}%")
        logger.info(f"  Within ±20%: {within_20:.1f}%")
        logger.info(f"  Within ±30%: {within_30:.1f}%")
        logger.info(f"  Model Size:  {model_sizes.get(name, 0):.1f} MB")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('r2', ascending=False)
    
    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("=" * 80)
    
    # Format nice table
    header = f"{'Model':<22} {'R²':>8} {'RMSE':>12} {'MAE':>10} {'MAPE':>8} {'MedAPE':>8} {'±10%':>7} {'±20%':>7} {'Size':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    
    for _, row in results_df.iterrows():
        line = (f"{row['model']:<22} "
                f"{row['r2']:>8.4f} "
                f"${row['rmse']:>10,.0f} "
                f"${row['mae']:>8,.0f} "
                f"{row['mape']:>7.1f}% "
                f"{row['median_ape']:>7.1f}% "
                f"{row['within_10pct']:>6.1f}% "
                f"{row['within_20pct']:>6.1f}% "
                f"{row['model_size_mb']:>8.1f}MB")
        logger.info(line)
    
    return results_df


def select_best_model(results_df, models):
    """Select the best model based on multiple criteria."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: SELECTING BEST MODEL")
    logger.info("=" * 80)
    
    # Primary criterion: R² score
    best_by_r2 = results_df.loc[results_df['r2'].idxmax()]
    logger.info(f"\nBest by R²:         {best_by_r2['model']} (R²={best_by_r2['r2']:.4f})")
    
    # Secondary: Lowest MAPE
    best_by_mape = results_df.loc[results_df['mape'].idxmin()]
    logger.info(f"Best by MAPE:       {best_by_mape['model']} (MAPE={best_by_mape['mape']:.2f}%)")
    
    # Best by Median APE (more robust)
    best_by_medape = results_df.loc[results_df['median_ape'].idxmin()]
    logger.info(f"Best by Median APE: {best_by_medape['model']} (MedAPE={best_by_medape['median_ape']:.2f}%)")
    
    # Best by within ±10%
    best_by_10pct = results_df.loc[results_df['within_10pct'].idxmax()]
    logger.info(f"Best by ±10%:       {best_by_10pct['model']} ({best_by_10pct['within_10pct']:.1f}%)")
    
    # Score-based selection: weighted combination
    # Normalize each metric to 0-1 range and combine
    df = results_df.copy()
    
    # Higher is better
    df['r2_norm'] = (df['r2'] - df['r2'].min()) / (df['r2'].max() - df['r2'].min() + 1e-10)
    df['w10_norm'] = (df['within_10pct'] - df['within_10pct'].min()) / (df['within_10pct'].max() - df['within_10pct'].min() + 1e-10)
    
    # Lower is better (invert)
    df['mape_norm'] = 1 - (df['mape'] - df['mape'].min()) / (df['mape'].max() - df['mape'].min() + 1e-10)
    df['rmse_norm'] = 1 - (df['rmse'] - df['rmse'].min()) / (df['rmse'].max() - df['rmse'].min() + 1e-10)
    df['size_norm'] = 1 - (df['model_size_mb'] - df['model_size_mb'].min()) / (df['model_size_mb'].max() - df['model_size_mb'].min() + 1e-10)
    
    # Weighted score: R² (30%) + MAPE (20%) + RMSE (15%) + Within10% (20%) + Size (15%)
    df['composite_score'] = (
        0.30 * df['r2_norm'] +
        0.20 * df['mape_norm'] +
        0.15 * df['rmse_norm'] +
        0.20 * df['w10_norm'] +
        0.15 * df['size_norm']
    )
    
    best_row = df.loc[df['composite_score'].idxmax()]
    best_model_name = best_row['model']
    
    logger.info(f"\n{'=' * 40}")
    logger.info(f"🏆 BEST OVERALL MODEL: {best_model_name}")
    logger.info(f"{'=' * 40}")
    logger.info(f"  Composite Score: {best_row['composite_score']:.4f}")
    logger.info(f"  R²:             {best_row['r2']:.4f}")
    logger.info(f"  RMSE:           ${best_row['rmse']:,.2f}")
    logger.info(f"  MAPE:           {best_row['mape']:.2f}%")
    logger.info(f"  Median APE:     {best_row['median_ape']:.2f}%")
    logger.info(f"  Within ±10%:    {best_row['within_10pct']:.1f}%")
    logger.info(f"  Model Size:     {best_row['model_size_mb']:.1f} MB")
    
    # Save best model info
    best_model = models[best_model_name]
    best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'metrics': {
            'r2': float(best_row['r2']),
            'rmse': float(best_row['rmse']),
            'mae': float(best_row['mae']),
            'mape': float(best_row['mape']),
            'median_ape': float(best_row['median_ape']),
            'within_10pct': float(best_row['within_10pct']),
            'within_20pct': float(best_row['within_20pct']),
            'within_30pct': float(best_row['within_30pct'])
        },
        'model_size_mb': float(best_row['model_size_mb']),
        'composite_score': float(best_row['composite_score'])
    }
    
    import json
    with open(os.path.join(MODELS_DIR, 'best_model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\nBest model saved to: {best_model_path}")
    logger.info(f"Metadata saved to: {os.path.join(MODELS_DIR, 'best_model_metadata.json')}")
    
    return best_model_name, best_model


def generate_improvement_report(results_df):
    """Generate a final improvement report."""
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVEMENT REPORT")
    logger.info("=" * 80)
    
    # Save detailed comparison to CSV
    results_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'), index=False)
    logger.info(f"Comparison saved to: {os.path.join(MODELS_DIR, 'model_comparison.csv')}")
    
    # Generate text report
    report_lines = [
        "=" * 80,
        "MODEL IMPROVEMENT REPORT",
        "=" * 80,
        "",
        "MODELS COMPARED:",
        "-" * 80,
    ]
    
    for _, row in results_df.iterrows():
        report_lines.extend([
            f"\n  {row['model'].upper()}:",
            f"    R² Score:      {row['r2']:.4f}",
            f"    RMSE:          ${row['rmse']:,.2f}",
            f"    MAE:           ${row['mae']:,.2f}",
            f"    MAPE:          {row['mape']:.2f}%",
            f"    Median APE:    {row['median_ape']:.2f}%",
            f"    Within ±10%:   {row['within_10pct']:.1f}%",
            f"    Within ±20%:   {row['within_20pct']:.1f}%",
            f"    Model Size:    {row['model_size_mb']:.1f} MB",
        ])
    
    report_lines.extend([
        "",
        "=" * 80,
        f"BEST MODEL: {results_df.iloc[0]['model']} (by R²)",
        "=" * 80,
    ])
    
    report_text = "\n".join(report_lines)
    
    with open(os.path.join(MODELS_DIR, 'improvement_report.txt'), 'w') as f:
        f.write(report_text)
    
    logger.info(f"Report saved to: {os.path.join(MODELS_DIR, 'improvement_report.txt')}")


def main():
    """Run the full improvement pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING MODEL IMPROVEMENT PIPELINE")
    logger.info("=" * 80)
    
    total_start = time.time()
    
    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Step 2: Check data leakage
    leakage_issues = check_data_leakage(X_train, X_test, y_train, y_test)
    
    # Step 3: Apply feature pipeline
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: APPLYING FEATURE PIPELINE")
    logger.info("=" * 80)
    X_train_sel, X_test_sel = apply_feature_pipeline(X_train, X_test)
    
    # Step 4: Train multiple models
    models, training_times, model_sizes = train_multiple_models(
        X_train_sel, y_train, X_test_sel, y_test
    )
    
    # Step 5: Compare models
    results_df = compare_models(models, X_test_sel, y_test, training_times, model_sizes)
    
    # Step 6: Select best model
    best_name, best_model = select_best_model(results_df, models)
    
    # Step 7: Generate report
    generate_improvement_report(results_df)
    
    total_time = time.time() - total_start
    
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVEMENT PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Best model: {best_name}")
    logger.info(f"Next step: Run validate_predictions.py to validate the best model")
    
    print("\n" + "=" * 80)
    print(f"✅ DONE! Best model: {best_name}")
    print(f"   Run 'python validate_predictions.py' to validate")
    print("=" * 80)


if __name__ == "__main__":
    main()
