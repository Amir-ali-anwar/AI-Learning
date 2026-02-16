"""
Main execution script for the Used Cars Price Prediction ML Pipeline.
Run this script to execute the end-to-end pipeline from data loading to model evaluation.

Features:
- End-to-end pipeline execution
- Multi-model training & comparison
- Model versioning with metadata
- Structured JSON logging option
- Cross-validation support
"""

import sys
import os
import logging
import json
import hashlib
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from preprocessing_pipeline import DataPreprocessor, split_data
from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

# Setup logging with structured format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production log aggregation systems."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(use_json: bool = False):
    """Configure logging handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)
    
    # File handler
    if use_json:
        file_handler = logging.FileHandler('pipeline.log')
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler = logging.FileHandler('pipeline.log')
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute a hash of the dataframe for versioning."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:12]


def save_model_metadata(models_dir: str, best_model_name: str, best_metrics: dict,
                        model_sizes: dict, data_hash: str, n_train: int, n_test: int,
                        selected_features: list, pipeline_version: str = "2.0.0"):
    """Save comprehensive model metadata for versioning and traceability."""
    metadata = {
        'model_name': best_model_name,
        'pipeline_version': pipeline_version,
        'trained_at': datetime.now().astimezone().isoformat(),
        'python_version': sys.version,
        'data_hash': data_hash,
        'n_train_samples': n_train,
        'n_test_samples': n_test,
        'n_features': len(selected_features),
        'selected_features': selected_features,
        'metrics': {k: float(v) for k, v in best_metrics.items() if isinstance(v, (int, float, np.floating))},
        'model_size_mb': model_sizes.get(best_model_name, 0),
    }
    
    metadata_path = os.path.join(models_dir, 'best_model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return metadata


def main():
    logger = setup_logging(use_json=False)
    logger.info("Starting Complete ML Pipeline v2.0")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # 1. Configuration Setup
    # ----------------------
    pipeline_config = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    
    root_dir = Path(__file__).parent.absolute()
    pipeline_config['RAW_DATA_PATH'] = str(root_dir / 'data' / 'raw' / 'vehicles.csv')
    pipeline_config['PROCESSED_DATA_PATH'] = str(root_dir / 'data' / 'processed')
    pipeline_config['MODELS_DIR'] = str(root_dir / 'models')
    
    os.makedirs(pipeline_config['PROCESSED_DATA_PATH'], exist_ok=True)
    os.makedirs(pipeline_config['MODELS_DIR'], exist_ok=True)
    
    # 2. Data Loading
    # ---------------
    logger.info(f"Loading data from {pipeline_config['RAW_DATA_PATH']}...")
    if not os.path.exists(pipeline_config['RAW_DATA_PATH']):
        logger.error(f"Data file not found at {pipeline_config['RAW_DATA_PATH']}")
        return
        
    df_raw = pd.read_csv(pipeline_config['RAW_DATA_PATH'])
    logger.info(f"Data loaded. Shape: {df_raw.shape}")
    
    # Compute data hash for versioning
    data_hash = compute_data_hash(df_raw)
    logger.info(f"Data hash: {data_hash}")
    
    # 3. Data Preprocessing
    # ---------------------
    logger.info("Initializing Preprocessor...")
    preprocessor = DataPreprocessor(pipeline_config)
    
    df_processed = preprocessor.fit_transform(df_raw)
    
    with open(os.path.join(pipeline_config['MODELS_DIR'], 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
        
    # 4. Train-Test Split
    # -------------------
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df_processed, pipeline_config)
    
    # Save train/test splits for later validation
    processed_dir = pipeline_config['PROCESSED_DATA_PATH']
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'))
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'))
    logger.info(f"Saved train/test splits to {processed_dir}")
    
    # 5. Feature Engineering
    # ----------------------
    logger.info("Initializing Feature Engineer...")
    feature_engineer = FeatureEngineer(pipeline_config)
    
    X_train_eng = feature_engineer.fit_transform(X_train)
    X_test_eng = feature_engineer.transform(X_test)
    
    with open(os.path.join(pipeline_config['MODELS_DIR'], 'feature_engineer.pkl'), 'wb') as f:
        pickle.dump(feature_engineer, f)
        
    # 6. Feature Selection
    # --------------------
    logger.info("Initializing Feature Selector...")
    feature_selector = FeatureSelector(pipeline_config)
    
    selection_method = 'random_forest'
    n_features = 30
    
    logger.info(f"Selecting top {n_features} features using {selection_method}...")
    X_train_sel = feature_selector.fit_transform(X_train_eng, y_train, 
                                                method=selection_method, 
                                                n_features=n_features)
    X_test_sel = feature_selector.transform(X_test_eng)
    
    selected_features = feature_selector.get_selected_features()
    logger.info(f"Selected features: {selected_features}")
    
    with open(os.path.join(pipeline_config['MODELS_DIR'], 'feature_selector.pkl'), 'wb') as f:
        pickle.dump(feature_selector, f)
        
    # 7. Model Training
    # -----------------
    logger.info("Initializing Model Trainer...")
    trainer = ModelTrainer(pipeline_config)
    
    models_to_train = ['xgboost', 'lightgbm']
    logger.info(f"Training models: {models_to_train}")
    
    import time
    training_times = {}
    model_sizes = {}
    
    for model_name in models_to_train:
        model_path = os.path.join(pipeline_config['MODELS_DIR'], f'{model_name}.pkl')
        
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        trained_models = trainer.train_all_models(X_train_sel, y_train, model_subset=[model_name])
        training_times[model_name] = time.time() - start_time
        
        # Save model
        trainer.save_model(model_name, model_path)
        model_sizes[model_name] = os.path.getsize(model_path) / (1024 * 1024)
        
        logger.info(f"  {model_name}: trained in {training_times[model_name]:.1f}s, "
                    f"size: {model_sizes[model_name]:.1f} MB")
    
    # 8. Model Evaluation
    # -------------------
    logger.info("Initializing Model Evaluator...")
    evaluator = ModelEvaluator(pipeline_config)
    
    predictions = trainer.predict_all(X_test_sel)
    
    results_df = evaluator.evaluate_all_models(predictions, y_test)
    
    # Select best model
    best_model_name, best_metrics = evaluator.get_best_model(metric='r2')
    trainer.set_best_model(best_model_name)
    
    # 9. Save Metadata & Report
    # -------------------------
    metadata = save_model_metadata(
        models_dir=pipeline_config['MODELS_DIR'],
        best_model_name=best_model_name,
        best_metrics=best_metrics,
        model_sizes=model_sizes,
        data_hash=data_hash,
        n_train=len(X_train),
        n_test=len(X_test),
        selected_features=selected_features,
    )
    
    report = evaluator.generate_evaluation_report(
        results_df, 
        save_path=os.path.join(pipeline_config['MODELS_DIR'], 'evaluation_report.txt')
    )
    
    # 10. Cross-Validation (quick 3-fold for pipeline confirmation)
    # -------------------------------------------------------------
    logger.info("Running 3-fold cross-validation for stability check...")
    from sklearn.model_selection import cross_val_score, KFold
    
    best_model = trainer.get_model(best_model_name)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    cv_r2 = cross_val_score(best_model, X_train_sel, y_train, cv=kf, scoring='r2', n_jobs=-1)
    logger.info(f"CV R² scores: {[f'{s:.4f}' for s in cv_r2]}")
    logger.info(f"CV R² mean ± std: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Best Model:    {best_model_name}")
    print(f"  R² Score:      {best_metrics['r2']:.4f}")
    print(f"  RMSE:          ${best_metrics['rmse']:,.2f}")
    print(f"  MAE:           ${best_metrics['mae']:,.2f}")
    print(f"  MAPE:          {best_metrics['mape']:.2f}%")
    print(f"  CV R² (3-fold):{cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  Data Hash:     {data_hash}")
    print(f"  Pipeline Ver:  2.0.0")
    print(f"  Models saved:  {pipeline_config['MODELS_DIR']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
