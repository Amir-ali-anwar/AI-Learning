"""
Model Evaluation module for evaluating model performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare model performance."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ModelEvaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,  # Convert to percentage
            'explained_variance': explained_variance_score(y_true, y_pred)
        }
        
        # Additional custom metrics
        metrics['mean_error'] = np.mean(y_pred - y_true)
        metrics['std_error'] = np.std(y_pred - y_true)
        metrics['max_error'] = np.max(np.abs(y_pred - y_true))
        
        return metrics
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, 
                      y_pred: np.ndarray, dataset: str = 'test') -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            dataset: Dataset name ('train', 'test', 'val')
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name} on {dataset} set...")
        
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Store results
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][dataset] = metrics
        
        # Log metrics
        logger.info(f"{model_name} - {dataset} metrics:")
        logger.info(f"  RMSE: ${metrics['rmse']:,.2f}")
        logger.info(f"  MAE: ${metrics['mae']:,.2f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def evaluate_all_models(self, predictions: Dict[str, np.ndarray], 
                           y_true: np.ndarray, dataset: str = 'test') -> pd.DataFrame:
        """
        Evaluate multiple models.
        
        Args:
            predictions: Dictionary of model name to predictions
            y_true: True values
            dataset: Dataset name
            
        Returns:
            DataFrame with all metrics
        """
        logger.info("="*80)
        logger.info(f"EVALUATING ALL MODELS ON {dataset.upper()} SET")
        logger.info("="*80)
        
        results_list = []
        
        for model_name, y_pred in predictions.items():
            metrics = self.evaluate_model(model_name, y_true, y_pred, dataset)
            metrics['model'] = model_name
            metrics['dataset'] = dataset
            results_list.append(metrics)
        
        results_df = pd.DataFrame(results_list)
        
        # Sort by R² score (descending)
        results_df = results_df.sort_values('r2', ascending=False)
        
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"\nTop 5 models by R² score:")
        for idx, row in results_df.head(5).iterrows():
            logger.info(f"  {row['model']}: R² = {row['r2']:.4f}, RMSE = ${row['rmse']:,.2f}")
        
        return results_df
    
    def compare_models(self, metric: str = 'r2', dataset: str = 'test') -> pd.DataFrame:
        """
        Compare models by a specific metric.
        
        Args:
            metric: Metric to compare
            dataset: Dataset to compare on
            
        Returns:
            DataFrame with model comparisons
        """
        comparison_data = []
        
        for model_name, datasets in self.results.items():
            if dataset in datasets:
                comparison_data.append({
                    'model': model_name,
                    metric: datasets[dataset][metric]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(metric, ascending=(metric in ['rmse', 'mae', 'mape']))
        
        return comparison_df
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str = None, save_path: str = None):
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price ($)', fontsize=12)
        plt.ylabel('Predicted Price ($)', fontsize=12)
        title = f'Predictions vs Actual Values'
        if model_name:
            title += f' - {model_name}'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str = None, save_path: str = None):
        """
        Plot residuals distribution.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save plot
        """
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Price ($)', fontsize=12)
        axes[0].set_ylabel('Residuals ($)', fontsize=12)
        axes[0].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals ($)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        if model_name:
            fig.suptitle(f'{model_name} - Residual Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results_df: pd.DataFrame, 
                             metric: str = 'r2', save_path: str = None):
        """
        Plot model comparison by metric.
        
        Args:
            results_df: DataFrame with model results
            metric: Metric to plot
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # Sort by metric
        ascending = metric in ['rmse', 'mae', 'mape']
        results_sorted = results_df.sort_values(metric, ascending=ascending)
        
        # Create bar plot
        plt.barh(range(len(results_sorted)), results_sorted[metric], color='steelblue')
        plt.yticks(range(len(results_sorted)), results_sorted['model'])
        
        metric_labels = {
            'r2': 'R² Score',
            'rmse': 'RMSE ($)',
            'mae': 'MAE ($)',
            'mape': 'MAPE (%)'
        }
        
        plt.xlabel(metric_labels.get(metric, metric), fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(f'Model Comparison by {metric_labels.get(metric, metric)}', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_learning_curve(self, train_sizes: List[int], train_scores: List[float],
                           val_scores: List[float], model_name: str = None,
                           save_path: str = None):
        """
        Plot learning curve.
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            val_scores: Validation scores
            model_name: Name of the model
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
        plt.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2)
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        title = 'Learning Curve'
        if model_name:
            title += f' - {model_name}'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, importances: pd.DataFrame, top_n: int = 20,
                               model_name: str = None, save_path: str = None):
        """
        Plot feature importance.
        
        Args:
            importances: DataFrame with feature importances
            top_n: Number of top features to plot
            model_name: Name of the model
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 8))
        
        top_features = importances.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        title = f'Top {top_n} Most Important Features'
        if model_name:
            title += f' - {model_name}'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = None, save_path: str = None):
        """
        Plot error distribution by price range.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save plot
        """
        errors = np.abs(y_pred - y_true)
        
        # Create price bins
        price_bins = pd.qcut(y_true, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Price Range': price_bins,
            'Absolute Error': errors
        })
        
        plt.figure(figsize=(12, 6))
        
        # Box plot
        df.boxplot(column='Absolute Error', by='Price Range', figsize=(12, 6))
        plt.xlabel('Price Range', fontsize=12)
        plt.ylabel('Absolute Error ($)', fontsize=12)
        title = 'Error Distribution by Price Range'
        if model_name:
            title += f' - {model_name}'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results_df: pd.DataFrame, 
                                   save_path: str = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results_df: DataFrame with model results
            save_path: Path to save report
            
        Returns:
            Report string
        """
        report = []
        report.append("="*80)
        report.append("MODEL EVALUATION REPORT")
        report.append("="*80)
        report.append("")
        
        # Best model by each metric
        report.append("BEST MODELS BY METRIC:")
        report.append("-"*80)
        
        metrics = ['r2', 'rmse', 'mae', 'mape']
        for metric in metrics:
            if metric in results_df.columns:
                ascending = metric in ['rmse', 'mae', 'mape']
                best_model = results_df.sort_values(metric, ascending=ascending).iloc[0]
                report.append(f"  {metric.upper()}: {best_model['model']} ({best_model[metric]:.4f})")
        
        report.append("")
        report.append("DETAILED RESULTS:")
        report.append("-"*80)
        report.append(results_df.to_string(index=False))
        
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to: {save_path}")
        
        return report_text
    
    def get_best_model(self, metric: str = 'r2', dataset: str = 'test') -> Tuple[str, Dict]:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for selection
            dataset: Dataset to evaluate on
            
        Returns:
            Tuple of (model name, metrics dictionary)
        """
        best_model = None
        best_score = float('-inf') if metric == 'r2' else float('inf')
        best_metrics = None
        
        for model_name, datasets in self.results.items():
            if dataset in datasets:
                score = datasets[dataset][metric]
                
                # Check if better
                if metric == 'r2':
                    is_better = score > best_score
                else:  # Lower is better for error metrics
                    is_better = score < best_score
                
                if is_better:
                    best_model = model_name
                    best_score = score
                    best_metrics = datasets[dataset]
        
        logger.info(f"Best model by {metric}: {best_model} ({metric}={best_score:.4f})")
        
        return best_model, best_metrics
    
    def get_results(self) -> Dict:
        """Get all evaluation results."""
        return self.results
    
    def get_results_dataframe(self, dataset: str = 'test') -> pd.DataFrame:
        """
        Get results as DataFrame.
        
        Args:
            dataset: Dataset to get results for
            
        Returns:
            DataFrame with results
        """
        results_list = []
        
        for model_name, datasets in self.results.items():
            if dataset in datasets:
                metrics = datasets[dataset].copy()
                metrics['model'] = model_name
                metrics['dataset'] = dataset
                results_list.append(metrics)
        
        return pd.DataFrame(results_list)
