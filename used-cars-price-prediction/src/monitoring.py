"""
Model Monitoring & Data Drift Detection Module.

Provides:
- Data drift detection using statistical tests
- Prediction distribution monitoring
- Feature distribution comparison
- Alert generation for significant drift

Usage:
    from monitoring import DriftDetector
    
    detector = DriftDetector(reference_data=X_train)
    report = detector.check_drift(new_data)
    if report['has_drift']:
        print("Data drift detected! Consider retraining.")
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data drift between training data and incoming production data."""
    
    def __init__(self, reference_data: pd.DataFrame = None, 
                 significance_level: float = 0.05,
                 psi_threshold: float = 0.2):
        """
        Initialize DriftDetector.
        
        Args:
            reference_data: Training/reference data to compare against
            significance_level: P-value threshold for statistical tests
            psi_threshold: PSI threshold (>0.2 = significant drift)
        """
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.psi_threshold = psi_threshold
        self.reference_stats_ = {}
        
        if reference_data is not None:
            self._compute_reference_stats()
    
    def _compute_reference_stats(self):
        """Compute statistics from reference data."""
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            self.reference_stats_[col] = {
                'mean': float(self.reference_data[col].mean()),
                'std': float(self.reference_data[col].std()),
                'min': float(self.reference_data[col].min()),
                'max': float(self.reference_data[col].max()),
                'median': float(self.reference_data[col].median()),
                'q25': float(self.reference_data[col].quantile(0.25)),
                'q75': float(self.reference_data[col].quantile(0.75)),
            }
        
        logger.info(f"Computed reference statistics for {len(self.reference_stats_)} features")
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                      n_bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant shift
        PSI 0.1-0.2: Moderate shift
        PSI > 0.2: Significant shift
        
        Args:
            expected: Reference distribution values
            actual: New distribution values
            n_bins: Number of bins to use
            
        Returns:
            PSI score
        """
        # Create bins from expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)
        
        if len(breakpoints) < 3:
            return 0.0
        
        # Calculate bin proportions
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # Add small constant to avoid division by zero
        expected_pct = (expected_counts + 1) / (len(expected) + n_bins)
        actual_pct = (actual_counts + 1) / (len(actual) + n_bins)
        
        # PSI formula
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return float(psi)
    
    def ks_test(self, reference: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution comparison.
        
        Args:
            reference: Reference distribution
            actual: New distribution
            
        Returns:
            Tuple of (statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, actual)
        return float(statistic), float(p_value)
    
    def check_drift(self, new_data: pd.DataFrame) -> Dict:
        """
        Check for data drift between reference and new data.
        
        Args:
            new_data: New data to check for drift
            
        Returns:
            Dictionary with drift report
        """
        if self.reference_data is None:
            raise ValueError("No reference data set. Provide reference_data in constructor.")
        
        logger.info(f"Checking drift on {len(new_data)} samples...")
        
        report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'n_samples_reference': len(self.reference_data),
            'n_samples_new': len(new_data),
            'features_checked': 0,
            'features_drifted': 0,
            'has_drift': False,
            'drift_severity': 'none',
            'feature_reports': {},
        }
        
        numerical_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numerical_cols if c in new_data.columns]
        
        report['features_checked'] = len(common_cols)
        drifted_features = []
        
        for col in common_cols:
            ref_values = self.reference_data[col].dropna().values
            new_values = new_data[col].dropna().values
            
            if len(new_values) == 0:
                continue
            
            # PSI
            psi = self.calculate_psi(ref_values, new_values)
            
            # KS Test
            ks_stat, ks_pvalue = self.ks_test(ref_values, new_values)
            
            # Mean shift
            ref_mean = np.mean(ref_values)
            new_mean = np.mean(new_values)
            mean_shift_pct = abs(new_mean - ref_mean) / (abs(ref_mean) + 1e-10) * 100
            
            # Determine drift status
            is_drifted = psi > self.psi_threshold or ks_pvalue < self.significance_level
            
            if is_drifted:
                drifted_features.append(col)
            
            report['feature_reports'][col] = {
                'psi': round(psi, 4),
                'ks_statistic': round(ks_stat, 4),
                'ks_pvalue': round(ks_pvalue, 4),
                'mean_shift_pct': round(mean_shift_pct, 2),
                'ref_mean': round(ref_mean, 4),
                'new_mean': round(new_mean, 4),
                'drifted': is_drifted,
            }
        
        report['features_drifted'] = len(drifted_features)
        report['drifted_features'] = drifted_features
        
        # Determine overall drift severity
        drift_pct = len(drifted_features) / max(len(common_cols), 1) * 100
        
        if drift_pct == 0:
            report['drift_severity'] = 'none'
        elif drift_pct < 20:
            report['drift_severity'] = 'low'
        elif drift_pct < 50:
            report['drift_severity'] = 'moderate'
        else:
            report['drift_severity'] = 'high'
        
        report['has_drift'] = drift_pct > 20
        
        # Log results
        if report['has_drift']:
            logger.warning(
                f"⚠️ DATA DRIFT DETECTED: {len(drifted_features)}/{len(common_cols)} features drifted "
                f"(severity: {report['drift_severity']})"
            )
            for feat in drifted_features:
                fr = report['feature_reports'][feat]
                logger.warning(f"  - {feat}: PSI={fr['psi']:.4f}, KS p-value={fr['ks_pvalue']:.4f}")
        else:
            logger.info(f"✓ No significant drift detected ({len(drifted_features)}/{len(common_cols)} features)")
        
        return report
    
    def monitor_predictions(self, predictions: np.ndarray, 
                           reference_predictions: np.ndarray = None) -> Dict:
        """
        Monitor prediction distribution for anomalies.
        
        Args:
            predictions: New predictions to monitor
            reference_predictions: Reference predictions (e.g., from training)
            
        Returns:
            Monitoring report
        """
        report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'n_predictions': len(predictions),
            'prediction_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'median': float(np.median(predictions)),
            },
            'anomalies': {},
        }
        
        # Check for anomalous predictions
        n_negative = int(np.sum(predictions < 0))
        n_extreme_low = int(np.sum(predictions < 500))
        n_extreme_high = int(np.sum(predictions > 100000))
        
        report['anomalies'] = {
            'negative_predictions': n_negative,
            'extreme_low_predictions': n_extreme_low,
            'extreme_high_predictions': n_extreme_high,
            'has_anomalies': n_negative > 0 or n_extreme_low > len(predictions) * 0.1,
        }
        
        # Compare with reference predictions if available
        if reference_predictions is not None:
            psi = self.calculate_psi(reference_predictions, predictions)
            report['prediction_drift'] = {
                'psi': round(psi, 4),
                'has_drift': psi > self.psi_threshold,
            }
        
        return report
    
    def save_report(self, report: Dict, save_path: str):
        """Save drift report to JSON file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Drift report saved to: {save_path}")
    
    def load_reference_from_csv(self, csv_path: str):
        """Load reference data from CSV file."""
        self.reference_data = pd.read_csv(csv_path, index_col=0)
        self._compute_reference_stats()
        logger.info(f"Loaded reference data: {self.reference_data.shape}")


if __name__ == "__main__":
    """Example: Check drift on test data vs training data."""
    
    train_path = 'data/processed/X_train.csv'
    test_path = 'data/processed/X_test.csv'
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        detector = DriftDetector()
        detector.load_reference_from_csv(train_path)
        
        test_data = pd.read_csv(test_path, index_col=0)
        report = detector.check_drift(test_data)
        
        detector.save_report(report, 'models/drift_report.json')
        
        print(f"\nDrift Report Summary:")
        print(f"  Features checked: {report['features_checked']}")
        print(f"  Features drifted: {report['features_drifted']}")
        print(f"  Severity: {report['drift_severity']}")
        print(f"  Has drift: {report['has_drift']}")
    else:
        print("Training/test data not found. Run the pipeline first.")
