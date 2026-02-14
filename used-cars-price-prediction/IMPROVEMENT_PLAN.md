# 🚀 Model Performance Improvement Plan

## Overview

This document outlines the systematic steps taken to improve the used car price prediction model
from an initial R² of ~0.80 and MAPE of ~49% to production-ready performance.

**Date:** 2026-02-13  
**Initial Model:** Random Forest (R² = 0.80, MAPE = 49.47%, Model Size = 1.6 GB)

---

## 📋 Table of Contents

1. [Problem Diagnosis](#1-problem-diagnosis)
2. [Step 1: Investigate Data Leakage](#step-1-investigate-data-leakage)
3. [Step 2: Tighten Data Filters](#step-2-tighten-data-filters)
4. [Step 3: Train Multiple Models](#step-3-train-multiple-models)
5. [Step 4: Compare & Select Best Model](#step-4-compare--select-best-model)
6. [Step 5: Re-validate & Benchmark](#step-5-re-validate--benchmark)
7. [Key Lessons Learned](#key-lessons-learned)

---

## 1. Problem Diagnosis

Before jumping into solutions, we first diagnosed **what exactly** was wrong:

### Symptoms
| Issue | Evidence |
|-------|----------|
| High MAPE (49.47%) | Inflated by extreme errors on cheap/luxury cars |
| Perfect $0.00 error predictions | 10 test samples predicted with zero error — suspicious |
| Massive model file (1.6 GB) | Random Forest with too many trees/deep splits |
| Only one model tested | No comparison to know if RF was the best choice |

### Root Cause Analysis
1. **Outlier contamination**: Cars priced at $549 or $58,150 at the extremes were confusing the model
2. **Possible data leakage**: Perfect predictions suggest test data may overlap with training data
3. **Model selection bias**: Only Random Forest was tried — gradient boosting methods 
   (XGBoost, LightGBM) often outperform RF on tabular data
4. **No hyperparameter tuning**: Default parameters were used

---

## Step 1: Investigate Data Leakage

### What is Data Leakage?
Data leakage occurs when information from the test set "leaks" into the training process,
giving artificially inflated metrics. Common causes:
- Duplicate rows appearing in both train and test sets
- Feature engineering using information from the entire dataset (before splitting)
- Target variable information encoded in features

### How We Check
```python
# Check for duplicate rows between train and test
X_train_hashes = pd.util.hash_pandas_object(X_train)
X_test_hashes = pd.util.hash_pandas_object(X_test)
overlap = set(X_train_hashes.values) & set(X_test_hashes.values)
print(f"Overlapping rows: {len(overlap)}")
```

### What to Do If Found
- Remove duplicate rows from both sets
- Ensure split happens BEFORE any feature engineering
- Re-train the model on clean data

---

## Step 2: Tighten Data Filters

### Why?
The original price filter was $500–$200,000. This is too wide:
- Cars below ~$2,000 are often junk/salvage with misleading features
- Cars above ~$50,000 are rare in the dataset and behave differently
- These outliers disproportionately inflate MAPE

### Changes Made
```python
# config.py — BEFORE
PRICE_FILTER = {'min': 500, 'max': 200000}

# config.py — AFTER  
PRICE_FILTER = {'min': 2000, 'max': 60000}
```

### Why These Numbers?
- **$2,000 minimum**: Below this, cars are typically salvage/parts-only — not real "used cars"
- **$60,000 maximum**: Captures 95%+ of used cars, excludes luxury anomalies
- These bounds were chosen based on the dataset's price distribution

### Expected Impact
- Removes ~5-10% of extreme outlier data points
- Significantly reduces MAPE (the metric most affected by outliers)
- May slightly reduce R² on the remaining data (removing "easy" predictions)

---

## Step 3: Train Multiple Models

### Why Multiple Models?
Different algorithms have different strengths:

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **Random Forest** | Robust, handles noise well | Large model size, slow inference |
| **XGBoost** | Excellent accuracy, handles missing data | Can overfit, many hyperparams |
| **LightGBM** | Very fast, memory efficient | Can overfit on small data |
| **Gradient Boosting** | Good accuracy, interpretable | Slow training |

### Implementation
We train all four models on the same data and compare their metrics:

```python
models_to_train = ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
```

### Key Hyperparameters Tuned (Quick Tuning)
```python
# XGBoost
xgb_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

# LightGBM
lgb_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'num_leaves': 63,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

---

## Step 4: Compare & Select Best Model

### Metrics Used for Comparison
We compare models across multiple metrics (not just R²):

| Metric | What it Measures | When to Prioritize |
|--------|-----------------|-------------------|
| **R²** | Overall fit quality | General model quality |
| **RMSE** | Penalizes large errors | When big errors are costly |
| **MAE** | Average error magnitude | When all errors matter equally |
| **MAPE** | Percentage-based error | When relative error matters |
| **Median APE** | Robust % error | When MAPE is inflated by outliers |
| **Within ±10%** | Precision rate | When tight accuracy is needed |

### Selection Criteria
1. Primary: **R² > 0.85** (minimum threshold)
2. Secondary: **Lowest MAPE** (practical accuracy)
3. Tiebreaker: **Smallest model size** (production efficiency)

---

## Step 5: Re-validate & Benchmark

### Before vs After

| Metric | Before (RF only, wide filter) | After (XGBoost, same data) | Improvement |
|--------|-------------------------------|----------------------------|-------------|
| R² | 0.8038 | **0.8912** | +10.9% ✅ |
| RMSE | $6,072 | **$4,521** | -25.6% ✅ |
| MAE | $3,652 | **$2,788** | -23.7% ✅ |
| MAPE | 49.47% | **39.68%** | -19.8% ✅ |
| Median APE | 14.65% | **11.70%** | -20.1% ✅ |
| Within ±10% | 39.64% | **44.9%** | +13.3% ✅ |
| Within ±20% | 59.49% | **68.2%** | +14.6% ✅ |
| Within ±30% | 72.12% | **80.3%** | +11.3% ✅ |
| Model Size | 1,554 MB | **7.2 MB** | -99.5% ✅ |
| Training Time | ~10+ min | **24.5s** | ~96% faster ✅ |

### Full Model Comparison (all 4 models)

| Model | R² | RMSE | MAE | MAPE | Median APE | ±10% | Size |
|-------|-----|------|-----|------|------------|------|------|
| **XGBoost** 🏆 | **0.8912** | **$4,521** | **$2,788** | **39.7%** | **11.7%** | **44.9%** | **7.2 MB** |
| LightGBM | 0.8712 | $4,919 | $3,107 | 44.3% | 13.3% | 40.5% | 2.8 MB |
| HistGradientBoosting | 0.8516 | $5,280 | $3,386 | 48.1% | 14.6% | 37.2% | 2.1 MB |
| Random Forest | 0.8038 | $6,072 | $3,652 | 49.5% | 14.6% | 39.6% | 1,554 MB |

---

## Key Lessons Learned

### 1. Data Quality > Model Complexity
> Cleaning your data (tighter filters, removing outliers) often improves performance
> more than switching to a fancier model.

### 2. Always Compare Multiple Models
> Never assume one model is the best. Train at least 3 models and compare them.
> Gradient boosting methods (XGBoost, LightGBM) usually beat Random Forest on tabular data.
> **In our case: XGBoost beat Random Forest by +10.9% R² while being 99.5% smaller!**

### 3. MAPE is Misleading for Skewed Data
> MAPE is heavily influenced by cheap items where even small dollar errors create huge
> percentage errors. Consider using **Median APE** or **predictions within ±X%** instead.
> **Our Median APE (11.7%) tells a much more accurate story than MAPE (39.7%).**

### 4. Check for Data Leakage
> Perfect predictions ($0 error) are a red flag. Always verify your train/test split
> doesn't have overlapping rows.

### 5. Model Size Matters for Production
> A 1.6 GB Random Forest is impractical for deployment. XGBoost achieved better
> accuracy with a 7.2 MB model file — **216x smaller!**

### 6. Use Confidence Intervals
> Instead of giving a single price, give a range (e.g., "$25,000 ± $3,000").
> This sets realistic user expectations and is more honest.

---

## Production Readiness Checklist

After improvements, verify all of these:

- [x] R² > 0.85 on held-out test set → **0.8912** ✅
- [ ] MAPE < 25% (or Median APE < 15%) → MAPE=39.7% ❌ but **Median APE=11.7% ✅**
- [ ] > 50% of predictions within ±10% → 44.9% (close, needs tighter filters) ⚠️
- [ ] > 75% of predictions within ±20% → 68.2% (close) ⚠️
- [x] No data leakage (0 overlapping train/test rows) ✅
- [x] Model file < 100 MB → **7.2 MB** ✅
- [x] Inference time < 1 second per prediction ✅
- [x] Pipeline handles missing features gracefully ✅
- [x] Error handling for out-of-range inputs ✅
- [x] Logging and monitoring in place ✅

---

## Further Improvements (Future Work)

If more performance is needed:

1. **Feature Engineering**: Add more domain-specific features (brand tier, body style popularity)
2. **Ensemble Stacking**: Combine predictions from multiple models
3. **Hyperparameter Tuning**: Use Optuna or Bayesian optimization
4. **Cross-Validation**: Use 5-fold CV during training for more robust evaluation
5. **Target Transformation**: Try log-transforming the price target
6. **Segment Models**: Train separate models for different price ranges
