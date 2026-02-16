# Complete Machine Learning Pipeline - Used Cars Price Prediction

A production-ready, end-to-end machine learning pipeline for predicting used car prices.

## 🎯 Project Overview

This project demonstrates a complete ML workflow from raw data to production-ready models, including:

- **Data Preprocessing** - Cleaning, filtering, and preparing data
- **Feature Engineering** - Creating meaningful features
- **Feature Selection** - Identifying most important features
- **Model Training** - Training multiple algorithms
- **Model Evaluation** - Comprehensive performance analysis
- **Hyperparameter Tuning** - Optimizing model parameters
- **Model Deployment** - Production-ready artifacts

## 📁 Project Structure

```
used-cars-price-prediction/
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data and reports
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing_production.ipynb
│   └── 03_complete_ml_pipeline.ipynb  # ⭐ Main pipeline
├── src/
│   ├── config.py               # Configuration
│   ├── data_validator.py       # Data validation
│   ├── preprocessing_pipeline.py  # Preprocessing
│   ├── feature_engineering.py  # Feature engineering
│   ├── feature_selection.py    # Feature selection
│   ├── model_training.py       # Model training
│   ├── model_evaluation.py     # Model evaluation
│   └── hyperparameter_tuning.py  # Hyperparameter tuning
├── models/                     # Saved models and artifacts
├── logs/                       # Execution logs
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn scipy
```

### 2. Run the Complete Pipeline

You can run the entire end-to-end pipeline (preprocessing, feature engineering, selection, training, and evaluation) with a single command:

```bash
python main.py
```

This script will:
1. Load data from `data/raw/`
2. Process and feature engineer the data
3. Select optimal features
4. Train multiple models (Random Forest, XGBoost, LightGBM, Linear Regression)
5. Evaluate all models and save the best one to `models/`
6. Generate a comprehensive report in `models/evaluation_report.txt`

Alternatively, you can run the interactive notebook:
Open and run `notebooks/03_complete_ml_pipeline.ipynb`.


Or run individual steps:

```python
# Step 1: Preprocessing
from src.preprocessing_pipeline import DataPreprocessor
preprocessor = DataPreprocessor(config)
df_processed = preprocessor.fit_transform(df_raw)

# Step 2: Feature Engineering
from src.feature_engineering import FeatureEngineer
feature_engineer = FeatureEngineer()
df_engineered = feature_engineer.fit_transform(df_processed)

# Step 3: Feature Selection
from src.feature_selection import FeatureSelector
feature_selector = FeatureSelector()
X_selected = feature_selector.fit_transform(X, y, method='ensemble', n_features=50)

# Step 4: Model Training
from src.model_training import ModelTrainer
trainer = ModelTrainer()
models = trainer.train_all_models(X_train, y_train)

# Step 5: Model Evaluation
from src.model_evaluation import ModelEvaluator
evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(predictions, y_test)

# Step 6: Hyperparameter Tuning
from src.hyperparameter_tuning import HyperparameterTuner
tuner = HyperparameterTuner()
best_model, best_params = tuner.tune_xgboost(X_train, y_train)
```

## 📊 Pipeline Steps

### 1. Data Preprocessing

**Module**: `preprocessing_pipeline.py`

- Drop unnecessary columns
- Filter by domain knowledge (price, year, odometer ranges)
- Remove duplicates
- Handle missing values (mode, median, constant)
- Detect and handle outliers (IQR/Z-score)
- Encode categorical variables
- Scale numerical features
- Train-test split

**Key Features**:
- ✅ Configurable strategies
- ✅ Comprehensive validation
- ✅ Detailed logging
- ✅ Reusable pipeline

### 2. Feature Engineering

**Module**: `feature_engineering.py`

Creates new features:
- **Age features**: vehicle_age
- **Price ratios**: price_per_mile, price_per_year
- **Mileage features**: avg_miles_per_year, mileage_category, odometer_log
- **Age categories**: age_category, is_vintage
- **Interaction features**: age_mileage_interaction
- **Boolean features**: is_low_mileage, is_high_mileage, is_recent_model
- **Statistical features**: price_mean_by_manufacturer, price_deviation

**Key Features**:
- ✅ Domain-specific features
- ✅ Automated feature creation
- ✅ Configurable options
- ✅ Transform consistency

### 3. Feature Selection

**Module**: `feature_selection.py`

Multiple selection methods:
- **Variance Threshold**: Remove low-variance features
- **Correlation Filter**: Remove highly correlated features
- **K-Best**: Statistical tests (f_regression)
- **Mutual Information**: Information gain
- **RFE**: Recursive Feature Elimination
- **Lasso**: L1 regularization
- **Random Forest**: Feature importance
- **Ensemble**: Combination of multiple methods

**Key Features**:
- ✅ Multiple algorithms
- ✅ Ensemble approach
- ✅ Feature importance visualization
- ✅ Configurable thresholds

### 4. Model Training

**Module**: `model_training.py`

Supported models:
- **Linear**: Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-based**: Decision Tree, Random Forest, Extra Trees
- **Boosting**: Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost
- **Other**: KNN, SVR
- **Ensemble**: Weighted ensemble of multiple models

**Key Features**:
- ✅ 15+ algorithms
- ✅ Baseline and optimized parameters
- ✅ Parallel training
- ✅ Model persistence

### 5. Model Evaluation

**Module**: `model_evaluation.py`

Metrics:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Explained Variance**: Variance explained by model

Visualizations:
- Predictions vs Actual
- Residuals Analysis
- Model Comparison
- Feature Importance
- Error Distribution

**Key Features**:
- ✅ Comprehensive metrics
- ✅ Rich visualizations
- ✅ Comparison tools
- ✅ Detailed reports

### 6. Hyperparameter Tuning

**Module**: `hyperparameter_tuning.py`

Methods:
- **Grid Search**: Exhaustive search
- **Random Search**: Random sampling (faster)

Supported models:
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

**Key Features**:
- ✅ Automated tuning
- ✅ Cross-validation
- ✅ Parameter grids
- ✅ Best model selection

## 🎓 Usage Examples

### Making Predictions on New Data

```python
import pickle
import pandas as pd

# Load all components
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('models/feature_engineer.pkl', 'rb') as f:
    feature_engineer = pickle.load(f)

with open('models/feature_selector.pkl', 'rb') as f:
    feature_selector = pickle.load(f)

with open('models/xgboost_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

# Load new data
df_new = pd.read_csv('new_cars.csv')

# Apply transformations
df_processed = preprocessor.transform(df_new)
df_engineered = feature_engineer.transform(df_processed)
df_selected = feature_selector.transform(df_engineered)

# Make predictions
predictions = model.predict(df_selected)

# Add predictions to dataframe
df_new['predicted_price'] = predictions
```

## 📚 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

---

**Happy Modeling! 🚀**
