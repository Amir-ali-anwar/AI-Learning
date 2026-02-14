# Machine Learning Notebooks - Comprehensive Guide

## 📚 Overview
This repository contains comprehensive machine learning notebooks covering supervised and unsupervised learning algorithms with best practices and real-world applications.

## 🎯 Learning Objectives
- Understand fundamental ML algorithms
- Implement proper data preprocessing pipelines
- Prevent data leakage using sklearn pipelines
- Perform hyperparameter tuning
- Evaluate and compare models
- Apply feature engineering techniques

## 📂 Repository Structure

### Supervised Learning
#### Regression
- **Linear Regression**
  - `00_Pipeline_Setup.ipynb` - Proper pipeline implementation to prevent data leakage
  - `01_EDA_and_Preprocessing.ipynb` - Exploratory data analysis
  - `02_Linear_Regression_Sklearn.ipynb` - Linear regression implementation
  - `03_Ridge_Regression.ipynb` - Regularized regression
  - `04_Advanced_Linear_Regression.ipynb` - ⭐ Model comparison, residual analysis, cross-validation

- **Decision Trees & Ensemble Methods**
  - `01_Decision_Tree_Regression.ipynb` - Decision trees and Random Forests
  
- **Gradient Boosting**
  - `01_gradient_boosting_regressor.ipynb` - Basic gradient boosting
  - `02_XGBoost_Complete_Guide.ipynb` - ⭐ XGBoost with hyperparameter tuning
  - `03_LightGBM_Complete_Guide.ipynb` - ⭐ LightGBM with categorical features

- **K-Nearest Neighbors (KNN)**
  - `03_KNN_Complete_Guide.ipynb` - ⭐ KNN for classification and regression

- **Support Vector Machines (SVM)**
  - `SVM_Classification.ipynb` - SVM for classification tasks
  - `SVM_Regression.ipynb` - SVM for regression tasks

### Unsupervised Learning
- **Clustering**
  - `01-k_means_all.ipynb` - Basic K-Means clustering
  - `02-k_means_advanced.ipynb` - ⭐ Advanced K-Means with elbow method and silhouette analysis

### Special Guides
- **`Feature_Engineering_Guide.ipynb`** - ⭐ Complete feature engineering techniques
- **`Model_Deployment_Guide.ipynb`** - ⭐ Production deployment with Flask API

## 🔑 Key Concepts

### Data Leakage Prevention
All notebooks follow best practices using sklearn pipelines:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Always split BEFORE preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Use pipelines to ensure proper preprocessing
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
```

### Model Evaluation
- Cross-validation for robust performance estimates
- Multiple metrics (RMSE, MAE, R²)
- Residual analysis
- Feature importance visualization

### Hyperparameter Tuning
- Grid Search CV
- Random Search CV
- Bayesian Optimization (advanced)

## 🛠️ Prerequisites

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Optional (for advanced features)
```bash
pip install xgboost lightgbm optuna
```

## 📊 Datasets Used
- **California Housing Dataset** - Regression tasks
- **Iris Dataset** - Classification examples
- **Diamonds Dataset** - Clustering and regression
- Custom synthetic datasets for demonstrations

## 🚀 Getting Started

1. **Clone the repository**
2. **Set up virtual environment**
   ```bash
   python -m venv ml_env
   source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Start with the basics**
   - Begin with `00_Pipeline_Setup.ipynb`
   - Follow the numbered sequence in each folder

## 📈 Best Practices Implemented

1. **Data Preprocessing**
   - Handle missing values appropriately
   - Scale features when necessary
   - Encode categorical variables properly

2. **Model Development**
   - Use pipelines to prevent data leakage
   - Implement cross-validation
   - Tune hyperparameters systematically

3. **Model Evaluation**
   - Use multiple metrics
   - Visualize results
   - Compare different models

4. **Code Quality**
   - Clear documentation
   - Reproducible results (set random_state)
   - Modular and reusable code

## 🎓 Learning Path

### Beginner
1. Start with Linear Regression notebooks
2. Understand EDA and preprocessing
3. Learn about pipelines and data leakage

### Intermediate
1. Explore Decision Trees and Random Forests
2. Learn about regularization techniques
3. Implement hyperparameter tuning

### Advanced
1. Master ensemble methods (Gradient Boosting)
2. Implement custom transformers
3. Build end-to-end ML pipelines

## 📝 Notes

- All notebooks include detailed explanations
- Code is well-commented for learning purposes
- Each notebook can be run independently
- Results are reproducible with fixed random states

## 🤝 Contributing
Feel free to add more examples, improve documentation, or suggest enhancements!

## 📧 Contact
For questions or suggestions, please open an issue in the repository.

---
**Last Updated:** January 2026
**Author:** AI Learning Repository
