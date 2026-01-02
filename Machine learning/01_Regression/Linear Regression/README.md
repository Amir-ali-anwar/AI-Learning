# Linear Regression - California Housing Dataset

## 1️⃣ Project Overview
This folder contains a **Linear Regression implementation** on the **California Housing dataset**.  
Linear Regression is a **supervised machine learning algorithm** used to predict a **continuous target variable** based on one or more independent features.

**Goal:** Predict the **Median House Value** of California block groups using socio-economic and geographical features.

---

## 2️⃣ Dataset Description
The dataset contains **20,640 observations** with **10 attributes**:

| Feature | Description |
|---------|-------------|
| Longitude | Longitude of block group |
| Latitude | Latitude of block group |
| Housing Median Age | Median age of houses |
| Total Rooms | Total rooms in houses |
| Total Bedrooms | Total bedrooms in houses |
| Population | Total population of block group |
| Households | Total households |
| Median Income | Median income of the block group |
| Median House Value | Median value of houses (**Target**) |
| Ocean Proximity | Categorical: distance to ocean or other water |

**Source:** California Housing dataset

---

## 3️⃣ Folder Structure

Linear_Regression/
├── data/
│   ├── dataset_imputed.csv        # Raw dataset after initial imputation
│   └── dataset_processed.csv      # Preprocessed dataset (scaled & encoded)
├── 01_EDA_and_Preprocessing.ipynb # Exploratory Data Analysis + preprocessing
├── 02_Linear_Regression_Sklearn.ipynb # Linear Regression implementation using scikit-learn
└── README.md

---

## 4️⃣ Preprocessing Steps
- Handled **missing values** in numeric features using median imputation  
- **One-hot encoded** the categorical feature (`Ocean Proximity`)  
- **Scaled numeric features** using StandardScaler  
- Saved the **processed dataset** for consistent training/testing  

> Preprocessing is separated into a dedicated notebook to ensure **reproducibility** and **clean pipeline**.

---

## 5️⃣ Model Implementation

- Linear Regression model implemented using **scikit-learn**  
- **Train-test split**: 80% train, 20% test  
- **Evaluation metrics**:  
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - Mean Absolute Error (MAE)  
  - R² Score  

- **Visualizations included**:  
  - Actual vs Predicted plot  
  - Residuals distribution plot  
  - Residuals vs Predicted plot  

- **Feature Importance:** Coefficients of each feature interpreted to understand impact on house prices  

---

## 6️⃣ Key Insights
- Median Income is the **most influential feature** on house prices  
- Proximity to the ocean affects house value significantly  
- Residual plots show that **errors are mostly random**, indicating a good model fit  

---

