import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = [10,5]


# Load dataset

df = pd.read_csv('./dataset/student-mat.csv')
print(df.head())


# Visualize missing values

sns.heatmap(df.isnull(),cbar=False, cmap='coolwarm')
plt.title("Missing Values Heatmap")
plt.show()


corr =df.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()
