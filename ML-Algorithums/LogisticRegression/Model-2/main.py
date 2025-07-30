import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = [10,5]

# Load dataset
df = pd.read_csv('./data/data.csv')
print('train data', df.shape)
print(df.info())
print(df.shape)
print(df.head())
print(df.describe())

# Visual heatmap


sns.heatmap(df.isnull(),cbar=False, cmap='coolwarm')
plt.title("Missing Values Heatmap")
plt.show()

# Count of missing values

print("df.isnull().sum()",df.isnull().sum())