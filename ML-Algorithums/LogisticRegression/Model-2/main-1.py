import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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

df.hist(figsize=(15,10),bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Correlation Analysis

corr= df.corr(numeric_only=True)
sns.heatmap(corr,annot=True ,cbar=False, cmap='coolwarm')
plt.title("Missing Values Heatmap")
plt.show()


# Drop useless columns
df=df.drop(['Unnamed: 32', 'id'], axis=1)

# Convert diagnosis to binary
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

x= df.drop('diagnosis', axis=1)
y= df['diagnosis']

x_train,x_test,y_train,y_Test= train_test_split(x,y,test_size=0.2,random_state=42)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)



def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {acc:.4f}")
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_predict))
    print('Classification Report:\n', classification_report(y_test, y_predict))
    print("-" * 40)
    
models=[
    LogisticRegression(),RandomForestClassifier(random_state=42),SVC(probability=True),KNeighborsClassifier(),GaussianNB()
]
    
for model in models:
    evaluate_model(model,x_train_scaled,y_train,x_test_scaled,y_Test)

