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
df = pd.read_csv('./data/train.csv')
print('train data', df.shape)
print(df.info())

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap='coolwarm')
plt.title("Missing Values Heatmap")
plt.show()

# Age distribution by Pclass
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=df, palette='GnBu_d').set_title('Age by Passenger Class')
plt.show()

# Impute Age based on Pclass
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

df['Age'] = df[['Age','Pclass']].apply(impute_age, axis=1)

# Drop unwanted columns
df.drop('Cabin', axis=1, inplace=True)
df.dropna(inplace=True)
df.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True)

# Convert categorical columns
objcat = ['Sex','Embarked']
for colName in objcat:
    df[colName] = df[colName].astype('category')

# One-hot encoding
sex = pd.get_dummies(df['Sex'], drop_first=True)
embarked = pd.get_dummies(df['Embarked'], drop_first=True)

# Concatenate encoded columns and drop originals
df = pd.concat([df.drop(['Sex','Embarked'], axis=1), sex, embarked], axis=1)

# ✅ Ensure column names match
# Final feature list (must match DataFrame after encoding)
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']
X = df[features]
y = df['Survived']  # ✅ Fix column name: it's 'Survived' with capital S in Titanic dataset

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
