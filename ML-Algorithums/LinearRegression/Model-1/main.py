import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


x= df.drop(['G3'],axis=1)
y= df['G3']

x= pd.get_dummies(x, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

ridge= Ridge(alpha=1)
ridge.fit(X_train,y_train)
print("Ridge R²:", ridge.score(X_test, y_test))

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
print("Lasso R²:", lasso.score(X_test, y_test))
y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


plt.scatter(y_test,y_pred)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Final Grades")
plt.plot([0, 20], [0, 20], '--', color='red')  # Line y=x for perfect prediction
plt.grid()
plt.show()





