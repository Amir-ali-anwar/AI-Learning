import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"

# Load dataset
df = pd.read_excel(url)

# Show first 5 rows
print(df.head())