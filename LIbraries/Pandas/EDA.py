import pandas as pd

df= pd.read_csv('data.csv')

df= pd.DataFrame(df)

df['Date']= pd.to_datetime(df['Date'], format="mixed")

df.dropna(subset=['Date','Calories'],inplace=True)
print(df.to_string())