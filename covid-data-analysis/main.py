import pandas as pd
import numpy as np


data= pd.read_csv('../Datasets/covid/covid_19_data.csv')
# print(data.head())
# print(data.shape)
# print(data.info())
# print(data.describe())

# print(type(data))
# print(data.isnull().sum())
# print(data)
# data['Province/State'] = data.groupby('Country/Region')['Province/State'].ffill().bfill()
# print(data)

data['ObservationDate'] = pd.to_datetime(data['ObservationDate'])
print(data.isnull().sum())
data = data.dropna()  # or fillna if appropriate

country_confirmed= data.groupby('Country/Region')['Confirmed'].sum().sort_values(ascending=False)

print(country_confirmed.head(5))

