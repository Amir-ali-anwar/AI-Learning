import pandas as pd
# import numpy as np

data = pd.read_csv('../Datasets/covid/covid_19_data.csv')

print(data.head())         
print(data.shape)         
print(data.columns)        
print(data.info())        
print(data.describe())  

print(data.isnull().sum())

data['Province/State'] = data.groupby('Country/Region')['Province/State'].ffill().bfill()

data = data.dropna()

data.columns = [col.strip().replace('/', '_').replace(' ', '_') for col in data.columns]

data['ObservationDate'] = pd.to_datetime(data['ObservationDate'])

print(data.isnull().sum())

country_confirmed = data.groupby('Country_Region')['Confirmed'].sum().sort_values(ascending=False)

print(country_confirmed.head(5))
